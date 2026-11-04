"""GPU-free SM90 compile of the fused attention kernel (_attention_fused) for each launch class.
Usage: .tools/venv/bin/python -B tools/compile_fused.py candidate.py"""
import importlib.util, os, sys, types, re, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
os.environ["TRITON_CACHE_DIR"] = str(root / ".tools" / "triton-cache")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
sys.modules["torch"] = types.ModuleType("torch")
sys.modules["sgl_kernel"] = types.ModuleType("sgl_kernel")
fa = types.ModuleType("sgl_kernel.flash_attn"); fa.flash_attn_with_kvcache = None
sys.modules[fa.__name__] = fa
spec = importlib.util.spec_from_file_location("cand", sys.argv[1]); mod = importlib.util.module_from_spec(spec)
sys.modules["cand"] = mod; spec.loader.exec_module(mod)
d = root / ".tools" / "inspect"; d.mkdir(parents=True, exist_ok=True)
ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
fn = mod._attention_fused
sig = dict(QKV="*bf16", POS="*i64", LOC="*i64", KC="*fp8e4nv", VC="*fp8e4nv", PT="*i32", LENS="*i32",
           KS="*fp32", VS="*fp32", PART="*fp32", LSE="*fp32")
attrs = {(fn.arg_names.index(a),): [["tt.divisibility", 16]] for a in sig}
cases = [  # (L, WINDOW, THETA, PAGE, CHUNK, SPLITS)
    (1, 2048, 500000.0, 1, 128, 16), (1, 2048, 500000.0, 1, 256, 8), (8, 2048, 500000.0, 1, 128, 17),
    (8, 2048, 500000.0, 64, 256, 9), (1, 0, 0.0, 1, 2048, 43), (1, 0, 0.0, 1, 256, 336),
    (8, 0, 0.0, 1, 256, 336), (8, 0, 0.0, 64, 2048, 43)]
bad = 0
for L, W, TH, PAGE, CHUNK, SPLITS in cases:
    const = dict(HQ=16, HK=2, D=64, L=L, LP=max(2, triton.next_power_of_2(L)), PAGE=PAGE, PT_STRIDE=86009,
                 SPLITS=SPLITS, CHUNK=CHUNK, WINDOW=W, SCALE=0.125, M=max(16, L * 8), N=128, STRIDE=2304, THETA=TH)
    k = triton.compile(ASTSource(fn, signature=sig, constexprs=const, attrs=attrs), target=GPUTarget("cuda", 90, 32),
                       options=dict(num_warps=4, num_stages=2, enable_fp_fusion=False))
    (d / "fused.ptx").write_text(k.asm["ptx"])
    rep = subprocess.run([str(ptxas), "-v", "--gpu-name=sm_90a", str(d / "fused.ptx"), "-o", str(d / "fused.cubin")],
                         text=True, capture_output=True, check=True).stderr
    sass = k.asm["sass"].splitlines()
    # Order check on the PTX (nvdisasm output can be cut short for large kernels).
    ptx = k.asm["ptx"].splitlines()
    trig = [i for i, l in enumerate(ptx) if "griddepcontrol.launch_dependents" in l]
    stores = [i for i, l in enumerate(ptx) if re.search(r"\bst\.global", l)]
    spills = sum(int(x) for x in re.findall(r"(\d+) bytes spill", rep))
    ok = trig and all(trig[0] > i for i in stores) and spills == 0
    bad += not ok
    print("PASS" if ok else "FAIL", {k2: const[k2] for k2 in ("L", "WINDOW", "PAGE", "CHUNK", "M")}, "shared", k.metadata.shared,
          re.search(r"Used \d+ registers", rep).group(0), "spills", spills, "bar", sum("BAR.SYNC" in l for l in sass))
print("FAILURES:", bad)
