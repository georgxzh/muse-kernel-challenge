import importlib.util, os, sys, types, re, subprocess
from pathlib import Path
root = Path("/home/georg/muse-kernel-challenge/muse_attention_decode")
os.environ["TRITON_CACHE_DIR"] = str(root / ".tools" / "triton-cache")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
sys.modules["torch"] = types.ModuleType("torch")
sys.modules["sgl_kernel"] = types.ModuleType("sgl_kernel")
fa = types.ModuleType("sgl_kernel.flash_attn"); fa.flash_attn_with_kvcache = None
sys.modules[fa.__name__] = fa
path = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("cand", path); mod = importlib.util.module_from_spec(spec)
sys.modules["cand"] = mod; spec.loader.exec_module(mod)
fn = mod._attention_pair
sig = dict(Q="*fp8e4nv", KC="*fp8e4nv", VC="*fp8e4nv", PT="*i32", LENS="*i32", KS="*fp32", VS="*fp32", PART="*fp32", LSE="*fp32")
attrs = {(fn.arg_names.index(a),): [["tt.divisibility", 16]] for a in sig}
d = root / ".tools" / "inspect"; d.mkdir(parents=True, exist_ok=True)
ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
for page, splits, chunk in ((1, 43, 2048), (1, 336, 256), (64, 43, 2048)):
    const = dict(HQ=16, HK=2, D=64, PAGE=page, PT_STRIDE=86008, SPLITS=splits, CHUNK=chunk, SCALE=0.125, N=mod.PAIR_N)
    k = triton.compile(ASTSource(fn, signature=sig, constexprs=const, attrs=attrs), target=GPUTarget("cuda", 90, 32),
                       options=dict(num_warps=int(os.environ.get("WARPS", 4)), num_stages=int(os.environ.get("STAGES", 2))))
    (d / "pair.ptx").write_text(k.asm["ptx"])
    rep = subprocess.run([str(ptxas), "-v", "--gpu-name=sm_90a", str(d / "pair.ptx"), "-o", str(d / "pair.cubin")], text=True, capture_output=True, check=True).stderr
    sass = k.asm["sass"].splitlines()
    idx = lambda pat: [i for i, l in enumerate(sass) if re.search(pat, l)]
    wait, trig, loads, stores = idx(r"\bACQBULK\b"), idx(r"\bPREEXIT\b"), idx(r"\b(LDG|LDGSTS)\b"), idx(r"\bSTG\b")
    ok = wait and trig and all(wait[0] < i for i in loads + stores) and all(trig[0] > i for i in stores)
    spills = sum(int(x) for x in re.findall(r"(\d+) bytes spill", rep))
    mma = len(idx(r"MMA"))
    print("PASS" if ok and spills == 0 else "FAIL", "_attention_pair", const, "shared", k.metadata.shared,
          re.search(r"Used \d+ registers", rep).group(0), "spills", spills, "mma_instr", mma,
          "ACQBULK@", wait[:1], "PREEXIT@", trig[:1], "first_load@", loads[:1], "last_store@", stores[-1:])
