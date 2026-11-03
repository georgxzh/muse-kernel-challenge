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
fn = mod._attention_parts
sig = dict(Q="*fp8e4nv", KC="*fp8e4nv", VC="*fp8e4nv", PT="*i32", LENS="*i32", KS="*fp32", VS="*fp32", PART="*fp32", LSE="*fp32")
L = int(sys.argv[3]) if len(sys.argv) > 3 else 1
const = dict(HQ=16, HK=2, D=64, L=L, PAGE=1, PT_STRIDE=86008, SPLITS=43, CHUNK=2048, WINDOW=0,
             SCALE=0.125, M=max(16, L * 8), N=128)
for extra in sys.argv[4:]:
    k, v = extra.split("="); const[k] = int(v)
attrs = {(fn.arg_names.index(a),): [["tt.divisibility", 16]] for a in sig}
stages = int(sys.argv[2])
k = triton.compile(ASTSource(fn, signature=sig, constexprs=const, attrs=attrs), target=GPUTarget("cuda", 90, 32),
                   options=dict(num_warps=4, num_stages=stages))
out = Path(sys.argv[1]).stem + f"_st{stages}_L{L}"
d = root / ".tools" / "inspect"; d.mkdir(parents=True, exist_ok=True)
(d / (out + ".ttgir")).write_text(k.asm["ttgir"]); (d / (out + ".sass")).write_text(k.asm["sass"])
(d / (out + ".ptx")).write_text(k.asm["ptx"])
ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
rep = subprocess.run([str(ptxas), "-v", "--gpu-name=sm_90a", str(d / (out + ".ptx")), "-o", str(d / (out + ".cubin"))], text=True, capture_output=True).stderr
print(out, "shared", k.metadata.shared, re.findall(r"Used \d+ registers.*", rep))
sass = k.asm["sass"].splitlines()
ops = [(i, l.split(";")[0].strip()) for i, l in enumerate(sass) if re.search(r"\b(LDG|LDGSTS|BRA|HGMMA|QGMMA|HMMA|STG|BAR|DEPBAR|ACQBULK|PREEXIT|WARPGROUP)\b", l)]
for i, l in ops: print(i, l[:110])
