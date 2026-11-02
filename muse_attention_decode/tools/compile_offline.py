"""GPU-free SM90 compilation check. Run with the problem-local Triton environment.

This checks code generation only, never correctness or performance. No judge modules
are imported. The torch/FA3 stubs only let us load the submitted Triton definitions.
"""
import importlib.util
import os
from pathlib import Path
import sys
import types

root = Path(__file__).resolve().parents[1]
os.environ["TRITON_CACHE_DIR"] = str(root / ".tools" / "triton-cache")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

sys.modules["torch"] = types.ModuleType("torch")
sys.modules["sgl_kernel"] = types.ModuleType("sgl_kernel")
fa = types.ModuleType("sgl_kernel.flash_attn")
fa.flash_attn_with_kvcache = None
sys.modules[fa.__name__] = fa
path = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("attention_candidate", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
target = GPUTarget("cuda", 90, 32)


def compile_one(name, signature, constants, **options):
    fn = getattr(module, name)
    source = ASTSource(fn, signature=signature, constexprs=constants)
    kernel = triton.compile(source, target=target, options=options)
    print(name, constants, "shared", kernel.metadata.shared, flush=True)


for theta in (0., 500000.):
    compile_one("_prepare", dict(QKV="*bf16", POS="*i64", LOC="*i64", KS="*fp32", VS="*fp32",
                                KC="*fp8e4nv", VC="*fp8e4nv", Q="*fp8e4nv"),
                dict(HQ=16, HK=2, D=64, STRIDE=2304, THETA=theta, **({"TOKENS": 256, "BLOCK": 4} if "TOKENS" in module._prepare.arg_names else {})),
                num_warps=4, enable_fp_fusion=False)
if hasattr(module, "_attention_parts"):
    for length, page in ((1, 1), (8, 64)):
        compile_one("_attention_parts", dict(Q="*fp8e4nv", KC="*fp8e4nv", VC="*fp8e4nv", PT="*i32",
                                            LENS="*i32", KS="*fp32", VS="*fp32", PART="*fp32", LSE="*fp32"),
                    dict(HQ=16, HK=2, D=64, L=length, PAGE=page, PT_STRIDE=86008,
                         SPLITS=42, CHUNK=2048, **({"WINDOW": 0} if "WINDOW" in module._attention_parts.arg_names else {}), SCALE=0.6841258107984375, M=max(16, length * 8), N=64),
                    num_warps=4, num_stages=2)
    for length, splits in ((1, 42), (8, 513)):
        compile_one("_combine_gate", dict(PART="*fp32", LSE="*fp32", QKV="*bf16", O="*bf16"),
                    dict(HQ=16, HK=2, D=64, L=length, STRIDE=2304, SPLITS=splits,
                         S=triton.next_power_of_2(splits)),
                    num_warps=8 if splits >= 128 else 4)

if hasattr(module, "_attention_parts") and "WINDOW" in module._attention_parts.arg_names:
    compile_one("_attention_parts", dict(Q="*fp8e4nv", KC="*fp8e4nv", VC="*fp8e4nv", PT="*i32",
                                        LENS="*i32", KS="*fp32", VS="*fp32", PART="*fp32", LSE="*fp32"),
                dict(HQ=16, HK=2, D=64, L=8, PAGE=1, PT_STRIDE=86008,
                     SPLITS=17, CHUNK=128, WINDOW=2048, SCALE=0.6841258107984375, M=64, N=64),
                num_warps=4, num_stages=2)
