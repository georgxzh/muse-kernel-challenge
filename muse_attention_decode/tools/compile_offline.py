"""GPU-free SM90 compilation check. Run with the problem-local Triton environment.

This checks code generation only, never correctness or performance. No judge modules
are imported. The torch/FA3 stubs only let us load the submitted Triton definitions.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
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
    constants = dict(constants)
    if name == "_attention_parts" and hasattr(module, "ATTENTION_M"):
        constants["M"] = min(constants["M"], module.ATTENTION_M)
    if "QGROUPS" in fn.arg_names:
        constants["QGROUPS"] = triton.cdiv(constants["L"] * (constants["HQ"] // constants["HK"]), constants["M"])
    if name == "_attention_parts" and len(sys.argv) > 2:
        options["num_stages"] = int(sys.argv[2])
    # CUDA allocator tensors passed to the real JIT have 16-byte aligned bases.
    attrs = {(fn.arg_names.index(arg),): [["tt.divisibility", 16]]
             for arg, dtype in signature.items() if dtype.startswith("*")}
    source = ASTSource(fn, signature=signature, constexprs=constants, attrs=attrs)
    kernel = triton.compile(source, target=target, options=options)
    print(name, constants, "shared", kernel.metadata.shared, flush=True)
    report_dir = root / ".tools" / "resource-reports"
    report_dir.mkdir(exist_ok=True)
    label = name + "-" + "-".join(str(constants.get(k, 0)) for k in ("M", "N", "S", "WINDOW")) + "-w" + str(options.get("num_warps")) + "-st" + str(options.get("num_stages", 0))
    ptx_path = report_dir / (label + ".ptx")
    ptx_path.write_text(kernel.asm["ptx"])
    ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
    report = subprocess.run([str(ptxas), "-v", "--gpu-name=sm_90a", str(ptx_path),
                             "-o", str(report_dir / (label + ".cubin"))],
                            text=True, capture_output=True, check=True)
    print(report.stderr, flush=True)


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
                         SPLITS=42, CHUNK=getattr(module, "ATTENTION_CHUNK", 2048), **({"WINDOW": 0} if "WINDOW" in module._attention_parts.arg_names else {}), SCALE=0.6841258107984375, M=max(16, length * 8), N=getattr(module, "ATTENTION_N", 64)),
                    num_warps=getattr(module, "ATTENTION_WARPS", 4), num_stages=getattr(module, "ATTENTION_STAGES", 2))
    for length, splits in ((1, 42), (8, 513)):
        compile_one("_combine_gate", dict(PART="*fp32", LSE="*fp32", QKV="*bf16", O="*bf16"),
                    dict(HQ=16, HK=2, D=64, L=length, STRIDE=2304, SPLITS=splits,
                         S=triton.next_power_of_2(splits)),
                    num_warps=8 if splits >= 128 else 4)

if hasattr(module, "_attention_parts") and "WINDOW" in module._attention_parts.arg_names:
    compile_one("_attention_parts", dict(Q="*fp8e4nv", KC="*fp8e4nv", VC="*fp8e4nv", PT="*i32",
                                        LENS="*i32", KS="*fp32", VS="*fp32", PART="*fp32", LSE="*fp32"),
                dict(HQ=16, HK=2, D=64, L=8, PAGE=1, PT_STRIDE=86008,
                     SPLITS=17, CHUNK=128, WINDOW=2048, SCALE=0.6841258107984375, M=64, N=getattr(module, "ATTENTION_N", 64)),
                num_warps=getattr(module, "ATTENTION_WARPS", 4), num_stages=getattr(module, "ATTENTION_STAGES", 2))
