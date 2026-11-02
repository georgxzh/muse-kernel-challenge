"""Compile submitted Triton kernels for H200 without a GPU.

Only checks code generation; okc remains the correctness and timing authority.
Usage: <a Triton-enabled Python> -B tools/compile_offline.py [candidate.py]
All generated source and compiler caches stay inside this problem's .tools/.
"""
import ast
import importlib.util
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
work = root / ".tools"
work.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(work / "triton-cache")
os.environ["TRITON_DUMP_DIR"] = str(work / "triton-dump")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "submission.py"
source = path.read_text()
tree = ast.parse(source)
parts = ["import triton", "import triton.language as tl"]
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and any(ast.unparse(d) == "triton.jit" for d in node.decorator_list):
        parts.append("@triton.jit\n" + ast.get_source_segment(source, node))
target_file = work / "offline_kernels.py"
target_file.write_text("\n\n".join(parts) + "\n")
spec = importlib.util.spec_from_file_location("prefill_offline", target_file)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
target = GPUTarget("cuda", 90, 32)
print("Triton", triton.__version__, flush=True)

def check(name, signature, constants, warps):
    src = ASTSource(getattr(module, name), signature=signature, constexprs=constants)
    kernel = triton.compile(src, target=target, options={"num_warps": warps})
    print(name, constants, "PASS, shared bytes:", kernel.metadata.shared, flush=True)

check("_expand_nvfp4", dict(W="*u8", SF="*fp8e4nv", GS="*fp32", OUT="*bf16"),
      dict(NK=8192*4096, BLOCK=2048), 4)
check("_pack_grouped", dict(X="*bf16", SORT="*i32", EXP="*i32", NP="*i32", PACK="*bf16", LAYOUT="*i32", MAP="*i32"),
      dict(CAP=36960, M=36992, K=4096, SLOTS=32768), 4)
check("_grouped_silu", dict(H="*bf16", I="*bf16", LAYOUT="*i32"), dict(D=4096), 4)
check("_grouped_finalize", dict(Y="*bf16", WEIGHTS="*fp32", IDS="*i32", MAP="*i32", NORM="*bf16", OUT="*bf16"),
      dict(D=4096), 8)
for rope in (False, True):
    for scaled in (False, True):
        check("_qk_cache", dict(QKV="*bf16", Q="*fp8e4nv", KC="*fp8e4nv", VC="*fp8e4nv", LOC="*i64", POS="*i64", CS="*fp32", KS="*fp32", VS="*fp32"),
              dict(T=4096, STRIDE=2304, ROPE=rope, SCALED=scaled), 4)
