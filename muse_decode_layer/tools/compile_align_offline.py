"""Compile a decode candidate's _align_local_routes for H200 (SM90) without a GPU.

Code generation only; okc remains the correctness and timing authority.
Usage: <Triton 3.5.1 python> -B tools/compile_align_offline.py candidate.py
Caches stay in this problem's .tools/ (gitignored).
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
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

source = Path(sys.argv[1]).read_text()
parts = ["import triton", "import triton.language as tl"]
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and any(ast.unparse(d) == "triton.jit" for d in node.decorator_list):
        parts.append("@triton.jit\n" + ast.get_source_segment(source, node))
target_file = work / "offline_kernels.py"
target_file.write_text("\n\n".join(parts) + "\n")
spec = importlib.util.spec_from_file_location("decode_offline", target_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("Triton", triton.__version__, flush=True)

# Every timed/test token count: decode T=1,16,32 and verify T=8,128,256, plus
# boundary counts; route tile M matches the candidate's block_size_m rule.
for t in (1, 2, 8, 16, 17, 32, 64, 128, 129, 256):
    bm = 8 if t <= 16 else (16 if t <= 128 else 32)
    n = t * 8
    const = dict(N=n, START=0, E=32, BM=bm, NP=triton.next_power_of_2(n), NE=32,
                 NB=triton.next_power_of_2(triton.cdiv(n, bm)))
    src = ASTSource(module._align_local_routes,
                    signature=dict(Ids="*i32", Local="*i32", Sorted="*i32", Experts="*i32", Total="*i32"),
                    constexprs=const)
    k = triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": 8})
    print("_align_local_routes T=%d" % t, const, "PASS, shared bytes:", k.metadata.shared, flush=True)
