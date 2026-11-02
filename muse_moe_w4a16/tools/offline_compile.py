"""Compile submission Triton kernels for H200 without a GPU or PyTorch.

Run from the problem folder with .tooling/venv/bin/python tools/offline_compile.py.
This checks compilation only; it does not establish numerical correctness.
"""
import ast
import importlib.util
from pathlib import Path
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

root = Path(__file__).resolve().parents[1]
source = root / (sys.argv[1] if len(sys.argv) > 1 else "attempts/attempt_03.py")
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and any(
    isinstance(d, ast.Attribute) and d.attr == "jit" for d in n.decorator_list)]
module_path = root / ".tooling/offline_kernels.py"
module_path.write_text("import triton\nimport triton.language as tl\n" +
                       "\n\n".join(ast.unparse(n) for n in nodes))
spec = importlib.util.spec_from_file_location("offline_kernels", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
target = GPUTarget("cuda", 90, 32)

if hasattr(module, "_align_local_routes"):
    fn = module._align_local_routes
    signature = {name: "*i32" for name in ["Ids", "Local", "Sorted", "Experts", "Total"]}
    for m in (1, 4, 8, 16, 32, 33, 128, 256, 257):
        n = m * 8
        bm = 8 if m <= 32 else (int(sys.argv[2]) if len(sys.argv) > 2 else 16)
        constants = dict(N=n, START=0, E=32, BM=bm,
                         NP=triton.next_power_of_2(n), NE=32,
                         NB=triton.next_power_of_2(triton.cdiv(n, bm)))
        result = triton.compile(ASTSource(fn, signature, constants), target=target,
                                options={"num_warps": 4})
        print(f"align M={m} BM={bm}: compiled, shared={result.metadata.shared}", flush=True)

if hasattr(module, "_align_mixed_routes"):
    fn = module._align_mixed_routes
    small = 16 if "S16" in fn.arg_names else 8
    large = 64 if "S64" in fn.arg_names else 32
    cut = 32 if large == 64 else 16
    names = ["Ids", "Local", f"S{small}", f"E{small}", f"T{small}",
             f"S{large}", f"E{large}", f"T{large}"]
    signature = {name: "*i32" for name in names}
    for m in (33, 65, 128, 256, 257):
        n = m * 8
        constants = dict(N=n, START=0, E=32, CUT=cut,
                         NP=triton.next_power_of_2(n), NE=32)
        constants[f"NB{small}"] = triton.next_power_of_2(triton.cdiv(n, small))
        constants[f"NB{large}"] = triton.next_power_of_2(triton.cdiv(n, large))
        result = triton.compile(ASTSource(fn, signature, constants),
                                target=target, options={"num_warps": 4})
        print(f"mixed align M={m} blocks={small}/{large}: compiled, shared={result.metadata.shared}", flush=True)

if hasattr(module, "_post_norm_sum"):
    signature = dict(Y="*bf16", Scores="*fp32", Norm="*bf16", Ids="*i32", Out="*bf16")
    result = triton.compile(ASTSource(module._post_norm_sum, signature,
                                      dict(H=4096, K=8, B=4096)), target=target,
                            options={"num_warps": 8, "enable_fp_fusion": False})
    print(f"post norm: compiled, shared={result.metadata.shared}", flush=True)
