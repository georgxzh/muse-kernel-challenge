"""Compile the skinny NVFP4 MoE kernel for SM90 without a GPU; report ptxas registers/spills.

Usage: <Triton 3.5.1 python> -B tools/compile_skinny.py attempts/attempt_XX.py
Code generation only; okc is the correctness and timing authority.
"""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
work = root / ".tooling"
work.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(work / "triton-cache-skinny")
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

source = (root / sys.argv[1]).read_text()
tree = ast.parse(source)
parts = ["import triton", "import triton.language as tl"]
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("_DECODE") for t in node.targets):
        parts.append(ast.get_source_segment(source, node))
    if isinstance(node, ast.FunctionDef) and any(ast.unparse(d) == "triton.jit" for d in node.decorator_list):
        parts.append("@triton.jit\n" + ast.get_source_segment(source, node))
path = work / "offline_skinny.py"
path.write_text("\n\n".join(parts) + "\n")
spec = importlib.util.spec_from_file_location("offline_skinny", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
ptxas = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
print("Triton", triton.__version__, flush=True)
consts = {name: getattr(module_src := None, "x", None) for name in ()}
glob = {}
exec(compile("\n".join(l for l in source.splitlines() if l.startswith("SKINNY_")), "c", "exec"), glob)
bn, bkp, warps = glob["SKINNY_BN"], glob["SKINNY_BKP"], glob.get("SKINNY_WARPS", 4)
sig = dict(X="*bf16", W="*u8", SF="*fp8e4nv", GS="*fp32", Sorted="*i32", Experts="*i32", Total="*i32", Out="*bf16")
for label, n, k, div in (("gemm1 w13", 8192, 4096, 8), ("gemm2 w2", 4096, 4096, 1)):
    for m in (1, 32):
        c = dict(N=n, K=k, ROUTES=m * 8, ROW_DIV=div, BM=16, BN=bn, BKP=bkp)
        kern = triton.compile(ASTSource(module._nvfp4_moe_mm, sig, c), target=GPUTarget("cuda", 90, 32),
                              options={"num_warps": warps})
        ptx = work / f"skinny_{n}_{m}.ptx"
        ptx.write_text(kern.asm["ptx"])
        r = subprocess.run([str(ptxas), "-v", "--gpu-name=sm_90a", str(ptx), "-o", str(ptx.with_suffix(".cubin"))],
                           capture_output=True, text=True)
        info = " | ".join(l.split("info    : ")[-1] for l in r.stderr.splitlines() if "registers" in l or "spill" in l)
        mma = kern.asm["ptx"].count("mma.sync") + kern.asm["ptx"].count("wgmma.mma_async")
        print(f"{label} M={m} BN={bn} BKP={bkp} warps={warps}: shared={kern.metadata.shared} mma_instr={mma} :: {info}", flush=True)
