"""Compile the fused NVFP4 grouped GEMM for SM90 offline; report registers, spills and SASS mix.

Usage: <Triton 3.5.1 python> -B tools/compile_fused.py attemptNN.py
Code generation only; okc is the correctness and timing authority.
"""
import ast
import collections
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
work = root / ".tools"
work.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(work / "triton-cache-fused")
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

source = (root / sys.argv[1]).read_text()
tree = ast.parse(source)
parts = ["import triton", "import triton.language as tl"]
consts = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name.startswith("_DECODE"):
            parts.append(ast.get_source_segment(source, node))
        if name.startswith("FUSED_"):
            consts[name] = ast.literal_eval(node.value)
    if isinstance(node, ast.FunctionDef) and any(ast.unparse(d) == "triton.jit" for d in node.decorator_list):
        parts.append("@triton.jit\n" + ast.get_source_segment(source, node))
path = work / "offline_fused.py"
path.write_text("\n\n".join(parts) + "\n")
spec = importlib.util.spec_from_file_location("offline_fused", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bin_dir = Path(triton.__file__).parent / "backends/nvidia/bin"
print("Triton", triton.__version__, consts, flush=True)
warps = consts.get("FUSED_WARPS", 8)
stages = consts.get("FUSED_STAGES", 3)
sig = dict(X="*bf16", W="*u8", SF="*fp8e4nv", GS="*fp32", LAYOUT="*i32", Y="*bf16")
for label, n, k in (("gemm1 w13", 8192, 4096), ("gemm2 w2", 4096, 4096)):
    c = dict(M=135168, N=n, K=k, BM=consts["FUSED_BM"], BN=consts["FUSED_BN"],
             BKP=consts["FUSED_BKP"], GROUP_M=consts["FUSED_GROUP_M"])
    # Runtime specialization marks 16-byte-aligned pointer arguments; mirror it offline.
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(len(sig))}
    kern = triton.compile(ASTSource(module._fused_grouped_mm, sig, c, attrs), target=GPUTarget("cuda", 90, 32),
                          options={"num_warps": warps, "num_stages": stages})
    ptx = work / f"fused_{n}.ptx"
    ptx.write_text(kern.asm["ptx"])
    cubin = ptx.with_suffix(".cubin")
    r = subprocess.run([str(bin_dir / "ptxas"), "-v", "--gpu-name=sm_90a", str(ptx), "-o", str(cubin)],
                       capture_output=True, text=True)
    info = " | ".join(l.split("info    : ")[-1] for l in r.stderr.splitlines() if "registers" in l or "spill" in l)
    sass = subprocess.run([str(bin_dir / "cuobjdump"), "-sass", str(cubin)], capture_output=True, text=True).stdout
    ops = collections.Counter(m.group(1).split(".")[0] for m in re.finditer(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", sass))
    ldg = collections.Counter(m.group(0) for m in re.finditer(r"LDG\.E(?:\.[A-Z0-9]+)*", sass))
    ldgsts = sass.count("LDGSTS")
    print(f"{label}: warps={warps} stages={stages} shared={kern.metadata.shared} :: {info}", flush=True)
    print("   top ops:", ", ".join(f"{o}={c}" for o, c in ops.most_common(16)), flush=True)
    print("   HGMMA:", ops.get("HGMMA", 0), " HMMA:", ops.get("HMMA", 0), " LDGSTS(cp.async):", ldgsts,
          " LDG forms:", dict(ldg), flush=True)
