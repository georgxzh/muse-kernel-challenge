"""Compile bf16_proj split-K kernels (serial in-kernel reduction) for SM90 offline.

Usage: <Triton 3.5.1 python> -B tools/compile_serial.py attempts/attemptNN.py
Reports registers/spills and whether ATOM/BAR are present. Code generation only.
"""
import ast
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
work = root / ".tooling"
work.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(work / "triton-cache-serial")
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

source = (root / sys.argv[1]).read_text()
parts = ["import triton", "import triton.language as tl"]
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and any(ast.unparse(d) == "triton.jit" for d in node.decorator_list):
        parts.append("@triton.jit\n" + ast.get_source_segment(source, node))
path = work / "offline_serial.py"
path.write_text("\n\n".join(parts) + "\n")
spec = importlib.util.spec_from_file_location("offline_serial", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
bin_dir = Path(triton.__file__).parent / "backends/nvidia/bin"
print("Triton", triton.__version__, flush=True)


def build(fn, sig, consts, warps, label):
    attrs = {(i,): [["tt.divisibility", 16]] for i, t in enumerate(sig.values()) if t.startswith("*")}
    k = triton.compile(ASTSource(fn, sig, consts, attrs), target=GPUTarget("cuda", 90, 32),
                       options={"num_warps": warps, "num_stages": 3})
    ptx = work / f"serial_{label}.ptx"
    ptx.write_text(k.asm["ptx"])
    cub = ptx.with_suffix(".cubin")
    r = subprocess.run([str(bin_dir / "ptxas"), "-v", "--gpu-name=sm_90a", str(ptx), "-o", str(cub)],
                       capture_output=True, text=True)
    info = " | ".join(l.split("info    : ")[-1] for l in r.stderr.splitlines() if "registers" in l or "spill" in l)
    sass = subprocess.run([str(bin_dir / "cuobjdump"), "-sass", str(cub)], capture_output=True, text=True).stdout
    atoms = sorted(set(re.findall(r"\b(ATOMG?\.[A-Z0-9.]+|RED\.[A-Z0-9.]+)", sass)))
    fences = sorted(set(re.findall(r"\b(MEMBAR\.[A-Z0-9.]+|FENCE\.[A-Z0-9.]+|CCTL\.[A-Z0-9.]+)", sass)))
    print(f"{label}: {info} | atomics={atoms} fences={fences} BAR={sass.count('BAR.SYNC')}", flush=True)


ptr = lambda *names: {n: "*fp32" for n in names}
for m in (2, 8, 9, 16, 32):
    bm = 16 if m <= 16 else 32
    sig = dict(X="*bf16", W="*bf16", L="*bf16", P="*fp32", Y="*fp32", CNT="*i32")
    build(mod._split_mm, sig, dict(M=m, N=256, K=8192, SPLIT=16, ROUTER=True, BM=bm, BN=32, BK=64, SERIAL=True),
          4, f"router_split_m{m}")
sig = dict(X="*bf16", W="*fp32", P="*fp32", Y="*fp32", CNT="*i32")
build(mod._router_small, sig, dict(M=1, N=256, K=8192, SPLIT=16, BN=4, BK=512, SERIAL=True), 4, "router_small_m1")
for serial in (False, True):
    for layer, n, k, m in (("qkv_gate", 2304, 8192, 1), ("qkv_gate", 2304, 8192, 16), ("pooler", 4096, 8192, 7),
                           ("unpooler", 1024, 4096, 1), ("unpooler", 1024, 4096, 5), ("unpooler", 1024, 4096, 8)):
        sig = dict(X="*bf16", T="*bf16", P="*bf16", Y="*bf16", CNT="*i32")
        build(mod._tiled_mm, sig, dict(M=m, N=n, K=k, SPLIT=8, BM=16, BN=64, BK=64, SERIAL=serial), 4,
              f"tiled_{layer}_m{m}_serial{int(serial)}")
