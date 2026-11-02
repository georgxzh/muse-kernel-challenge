"""Compile expansion variants and inspect register/spill use without a GPU."""
import os
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parents[1]
work = root / ".tools"
os.environ["TRITON_CACHE_DIR"] = str(work / "triton-cache")
sys.path.insert(0, str(work))
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from offline_kernels import _expand_nvfp4
ptxas = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
for block, warps in ((2048,8), (4096,8), (4096,16), (8192,16), (8192,32)):
    kernel = triton.compile(ASTSource(_expand_nvfp4,
        signature=dict(W="*u8", SF="*fp8e4nv", GS="*fp32", OUT="*bf16"),
        constexprs=dict(NK=8192*4096, BLOCK=block)),
        target=GPUTarget("cuda",90,32), options={"num_warps":warps})
    path = work / f"expand-{block}-{warps}.ptx"
    path.write_text(kernel.asm["ptx"])
    result = subprocess.run([str(ptxas), "-v", "--gpu-name=sm_90a", str(path),
                             "-o", str(path.with_suffix(".cubin"))], text=True, capture_output=True)
    print(block, warps, result.returncode, result.stderr, flush=True)
