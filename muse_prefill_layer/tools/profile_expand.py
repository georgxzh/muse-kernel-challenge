"""Offline SM90 register/spill inspection; no GPU correctness or timing validation."""
"""Compile expansion variants and inspect register/spill use without a GPU."""
import os
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parents[1]
work = root / ".tools"
os.environ["TRITON_CACHE_DIR"] = str(work / "triton-cache")
import ast
import importlib.util
source_path=Path(sys.argv[1]) if len(sys.argv)>1 else root/'attempt9.py'
source=source_path.read_text()
parts=["import triton", "import triton.language as tl"]
for node in ast.parse(source).body:
    if isinstance(node,ast.FunctionDef) and any(ast.unparse(d)=="triton.jit" for d in node.decorator_list):
        parts.append("@triton.jit\n"+ast.get_source_segment(source,node))
isolated=work/'resource_expand_kernels.py'
isolated.write_text("\n\n".join(parts)+"\n")
spec=importlib.util.spec_from_file_location('resource_expand_kernels',isolated)
resource_module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=resource_module
spec.loader.exec_module(resource_module)

import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
offline_kernels=resource_module
ptxas = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
for name, block, warps in (("_expand_nvfp4",2048,4), ("_expand_pairs",1024,4), ("_expand_pairs",2048,8)):
    kernel = triton.compile(ASTSource(getattr(offline_kernels,name),
        signature=dict(W="*u8", SF="*fp8e4nv", GS="*fp32", OUT="*i32" if name=="_expand_pairs" else "*bf16"),
        constexprs=dict(NK=8192*4096, BLOCK=block)),
        target=GPUTarget("cuda",90,32), options={"num_warps":warps})
    path = work / f"{name}-{block}-{warps}.ptx"
    path.write_text(kernel.asm["ptx"])
    result = subprocess.run([str(ptxas), "-v", "--gpu-name=sm_90a", str(path),
                             "-o", str(path.with_suffix(".cubin"))], text=True, capture_output=True)
    print(name, block, warps, result.returncode, result.stderr, flush=True)
