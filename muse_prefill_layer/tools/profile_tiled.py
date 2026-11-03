"""Offline SM90 register/spill inspection; no GPU correctness or timing validation."""
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
work=root/".tools"
os.environ["TRITON_CACHE_DIR"]=str(work/"triton-cache")
import ast
import importlib.util
source_path=Path(sys.argv[1]) if len(sys.argv)>1 else root/'prototypes/fused_nvfp4_mm.py'
source=source_path.read_text()
parts=["import triton", "import triton.language as tl"]
for node in ast.parse(source).body:
    if isinstance(node,ast.FunctionDef) and any(ast.unparse(d)=="triton.jit" for d in node.decorator_list):
        parts.append("@triton.jit\n"+ast.get_source_segment(source,node))
isolated=work/'resource_tiled_kernels.py'
isolated.write_text("\n\n".join(parts)+"\n")
spec=importlib.util.spec_from_file_location('resource_tiled_kernels',isolated)
resource_module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=resource_module
spec.loader.exec_module(resource_module)

import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
_nvfp4_tiled_mm=resource_module._nvfp4_tiled_mm
ptxas=Path(triton.__file__).parent/"backends/nvidia/bin/ptxas"
for bm,bn,bk,nw,stages in ((64,128,64,8,2),(64,64,64,8,2),(64,128,32,8,2),(64,64,32,8,2),(128,128,64,8,2)):
    c=dict(M=135168,N=8192,K=4096,BM=bm,BN=bn,BK=bk,GROUP_M=8)
    k=triton.compile(ASTSource(_nvfp4_tiled_mm,
        signature=dict(X="*bf16",W="*u8",SF="*fp8e4nv",GS="*fp32",LAYOUT="*i32",Y="*bf16"),constexprs=c),
        target=GPUTarget("cuda",90,32),options=dict(num_warps=nw,num_stages=stages))
    path=work/f"tiled-{bm}-{bn}-{bk}-{nw}-{stages}.ptx"
    path.write_text(k.asm["ptx"])
    r=subprocess.run([str(ptxas),"-v","--gpu-name=sm_90a",str(path),"-o",str(path.with_suffix(".cubin"))],capture_output=True,text=True)
    print(c,"warps",nw,"stages",stages,"shared",k.metadata.shared,"wgmma",k.asm["ptx"].count("wgmma.mma_async"),"mma.sync",k.asm["ptx"].count("mma.sync"),flush=True)
    print(r.stderr,flush=True)
