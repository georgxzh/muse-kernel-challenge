"""Compile a bf16_proj candidate's Triton kernels for H200 (SM90) without a GPU,
and check the PDL instruction order in SASS.

Code generation only; okc remains the correctness and timing authority.
Usage: <Triton 3.5.1 python> -B tools/compile_pdl.py candidate.py
For each kernel: ACQBULK (griddepcontrol.wait) must come before every global
load/store, and PREEXIT (griddepcontrol.launch_dependents) after every global store.
"""
import ast
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
work = root / ".tools"
work.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(work / "triton-cache")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

source = Path(sys.argv[1]).read_text()
parts = ["import triton", "import triton.language as tl",
         "from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait"]
for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and any(ast.unparse(d) == "triton.jit" for d in node.decorator_list):
        parts.append("@triton.jit\n" + ast.get_source_segment(source, node))
target_file = work / "offline_pdl_kernels_bf16.py"
target_file.write_text("\n\n".join(parts) + "\n")
spec = importlib.util.spec_from_file_location("bf16_pdl_offline", target_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
target = GPUTarget("cuda", 90, 32)
print("Triton", triton.__version__, flush=True)
failures = []


def compile_one(name, signature, constants, expect_wait=True, expect_trigger=True, **options):
    fn = getattr(module, name)
    # CUDA allocator tensors passed to the real JIT have 16-byte aligned bases.
    attrs = {(fn.arg_names.index(arg),): [["tt.divisibility", 16]]
             for arg, dtype in signature.items() if dtype.startswith("*")}
    kernel = triton.compile(ASTSource(fn, signature=signature, constexprs=constants, attrs=attrs),
                            target=target, options=options)
    sass = kernel.asm["sass"].splitlines()
    idx = lambda pat: [i for i, line in enumerate(sass) if re.search(pat, line)]
    wait, trig = idx(r"\bACQBULK\b"), idx(r"\bPREEXIT\b")
    loads, stores = idx(r"\b(LDG|LDGSTS|UBLKCP|UTMALDG)\b"), idx(r"\b(STG|RED|ATOMG)\b")
    ok = True
    if expect_wait:
        ok &= bool(wait) and all(wait[0] < i for i in loads + stores)
    else:
        ok &= not wait
    if expect_trigger:
        ok &= bool(trig) and all(trig[0] > i for i in stores)
    else:
        ok &= not trig
    ptx = work / "pdl_check.ptx"
    ptx.write_text(kernel.asm["ptx"])
    ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
    rep = subprocess.run([str(ptxas), "-v", "--gpu-name=sm_90a", str(ptx), "-o", str(work / "pdl_check.cubin")],
                         text=True, capture_output=True, check=True).stderr
    regs = int(re.search(r"Used (\d+) registers", rep).group(1))
    spills = sum(int(x) for x in re.findall(r"(\d+) bytes spill (?:stores|loads)", rep))
    tag = "PASS" if ok and spills == 0 else "FAIL"
    if tag == "FAIL":
        failures.append((name, constants))
    print(f"{tag} {name} {constants} regs={regs} spills={spills} shared={kernel.metadata.shared} "
          f"ACQBULK@{wait[:1]} PREEXIT@{trig[:1]} first_load@{loads[:1]} last_store@{stores[-1:]}",
          flush=True)


# Router m=32: _split_mm (not PDL-launched, triggers) -> _finish (PDL, waits).
for m in (17, 24, 32):
    compile_one("_split_mm", dict(X="*bf16", W="*bf16", L="*bf16", P="*fp32", Y="*fp32", CNT="*i32"),
                dict(M=m, N=256, K=8192, SPLIT=16, ROUTER=True, BM=32, BN=32, BK=64, SERIAL=False),
                expect_wait=False, num_warps=4, num_stages=3)
    compile_one("_finish", dict(P="*fp32", Y="*fp32"), dict(SIZE=m * 256, SPLIT=16, BLOCK=256),
                expect_trigger=False)
# Router m<=16 serial path: no PDL (single kernel).
for m in (2, 8, 16):
    compile_one("_split_mm", dict(X="*bf16", W="*bf16", L="*bf16", P="*fp32", Y="*fp32", CNT="*i32"),
                dict(M=m, N=256, K=8192, SPLIT=16, ROUTER=True, BM=16, BN=32, BK=64, SERIAL=True),
                expect_wait=False, expect_trigger=False, num_warps=4, num_stages=3)
# Router m>32: cuBLAS -> _join_router (PDL, waits).
for m in (33, 128, 256):
    compile_one("_join_router", dict(P="*fp32", Y="*fp32"), dict(SIZE=m * 256, N=256, BLOCK=256),
                expect_trigger=False)
print("FAILURES:", failures if failures else "none")
