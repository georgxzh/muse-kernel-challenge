"""Compile a decode-layer candidate's Triton kernels for H200 (SM90) without a GPU,
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
target_file = work / "offline_pdl_kernels.py"
target_file.write_text("\n\n".join(parts) + "\n")
spec = importlib.util.spec_from_file_location("decode_pdl_offline", target_file)
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


# Triton projections at T <= 8 (decode 1 and verify 1): qkv_gate, o, pooler, unpooler slice.
for m in (1, 2, 8):
    for n, k in ((2304, 8192), (8192, 1024), (4096, 8192), (1024, 4096)):
        split = 8 if k >= 4096 else 4
        compile_one("_dense_partials", dict(X="*bf16", W="*bf16", P="*fp32"),
                    dict(M=m, N=n, K=k, BM=16, BN=64, BK=128, SPLIT=split),
                    num_warps=4, num_stages=2)
        trigger = n == 2304
        compile_one("_dense_reduce", dict(P="*fp32", O="*bf16"),
                    dict(SIZE=m * n, SPLIT=split, BLOCK=256, TRIGGER=trigger),
                    expect_trigger=trigger, num_warps=4)
# Router at 1 < T <= 32 (side stream).
for m in (2, 8, 16, 17, 32):
    bm = 16 if m <= 16 else 32
    compile_one("_router_partials", dict(X="*bf16", HI="*bf16", LO="*bf16", P="*fp32"),
                dict(M=m, BM=bm), num_warps=4, num_stages=2)
    compile_one("_dense_reduce", dict(P="*fp32", O="*fp32"),
                dict(SIZE=m * 256, SPLIT=16, BLOCK=256, TRIGGER=False),
                expect_trigger=False, num_warps=4)
# Cache store + q, every timed token count.
for t in (1, 8, 16, 32, 128, 256):
    for rope in (False, True):
        for scaled in (False, True):
            compile_one("_qkv_to_cache",
                        dict(QKV="*bf16", Q="*fp8e4nv", KC="*fp8e4nv", VC="*fp8e4nv", LOC="*i64",
                             POS="*i64", CS="*fp32", KS="*fp32", VS="*fp32"),
                        dict(T=t, STRIDE=2304, ROPE=rope, SCALED=scaled, BM=4),
                        expect_trigger=False, num_warps=4, enable_fp_fusion=False)
# Unchanged kernel (no PDL): compile only.
compile_one("_norm_and_local_routes",
            dict(X="*bf16", W="*bf16", TOP="*i32", START="*i32", Y="*bf16", IDS="*i32"), {},
            expect_wait=False, expect_trigger=False, num_warps=4, enable_fp_fusion=False)
print("FAILURES:", failures if failures else "none")
