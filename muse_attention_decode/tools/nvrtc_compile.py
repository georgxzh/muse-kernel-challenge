"""Compile a CUDA source to PTX with NVRTC and assemble it for sm_90a with ptxas (no GPU).
Usage: .tools/venv/bin/python tools/nvrtc_compile.py file.cu"""
import ctypes, glob, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1] / ".tools" / "cuda-nvcc" / "nvidia"
lib = ctypes.CDLL(sorted(glob.glob(str(root / "cuda_nvrtc/lib/libnvrtc.so*")))[0])
src = Path(sys.argv[1]).read_bytes()
prog = ctypes.c_void_p()
assert lib.nvrtcCreateProgram(ctypes.byref(prog), src, b"k.cu", 0, None, None) == 0
opts = [b"-arch=compute_90a", b"-std=c++17", ("-I" + str(root / "cuda_runtime/include")).encode(), b"-lineinfo"]
arr = (ctypes.c_char_p * len(opts))(*opts)
rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
n = ctypes.c_size_t(); lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
log = ctypes.create_string_buffer(n.value); lib.nvrtcGetProgramLog(prog, log)
print("nvrtc rc", rc); print(log.value.decode()[-4000:])
if rc != 0: sys.exit(1)
lib.nvrtcGetPTXSize(prog, ctypes.byref(n)); ptx = ctypes.create_string_buffer(n.value); lib.nvrtcGetPTX(prog, ptx)
out = Path(__file__).resolve().parents[1] / ".tools" / "inspect"; out.mkdir(parents=True, exist_ok=True)
(out / "cuda.ptx").write_bytes(ptx.value)
r = subprocess.run([str(root / "cuda_nvcc/bin/ptxas"), "-v", "--gpu-name=sm_90a", str(out / "cuda.ptx"), "-o", str(out / "cuda.cubin")],
                   capture_output=True, text=True)
print(r.stderr[-3000:]); sys.exit(r.returncode)
