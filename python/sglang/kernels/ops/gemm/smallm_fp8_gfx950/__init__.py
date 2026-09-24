"""gfx950 small-M W8A8 FP8 GEMM (per-token x per-channel, bf16 out) on aiter's (16, 16)-preshuffled weights.

`smallm_fp8_gemm.hip` is built with hipcc at first use and launched like smallm_moe_gfx950. On by default for the
Qwen3.5-397B AttnFP8 TP4 projection shapes below; SGLANG_ROCM_SMALLM_FP8_PROJ=0 turns it off.
"""

import ctypes
import os
import subprocess
import tempfile

import torch

from sglang.kernels.ops.moe.smallm_moe_gfx950 import (
    _check,
    _hip_lib,
    _hip_version,
    _hipcc,
    _Kernel,
)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smallm_fp8_gemm.hip")
# (N, K) -> ((max M, n-tiles per block, k steps per wave, waves, LDS-staged activations), ...), first match wins.
# Measured faster than aiter gemm_a8w8_bpreshuffle: packed GDN in_proj, attention qkv_proj, out_proj / o_proj.
SHAPES = {
    (5184, 4096): ((16, 2, 8, 8, 1), (28, 2, 4, 16, 0)),
    (4608, 4096): ((16, 2, 8, 8, 1), (28, 2, 4, 16, 0)),
    (4096, 2048): ((12, 1, 8, 4, 0), (36, 1, 4, 8, 0)),
}
_mod = None
_kernels = {}


class Args(ctypes.Structure):
    _pack_ = 1
    _fields_ = [(n, ctypes.c_void_p) for n in ("x", "w", "xs", "ws", "out")] + [
        (n, ctypes.c_int32) for n in ("M", "N", "lda", "xs_stride")
    ]


def smallm_fp8_gemm_enabled() -> bool:
    global _mod
    if os.environ.get("SGLANG_ROCM_SMALLM_FP8_PROJ", "1") == "0":
        return False
    if _mod is None:
        _mod = False
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if arch == "gfx950" and _hip_version() >= (7, 2, 0):
            co = os.path.join(tempfile.mkdtemp(prefix="sglang_smallm_fp8_"), "k.co")
            cmd = [_hipcc(), "--genco", "--offload-arch=gfx950", "-O3", "-o", co, _SRC]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                mod = ctypes.c_void_p()
                _check(_hip_lib().hipModuleLoad(ctypes.byref(mod), co.encode()), "load")
                _mod = mod
            except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
                print(f"[smallm_fp8_gemm] disabled: {e}", flush=True)
    return _mod is not False


def _config(M, N, K):
    for max_m, *cfg in SHAPES.get((N, K), ()):
        if M <= max_m:
            return cfg
    return None


def smallm_fp8_gemm_supported(XQ, WQ, x_scale, dtype) -> bool:
    """XQ fp8 [M, K], WQ the (16, 16)-shuffled fp8 [N, K], fp32 x_scale [M, 1], bf16 out."""
    return (
        XQ.dim() == 2
        and XQ.dtype == WQ.dtype == torch.float8_e4m3fn
        and dtype == torch.bfloat16
        and _config(XQ.shape[0], *WQ.shape) is not None
        and XQ.shape[1] == WQ.shape[1]
        and XQ.stride(1) == 1
        and WQ.is_contiguous()
        and x_scale.numel() == XQ.shape[0]
        and x_scale.dtype == torch.float32
        and not torch.compiler.is_compiling()
        and smallm_fp8_gemm_enabled()
    )


def smallm_fp8_gemm(XQ, WQ, x_scale, w_scale, dtype):
    """Drop-in for aiter gemm_a8w8_bpreshuffle where smallm_fp8_gemm_supported(); w_scale fp32 [N, 1]."""
    (M, K), N = XQ.shape, WQ.shape[0]
    nt, steps, waves, xlds = cfg = _config(M, N, K)
    mt = (M + 15) // 16
    kernel = _kernels.get((mt, *cfg))
    if kernel is None:
        name = f"smallm_fp8_gemm_m{mt}_n{nt}_s{steps}_w{waves}_x{xlds}"
        kernel = _kernels[(mt, *cfg)] = _Kernel(_mod, name)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=XQ.device)
    xs = x_scale.view(M, -1)[:, 0]
    ptrs = (XQ, WQ, xs, w_scale.contiguous(), out)
    args = Args(*(t.data_ptr() for t in ptrs), M, N, XQ.stride(0), xs.stride(0))
    stream = torch.cuda.current_stream().cuda_stream
    kernel.launch(N // (16 * nt), args, stream, 64 * waves, M * (K + 16) * xlds)
    return out
