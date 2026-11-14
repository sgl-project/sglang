"""gfx950 small-M FP8 block-scale GEMM with fused activation quant."""

import ctypes
import logging
import os
import subprocess

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils.hipcc import (
    HipKernel,
    build_gfx950_module,
    check_hip,
    hip_lib,
)
from sglang.srt.environ import envs

_SRC = os.path.join(os.path.dirname(__file__), "smallm_fp8_blockscale_gemm.hip")
# (N, K) -> (max M, N tiles/block, K steps/wave, waves), first match wins.
SHAPES = {
    (5120, 4096): ((32, 2, 4, 16),),
    (4608, 4096): ((32, 2, 4, 16),),
    (4096, 2048): ((8, 1, 4, 8), (16, 1, 2, 16), (32, 1, 4, 8)),
    (2560, 4096): ((32, 1, 4, 16),),
    (4096, 1024): ((32, 1, 2, 8),),
}
_mod = None
_fns = {}
logger = logging.getLogger(__name__)


class Args(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("x", "w", "ws", "out")] + [
        (n, ctypes.c_int32) for n in ("M", "N", "lda")
    ]


# (N, K) -> max M at which this kernel on a bf16 input beats the bpreshuffle GEMM
# on an input its producer already quantized (fused RMSNorm or SiLU-and-mul).
# Measured on MI355X; re-measure when either GEMM or the shape table changes.
_BEATS_PREQUANTIZED_MAX_M = {
    (5120, 4096): 4,
    (4608, 4096): 4,
    (4096, 2048): 16,
    (2560, 4096): 12,
    (4096, 1024): 16,
}


def _config(M, N, K):
    return next((cfg for max_m, *cfg in SHAPES.get((N, K), ()) if M <= max_m), None)


def prefers_bf16_input(M, N, K) -> bool:
    """Whether an (M, K) input to an (N, K) preshuffled weight should reach this
    kernel in bf16 rather than be quantized by its producer: the kernel takes
    it, and this pair is faster than the pre-quantized one."""
    return (
        _mod is not False
        and envs.SGLANG_ROCM_SMALLM_FP8_BS.get()
        and _config(M, N, K) is not None
        and M <= _BEATS_PREQUANTIZED_MAX_M.get((N, K), 0)
    )


def try_smallm_fp8_bs_linear(x, WQ, w_scale):
    """Run the fused kernel, or return None for fallback."""
    global _mod
    (M, K), N = x.shape, WQ.shape[0]
    cfg = _config(M, N, K)
    if not (
        _mod is not False
        and envs.SGLANG_ROCM_SMALLM_FP8_BS.get()
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and getattr(WQ, "is_shuffled", False)
        and w_scale.shape == (N // 128, K // 128)
        and w_scale.is_contiguous()
        and cfg is not None
        and not torch.compiler.is_compiling()
    ):
        return None
    if _mod is None:
        hip = current_arch().hip
        if hip is None or hip.gcn_arch != "gfx950":
            _mod = False
            return None
        try:
            _mod = build_gfx950_module(_SRC, "sglang_smallm_fp8_bs_")
        except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
            _mod = False
            logger.warning("smallm_fp8_bs disabled: %s", e)
            return None
    key = ((M + 15) // 16, *cfg)
    hip, fn = hip_lib(), _fns.get(key)
    if fn is None:
        fn = _fns[key] = HipKernel(
            _mod, "smallm_fp8_bs_m{}_n{}_s{}_w{}".format(*key)
        ).fn
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    args = Args(*(t.data_ptr() for t in (x, WQ, w_scale, out)), M, N, x.stride(0))
    params = (ctypes.c_void_p * 1)(ctypes.addressof(args))
    stream = torch.cuda.current_stream().cuda_stream
    grid, block = N // (16 * key[1]), 64 * key[3]
    rc = hip.hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, params, None)
    check_hip(rc, "launch")
    return out
