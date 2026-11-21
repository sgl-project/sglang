from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import is_hip_runtime, is_musa_runtime


@triton.jit
def ue8m0_to_fp32(scale_u8):
    # Byte 0 is 2**-127, byte 255 is NaN, otherwise 2**(byte - 127).
    normal = (scale_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    denormal = tl.full(scale_u8.shape, 0x00400000, tl.int32).to(
        tl.float32, bitcast=True
    )
    scale = tl.where(scale_u8 == 0, denormal, normal)
    return tl.where(scale_u8 == 255, float("nan"), scale)


def use_fp8_e4b15_for_e4m3fn() -> bool:
    """Triton names E4M3 as fp8e4b15 on CUDA architectures before SM89."""
    if is_hip_runtime() or is_musa_runtime() or not torch.cuda.is_available():
        return False
    return current_arch().cuda.capability < (8, 9)


def fp8_dtype_to_triton(fp8_dtype: torch.dtype):
    if fp8_dtype == torch.float8_e4m3fn:
        if use_fp8_e4b15_for_e4m3fn():
            return tl.float8e4b15
        return tl.float8e4nv
    if fp8_dtype == torch.float8_e4m3fnuz:
        return tl.float8e4b8
    if fp8_dtype == torch.float8_e5m2:
        return tl.float8e5
    raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
