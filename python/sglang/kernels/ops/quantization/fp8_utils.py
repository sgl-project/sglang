from __future__ import annotations

from typing import Optional, Tuple

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


def cuda_capability_uses_fp8_e4b15(cuda_capability: Tuple[int, int]) -> bool:
    """Triton names E4M3 as fp8e4b15 on CUDA architectures before SM89."""
    return cuda_capability < (8, 9)


def use_fp8_e4b15_for_e4m3fn(
    device: Optional[int] = None,
    cuda_capability: Optional[Tuple[int, int]] = None,
) -> bool:
    if cuda_capability is None:
        if is_hip_runtime() or is_musa_runtime() or not torch.cuda.is_available():
            return False
        cuda_capability = current_arch().cuda.capability

    return cuda_capability_uses_fp8_e4b15(cuda_capability)


def fp8_dtype_to_triton(
    fp8_dtype: torch.dtype,
    *,
    device: Optional[int] = None,
    cuda_capability: Optional[Tuple[int, int]] = None,
):
    if fp8_dtype == torch.float8_e4m3fn:
        if use_fp8_e4b15_for_e4m3fn(device, cuda_capability):
            return tl.float8e4b15
        return tl.float8e4nv
    if fp8_dtype == torch.float8_e4m3fnuz:
        return tl.float8e4b8
    if fp8_dtype == torch.float8_e5m2:
        return tl.float8e5
    raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
