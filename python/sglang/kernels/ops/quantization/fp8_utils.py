from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    get_jit_cuda_arch,
    is_hip_runtime,
    is_musa_runtime,
)


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
        if device is None:
            arch = get_jit_cuda_arch()
            cuda_capability = (arch.major, arch.minor)
        else:
            cuda_capability = torch.cuda.get_device_capability(device)

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


@triton.jit
def load_fp8_e4m3fn(raw_ptr, offsets, mask, out_dtype: tl.constexpr):
    """Decode OCP e4m3fn bytes without using the unsupported e4m3nv type."""
    raw = tl.load(raw_ptr + offsets, mask=mask, other=0).to(tl.int32)
    sign = (raw & 0x80) << 8
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    normal_bits = ((exponent + 120) << 7) | (mantissa << 4) | sign
    subnormal_bits = (
        tl.where(
            mantissa == 0,
            0,
            tl.where(
                mantissa == 1,
                0x3B00,
                tl.where(
                    mantissa == 2,
                    0x3B80,
                    tl.where(
                        mantissa == 3,
                        0x3BC0,
                        tl.where(
                            mantissa == 4,
                            0x3C00,
                            tl.where(
                                mantissa == 5,
                                0x3C20,
                                tl.where(mantissa == 6, 0x3C40, 0x3C60),
                            ),
                        ),
                    ),
                ),
            ),
        )
        | sign
    )
    bits = tl.where(exponent == 0, subnormal_bits, normal_bits)
    bits = tl.where((exponent == 15) & (mantissa == 7), 0x7FC0 | sign, bits)
    return (bits << 16).to(tl.float32, bitcast=True).to(out_dtype)
