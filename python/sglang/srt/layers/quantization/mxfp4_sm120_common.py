"""Shared SM120 MXFP4 weight-scale swizzle helpers.

The MXFP4 MoE methods (W4A4 fused and W4A8 grouped-GEMM) load DeepSeek-V4
experts with E8M0 32-element block scales and must move those scales into the
128x4 block layout the FlashInfer kernels consume. This module holds the
load-time swizzle shared by both methods.
"""

from __future__ import annotations

import torch

# MXFP4/MXFP8 block size (E8M0 32-element block scale).
_MXFP4_BLOCK = 32


def _flashinfer_scale_helpers():
    """FlashInfer scale-factor padding + the device fp4 quantization module."""
    from flashinfer.fp4_quantization import (
        _pad_scale_factors,
        get_fp4_quantization_module,
    )

    return _pad_scale_factors, get_fp4_quantization_module


def swizzle_weight_scale_mxf4(
    scale_u8: torch.Tensor, b: int, m: int, n: int
) -> torch.Tensor:
    """Swizzle a batched E8M0 weight-scale tensor into the 128x4 block layout.

    Mirrors the FlashInfer reference path: per-expert ``_pad_scale_factors``
    then a single ``block_scale_interleave_sm100``.

    scale_u8 : ``[B, M, n // 32]`` uint8 (unswizzled E8M0)
    Returns  : ``[B, M_pad128, ceil(n/32 to mult-4)]`` uint8 swizzled.

    Done once at load time (not under CUDA graph), so a per-expert Python loop
    is fine.
    """
    pad_scale_factors, get_fp4_mod = _flashinfer_scale_helpers()
    fp4_mod = get_fp4_mod(
        "".join(str(c) for c in torch.cuda.get_device_capability(scale_u8.device))
    )
    padded = torch.stack(
        [pad_scale_factors(scale_u8[i], m, n, _MXFP4_BLOCK) for i in range(b)]
    )
    out = fp4_mod.block_scale_interleave_sm100(padded)
    return out.view(padded.shape)
