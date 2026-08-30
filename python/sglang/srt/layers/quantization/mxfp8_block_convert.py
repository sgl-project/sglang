"""MXFP8 conversion helpers.

MXFP8 checkpoints store e4m3fn weights with 1x32 UE8M0 scales. Platforms
without native MX-scaled matmul can dequantize those weights exactly to BF16
inside a graph-captured forward without permanently expanding model storage.
An optional compatibility/performance path can instead requantize the BF16
values to block-wise FP8 [128,128] at load time. The conversion is:

    bf16 = e4m3.to(f32) * exp2(ue8m0_scale.to(f32) - 127.0)   # dequant 1x32
    block-fp8 = per-128x128-block quantize(bf16)              # optional requant
"""

from __future__ import annotations

from typing import Tuple

import torch

MXFP8_BLOCK_SIZE = 32


def _ue8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """UE8M0 uint8 (biased exponent, bias 127) -> fp32 multiplier 2^(v-127)."""
    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def dequant_mxfp8_to_bf16(weight: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """Dequant an MXFP8 tensor to BF16 along its final dimension.

    ``weight`` has shape ``[..., K]`` and ``scale_u8`` has shape
    ``[..., K // 32]``. Supporting arbitrary leading dimensions lets dense
    linears and all experts in an MoE layer share the same exact conversion.
    """
    k = weight.shape[-1]
    if k % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(f"MXFP8 weight K={k} must be divisible by {MXFP8_BLOCK_SIZE}.")
    expected_scale_shape = (*weight.shape[:-1], k // MXFP8_BLOCK_SIZE)
    if tuple(scale_u8.shape) != expected_scale_shape:
        raise ValueError(
            "MXFP8 scale shape must match weight leading dimensions and have "
            f"K/{MXFP8_BLOCK_SIZE} columns: expected {expected_scale_shape}, "
            f"got {tuple(scale_u8.shape)}."
        )

    descale = _ue8m0_to_fp32(scale_u8).unsqueeze(-1)
    deq = weight.to(torch.float32).view(
        *weight.shape[:-1], k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE
    )
    return (deq * descale).view_as(weight).to(torch.bfloat16)


def dequant_mxfp8_2d_to_bf16(
    weight: torch.Tensor, scale_u8: torch.Tensor
) -> torch.Tensor:
    """Backward-compatible 2D wrapper around :func:`dequant_mxfp8_to_bf16`."""
    if weight.ndim != 2 or scale_u8.ndim != 2:
        raise ValueError(
            "dequant_mxfp8_2d_to_bf16 requires 2D weight and scale tensors."
        )
    return dequant_mxfp8_to_bf16(weight, scale_u8)


def bf16_to_block_fp8_128(
    weight: torch.Tensor, block: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D bf16/fp32 weight to block-wise FP8 (e4m3fn) + fp32 scales.

    Returns (qweight [N,K] e4m3fn, scale [ceil(N/block), ceil(K/block)] fp32).
    Mirrors the DeepSeek-V3 block-fp8 contract (divide by e4m3fn max 448).
    The downstream gfx942 path normalizes e4m3fn -> e4m3fnuz separately.
    """
    n, k = weight.shape
    pn = ((n + block - 1) // block) * block
    pk = ((k + block - 1) // block) * block
    xp = torch.zeros((pn, pk), dtype=torch.float32, device=weight.device)
    xp[:n, :k] = weight.to(torch.float32)
    xv = xp.view(pn // block, block, pk // block, block)
    amax = xv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    sf = amax / 448.0
    xq = (xv / sf).to(torch.float8_e4m3fn)
    qweight = xq.view(pn, pk)[:n, :k].contiguous()
    scale = sf.view(pn // block, pk // block).contiguous()
    return qweight, scale


def convert_mxfp8_weight_to_block_fp8(
    weight: torch.Tensor, scale_u8: torch.Tensor, block: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 (e4m3fn + 1x32 UE8M0) -> block-fp8 [block,block] (e4m3fn + fp32).

    Used on gfx942 to run MXFP8 checkpoints through the fast native block-fp8
    kernels.
    """
    bf16 = dequant_mxfp8_2d_to_bf16(weight, scale_u8)
    return bf16_to_block_fp8_128(bf16, block=block)
