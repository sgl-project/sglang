"""Convert MXFP8 weights to block-fp8 [128,128] for AMD gfx942 (CDNA3 / MI300).

gfx942 has no hardware MX-scaled matmul: Triton's ``tl.dot_scaled`` fails to
lower and the gfx950 ``mfma_scale`` intrinsics are unavailable. So MXFP8
checkpoints (e4m3fn weights + 1x32 UE8M0 scales) are converted at load time to
block-wise FP8 [128,128] (e4m3fn + fp32 scales), which runs through SGLang's
native DeepSeek-V3 block-fp8 kernels (aiter / triton). The conversion is:

    bf16 = e4m3.to(f32) * exp2(ue8m0_scale.to(f32) - 127.0)   # dequant 1x32
    block-fp8 = per-128x128-block quantize(bf16)              # requant 128x128
"""

from __future__ import annotations

from typing import Tuple

import torch

MXFP8_BLOCK_SIZE = 32
DEFAULT_MAX_CONVERSION_WORKSPACE_BYTES = 256 * 1024 * 1024


def _ue8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """UE8M0 uint8 (biased exponent, bias 127) -> fp32 multiplier 2^(v-127)."""
    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def dequant_mxfp8_2d_to_bf16(
    weight: torch.Tensor, scale_u8: torch.Tensor
) -> torch.Tensor:
    """Dequant a 2D MXFP8 tensor (e4m3fn + 1x32 UE8M0 scales) to bf16.

    weight: [N, K] float8_e4m3fn; scale_u8: [N, K//32] uint8.
    """
    n, k = weight.shape
    descale = _ue8m0_to_fp32(scale_u8).unsqueeze(-1)  # [N, K//32, 1]
    deq = weight.to(torch.float32).view(n, k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (deq * descale).view(n, k).to(torch.bfloat16)


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


def _conversion_chunk_rows(
    k: int, block: int, max_workspace_bytes: int
) -> int:
    """Choose a block-aligned row chunk under a conservative memory budget."""
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}.")
    if max_workspace_bytes <= 0:
        raise ValueError(
            "max_workspace_bytes must be positive, "
            f"got {max_workspace_bytes}."
        )

    padded_k = ((k + block - 1) // block) * block
    # Conversion can briefly hold BF16 dequantized values, FP32 padded and
    # normalized values, and FP8 output for the active chunk.
    bytes_per_row = max(1, padded_k * 12)
    row_blocks = max(1, max_workspace_bytes // (block * bytes_per_row))
    return row_blocks * block


def convert_mxfp8_weight_to_block_fp8(
    weight: torch.Tensor,
    scale_u8: torch.Tensor,
    block: int = 128,
    *,
    max_workspace_bytes: int = DEFAULT_MAX_CONVERSION_WORKSPACE_BYTES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 (e4m3fn + 1x32 UE8M0) -> block-fp8 [block,block] (e4m3fn + fp32).

    Used on gfx942 to run MXFP8 checkpoints through the fast native block-fp8
    kernels. Conversion is row-chunked on block boundaries so loading a large
    matrix does not materialize full-size BF16 and FP32 copies simultaneously.
    """
    if weight.ndim != 2 or scale_u8.ndim != 2:
        raise ValueError(
            "convert_mxfp8_weight_to_block_fp8 requires 2D weight and scale "
            "tensors."
        )

    n, k = weight.shape
    if n <= 0 or k <= 0:
        raise ValueError(f"MXFP8 weight must be non-empty, got shape {(n, k)}.")
    if k % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(f"MXFP8 weight K={k} must be divisible by {MXFP8_BLOCK_SIZE}.")
    expected_scale_shape = (n, k // MXFP8_BLOCK_SIZE)
    if tuple(scale_u8.shape) != expected_scale_shape:
        raise ValueError(
            "MXFP8 scale shape must be [N, K/32]: "
            f"expected {expected_scale_shape}, got {tuple(scale_u8.shape)} "
            f"for weight shape {(n, k)}."
        )

    chunk_rows = _conversion_chunk_rows(k, block, max_workspace_bytes)
    qweight = torch.empty_like(weight)
    scale = torch.empty(
        ((n + block - 1) // block, (k + block - 1) // block),
        dtype=torch.float32,
        device=weight.device,
    )

    for row_start in range(0, n, chunk_rows):
        row_end = min(row_start + chunk_rows, n)
        bf16_chunk = dequant_mxfp8_2d_to_bf16(
            weight[row_start:row_end], scale_u8[row_start:row_end]
        )
        q_chunk, scale_chunk = bf16_to_block_fp8_128(bf16_chunk, block=block)
        qweight[row_start:row_end].copy_(q_chunk)
        scale_row_start = row_start // block
        scale[scale_row_start : scale_row_start + scale_chunk.shape[0]].copy_(
            scale_chunk
        )
        del bf16_chunk, q_chunk, scale_chunk

    return qweight, scale
