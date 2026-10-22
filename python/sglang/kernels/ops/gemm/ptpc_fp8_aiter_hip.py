"""Fail-closed PTPC (per-token activation, per-channel weight) FP8 GEMM adapter
for AITER's preshuffled a8w8 kernel on ROCm."""

from __future__ import annotations

import functools
from typing import Optional

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip

# AITER's a8w8 bpreshuffle instances reject N that is not a multiple of 64, so
# short weights are zero-padded and the padding columns sliced off the result.
_N_ALIGN = 64
_SHUFFLE_LAYOUT = (16, 16)


@functools.cache
def _ops():
    # Quark ships OCP e4m3fn weights; gfx942's FNUZ fp8 cannot take them as-is.
    if not (is_hip() and is_gfx95_supported()):
        return None
    try:
        from aiter import dtypes
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
        from aiter.ops.quant import per_token_quant_hip
        from aiter.ops.shuffle import shuffle_weight
    except ImportError:
        return None
    return dtypes.fp8, gemm_a8w8_bpreshuffle, per_token_quant_hip, shuffle_weight


@functools.cache
def _gemm_out():
    """AITER's gemm_a8w8_bpreshuffle_out (ROCm/aiter#6437), or None without it."""
    try:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_out
    except ImportError:
        return None
    return gemm_a8w8_bpreshuffle_out


def available() -> bool:
    return _ops() is not None


def _pad_rows(
    weight: torch.Tensor, scale: Optional[torch.Tensor]
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    pad = (-weight.shape[0]) % _N_ALIGN
    if pad:
        weight = torch.cat([weight, weight.new_zeros((pad, weight.shape[1]))])
        if scale is not None:
            # Zero rows produce zero at any scale.
            scale = torch.cat([scale, scale.new_ones(pad)])
    return weight, scale


def pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(preshuffled FP8 weight, [N_pad, 1] FP32 scales, N) for a BF16 [N, K] weight."""
    fp8, _, per_token_quant, shuffle = _ops()
    out_features = weight.shape[0]
    padded, _ = _pad_rows(weight, None)
    quantized, scale = per_token_quant(padded.contiguous(), quant_dtype=fp8)
    return (
        shuffle(quantized, layout=_SHUFFLE_LAYOUT),
        scale.view(-1, 1).float().contiguous(),
        out_features,
    )


def pack_prequantized(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Same layout as pack() from an FP8 [N, K] weight that already has [N]
    per-channel scales, so the GEMM stays bit-identical to the checkpoint."""
    fp8, _, _, shuffle = _ops()
    if weight.dtype != fp8 or scale.numel() != weight.shape[0]:
        raise ValueError(
            f"expected {fp8} weight with one scale per row, got {weight.dtype} "
            f"{tuple(weight.shape)} and {scale.numel()} scales"
        )
    out_features = weight.shape[0]
    padded, padded_scale = _pad_rows(weight, scale.reshape(-1))
    return (
        shuffle(padded.contiguous(), layout=_SHUFFLE_LAYOUT),
        padded_scale.view(-1, 1).float().contiguous(),
        out_features,
    )


def covered(x: torch.Tensor) -> bool:
    return (
        available()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[0] > 0
    )


def covered_prequant(x_q: torch.Tensor, x_scale: torch.Tensor) -> bool:
    return (
        available()
        and x_q.dim() == 2
        and x_q.dtype == _ops()[0]
        and x_q.is_contiguous()
        and x_q.shape[0] > 0
        and x_scale.dtype == torch.float32
        and x_scale.numel() == x_q.shape[0]
    )


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    *,
    x_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BF16 x @ weight.T[:, :out_features]; x is BF16, or FP8 with x_scale."""
    fp8, gemm, per_token_quant, _ = _ops()
    if x_scale is None:
        x, x_scale = per_token_quant(x, quant_dtype=fp8)
    x_scale = x_scale.view(x.shape[0], 1)
    if out is None:
        result = gemm(x, weight, x_scale, scale, dtype=torch.bfloat16)
        return result if result.shape[1] == out_features else result[:, :out_features]
    if weight.shape[0] != out_features:
        raise ValueError("out= needs a weight without padded rows")
    gemm_out = _gemm_out()
    if gemm_out is None:
        out.copy_(gemm(x, weight, x_scale, scale, dtype=torch.bfloat16))
    else:
        gemm_out(out, x, weight, x_scale, scale)
    return out


def warmup(
    *,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
    max_tokens: int,
) -> None:
    """Pick and compile kernels for every power-of-two batch up to max_tokens,
    outside graph capture."""
    num_tokens = 1
    while num_tokens <= max_tokens:
        x = torch.zeros(
            (num_tokens, in_features), dtype=torch.bfloat16, device=weight.device
        )
        run(x, weight, scale, out_features)
        num_tokens *= 2
    torch.cuda.synchronize(weight.device)
