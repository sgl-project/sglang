"""Fail-closed PTPC FP8 adapter for Kimi-K3's BF16 decode projections.

The checkpoint keeps `self_attn.*`, `shared_experts.*` and the latent
projections in BF16, and at decode those GEMMs are HBM-bound on weight bytes
(measured ~4 TB/s on MI355X, which is where the machine tops out). Quantizing
the weight to FP8 per output channel and the activation per token halves the
bytes the GEMM has to stream, which is the only lever that helps once the
kernel is already bandwidth-saturated.

Routing is deliberately out of scope: FP8 router logits move the top-k
selection (measured ~15.3/16 agreement), so callers must keep the gate BF16.
"""

from __future__ import annotations

import torch

from sglang.srt.utils import is_hip

# aiter's a8w8 bpreshuffle instances reject N that is not a multiple of 64
# ("This GEMM is not supported"), so short weights are zero-padded and the
# padding columns are dropped from the result.
_N_ALIGN = 64
_SHUFFLE_LAYOUT = (16, 16)


def _ops():
    try:
        from aiter import dtypes
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
        from aiter.ops.quant import per_token_quant_hip
        from aiter.ops.shuffle import shuffle_weight
    except (ImportError, ModuleNotFoundError):
        return None
    return dtypes.fp8, gemm_a8w8_bpreshuffle, per_token_quant_hip, shuffle_weight


def available() -> bool:
    return is_hip() and _ops() is not None


def pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize [out, in] BF16 -> (preshuffled FP8 weight, row scales, out).

    Returns the logical `out` alongside the padded tensors so `run` can slice
    the padding away.
    """
    ops = _ops()
    if ops is None:
        raise RuntimeError("aiter PTPC FP8 GEMM is unavailable")
    fp8, _, per_token_quant, shuffle = ops
    if weight.ndim != 2 or not weight.is_cuda:
        raise ValueError(f"expected a 2D CUDA weight, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    padded = out_features + (-out_features) % _N_ALIGN
    if padded != out_features:
        weight = torch.cat(
            [
                weight,
                weight.new_zeros((padded - out_features, in_features)),
            ]
        )
    quantized, scale = per_token_quant(weight.contiguous(), quant_dtype=fp8)
    return (
        shuffle(quantized, layout=_SHUFFLE_LAYOUT),
        scale.view(padded, 1).contiguous().float(),
        out_features,
    )


def covered(x: torch.Tensor, weight: torch.Tensor | None) -> bool:
    return (
        weight is not None
        and available()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[0] > 0
    )


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """out[:, :out_features] = (x @ weight.T) with per-token / per-channel FP8."""
    ops = _ops()
    if ops is None:
        raise RuntimeError("aiter PTPC FP8 GEMM is unavailable")
    fp8, gemm, per_token_quant, _ = ops
    xq, xs = per_token_quant(x, quant_dtype=fp8)
    if out is not None and weight.shape[0] != out_features:
        raise ValueError("out= is unavailable when the packed weight has padded rows")
    result = gemm(
        xq,
        weight,
        xs.view(x.shape[0], 1),
        scale,
        dtype=x.dtype,
        out=out,
    )
    return result if result.shape[1] == out_features else result[:, :out_features]


def warmup(
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
    token_buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
) -> None:
    """Force kernel selection/compile outside graph capture."""
    device = weight.device
    for num_tokens in token_buckets:
        x = torch.zeros((num_tokens, in_features), dtype=torch.bfloat16, device=device)
        if covered(x, weight):
            run(x, weight, scale, out_features)
    torch.cuda.synchronize(device)
