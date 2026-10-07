"""Fail-closed PTPC FP8 adapter for Kimi-K3's BF16 decode projections.

The checkpoint keeps `self_attn.*`, `shared_experts.*` and the latent
projections in BF16, and at decode those GEMMs are HBM-bound on weight bytes
(measured ~4 TB/s on MI355X, which is where the machine tops out). Quantizing
the weight to FP8 per output channel and the activation per token halves the
bytes the GEMM has to stream, which is the only lever that helps once the
kernel is already bandwidth-saturated.

The GEMM is hipBLASLt ``torch._scaled_mm`` (the ``kernel_gemm_0`` ATOM uses
for dense PTPC). CK ``gemm_a8w8_bpreshuffle`` is slower on the same shapes
and rejects N that is not a multiple of 64. hipBLASLt needs N % 16 == 0, so
short weights are zero-padded and the padding columns are dropped.

Activation quant is aiter ``per_token_quant_hip``
(``dynamic_per_token_scaled_quant``), the same kernel ATOM launches.

Routing is deliberately out of scope: FP8 router logits move the top-k
selection (measured ~15.3/16 agreement), so callers must keep the gate BF16.
"""

from __future__ import annotations

import torch

from sglang.srt.utils import is_hip

# hipBLASLt rejects N that is not a multiple of 16.
_N_ALIGN = 16


def _ops():
    try:
        from aiter import dtypes
        from aiter.ops.quant import per_token_quant_hip
    except (ImportError, ModuleNotFoundError):
        return None
    return dtypes.fp8, per_token_quant_hip


def available() -> bool:
    return is_hip() and _ops() is not None


def _pad_rows(weight: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    # Contiguous [N, K] so the transpose below is column-major. hipBLASLt
    # accepts only row-major A times column-major B; calling contiguous() on
    # the transpose turns B row-major and _scaled_mm raises.
    weight = weight.contiguous()
    out_features, in_features = weight.shape
    padded = out_features + (-out_features) % _N_ALIGN
    if padded != out_features:
        weight = torch.cat(
            [
                weight,
                weight.new_zeros((padded - out_features, in_features)),
            ]
        )
    return weight, out_features, padded


def _as_scaled_mm_weight(weight_nk: torch.Tensor) -> torch.Tensor:
    """[N, K] contiguous FP8 -> [K, N] column-major, unshuffled."""
    return weight_nk.t()


def pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize [out, in] BF16 -> ([K, N] FP8 weight, [1, N] scales, out).

    Returns the logical `out` alongside the padded tensors so `run` can slice
    the padding away. The returned ``[K, N]`` weight is column-major:
    hipBLASLt rejects a row-major B.
    """
    ops = _ops()
    if ops is None:
        raise RuntimeError("aiter PTPC FP8 GEMM is unavailable")
    fp8, per_token_quant = ops
    if weight.ndim != 2 or not weight.is_cuda:
        raise ValueError(f"expected a 2D CUDA weight, got {tuple(weight.shape)}")
    weight, out_features, padded = _pad_rows(weight)
    quantized, scale = per_token_quant(weight.contiguous(), quant_dtype=fp8)
    return (
        _as_scaled_mm_weight(quantized),
        scale.reshape(1, padded).contiguous().float(),
        out_features,
    )


def pack_prequantized(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack an [out, in] FP8 weight that already carries [out] channel scales.

    A Quark per-channel checkpoint stores what pack() would otherwise derive, so
    reusing it keeps the GEMM bit-identical to the unfused linears. Padding rows
    quantize to zero at any scale, so the pad scale is arbitrary.
    """
    ops = _ops()
    if ops is None:
        raise RuntimeError("aiter PTPC FP8 GEMM is unavailable")
    fp8, _ = ops
    if weight.ndim != 2 or not weight.is_cuda:
        raise ValueError(f"expected a 2D CUDA weight, got {tuple(weight.shape)}")
    if weight.dtype != fp8:
        raise ValueError(f"expected {fp8} weight, got {weight.dtype}")
    out_features, _in_features = weight.shape
    if scale.numel() != out_features:
        raise ValueError(f"expected {out_features} channel scales, got {scale.numel()}")
    weight, out_features, padded = _pad_rows(weight)
    if padded != out_features:
        scale = torch.cat(
            [scale.reshape(-1), scale.reshape(-1).new_ones(padded - out_features)]
        )
    return (
        _as_scaled_mm_weight(weight),
        scale.reshape(1, padded).contiguous().float(),
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


def covered_prequant(
    x_q: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor | None
) -> bool:
    ops = _ops()
    if ops is None or weight is None:
        return False
    fp8 = ops[0]
    return (
        x_q.dim() == 2
        and x_q.dtype == fp8
        and x_q.is_contiguous()
        and x_q.shape[0] > 0
        and x_scale is not None
        and x_scale.numel() >= x_q.shape[0]
    )


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    out: torch.Tensor | None = None,
    x_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """out[:, :out_features] = (x @ weight) with per-token / per-channel FP8.

    ``weight`` is the packed ``[K, N]`` tensor from ``pack``.
    """
    ops = _ops()
    if ops is None:
        raise RuntimeError("aiter PTPC FP8 GEMM is unavailable")
    fp8, per_token_quant = ops
    if x_scale is None:
        xq, xs = per_token_quant(x, quant_dtype=fp8)
        out_dtype = x.dtype
    else:
        xq, xs = x, x_scale
        out_dtype = torch.bfloat16
    padded_n = weight.shape[1]
    if out is not None and padded_n != out_features:
        raise ValueError(
            "out= is unavailable when the packed weight has padded columns"
        )
    result = torch._scaled_mm(
        xq,
        weight,
        scale_a=xs.reshape(xq.shape[0], 1),
        scale_b=scale.reshape(1, padded_n),
        out_dtype=out_dtype,
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
