"""Fail-closed PTPC FP8 adapter for Kimi-K3's BF16 decode projections.

The checkpoint keeps `self_attn.*`, `shared_experts.*` and the latent
projections in BF16, and at decode those GEMMs are HBM-bound on weight bytes
(measured ~4 TB/s on MI355X, which is where the machine tops out). Quantizing
the weight to FP8 per output channel and the activation per token halves the
bytes the GEMM has to stream, which is the only lever that helps once the
kernel is already bandwidth-saturated.

The GEMM is aiter ``gemm_a8w8_bpreshuffle``. On the tuned gfx950 shapes that
is the ``kernel_gemm_0`` ATOM uses for dense PTPC (about 6 us on the latent
``[3584, 7168]`` projection). hipBLASLt ``torch._scaled_mm`` picks the F8BS
solution instead, which measured 13-17 us on the same shape, so it is only
the fallback for shapes the preshuffle kernel rejects.

``gemm_a8w8_bpreshuffle`` needs N % 64 == 0 and a (16, 16) shuffled weight,
which also needs K % 32 == 0. Short weights are zero-padded and the padding
columns are dropped. A shape the kernel still rejects is packed for
``torch._scaled_mm`` (column-major B, N padded to 16).

Activation quant is aiter ``per_token_quant_hip``
(``dynamic_per_token_scaled_quant``), the same kernel ATOM launches.

Routing is deliberately out of scope: FP8 router logits move the top-k
selection (measured ~15.3/16 agreement), so callers must keep the gate BF16.
"""

from __future__ import annotations

import torch

from sglang.srt.utils import is_hip

# gemm_a8w8_bpreshuffle tiles N by 64. The (16, 16) shuffle also needs K % 32.
_N_ALIGN = 64
_K_ALIGN = 32


def _ops():
    try:
        from aiter import dtypes
        from aiter.ops.quant import per_token_quant_hip
    except (ImportError, ModuleNotFoundError):
        return None
    return dtypes.fp8, per_token_quant_hip


def available() -> bool:
    return is_hip() and _ops() is not None


def _pad(weight: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Zero-pad [N, K] so N % 64 == 0 and K % 32 == 0."""
    weight = weight.contiguous()
    out_features, in_features = weight.shape
    pad_n = (-out_features) % _N_ALIGN
    pad_k = (-in_features) % _K_ALIGN
    if pad_n or pad_k:
        weight = torch.nn.functional.pad(weight, (0, pad_k, 0, pad_n))
    return weight, out_features


def _as_scaled_mm_weight(weight_nk: torch.Tensor) -> torch.Tensor:
    """[N, K] contiguous FP8 -> [K, N] column-major, unshuffled.

    hipBLASLt accepts only row-major A times column-major B. contiguous() on
    this transpose would make B row-major and ``_scaled_mm`` would raise.
    """
    return weight_nk.t()


def _bpreshuffle_weight(
    weight_nk: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor | None:
    """(16, 16)-shuffled [N, K], or None when the kernel rejects the shape."""
    n, k = weight_nk.shape
    if n % _N_ALIGN or k % _K_ALIGN:
        return None
    try:
        from aiter import gemm_a8w8_bpreshuffle
        from aiter.ops.shuffle import shuffle_weight
    except (ImportError, ModuleNotFoundError):
        return None
    shuffled = shuffle_weight(weight_nk, layout=(16, 16))
    probe_x = torch.zeros((1, k), device=weight_nk.device, dtype=weight_nk.dtype)
    probe_xs = torch.ones((1, 1), device=weight_nk.device, dtype=torch.float32)
    probe_ws = scale.reshape(n, 1).contiguous().float()
    try:
        gemm_a8w8_bpreshuffle(
            probe_x, shuffled, probe_xs, probe_ws, dtype=torch.bfloat16
        )
    except RuntimeError:
        return None
    return shuffled


def _finish(
    quantized: torch.Tensor, scale: torch.Tensor, out_features: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    shuffled = _bpreshuffle_weight(quantized, scale)
    if shuffled is not None:
        # Contiguous [N, K] is the preshuffle layout. run() tells the two
        # layouts apart by contiguity: the scaled_mm fallback is a transpose.
        return shuffled, scale.reshape(-1, 1).contiguous().float(), out_features
    return (
        _as_scaled_mm_weight(quantized),
        scale.reshape(1, -1).contiguous().float(),
        out_features,
    )


def pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize [out, in] BF16 and pack it for the PTPC GEMM.

    Returns the logical `out` alongside the padded tensors so `run` can slice
    the padding away.
    """
    ops = _ops()
    if ops is None:
        raise RuntimeError("aiter PTPC FP8 GEMM is unavailable")
    fp8, per_token_quant = ops
    if weight.ndim != 2 or not weight.is_cuda:
        raise ValueError(f"expected a 2D CUDA weight, got {tuple(weight.shape)}")
    weight, out_features = _pad(weight)
    quantized, scale = per_token_quant(weight, quant_dtype=fp8)
    return _finish(quantized, scale, out_features)


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
    weight, out_features = _pad(weight)
    padded_n = weight.shape[0]
    if padded_n != out_features:
        scale = torch.cat(
            [scale.reshape(-1), scale.reshape(-1).new_ones(padded_n - out_features)]
        )
    return _finish(weight, scale, out_features)


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


def _match_k(x: torch.Tensor, packed_k: int) -> torch.Tensor:
    if x.shape[-1] == packed_k:
        return x
    if x.shape[-1] > packed_k:
        raise ValueError(
            f"activation K {x.shape[-1]} is wider than the packed weight K {packed_k}"
        )
    return torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    out: torch.Tensor | None = None,
    x_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """out[:, :out_features] = (x @ weight) with per-token / per-channel FP8.

    A contiguous ``weight`` is the shuffled ``[N, K]`` preshuffle pack. A
    non-contiguous one is the column-major ``[K, N]`` hipBLASLt fallback.
    """
    ops = _ops()
    if ops is None:
        raise RuntimeError("aiter PTPC FP8 GEMM is unavailable")
    fp8, per_token_quant = ops
    preshuffle = weight.is_contiguous()
    if preshuffle:
        _packed_n, packed_k = weight.shape
    else:
        packed_k, _packed_n = weight.shape
    if x_scale is None:
        xq, xs = per_token_quant(_match_k(x, packed_k), quant_dtype=fp8)
        out_dtype = x.dtype
    else:
        xq, xs = _match_k(x, packed_k), x_scale
        out_dtype = torch.bfloat16
    if preshuffle:
        from aiter import gemm_a8w8_bpreshuffle

        result = gemm_a8w8_bpreshuffle(
            xq,
            weight,
            xs.reshape(xq.shape[0], 1).contiguous().float(),
            scale.reshape(weight.shape[0], 1),
            dtype=out_dtype,
        )
    else:
        padded_n = weight.shape[1]
        result = torch._scaled_mm(
            xq,
            weight,
            scale_a=xs.reshape(xq.shape[0], 1),
            scale_b=scale.reshape(1, padded_n),
            out_dtype=out_dtype,
            out=out if padded_n == out_features else None,
        )
    if result.shape[1] != out_features:
        result = result[:, :out_features]
    if out is not None and result.data_ptr() != out.data_ptr():
        out.copy_(result)
        return out
    return result


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
