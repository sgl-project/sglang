"""Portable Triton kernels for the adaLN block pattern of Qwen-Image 2.1.

``r = residual + update * gate`` and ``y = LayerNorm(r) * (1 + scale)``, with
an affine-free LayerNorm, BF16 activations and one gate/scale row per batch.
The kernels keep every rounding point of the eager chain: the product and the
sum of the residual update, ``1 + scale`` and the LayerNorm output are each
rounded to the activation dtype. The residual result is therefore identical
to eager; only the order of the LayerNorm mean and variance reduction differs.
The kernels use plain Triton, so they run on CUDA and ROCm.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_MAX_HIDDEN = 8192


@triton.jit
def _gated_residual_layernorm_modulate_kernel(
    residual_ptr,
    update_ptr,
    gate_ptr,
    scale_ptr,
    residual_out_ptr,
    out_ptr,
    seq_len,
    hidden,
    gate_batch_stride,
    scale_batch_stride,
    eps,
    HAS_UPDATE: tl.constexpr,
    HAS_NORM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // seq_len
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden
    offsets = row * hidden + cols
    dtype = residual_out_ptr.dtype.element_ty

    x = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    if HAS_UPDATE:
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(
            gate_ptr + batch * gate_batch_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        product = (update * gate).to(dtype).to(tl.float32)
        x = (x.to(tl.float32) + product).to(dtype)
        tl.store(residual_out_ptr + offsets, x, mask=mask)

    if HAS_NORM:
        xf = x.to(tl.float32)
        mean = tl.sum(xf, axis=0) / hidden
        centered = tl.where(mask, xf - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / hidden
        rstd = 1.0 / tl.sqrt(var + eps)
        normed = (centered * rstd).to(dtype).to(tl.float32)
        scale = tl.load(
            scale_ptr + batch * scale_batch_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        one_plus_scale = (1.0 + scale).to(dtype).to(tl.float32)
        tl.store(out_ptr + offsets, (normed * one_plus_scale).to(dtype), mask=mask)


def _batch_row_stride(t: torch.Tensor, batch: int, hidden: int) -> int | None:
    """Row stride of a [B, 1, D] (or [B, D]) per-batch tensor, or None."""
    if t.dtype is not torch.bfloat16 or t.shape[-1] != hidden or t.stride(-1) != 1:
        return None
    if t.numel() != batch * hidden:
        return None
    if t.dim() == 3 and t.shape[:2] == (batch, 1):
        return t.stride(0) if batch > 1 else hidden
    if t.dim() == 2 and t.shape[0] == batch:
        return t.stride(0) if batch > 1 else hidden
    return None


def can_use_gated_residual_layernorm_modulate(
    x: torch.Tensor, *per_batch: torch.Tensor
) -> bool:
    """Whether the kernels support ``x`` [B, L, D] with these per-batch rows."""
    if not (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.dim() == 3
        and x.is_contiguous()
        and x.numel() > 0
        and 0 < x.shape[-1] <= _MAX_HIDDEN
    ):
        return False
    batch, _, hidden = x.shape
    return all(
        t.device == x.device and _batch_row_stride(t, batch, hidden) is not None
        for t in per_batch
    )


def _launch(residual, update, gate, scale, eps, residual_out, out):
    batch, seq_len, hidden = residual.shape
    block = triton.next_power_of_2(hidden)
    has_update = update is not None
    has_norm = scale is not None
    gate_stride = _batch_row_stride(gate, batch, hidden) if has_update else 0
    scale_stride = _batch_row_stride(scale, batch, hidden) if has_norm else 0
    with torch.cuda.device(residual.device):
        _gated_residual_layernorm_modulate_kernel[(batch * seq_len,)](
            residual,
            update if has_update else residual,
            gate if has_update else residual,
            scale if has_norm else residual,
            residual_out if residual_out is not None else residual,
            out if out is not None else residual,
            seq_len,
            hidden,
            gate_stride,
            scale_stride,
            eps,
            HAS_UPDATE=has_update,
            HAS_NORM=has_norm,
            BLOCK=block,
            num_warps=8 if block >= 4096 else 4,
        )


def layernorm_modulate(x: torch.Tensor, scale: torch.Tensor, eps: float):
    """``LayerNorm(x) * (1 + scale)``."""
    out = torch.empty_like(x)
    _launch(x, None, None, scale, eps, None, out)
    return out


def gated_residual_add(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """``residual + update * gate``."""
    if update.shape != residual.shape or not update.is_contiguous():
        raise RuntimeError("update must be contiguous and match residual")
    residual_out = torch.empty_like(residual)
    _launch(residual, update, gate, None, 0.0, residual_out, None)
    return residual_out


def gated_residual_layernorm_modulate(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``r = residual + update * gate``; returns ``(r, LayerNorm(r) * (1 + scale))``."""
    if update.shape != residual.shape or not update.is_contiguous():
        raise RuntimeError("update must be contiguous and match residual")
    residual_out = torch.empty_like(residual)
    out = torch.empty_like(residual)
    _launch(residual, update, gate, scale, eps, residual_out, out)
    return residual_out, out
