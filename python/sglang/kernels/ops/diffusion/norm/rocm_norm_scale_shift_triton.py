"""Fused (residual + gate * x) -> LayerNorm -> * (1 + scale) + shift, for ROCm.

FP32 compute with BF16/FP16 storage, the contract of the CUDA
fused_norm_scale_shift / fused_scale_residual_norm_scale_shift kernels.
Per-token modulation (4D Wan2.2 TI2V tables) is left to the native path.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

MAX_FUSED_HIDDEN = 8192


@triton.jit
def _norm_scale_shift_kernel(
    residual_out_ptr,
    out_ptr,
    residual_ptr,
    x_ptr,
    gate_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_GATE: tl.constexpr,
    HAS_AFFINE: tl.constexpr,
    SCALE_VEC: tl.constexpr,
    SHIFT_VEC: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    off = row * D + cols

    h = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        res = tl.load(residual_ptr + off, mask=mask, other=0.0).to(tl.float32)
        if HAS_GATE:
            h = res + h * tl.load(gate_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            h = res + h
        tl.store(
            residual_out_ptr + off, h.to(residual_out_ptr.dtype.element_ty), mask=mask
        )

    mean = tl.sum(h, axis=0) / D
    centered = tl.where(mask, h - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / D
    normed = centered * tl.rsqrt(var + eps)
    if HAS_AFFINE:
        normed = normed * tl.load(weight_ptr + cols, mask=mask, other=0.0).to(
            tl.float32
        ) + tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if SCALE_VEC:
        sc = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        sc = tl.load(scale_ptr).to(tl.float32)
    if SHIFT_VEC:
        sh = tl.load(shift_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        sh = tl.load(shift_ptr).to(tl.float32)
    tl.store(
        out_ptr + off,
        (normed * (1.0 + sc) + sh).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _vec(t, hidden: int) -> bool | None:
    """True for a per-channel vector, False for a scalar, None if unsupported."""
    if not isinstance(t, torch.Tensor) or not t.is_contiguous():
        return None
    if t.numel() == 1:
        return False
    if t.numel() == hidden and t.shape[-1] == hidden:
        return True
    return None


def rocm_norm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    gate: torch.Tensor | int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """(modulated, residual + gate * x), both in x.dtype; None when unsupported."""
    if (
        x.dtype not in (torch.bfloat16, torch.float16)
        or x.dim() != 3
        or x.shape[0] != 1
    ):
        return None
    if not x.is_contiguous() or x.numel() == 0:
        return None
    seq_len, hidden = x.shape[1], x.shape[2]
    if hidden > MAX_FUSED_HIDDEN:
        return None
    if (weight is None) != (bias is None):
        return None
    if weight is not None and (weight.numel() != hidden or bias.numel() != hidden):
        return None
    scale_vec, shift_vec = _vec(scale, hidden), _vec(shift, hidden)
    if scale_vec is None or shift_vec is None:
        return None
    has_res = residual is not None
    has_gate = False
    if has_res:
        if (
            residual.shape != x.shape
            or residual.dtype != x.dtype
            or not residual.is_contiguous()
        ):
            return None
        if isinstance(gate, torch.Tensor):
            if _vec(gate, hidden) is not True:
                return None
            has_gate = True
        elif gate != 1:
            return None

    out = torch.empty_like(x)
    residual_out = torch.empty_like(x) if has_res else None
    _norm_scale_shift_kernel[(seq_len,)](
        residual_out if has_res else out,
        out,
        residual if has_res else x,
        x,
        gate if has_gate else x,
        weight if weight is not None else x,
        bias if bias is not None else x,
        scale,
        shift,
        eps,
        D=hidden,
        BLOCK_D=triton.next_power_of_2(hidden),
        HAS_RESIDUAL=has_res,
        HAS_GATE=has_gate,
        HAS_AFFINE=weight is not None,
        SCALE_VEC=scale_vec,
        SHIFT_VEC=shift_vec,
        num_warps=8,
    )
    return out, residual_out
