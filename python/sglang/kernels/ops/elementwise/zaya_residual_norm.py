"""Fused ZAYA1 residual-scale + accumulate + RMSNorm.

Every ZAYA1 layer -- 120 of them on the 74B -- opens with the same chain, which
costs ~5 elementwise launches::

    residual, hidden = res_scale(residual, hidden)   # 2 addcmul (fp32)
    residual = residual + hidden                     # 1 add     (fp32)
    hidden   = norm(residual.to(model_dtype))        # 1 cast + 1 rmsnorm

This collapses it to one kernel. The fp32 residual stream is preserved exactly
(``residual_in_fp32`` is a checkpoint property, not a choice), the two affine
scales fold in as the ``x*s + b*s`` form that ``ResidualScaling`` already
precomputes, and only the normalized output is rounded to the model dtype -- the
same single rounding the torch chain performs at its ``.to()``.

sglang's ``RMSNorm.forward(x, residual)`` already offers a fused add+norm, but
ZAYA1 cannot use it: it would run at the model dtype, discarding the fp32
residual, and it has no notion of the per-channel residual scaling that comes
first.

Follows ``kda_fused_decode``'s structure: a ``covered()`` predicate gates
supported inputs and the caller falls back to the torch chain. Triton, so it runs
on ROCm.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_norm_kernel(
    hidden_ptr,  # [T, H] model dtype
    residual_ptr,  # [T, H] fp32, or unused when HAS_RESIDUAL is False
    hs_scale_ptr,  # [H] fp32
    hs_bias_scaled_ptr,  # [H] fp32  (bias * scale)
    res_scale_ptr,  # [H] fp32
    res_bias_scaled_ptr,  # [H] fp32
    norm_weight_ptr,  # [H] model dtype
    out_residual_ptr,  # [T, H] fp32   out
    out_hidden_ptr,  # [T, H] model dtype  out
    s_hidden_t,
    s_residual_t,
    s_out_res_t,
    s_out_hid_t,
    hidden_size,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    HAS_RES_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    mask = h < hidden_size

    hidden = tl.load(hidden_ptr + t * s_hidden_t + h, mask=mask, other=0.0).to(
        tl.float32
    )
    # (x + b) * s == x * s + (b * s), with (b * s) precomputed by fold_scales.
    hs_scale = tl.load(hs_scale_ptr + h, mask=mask, other=0.0)
    hs_bias_scaled = tl.load(hs_bias_scaled_ptr + h, mask=mask, other=0.0)
    hidden = hidden * hs_scale + hs_bias_scaled

    if HAS_RESIDUAL:
        residual = tl.load(
            residual_ptr + t * s_residual_t + h, mask=mask, other=0.0
        ).to(tl.float32)
        if HAS_RES_SCALE:
            res_scale = tl.load(res_scale_ptr + h, mask=mask, other=0.0)
            res_bias_scaled = tl.load(res_bias_scaled_ptr + h, mask=mask, other=0.0)
            residual = residual * res_scale + res_bias_scaled
        new_residual = residual + hidden
    else:
        # Layer 0: no incoming residual stream, the scaled hidden state seeds it.
        new_residual = hidden

    tl.store(out_residual_ptr + t * s_out_res_t + h, new_residual, mask=mask)

    # RMSNorm over the hidden dim, computed on the fp32 residual and rounded once
    # on store -- matching norm(residual.to(model_dtype)).
    inv = tl.rsqrt(tl.sum(new_residual * new_residual, axis=0) / hidden_size + eps)
    weight = tl.load(norm_weight_ptr + h, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_hidden_ptr + t * s_out_hid_t + h, new_residual * inv * weight, mask=mask
    )


# The hidden dim is reduced in one block, so it must fit one. ZAYA1 uses 2048
# (base) / 4096 (74B).
_MAX_HIDDEN = 8192


def covered(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    norm_weight: Optional[torch.Tensor],
    folded: bool,
) -> bool:
    """Whether the fused chain can serve these inputs. ``folded`` says the
    ``bias * scale`` constants exist, i.e. ``fold_scales`` has run."""
    if not folded or norm_weight is None:
        return False
    if not hidden_states.is_cuda or hidden_states.ndim != 2:
        return False
    hidden_size = hidden_states.shape[-1]
    if hidden_size > _MAX_HIDDEN or hidden_states.stride(-1) != 1:
        return False
    if norm_weight.numel() != hidden_size or not norm_weight.is_contiguous():
        return False
    if residual is not None:
        if residual.shape != hidden_states.shape:
            return False
        # The fp32 residual stream is the whole reason this kernel exists.
        if residual.dtype != torch.float32 or residual.stride(-1) != 1:
            return False
    return hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16)


def residual_scale_accumulate_norm(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    *,
    hs_scale: torch.Tensor,
    hs_bias_scaled: torch.Tensor,
    res_scale: Optional[torch.Tensor],
    res_bias_scaled: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normed_hidden, new_residual)``. Caller must have checked
    :func:`covered`."""
    num_tokens, hidden_size = hidden_states.shape
    new_residual = torch.empty(
        (num_tokens, hidden_size), dtype=torch.float32, device=hidden_states.device
    )
    normed = torch.empty(
        (num_tokens, hidden_size), dtype=out_dtype, device=hidden_states.device
    )
    if num_tokens == 0:
        return normed, new_residual

    has_residual = residual is not None
    has_res_scale = has_residual and res_scale is not None
    _residual_norm_kernel[(num_tokens,)](
        hidden_states,
        residual if has_residual else hidden_states,
        hs_scale,
        hs_bias_scaled,
        res_scale if has_res_scale else hs_scale,
        res_bias_scaled if has_res_scale else hs_bias_scaled,
        norm_weight,
        new_residual,
        normed,
        hidden_states.stride(0),
        residual.stride(0) if has_residual else 0,
        new_residual.stride(0),
        normed.stride(0),
        hidden_size,
        float(eps),
        HAS_RESIDUAL=has_residual,
        HAS_RES_SCALE=has_res_scale,
        BLOCK=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )
    return normed, new_residual
