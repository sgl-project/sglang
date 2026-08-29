"""Fused ZAYA1 Mixture-of-Depths (MOD) skip-path kernels.

ZAYA1's router can send a token to a "skip" slot instead of an expert, and the
block blends the two outcomes. The torch form costs six elementwise launches per
MoE layer, split either side of the cross-rank expert reduce::

    mod_mask = (indices != num_experts).to(dtype)   # ne, to
    masked   = mod_mask * experts_out               # mul     <- before reduce
    # ... all-reduce(masked) ...
    mod_out  = hidden_states * probs                # mul
    out      = masked + (1 - mod_mask) * mod_out    # sub, addcmul

The reduce sits in the middle, so this is two kernels rather than one:

    premask : masked[t,h] = (indices[t] != N) ? experts_out[t,h] : 0
    blend   : out[t,h]    = masked[t,h] + ((indices[t] == N) ? hs[t,h]*probs[t] : 0)

Both recompute the predicate from ``indices`` directly, so the ``mod_mask``
tensor never has to be materialized or passed between them.

Why the mask is applied before the reduce (and the skip term after): ``mod_out``
is replicated on every rank, so folding it in first would sum it ``tp_size``
times. See ``mod_premask_experts`` / ``mod_blend`` in models/zaya.py, which stay
as the reference implementation and the fallback.

Follows ``kda_fused_decode``'s structure -- a ``covered()`` predicate gates
supported inputs and the caller falls back to the torch chain -- in Triton so it
runs on ROCm.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mod_premask_kernel(
    experts_ptr,
    indices_ptr,
    out_ptr,
    s_experts_t,
    s_out_t,
    hidden_size,
    skip_id,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    routed = tl.load(indices_ptr + t) != skip_id
    for h0 in tl.range(0, hidden_size, BLOCK):
        h = h0 + tl.arange(0, BLOCK)
        m = h < hidden_size
        v = tl.load(experts_ptr + t * s_experts_t + h, mask=m, other=0.0)
        tl.store(out_ptr + t * s_out_t + h, tl.where(routed, v, 0.0), mask=m)


@triton.jit
def _mod_blend_kernel(
    masked_ptr,
    hidden_ptr,
    probs_ptr,
    indices_ptr,
    out_ptr,
    s_masked_t,
    s_hidden_t,
    s_out_t,
    hidden_size,
    skip_id,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    skipped = tl.load(indices_ptr + t) == skip_id
    prob = tl.load(probs_ptr + t)
    for h0 in tl.range(0, hidden_size, BLOCK):
        h = h0 + tl.arange(0, BLOCK)
        m = h < hidden_size
        acc = tl.load(masked_ptr + t * s_masked_t + h, mask=m, other=0.0)
        hs = tl.load(hidden_ptr + t * s_hidden_t + h, mask=m, other=0.0)
        # Accumulate in fp32 then round once on store, matching the torch chain
        # (whose mul/addcmul run at the tensors' own dtype after promotion).
        acc = acc.to(tl.float32) + tl.where(
            skipped, hs.to(tl.float32) * prob.to(tl.float32), 0.0
        )
        tl.store(out_ptr + t * s_out_t + h, acc, mask=m)


def _rows_ok(indices: torch.Tensor, num_tokens: int) -> bool:
    """``indices`` must be one routed expert per token: ``[T]`` or ``[T, 1]``."""
    if indices.shape[0] != num_tokens:
        return False
    if indices.ndim == 1:
        return True
    return indices.ndim == 2 and indices.shape[1] == 1


def covered(
    experts_out: torch.Tensor,
    indices: torch.Tensor,
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
) -> bool:
    """Whether the fused MOD kernels can serve these inputs.

    Restricted to top-1 routing (``indices`` / ``probs`` one column wide), which
    is what ZAYA1 ships; wider top-k needs the cumulative-skip logic in
    ``ZayaBlock.forward`` and falls back.
    """
    if not all(t.is_cuda for t in (experts_out, indices, hidden_states, probs)):
        return False
    if experts_out.ndim != 2 or hidden_states.ndim != 2:
        return False
    num_tokens = experts_out.shape[0]
    if hidden_states.shape != experts_out.shape:
        return False
    if not (_rows_ok(indices, num_tokens) and _rows_ok(probs, num_tokens)):
        return False
    if not (indices.is_contiguous() and probs.is_contiguous()):
        return False
    return (
        experts_out.stride(-1) == 1
        and hidden_states.stride(-1) == 1
        and experts_out.dtype == hidden_states.dtype
    )


def mod_premask(
    experts_out: torch.Tensor, indices: torch.Tensor, num_moe_experts: int
) -> torch.Tensor:
    """Zero the rows routed to the skip slot. Caller must have checked
    :func:`covered`."""
    out = torch.empty_like(experts_out)
    num_tokens, hidden_size = experts_out.shape
    if num_tokens == 0:
        return out
    _mod_premask_kernel[(num_tokens,)](
        experts_out,
        indices,
        out,
        experts_out.stride(0),
        out.stride(0),
        hidden_size,
        num_moe_experts,
        BLOCK=1024,
        num_warps=4,
    )
    return out


def mod_blend(
    masked_experts_reduced: torch.Tensor,
    indices: torch.Tensor,
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    num_moe_experts: int,
) -> torch.Tensor:
    """Add the skip-path term back for the rows the router skipped. Caller must
    have checked :func:`covered`."""
    out = torch.empty_like(masked_experts_reduced)
    num_tokens, hidden_size = masked_experts_reduced.shape
    if num_tokens == 0:
        return out
    _mod_blend_kernel[(num_tokens,)](
        masked_experts_reduced,
        hidden_states,
        probs,
        indices,
        out,
        masked_experts_reduced.stride(0),
        hidden_states.stride(0),
        out.stride(0),
        hidden_size,
        num_moe_experts,
        BLOCK=1024,
        num_warps=4,
    )
    return out
