"""Fused ZAYA1 CCA prefill conv: varlen, driven entirely by device tensors.

Replaces the per-request host loop in :func:`cca_extend
<sglang.srt.models.zaya.cca_extend>`, whose launch count grows with the batch and
whose reads of ``extend_seq_lens_cpu`` also block prefill CUDA-graph capture.
This path is four launches per layer at any batch size:

1. a shifted copy for ``lag_prev[1:] = lag_now[:-1]``,
2. :func:`_boundary_state_kernel` -- fix ``lag_prev`` at each request start from
   that request's cached lag slot, and carry its last row into the slot,
3. :func:`_cca_conv1d_varlen_kernel` -- the conv, tiled over tokens,
4. :func:`_cca_conv_state_tail_kernel` -- write each request's trailing window
   back to ``conv_state``. Must follow step 3, which reads the *incoming*
   ``conv_state`` for the halo taps of a resumed request.

The lag stream carries the *projected* ``val_proj2`` value, not the raw hidden
state: the projection is linear and bias-free, so shifting commutes with it, and
the stream is ``latent_k_dim / 2`` wide instead of ``hidden_size``. Passing
``lag_now`` / ``lag_state`` as ``None`` drops steps 1-2 for a rank whose K heads
all come from ``val_proj1``.

The conv consumes the folded grouped weight (:meth:`CCA.fold_decode_conv`), not
the two-stage ``conv_qk``; the fold is itself a convolution, so one 3-tap grouped
kernel is equivalent. Like ``cca_state_step``, a ``covered()`` predicate gates
supported inputs and the caller falls back to the reference torch path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _boundary_state_kernel(
    lag_ptr,  # [T, D]      this chunk's val_proj2 values
    lag_state_ptr,  # [S, D, 1]   per-slot carried val_proj2 value
    out_ptr,  # [T, D]      in/out: already holds the shifted copy
    cu_ptr,  # [B+1]       request token offsets
    slots_ptr,  # [B]         pool slot per request
    prefix_ptr,  # [B]         request resumes a cached prefix
    s_lag_t,
    s_ls_s,
    s_out_t,
    lag_dim,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = d < lag_dim

    start = tl.load(cu_ptr + b)
    end = tl.load(cu_ptr + b + 1)
    # Zero-length (padded) requests own no tokens; leave their slot alone.
    if end > start:
        slot = tl.load(slots_ptr + b)
        prefix = tl.load(prefix_ptr + b)

        # The first token reads the value carried over from the previous chunk
        # (zero on a fresh one). Read before the store below overwrites the row.
        prev = tl.load(
            lag_state_ptr + slot * s_ls_s + d, mask=dmask & prefix, other=0.0
        )
        tl.store(out_ptr + start * s_out_t + d, prev, mask=dmask)

        # Carry the projected value, not the raw hidden state (see module doc).
        last = tl.load(lag_ptr + (end - 1) * s_lag_t + d, mask=dmask, other=0.0)
        tl.store(lag_state_ptr + slot * s_ls_s + d, last, mask=dmask)


@triton.jit
def _cca_conv1d_varlen_kernel(
    qk_ptr,  # [T, C]        pre-conv q/k rows
    w_ptr,  # [C, CG, TAPS] folded grouped-conv weight
    bias_ptr,  # [C]
    conv_state_ptr,  # [S, C, PAD]   incoming per-slot history
    out_ptr,  # [T, C]        out: conv output
    cu_ptr,  # [B+1]
    slots_ptr,  # [B]
    prefix_ptr,  # [B]
    s_qk_t,
    s_w_o,
    s_w_i,
    s_cs_s,
    s_cs_c,
    s_out_t,
    num_tokens,
    num_requests,
    CG: tl.constexpr,  # channels per group
    TAPS: tl.constexpr,  # PAD + 1
    PAD: tl.constexpr,  # total_padding
    BLOCK_T: tl.constexpr,
    SEARCH: tl.constexpr,  # ceil(log2(num_requests)) binary-search steps
):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    g = tl.program_id(1)

    tmask = t < num_tokens
    tsafe = tl.where(tmask, t, 0)

    # Request owning each token: the largest b with cu[b] <= t, so zero-length
    # (padded) requests lose the tie and fall out.
    lo = tl.zeros([BLOCK_T], dtype=tl.int32)
    hi = tl.full([BLOCK_T], num_requests, dtype=tl.int32)
    for _ in tl.static_range(SEARCH):
        mid = (lo + hi) // 2
        take = tl.load(cu_ptr + mid) <= tsafe
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid)

    req_start = tl.load(cu_ptr + lo)
    slot = tl.load(slots_ptr + lo)
    prefix = tl.load(prefix_ptr + lo)
    local = tsafe - req_start

    co = g * CG + tl.arange(0, CG)  # this group's output channels
    ci = tl.arange(0, CG)  # in-channel index within the group

    acc = tl.zeros([BLOCK_T, CG], dtype=tl.float32)
    for m in tl.static_range(TAPS):
        # Tap m of the causal window reads request position local-(TAPS-1)+m;
        # inside the chunk its global row is t-(TAPS-1)+m.
        pos = local - (TAPS - 1) + m
        in_seq = pos >= 0
        src = tl.maximum(tsafe - (TAPS - 1) + m, 0)
        x = tl.load(
            qk_ptr + src[:, None] * s_qk_t + co[None, :],
            mask=tmask[:, None] & in_seq[:, None],
            other=0.0,
        )
        # Otherwise a halo tap from the carried history, at tap index PAD+pos.
        halo = tl.load(
            conv_state_ptr
            + slot[:, None] * s_cs_s
            + co[None, :] * s_cs_c
            + tl.maximum(PAD + pos, 0)[:, None],
            mask=tmask[:, None] & (~in_seq[:, None]) & prefix[:, None],
            other=0.0,
        )
        xm = tl.where(in_seq[:, None], x, halo)
        # Load the tap's weight slice as [ci, co] so the dot contracts over ci.
        wm = tl.load(w_ptr + co[None, :] * s_w_o + ci[:, None] * s_w_i + m)
        acc += tl.dot(xm, wm, out_dtype=tl.float32)

    acc = acc + tl.load(bias_ptr + co)[None, :]
    tl.store(
        out_ptr + tsafe[:, None] * s_out_t + co[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=tmask[:, None],
    )


@triton.jit
def _cca_conv_state_tail_kernel(
    qk_ptr,  # [T, C]
    conv_state_ptr,  # [S, C, PAD]  in/out
    cu_ptr,  # [B+1]
    slots_ptr,  # [B]
    prefix_ptr,  # [B]
    s_qk_t,
    s_cs_s,
    s_cs_c,
    num_channels,
    PAD: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < num_channels

    start = tl.load(cu_ptr + b)
    end = tl.load(cu_ptr + b + 1)
    if end > start:
        slot = tl.load(slots_ptr + b)
        prefix = tl.load(prefix_ptr + b)
        seq_len = end - start

        for i in tl.static_range(PAD):
            # Outgoing tap i is request position seq_len-PAD+i.
            pos = seq_len - PAD + i
            from_chunk = pos >= 0
            v_chunk = tl.load(
                qk_ptr + tl.maximum(start + pos, 0) * s_qk_t + c,
                mask=cmask & from_chunk,
                other=0.0,
            )
            # A chunk shorter than the window keeps its oldest taps from the
            # incoming history. Reads index seq_len+i, always above the taps
            # already written (0..i-1), so the in-place update does not alias.
            v_carry = tl.load(
                conv_state_ptr
                + slot * s_cs_s
                + c * s_cs_c
                + tl.minimum(pos + PAD, PAD - 1),
                mask=cmask & (~from_chunk) & prefix,
                other=0.0,
            )
            tl.store(
                conv_state_ptr + slot * s_cs_s + c * s_cs_c + i,
                tl.where(from_chunk, v_chunk, v_carry),
                mask=cmask,
            )


def covered(
    qk: torch.Tensor,
    lag_now: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    lag_state: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    has_prefix: Optional[torch.Tensor],
    slots: Optional[torch.Tensor],
    total_padding: int,
    groups: int,
) -> bool:
    """Whether the fused prefill conv can serve these inputs.

    ``lag_now`` / ``lag_state`` must be both present or both ``None``: a
    half-specified pair is refused rather than guessed at, since guessing "no
    lag" silently skips the chunk-boundary carry and the next chunk then resumes
    from a stale slot.
    """
    if weight is None or bias is None:
        return False
    if query_start_loc is None or has_prefix is None or slots is None:
        return False
    if total_padding < 1 or groups < 1:
        return False
    if (lag_now is None) != (lag_state is None):
        return False

    tensors = [
        qk,
        weight,
        bias,
        conv_state,
        query_start_loc,
        has_prefix,
        slots,
    ]
    if lag_now is not None:
        tensors += [lag_now, lag_state]
    if not all(t.is_cuda for t in tensors):
        return False
    if qk.ndim != 2:
        return False
    if conv_state.ndim != 3:
        return False
    if lag_now is not None:
        if lag_now.ndim != 2 or lag_state.ndim != 3:
            return False
        if qk.shape[0] != lag_now.shape[0]:
            return False

    num_channels = qk.shape[-1]
    if num_channels % groups != 0:
        return False
    ch_per_group = num_channels // groups
    # tl.dot tiles are power-of-two and at least 16 wide; ZAYA1's is head_dim.
    if ch_per_group < 16 or (ch_per_group & (ch_per_group - 1)) != 0:
        return False

    taps = total_padding + 1
    if tuple(weight.shape) != (num_channels, ch_per_group, taps):
        return False
    if bias.ndim != 1 or bias.shape[0] != num_channels:
        return False
    if conv_state.shape[-1] != total_padding or conv_state.shape[-2] != num_channels:
        return False
    if lag_now is not None:
        # The lag row is addressed as ``slot * stride(0) + d``, so its feature
        # axis must be unit-strided (a narrowed view of a wider entry still is).
        if lag_state.shape[-2] != lag_now.shape[-1]:
            return False
        if lag_state.shape[-1] != 1:
            return False
        if lag_state.stride(-2) != 1:
            return False

    num_requests = query_start_loc.shape[0] - 1
    if num_requests < 1:
        return False
    if slots.shape[0] < num_requests or has_prefix.shape[0] < num_requests:
        return False
    if not (
        query_start_loc.is_contiguous()
        and slots.is_contiguous()
        and has_prefix.is_contiguous()
    ):
        return False

    # No dtype conversion is modelled: qk goes straight into the conv pool.
    if conv_state.dtype != qk.dtype:
        return False
    if weight.dtype != qk.dtype or bias.dtype != qk.dtype:
        return False
    if not (
        qk.stride(-1) == 1 and conv_state.stride(-1) == 1 and weight.stride(-1) == 1
    ):
        return False
    if lag_now is None:
        return True
    if lag_state.dtype != lag_now.dtype:
        return False
    return lag_now.stride(-1) == 1 and lag_state.stride(-1) == 1


def cca_conv1d_fn(
    qk: torch.Tensor,
    lag_now: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    lag_state: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    has_prefix: torch.Tensor,
    slots: torch.Tensor,
    total_padding: int,
    groups: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(qk_out [T, C], lag_prev [T, D])``, updating both pool slots in
    place. ``lag_prev`` is ``None`` when the lag stream is off. Caller must have
    checked :func:`covered`.
    """
    num_tokens, num_channels = qk.shape
    num_requests = query_start_loc.shape[0] - 1
    ch_per_group = num_channels // groups
    taps = total_padding + 1
    has_lag = lag_now is not None

    qk_out = torch.empty_like(qk)
    lag_prev = torch.empty_like(lag_now) if has_lag else None
    if num_tokens == 0:
        return qk_out, lag_prev

    if has_lag:
        # val_proj2 reads the previous token: inside a request a plain shift.
        # The row at each request start is wrong here and is fixed next.
        lag_prev[1:].copy_(lag_now[:-1])

        lag_dim = lag_now.shape[-1]
        _boundary_state_kernel[(num_requests, triton.cdiv(lag_dim, 512))](
            lag_now,
            lag_state,
            lag_prev,
            query_start_loc,
            slots,
            has_prefix,
            lag_now.stride(0),
            lag_state.stride(0),
            lag_prev.stride(0),
            lag_dim,
            BLOCK_D=512,
            num_warps=4,
        )

    block_t = 64
    search_steps = max(1, (num_requests - 1).bit_length())
    _cca_conv1d_varlen_kernel[(triton.cdiv(num_tokens, block_t), groups)](
        qk,
        weight,
        bias,
        conv_state,
        qk_out,
        query_start_loc,
        slots,
        has_prefix,
        qk.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        qk_out.stride(0),
        num_tokens,
        num_requests,
        CG=ch_per_group,
        TAPS=taps,
        PAD=total_padding,
        BLOCK_T=block_t,
        SEARCH=search_steps,
        num_warps=4,
    )

    # Strictly after the conv: it reads the incoming history for halo taps.
    _cca_conv_state_tail_kernel[(num_requests, triton.cdiv(num_channels, 256))](
        qk,
        conv_state,
        query_start_loc,
        slots,
        has_prefix,
        qk.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        num_channels,
        PAD=total_padding,
        BLOCK_C=256,
        num_warps=4,
    )
    return qk_out, lag_prev
