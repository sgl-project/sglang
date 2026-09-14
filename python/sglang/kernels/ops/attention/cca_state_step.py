"""Fused ZAYA1 CCA decode conv-state step.

One kernel replaces the five state-plumbing launches around the decode conv in
:func:`cca_decode <sglang.srt.models.zaya.cca_decode>` -- two gathers, a
concatenate and two scatters::

    left_pad = conv_state.index_select(0, slots)          # gather
    padded   = cat([left_pad, qk.unsqueeze(-1)], dim=-1)  # concat
    conv_state.index_copy_(0, slots, padded[..., -W+1:])   # scatter (shift)
    lag_prev = lag_state.index_select(0, slots)            # gather
    lag_state.index_copy_(0, slots, lag_now)               # scatter

It emits the ``[T, C, W]`` conv window the matmul consumes, returns the previous
step's ``val_proj2`` value, and shifts both pool slots in place -- reading each
slot before overwriting it, so the gather/scatter pair on the same row is safe
within one program.

``lag_now`` is the *projected* ``val_proj2`` value (``latent_k_dim / 2`` wide),
not the raw hidden state. Passing ``lag_now`` / ``lag_state`` as ``None``
compiles the stream out (``HAS_LAG``) and drops its tiles from the grid, which is
what a rank whose K heads all come from ``val_proj1`` wants.

``ones_column`` appends a constant-1.0 tap to the emitted window, which lets the
caller fold the conv bias into the matmul's weight (see ``CCA.fold_decode_conv``)
and drop the separate bias add. The bias then lands in the matmul's fp32
accumulator rather than being added after the output was rounded to bf16.

The grid is ``(T, n_channel_tiles + n_lag_tiles)``: one flat tile axis where the
low ``n_channel_tiles`` entries do the conv window and history shift for their
slice of ``C`` and the rest do the lag read-then-overwrite for their slice of
``D``. Tiling that axis rather than looping it inside one program is what fills
the GPU at decode batch sizes. It never puts two programs on the same
``(slot, column)`` -- within a token the tiles partition ``C`` and ``D``
disjointly, so the read-before-write ordering stays inside one program. Slot ids
must therefore be distinct, apart from the negative padding ids, which touch
nothing; that contract is unchanged from the ``grid=(T,)`` form.

The grouped matmul is deliberately left outside: folding it in measured a LOSS
(TPOT +2.4% at C=32 on MI355X), so do not re-fold it without a measurement.

Follows ``kda_fused_decode``'s structure -- a ``covered()`` predicate gates
supported inputs and the caller falls back to the unfused chain. Triton, so it
runs on ROCm.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _cca_state_step_kernel(
    qk_ptr,  # [T, C]        this step's pre-conv q/k row
    lag_ptr,  # [T, D]       this step's val_proj2 value (lag input)
    conv_state_ptr,  # [S, C, W-1]  per-slot conv history
    lag_state_ptr,  # [S, D, 1]    per-slot previous val_proj2 value
    slots_ptr,  # [T]           mamba slot per token (<0 == padding)
    window_ptr,  # [T, C, W(+1)] out: conv window
    lag_out_ptr,  # [T, D]       out: previous val_proj2 value
    s_qk_t,
    s_lag_t,
    s_cs_s,
    s_cs_c,
    s_ls_s,
    s_win_t,
    s_win_c,
    s_lo_t,
    num_channels,
    lag_dim,
    W: tl.constexpr,  # taps == total_padding + 1
    NC_TILES: tl.constexpr,  # ceil(num_channels / BLOCK_C)
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_LAG: tl.constexpr,
    ONES_COL: tl.constexpr,  # append a constant-1.0 tap at index W
):
    t = tl.program_id(0)
    tile = tl.program_id(1)
    slot = tl.load(slots_ptr + t)

    if tile < NC_TILES:
        # ---- conv window + in-place history shift ---------------------------
        c = tile * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = c < num_channels

        qk = tl.load(qk_ptr + t * s_qk_t + c, mask=cmask, other=0.0)

        # Taps [0, W-1) come from the cached history; tap W-1 is this token.
        for w in tl.static_range(W - 1):
            hist = tl.load(
                conv_state_ptr + slot * s_cs_s + c * s_cs_c + w,
                mask=cmask & (slot >= 0),
                other=0.0,
            )
            tl.store(window_ptr + t * s_win_t + c * s_win_c + w, hist, mask=cmask)
        tl.store(window_ptr + t * s_win_t + c * s_win_c + (W - 1), qk, mask=cmask)
        if ONES_COL:
            # The bias column's activation. Exactly 1.0 in every float format,
            # so ``1.0 * b`` contributes ``b`` to the fp32 accumulator unrounded.
            ones = tl.full([BLOCK_C], 1.0, dtype=window_ptr.dtype.element_ty)
            tl.store(window_ptr + t * s_win_t + c * s_win_c + W, ones, mask=cmask)

        # Shift the history left by one: new[w] = old[w+1], new[W-2] = qk. Read
        # every old tap above before storing so the two do not alias.
        for w in tl.static_range(W - 2):
            nxt = tl.load(
                conv_state_ptr + slot * s_cs_s + c * s_cs_c + (w + 1),
                mask=cmask & (slot >= 0),
                other=0.0,
            )
            tl.store(
                conv_state_ptr + slot * s_cs_s + c * s_cs_c + w,
                nxt,
                mask=cmask & (slot >= 0),
            )
        tl.store(
            conv_state_ptr + slot * s_cs_s + c * s_cs_c + (W - 2),
            qk,
            mask=cmask & (slot >= 0),
        )
    else:
        # ---- val_proj2 lag: read before overwrite ---------------------------
        # Only reachable when HAS_LAG: the caller sets the lag tile count to 0
        # otherwise, so this arm is both compiled out and never scheduled.
        if HAS_LAG:
            d = (tile - NC_TILES) * BLOCK_D + tl.arange(0, BLOCK_D)
            dmask = d < lag_dim
            prev = tl.load(
                lag_state_ptr + slot * s_ls_s + d, mask=dmask & (slot >= 0), other=0.0
            )
            tl.store(lag_out_ptr + t * s_lo_t + d, prev, mask=dmask)
            cur = tl.load(lag_ptr + t * s_lag_t + d, mask=dmask, other=0.0)
            tl.store(lag_state_ptr + slot * s_ls_s + d, cur, mask=dmask & (slot >= 0))


def covered(
    qk: torch.Tensor,
    lag_now: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    lag_state: Optional[torch.Tensor],
    slots: Optional[torch.Tensor],
    total_padding: int,
) -> bool:
    """Whether the fused state step can serve these inputs.

    ``lag_now`` / ``lag_state`` must be both present or both ``None`` -- the
    latter compiles the lag stream out for a rank that never reads it. A
    half-specified pair is rejected rather than guessed at, since guessing wrong
    means a skipped pool write (stale lag next step) or a write of the wrong
    width.
    """
    if slots is None or total_padding < 1:
        return False
    if (lag_now is None) != (lag_state is None):
        return False
    tensors = [qk, conv_state, slots]
    if lag_now is not None:
        tensors += [lag_now, lag_state]
    if not all(t.is_cuda for t in tensors):
        return False
    if qk.ndim != 2:
        return False
    if conv_state.ndim != 3:
        return False
    if conv_state.shape[-1] != total_padding:
        return False
    if conv_state.shape[-2] != qk.shape[-1]:
        return False
    if slots.ndim != 1 or slots.shape[0] != qk.shape[0]:
        return False
    if not slots.is_contiguous():
        return False
    # The kernel stores qk straight into the conv pool and lag_now into the lag
    # pool, so no dtype conversion is modelled.
    if conv_state.dtype != qk.dtype:
        return False
    if conv_state.stride(-1) != 1 or qk.stride(-1) != 1:
        return False
    if lag_now is None:
        return True
    if lag_now.ndim != 2 or lag_state.ndim != 3:
        return False
    if lag_state.shape[-2] != lag_now.shape[-1]:
        return False
    if lag_state.shape[-1] != 1:
        return False
    if lag_now.shape[0] != qk.shape[0]:
        return False
    if lag_state.dtype != lag_now.dtype:
        return False
    # The lag pool row is addressed as ``slot * stride(0) + d``, so its feature
    # axis must be unit-strided too -- a narrowed view of a wider pool entry
    # (a rank that owns only part of val_proj2's output) still satisfies this.
    return (
        lag_state.stride(-1) == 1
        and lag_state.stride(-2) == 1
        and lag_now.stride(-1) == 1
    )


def cca_state_step(
    qk: torch.Tensor,
    lag_now: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    lag_state: Optional[torch.Tensor],
    slots: torch.Tensor,
    total_padding: int,
    ones_column: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(window, lag_prev)`` and shift both pool slots in place.

    ``window`` is ``[T, C, total_padding + 1 (+ 1 if ones_column)]`` in ``qk``'s
    dtype; ``lag_prev`` is ``[T, D]`` in ``lag_now``'s dtype, or ``None`` when the
    lag stream is compiled out. Caller must have checked :func:`covered`."""
    num_tokens, num_channels = qk.shape
    taps = total_padding + 1
    win_taps = taps + 1 if ones_column else taps
    has_lag = lag_now is not None
    lag_dim = lag_now.shape[-1] if has_lag else 0

    window = torch.empty(
        (num_tokens, num_channels, win_taps), dtype=qk.dtype, device=qk.device
    )
    lag_prev = (
        torch.empty((num_tokens, lag_dim), dtype=lag_now.dtype, device=lag_now.device)
        if has_lag
        else None
    )
    if num_tokens == 0:
        return window, lag_prev

    # One flat tile axis: [0, nc_tiles) walk the channels, the rest the lag dim.
    # See the module docstring for why no two programs share a (slot, column).
    block_c, block_d = 256, 512
    nc_tiles = triton.cdiv(num_channels, block_c)
    nd_tiles = triton.cdiv(lag_dim, block_d) if has_lag else 0

    _cca_state_step_kernel[(num_tokens, nc_tiles + nd_tiles)](
        qk,
        lag_now if has_lag else qk,
        conv_state,
        lag_state if has_lag else conv_state,
        slots,
        window,
        lag_prev if has_lag else window,
        qk.stride(0),
        lag_now.stride(0) if has_lag else 0,
        conv_state.stride(0),
        conv_state.stride(1),
        lag_state.stride(0) if has_lag else 0,
        window.stride(0),
        window.stride(1),
        lag_prev.stride(0) if has_lag else 0,
        num_channels,
        lag_dim,
        W=taps,
        NC_TILES=nc_tiles,
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        HAS_LAG=has_lag,
        ONES_COL=ones_column,
        num_warps=4,
    )
    return window, lag_prev
