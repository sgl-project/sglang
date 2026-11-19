"""State-row copies of the eager GDN extend, one launch each.

The eager extend moves state-pool rows with chains of index ops: the FlashInfer
prefill reads its initial states through ``gather_pool_states`` and writes the
final states back through ``scatter_pool_states``; prefix tracking copies the
chunk-boundary / final states and the conv windows into the track slots (index
plus index_put each). These kernels make the same copies in one launch each, with
no arithmetic and no dtype change, so every byte equals the index ops' result.
Slot 0 stays the padding sink: a row whose index is <= 0 reads zeros and writes
zeros to slot 0, as ``gather_pool_states`` / ``scatter_pool_states`` do.

The ``*_ok`` predicates hold only on SM100+ CUDA devices and only where the copy
keeps the dtype and the kernel addresses the layout; callers keep the index ops
otherwise. On SM100+ they log the first engagement and each site's decline reasons
once. ``gdn_prefill_track`` holds the CUDA-graph variant of the track copies, which
reads per-row modes over a fixed row bound instead of the eager path's index lists.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch
from sglang.srt.utils import print_info_once

_BLOCK = 4096
_BLOCK_D = 1024


@triton.jit
def _gather_pool_rows_kernel(
    states,  # [slots, ...] per-slot contiguous
    indices,  # [rows] pool slot, <= 0 = padding
    out,  # [rows, ...] contiguous
    stride_slot,
    numel,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    slot = tl.load(indices + r).to(tl.int64)
    live = slot > 0
    val = tl.load(
        states + tl.maximum(slot, 0) * stride_slot + offs, mask=mask & live, other=0.0
    )
    tl.store(out + r.to(tl.int64) * numel + offs, val, mask=mask)


@triton.jit
def _scatter_pool_rows_kernel(
    states,  # [slots, ...] per-slot contiguous
    indices,  # [rows] pool slot, <= 0 = padding (writes zeros to slot 0)
    updates,  # [rows, ...] contiguous
    stride_slot,
    numel,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    slot = tl.load(indices + r).to(tl.int64)
    live = slot > 0
    val = tl.load(updates + r.to(tl.int64) * numel + offs, mask=mask & live, other=0.0)
    tl.store(states + tl.maximum(slot, 0) * stride_slot + offs, val, mask=mask)


@triton.jit
def _track_state_rows_kernel(
    states,  # [slots, ...] per-slot contiguous
    h,  # [chunks, ...] per-row contiguous, states' dtype
    h_src,  # [n_h] row of h
    h_dst,  # [n_h] track slot
    final_src,  # [n_final] cache slot (already holds the row's final state)
    final_dst,  # [n_final] track slot
    n_h,
    stride_slot,
    stride_h,
    numel,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    if r < n_h:
        src = h + tl.load(h_src + r).to(tl.int64) * stride_h
        dst = tl.load(h_dst + r).to(tl.int64)
    else:
        j = r - n_h
        src = states + tl.load(final_src + j).to(tl.int64) * stride_slot
        dst = tl.load(final_dst + j).to(tl.int64)
    tl.store(
        states + dst * stride_slot + offs, tl.load(src + offs, mask=mask), mask=mask
    )


@triton.jit
def _track_conv_rows_kernel(
    conv,  # [slots, dim, STATE_LEN]
    x,  # [tokens, dim] raw pre-conv input
    tokens,  # [rows, STATE_LEN] token row per window position
    dst,  # [rows] track slot
    stride_slot,
    stride_dim,
    stride_pos,
    stride_x_token,
    dim,
    STATE_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < dim
    slot = tl.load(dst + r).to(tl.int64)
    for j in tl.static_range(STATE_LEN):
        t = tl.load(tokens + r * STATE_LEN + j).to(tl.int64)
        val = tl.load(x + t * stride_x_token + offs, mask=mask)
        tl.store(
            conv + slot * stride_slot + offs.to(tl.int64) * stride_dim + j * stride_pos,
            val,
            mask=mask,
        )


_ON = "GDN extend state copies: on (one copy kernel per state-row move)"
_DTYPE = "a dtype change"
_LAYOUT = "a layout the copy kernel does not address"


def _engage(*, site: str, why: str) -> bool:
    """True when ``why`` is empty; logs the first engagement and, per site, each
    decline reason once."""
    if why:
        print_info_once(f"GDN extend state copies: {site} keeps the index ops ({why})")
        return False
    print_info_once(_ON)
    return True


def _slots_ok(states: torch.Tensor) -> bool:
    return states.dim() >= 2 and states[0].is_contiguous()


def _on_cuda(*tensors: Optional[torch.Tensor]) -> bool:
    """The copy kernels are Triton launches: every tensor they address is on CUDA."""
    return all(t is None or t.is_cuda for t in tensors)


def gather_pool_rows_ok(
    *, states: torch.Tensor, indices: torch.Tensor, dtype: torch.dtype
) -> bool:
    """Whether ``gather_pool_rows`` serves a gather whose rows are wanted in ``dtype``."""
    if not current_arch().is_sm_at_least(10) or not _on_cuda(states, indices):
        return False
    if dtype != states.dtype:
        why = _DTYPE
    elif indices.dim() != 1 or not _slots_ok(states):
        why = _LAYOUT
    else:
        why = ""
    return _engage(site="pool gather", why=why)


def gather_pool_rows(*, states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``gather_pool_states(states, indices)[2]`` in one launch."""
    rows = indices.shape[0]
    numel = states[0].numel()
    out = states.new_empty((rows, *states.shape[1:]))
    if rows:
        _gather_pool_rows_kernel[(rows, triton.cdiv(numel, _BLOCK))](
            states, indices, out, states.stride(0), numel, BLOCK=_BLOCK
        )
    return out


def scatter_pool_rows_ok(
    *, states: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor
) -> bool:
    if not current_arch().is_sm_at_least(10) or not _on_cuda(states, indices, updates):
        return False
    if updates.dtype != states.dtype:
        why = _DTYPE
    elif (
        indices.dim() != 1
        or not updates.is_contiguous()
        or tuple(updates.shape) != (indices.shape[0], *states.shape[1:])
        or not _slots_ok(states)
    ):
        why = _LAYOUT
    else:
        why = ""
    return _engage(site="pool write-back", why=why)


def scatter_pool_rows(
    *, states: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor
) -> None:
    """``scatter_pool_states`` of ``updates`` at ``indices`` in one launch."""
    rows = indices.shape[0]
    numel = states[0].numel()
    if rows:
        _scatter_pool_rows_kernel[(rows, triton.cdiv(numel, _BLOCK))](
            states, indices, updates, states.stride(0), numel, BLOCK=_BLOCK
        )


def track_state_rows_ok(
    *,
    states: torch.Tensor,
    h: Optional[torch.Tensor],
    h_rows: int,
    snapshot: bool,
    recompute_rows: int,
) -> bool:
    """Whether ``track_state_rows`` serves the track rows: ``h_rows`` from the per-chunk
    states ``h``, none from an in-kernel snapshot and none recomputed."""
    if not current_arch().is_sm_at_least(10) or not _on_cuda(states, h):
        return False
    rows = None if h is None else h.squeeze(0)
    if snapshot:
        why = "rows read the in-kernel snapshot"
    elif recompute_rows:
        why = "recomputed rows"
    elif h_rows and rows is None:
        why = "no per-chunk states"
    elif h_rows and rows.dtype != states.dtype:
        why = _DTYPE
    elif not _slots_ok(states) or (
        h_rows and (rows.shape[1:] != states.shape[1:] or not rows[0].is_contiguous())
    ):
        why = _LAYOUT
    else:
        why = ""
    return _engage(site="state track", why=why)


def track_state_rows(
    *,
    states: torch.Tensor,
    h: Optional[torch.Tensor],
    h_src: torch.Tensor,
    h_dst: torch.Tensor,
    final_src: torch.Tensor,
    final_dst: torch.Tensor,
) -> None:
    """``states[h_dst] = h[h_src]`` then ``states[final_dst] = states[final_src]`` in
    one launch. The destinations are distinct track slots and no source is a track
    slot, so the order of the rows does not matter."""
    n_h = h_src.numel()
    rows = n_h + final_src.numel()
    if rows == 0:
        return
    numel = states[0].numel()
    if n_h:
        h = h.squeeze(0)
        stride_h = h.stride(0)
    else:
        h, stride_h = states, states.stride(0)
    _track_state_rows_kernel[(rows, triton.cdiv(numel, _BLOCK))](
        states,
        h,
        h_src if n_h else final_src,
        h_dst if n_h else final_dst,
        final_src if final_src.numel() else h_src,
        final_dst if final_src.numel() else h_dst,
        n_h,
        states.stride(0),
        stride_h,
        numel,
        BLOCK=_BLOCK,
    )


def track_conv_rows_ok(
    *, conv: torch.Tensor, x: torch.Tensor, tokens: torch.Tensor
) -> bool:
    if not current_arch().is_sm_at_least(10) or not _on_cuda(conv, x, tokens):
        return False
    if conv.dtype != x.dtype:
        why = _DTYPE
    elif (
        conv.dim() != 3
        or x.dim() != 2
        or x.stride(1) != 1
        or conv.shape[1] != x.shape[1]
        or tokens.dim() != 2
        or tokens.shape[1] != conv.shape[2]
        or not tokens.is_contiguous()
    ):
        why = _LAYOUT
    else:
        why = ""
    return _engage(site="conv track", why=why)


def track_conv_rows(
    *, conv: torch.Tensor, x: torch.Tensor, tokens: torch.Tensor, dst: torch.Tensor
) -> None:
    """``conv[dst] = x[tokens].transpose(1, 2)`` in one launch."""
    rows = tokens.shape[0]
    if rows == 0:
        return
    dim = conv.shape[1]
    _track_conv_rows_kernel[(rows, triton.cdiv(dim, _BLOCK_D))](
        conv,
        x,
        tokens,
        dst,
        conv.stride(0),
        conv.stride(1),
        conv.stride(2),
        x.stride(0),
        dim,
        STATE_LEN=conv.shape[2],
        BLOCK_D=_BLOCK_D,
    )
