"""Prefix-cache track copies for the CUDA-graph captured KDA extend.

The eager extend selects tracked rows on the host (``nonzero`` / masked index
lists) and scatters with data-dependent shapes. Under a breakable prefill CUDA
graph the batch changes on every replay, so these kernels walk a fixed number of
rows (the bucket's sequence bound) and read a per-row mode instead:

* ``track_mode[r] == 0``: row ``r`` is not tracked (or is padding); no-op.
* ``track_mode[r] == 1``: the tracked length is chunk aligned; the snapshot is
  the post-extend state of the row's own cache slot.
* ``track_mode[r] == 2``: the snapshot is the chunk-boundary state the chunked
  delta-rule kernel wrote to ``h_track[r]`` (fp32).

Every tracked row (mode 1 or 2) also snapshots its conv window: the
``STATE_LEN`` raw pre-conv input rows ending at the tracked boundary.
"""

import torch
import triton
import triton.language as tl

TRACK_NONE = 0
TRACK_FINAL_STATE = 1
TRACK_CHUNK_STATE = 2


@triton.jit
def _kda_track_conv_window_kernel(
    conv_pool,  # [slots, STATE_LEN, dim]
    x,  # [tokens, dim] raw pre-conv input
    track_mode,  # [rows] int32
    track_conv_start,  # [rows] int32, first token row of the window (unclamped)
    track_dst,  # [rows] int64 physical track slot
    num_tokens,  # [1] int32 logical tokens of the batch
    stride_pool_slot,
    stride_pool_row,
    stride_x_token,
    dim,
    STATE_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    r = tl.program_id(0)
    mode = tl.load(track_mode + r)
    if mode == 0:
        return
    dst = tl.load(track_dst + r).to(tl.int64)
    start = tl.load(track_conv_start + r)
    total = tl.load(num_tokens)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < dim
    for j in tl.static_range(STATE_LEN):
        # Same clamp as the eager index list: clamp(0, total - 1).
        t = tl.minimum(tl.maximum(start + j, 0), total - 1).to(tl.int64)
        val = tl.load(x + t * stride_x_token + offs, mask=mask)
        tl.store(
            conv_pool + dst * stride_pool_slot + j * stride_pool_row + offs,
            val.to(conv_pool.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _kda_track_ssm_kernel(
    ssm_pool,  # [slots, ...] per-slot contiguous
    h_track,  # [rows, ...] fp32 chunk-boundary snapshots
    track_mode,  # [rows] int32
    track_dst,  # [rows] int64 physical track slot
    cache_indices,  # [rows] physical cache slot of the row
    stride_ssm_slot,
    stride_h_row,
    numel,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    mode = tl.load(track_mode + r)
    if mode == 0:
        return
    dst = tl.load(track_dst + r).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    if mode == 1:
        src = tl.load(cache_indices + r).to(tl.int64)
        val = tl.load(ssm_pool + src * stride_ssm_slot + offs, mask=mask)
    else:
        val = tl.load(h_track + r.to(tl.int64) * stride_h_row + offs, mask=mask)
    tl.store(
        ssm_pool + dst * stride_ssm_slot + offs,
        val.to(ssm_pool.dtype.element_ty),
        mask=mask,
    )


def kda_track_conv_window(
    conv_pool: torch.Tensor,
    x: torch.Tensor,
    track_mode: torch.Tensor,
    track_conv_start: torch.Tensor,
    track_dst: torch.Tensor,
    num_tokens: torch.Tensor,
) -> None:
    """``conv_pool[track_dst[r]] = x[clamp(start[r] + arange(STATE_LEN))]`` for
    every tracked row ``r``; the grid depends only on the row bound."""
    rows = track_mode.shape[0]
    state_len, dim = conv_pool.shape[-2], conv_pool.shape[-1]
    assert x.shape[-1] == dim and x.stride(-1) == 1 and conv_pool.stride(-1) == 1
    block_d = 1024
    _kda_track_conv_window_kernel[(rows, triton.cdiv(dim, block_d))](
        conv_pool,
        x,
        track_mode,
        track_conv_start,
        track_dst,
        num_tokens,
        conv_pool.stride(0),
        conv_pool.stride(1),
        x.stride(0),
        dim,
        STATE_LEN=state_len,
        BLOCK_D=block_d,
    )


def kda_track_ssm_state(
    ssm_pool: torch.Tensor,
    h_track: torch.Tensor,
    track_mode: torch.Tensor,
    track_dst: torch.Tensor,
    cache_indices: torch.Tensor,
) -> None:
    """Snapshot the tracked rows' SSM state into their track slots (mode 1:
    the row's final state, mode 2: its fp32 chunk-boundary state)."""
    rows = track_mode.shape[0]
    numel = ssm_pool[0].numel()
    assert ssm_pool[0].is_contiguous(), "per-slot SSM state must be contiguous"
    assert h_track.shape[0] >= rows and h_track[0].is_contiguous()
    assert h_track[0].numel() == numel
    block = 1024
    _kda_track_ssm_kernel[(rows, triton.cdiv(numel, block))](
        ssm_pool,
        h_track,
        track_mode,
        track_dst,
        cache_indices,
        ssm_pool.stride(0),
        h_track.stride(0),
        numel,
        BLOCK=block,
    )
