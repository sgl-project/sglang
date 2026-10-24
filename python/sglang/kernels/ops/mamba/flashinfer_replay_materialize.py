"""Eager accepted-state commits using FlashInfer 0.7's native ring materializer.

Ring history is transient, with start=pending=0 at every verification entry.
Tracking endpoints must be written before the accepted live state overwrites
their common source. Live slots are unique; tracking destinations are distinct
from every live slot, as guaranteed by the request/state allocator.
"""

import torch


def make_replay_pointer_table(state, x, B, dt, A):
    """Build persistent tables at pool construction, never inside graph capture.

    Return [layers, 11] with contiguous *columns*: the State layer-view convention
    uses axis zero, while FlashInfer consumes each whole-layer column separately.
    """
    tables = []
    for tensor in (state, x, B, dt):
        tables.extend(
            ([layer.data_ptr() for layer in tensor], [tensor.stride(1)] * len(tensor))
        )
    tables.extend(([layer.data_ptr() for layer in A], [0] * len(A), [0] * len(A)))
    return torch.tensor(tables, dtype=torch.int64, device=state.device).T


def materialize_flashinfer_mamba2(
    state,
    x,
    B,
    dt,
    A,
    pointer_table,
    slots,
    last,
    tracks=None,
    track_steps=None,
    *,
    seed=None,
    philox_rounds=0,
):
    """Commit zero-based accepted/tracking endpoints, masking padded requests.

    All metadata transforms are device operations (also CUDA-graph capturable).
    No host reads of acceptance lengths and no full-state staging allocation.
    The native kernel supports identical source/destination slots for a live
    commit; no two physical request rows may write/read the same live slot.
    """
    from flashinfer.mamba import replayssm_materialize

    if slots.numel() == 0:
        return
    layers, capacity, heads, dim, dstate = state.shape
    ring_len = x.shape[3]
    width = ring_len // 2
    assert pointer_table.shape == (layers, 11)
    assert (tracks is None) == (track_steps is None)
    valid = (slots >= 0) & (slots < capacity) & (last >= 0) & (last < width)
    src = torch.where(valid, slots, -1).to(torch.int32).repeat(layers, 1)
    start = torch.zeros(slots.numel(), dtype=torch.int32, device=slots.device)
    active = torch.arange(slots.numel(), dtype=torch.int32, device=slots.device)

    def materialize(dst, lengths):
        replayssm_materialize(
            *pointer_table.unbind(dim=1),
            src,
            dst,
            start,
            lengths,
            active,
            state_dtype=state.dtype,
            input_dtype=x.dtype,
            matrixA_dtype=A.dtype,
            dim=dim,
            dstate=dstate,
            num_heads=heads,
            heads_per_group=heads // B.shape[2],
            max_window=width,
            ring_buffer_len=ring_len,
            rand_seed=seed,
            philox_rounds=philox_rounds,
            dependency_inputs=[x, B, dt, A],
            dependency_outputs=[state],
        )

    if tracks is not None:
        track_valid = (
            valid
            & (tracks >= 0)
            & (tracks < capacity)
            & (track_steps >= 0)
            & (track_steps <= last)
        )
        dst = torch.where(track_valid, tracks, -1).to(torch.int32).repeat(layers, 1)
        lengths = torch.where(track_valid, track_steps + 1, 0).to(torch.int32)
        materialize(dst, lengths)
    lengths = torch.where(valid, last + 1, 0).to(torch.int32)
    materialize(src, lengths)
    if tracks is not None:
        # SR uses destination-address-based counters. When endpoints coincide,
        # copy the accepted state exactly instead of retaining an independently
        # rounded tracking state. Other tracking endpoints keep the old-base
        # result produced before the in-place live commit.
        same_endpoint = track_valid & (track_steps == last)
        dst = torch.where(same_endpoint, tracks, -1).to(torch.int32).repeat(layers, 1)
        materialize(dst, torch.zeros_like(lengths))
