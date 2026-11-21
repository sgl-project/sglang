"""Ownership and materialization boundaries for deferred GDN spec commits.

The active state is a checkpoint plus request-keyed accepted factors. Cached
tracking slots remain self-contained checkpoints. Slot operations run on the
same forward stream as the state pool; CPU export also synchronizes it.

Numerical contract: the GDN recurrence is unchanged, but BF16 checkpoint
rounding is part of the execution schedule. Deferred commit rounds the active
checkpoint only at a capacity fold or an external checkpoint-only read. A
tracking checkpoint is rounded independently and does not drain active history.
Verify/commit retains only accepted factors; rejected drafts are overwritten.
An explicit flush is a BF16 rounding boundary, so inserting a flush need not
preserve subsequent states, logits, or generated tokens bit for bit. Compare
against a sequential reference with the same flush schedule, not eager's
different rounding schedule. Kernel arithmetic also has finite precision;
unit tests bound that error on fixed inputs rather than promise a universal
model-level accuracy or token-equivalence guarantee.
"""

from collections import Counter

import torch
import triton
import triton.language as tl


@triton.jit
def _bind_rows(
    rows, slots, row_to_slot, slot_to_row, N: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(rows + i, i < N, other=-1).to(tl.int64)
    slot = tl.load(slots + i, i < N, other=-1).to(tl.int64)
    valid = (i < N) & (row >= 0) & (slot > 0)
    tl.store(row_to_slot + row, slot, valid)
    tl.store(slot_to_row + slot, row, valid)


@triton.jit
def _reset_owned_slots(
    slots,
    row_to_slot,
    slot_to_row,
    write_pos,
    cache_base,
    is_flush,
    stats,
    N: tl.constexpr,
    FORGET: tl.constexpr,
    HAS_STATS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + i, i < N, other=-1).to(tl.int64)
    valid = (i < N) & (slot > 0)
    row = tl.load(slot_to_row + slot, valid, other=-1).to(tl.int64)
    current = tl.load(row_to_slot + row, valid & (row >= 0), other=-1)
    owned = valid & (row >= 0) & (current == slot)
    if HAS_STATS:
        pending = tl.load(write_pos + row, owned, other=0).to(tl.int64)
        old = tl.load(stats + row * 9 + 7, owned, other=0)
        tl.store(stats + row * 9 + 7, old + pending, owned)
    tl.store(write_pos + row, 0, owned)
    tl.store(cache_base + row, 0, owned)
    tl.store(is_flush + row, 0, owned)
    if FORGET:
        tl.store(row_to_slot + row, -1, owned)
        tl.store(slot_to_row + slot, -1, valid)


@triton.jit
def _prepare_materialize(
    slots,
    row_to_slot,
    slot_to_row,
    write_pos,
    is_flush,
    state_indices,
    replay_indices,
    accept_lens,
    stats,
    N: tl.constexpr,
    HAS_STATS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + i, i < N, other=-1).to(tl.int64)
    valid = (i < N) & (slot > 0)
    row = tl.load(slot_to_row + slot, valid, other=-1).to(tl.int64)
    current = tl.load(row_to_slot + row, valid & (row >= 0), other=-1)
    owned = valid & (row >= 0) & (current == slot)
    pending = tl.load(write_pos + row, owned, other=0)
    dirty = owned & (pending > 0)
    # A batched copy may name an active source more than once. Fold it once;
    # two commit programs must never read/write the same checkpoint together.
    for start in range(0, N, BLOCK):
        j = start + tl.arange(0, BLOCK)
        other = tl.load(slots + j, j < N, other=-1)
        duplicate = (
            tl.sum(
                ((other[None, :] == slot[:, None]) & (j[None, :] < i[:, None])).to(
                    tl.int32
                ),
                axis=1,
            )
            > 0
        )
        dirty = dirty & ~duplicate
    tl.store(is_flush + row, 1, dirty)
    tl.store(state_indices + i, tl.where(dirty, slot, -1), i < N)
    tl.store(replay_indices + i, tl.maximum(row, 0), i < N)
    tl.store(accept_lens + i, 0, i < N)
    if HAS_STATS:
        old = tl.load(stats + row * 9 + 6, dirty, other=0)
        tl.store(stats + row * 9 + 6, old + 1, dirty)


@triton.jit
def _record_commit(
    stats,
    write_pos,
    is_flush,
    rows,
    slots,
    accepted,
    tracks,
    steps,
    N: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    slot = tl.load(slots + i, i < N, other=-1)
    valid = (i < N) & (slot >= 0)
    row = tl.load(rows + i, valid, other=0).to(tl.int64)
    length = tl.load(write_pos + row, valid, other=0).to(tl.int64)
    acc = tl.load(accepted + i, valid, other=0).to(tl.int64)
    fold = tl.load(is_flush + row, valid, other=0) != 0
    track = tl.full((BLOCK,), False, tl.int1)
    if HAS_TRACK:
        track = (tl.load(tracks + i, valid, other=-1) >= 0) & (
            tl.load(steps + i, valid, other=-1) >= 0
        )
    for column in tl.static_range(6):
        value = tl.full((BLOCK,), 1, tl.int64)
        if column == 1:
            value = acc
        elif column == 2:
            value = (fold & (length > 0)).to(tl.int64)
        elif column == 3:
            value = track.to(tl.int64)
        elif column == 4:
            value = tl.maximum(length - acc, 0)
        elif column == 5:
            value = length
        old = tl.load(stats + row * 9 + column, valid, other=0)
        new = tl.maximum(old, value) if column == 5 else old + value
        tl.store(stats + row * 9 + column, new, valid)
    # Column 8 is global, in reserved row zero; no per-request writer uses it.
    tl.atomic_add(stats + 8, 1)


def record_gdn_commit(stats, write_pos, is_flush, rows, slots, accepted, tracks, steps):
    """Diagnostic only; enable in separate probes, outside formal timing."""
    n = rows.numel()
    _record_commit[(1,)](
        stats,
        write_pos,
        is_flush,
        rows,
        slots,
        accepted,
        tracks if tracks is not None else slots,
        steps if steps is not None else accepted,
        N=n,
        HAS_TRACK=tracks is not None and steps is not None,
        BLOCK=triton.next_power_of_2(max(1, n)),
    )


def read_gdn_commit_stats(pool):
    stats = getattr(pool, "replayssm_commit_stats", None)
    if stats is None:
        return None
    values = stats.cpu()
    names = (
        "commit_rows",
        "accepted_tokens",
        "active_fold_rows",
        "track_rows",
        "replayed_history_tokens",
        "max_history_depth",
        "forced_materializations",
        "discarded_history_tokens",
        "commit_kernel_launches",
    )
    result = {name: int(values[:, i].sum()) for i, name in enumerate(names)}
    result["max_history_depth"] = int(values[:, 5].max())
    state = getattr(pool, "replayssm_deferred_state", None)
    result["materialize_calls"] = dict(state.materialize_calls) if state else {}
    result["state_bytes_per_row"] = (
        pool.mamba_cache.temporal.numel()
        // pool.mamba_cache.temporal.shape[1]
        * pool.mamba_cache.temporal.element_size()
    )
    result["full_state_write_bytes"] = result["state_bytes_per_row"] * (
        result["active_fold_rows"]
        + result["track_rows"]
        + result["forced_materializations"]
    )
    return result


class DeferredGDNState:
    def __init__(self, pool):
        self.pool = pool
        rows = pool.replayssm_spec_write_pos
        self.row_to_slot = torch.full_like(rows, -1)
        self.slot_to_row = torch.full(
            (pool.mamba_cache.temporal.shape[1],),
            -1,
            device=rows.device,
            dtype=torch.int32,
        )
        self.materialize_calls = Counter()

    def bind(self, rows, slots):
        if rows.numel():
            _bind_rows[(triton.cdiv(rows.numel(), 128),)](
                rows.contiguous(),
                slots.contiguous(),
                self.row_to_slot,
                self.slot_to_row,
                N=rows.numel(),
                BLOCK=128,
            )

    def reset(self, slots, *, forget=False):
        if not slots.numel():
            return
        pool = self.pool
        stats = getattr(pool, "replayssm_commit_stats", None)
        _reset_owned_slots[(triton.cdiv(slots.numel(), 128),)](
            slots.contiguous(),
            self.row_to_slot,
            self.slot_to_row,
            pool.replayssm_spec_write_pos,
            pool.replayssm_cache_base,
            pool.replayssm_is_flush,
            stats if stats is not None else self.row_to_slot,
            N=slots.numel(),
            FORGET=forget,
            HAS_STATS=stats is not None,
            BLOCK=128,
        )

    def reset_rows(self, rows):
        # New rows are reset on the forward stream after acquire bound them.
        self.reset(self.row_to_slot[rows.long()])

    def clear(self):
        self.row_to_slot.fill_(-1)
        self.slot_to_row.fill_(-1)
        self.pool.replayssm_spec_write_pos.zero_()
        self.pool.replayssm_cache_base.zero_()
        self.pool.replayssm_is_flush.zero_()

    def materialize(self, slots, *, reason):
        """Drain owned active slots before a reader consumes only temporal."""
        if not slots.numel():
            return
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
            commit_gdn_replayssm_circular,
        )

        pool = self.pool
        stats = getattr(pool, "replayssm_commit_stats", None)
        if stats is not None:
            self.materialize_calls[reason] += 1
        indices = torch.empty_like(slots, dtype=torch.int64)
        rows = torch.empty_like(indices)
        accepted = torch.empty_like(indices, dtype=torch.int32)
        block = triton.next_power_of_2(min(128, slots.numel()))
        _prepare_materialize[(triton.cdiv(slots.numel(), block),)](
            slots.contiguous(),
            self.row_to_slot,
            self.slot_to_row,
            pool.replayssm_spec_write_pos,
            pool.replayssm_is_flush,
            indices,
            rows,
            accepted,
            stats if stats is not None else self.row_to_slot,
            N=slots.numel(),
            HAS_STATS=stats is not None,
            BLOCK=block,
        )
        state = pool.get_speculative_mamba2_params_all_layers()
        commit_gdn_replayssm_circular(
            checkpoint_state=state.temporal,
            d_cache=state.replayssm_d,
            k_cache=state.replayssm_k,
            g_cache=state.replayssm_g,
            d_residual_cache=state.replayssm_rawv,
            k_residual_cache=state.replayssm_rawk,
            state_batch_indices=indices,
            replay_indices=rows,
            write_pos=pool.replayssm_spec_write_pos,
            cache_base=pool.replayssm_cache_base,
            is_flush=pool.replayssm_is_flush,
            accept_lens=accepted,
            null_block_id=-1,
        )
