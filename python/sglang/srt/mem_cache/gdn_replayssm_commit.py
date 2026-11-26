"""Ownership and materialization boundaries for deferred GDN spec commits.

The active state is a checkpoint plus request-keyed accepted factors. Cached
tracking slots remain self-contained checkpoints. Slot operations run on the
same forward stream as the state pool; CPU export also synchronizes it.

Checkpoint rounding happens during capacity folds or checkpoint-only reads.
A tracking snapshot rounds independently and keeps the active history intact.
Verify and commit retain only the accepted prefix.
"""

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
    N: tl.constexpr,
    FORGET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + i, i < N, other=-1).to(tl.int64)
    valid = (i < N) & (slot > 0)
    row = tl.load(slot_to_row + slot, valid, other=-1).to(tl.int64)
    current = tl.load(row_to_slot + row, valid & (row >= 0), other=-1)
    owned = valid & (row >= 0) & (current == slot)
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
    N: tl.constexpr,
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
        _reset_owned_slots[(triton.cdiv(slots.numel(), 128),)](
            slots.contiguous(),
            self.row_to_slot,
            self.slot_to_row,
            pool.replayssm_spec_write_pos,
            pool.replayssm_cache_base,
            pool.replayssm_is_flush,
            N=slots.numel(),
            FORGET=forget,
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
            N=slots.numel(),
            BLOCK=block,
        )
        state = pool.get_speculative_mamba2_params_all_layers()
        commit_gdn_replayssm_circular(
            checkpoint_state=state.temporal,
            checkpoint_scale=state.temporal_scale,
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
