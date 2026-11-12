"""A request's recurrent state: the state sub-pool's storage and its slots.

The storage (the state pool, the PLE side states, the int8 checkpoint pool),
the slot allocator and the layer map live here; the request table holds only
the slot ids, in its mapping columns. The slot operations of a request's
lifecycle (take slots for new rows, the ping-pong tracking slots, give them
back) take the table whose columns they write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.mem_cache.kv.decl import STATE_PART
from sglang.srt.utils.device import async_h2d

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.kv.router import KVLayerRouter


class StateSlotsExhausted(RuntimeError):
    """No state slot was left for a request that was admitted for one."""


class StateStorage:
    """The state sub-pool's storage, its slot allocator and its layer map."""

    def __init__(
        self,
        *,
        sub_pool: str,
        pool,
        slots,
        layer_map: Dict[int, int],
        short_conv_pool,
        ngram_pool,
        ckpt_pool=None,
    ):
        self.sub_pool = sub_pool
        self.pool = pool
        # The slot allocator: alloc/free and capacity of the state slots. The
        # unified builder installs it once the chain allocator exists.
        self.slots = slots
        self.layer_map = layer_map
        self.short_conv_pool = short_conv_pool
        self.ngram_pool = ngram_pool
        self.ckpt_pool = ckpt_pool

    @classmethod
    def empty(cls, sub_pool: str) -> StateStorage:
        """Storage whose parts are filled in as they are built."""
        return cls(
            sub_pool=sub_pool,
            pool=None,
            slots=None,
            layer_map={},
            short_conv_pool=None,
            ngram_pool=None,
        )

    # -- routes --

    def bind_routes(self, router: KVLayerRouter) -> None:
        """Routes the state layers through the KV pool's router, which owns the
        layer-wise load-back wait: the recurrent state, and the PLE side states
        when the model has them. The N-gram context is read once before the
        layer loop and restored with the first state layer, so it is routed
        there."""
        from sglang.srt.mem_cache.ple_state_pool import NGRAM_PART, SHORT_CONV_PART

        router.bind_part(
            sub_pool=self.sub_pool,
            part=STATE_PART,
            view=self.pool,
            indices=self.layer_map,
        )
        if self.short_conv_pool.enabled:
            router.bind_part(
                sub_pool=self.sub_pool,
                part=SHORT_CONV_PART,
                view=self.short_conv_pool,
                indices=self.short_conv_pool.layer_map,
                view_layer_ids={
                    layer_id: layer_id for layer_id in self.short_conv_pool.layer_map
                },
            )
        if self.ngram_pool.enabled:
            router.bind_part(
                sub_pool=self.sub_pool,
                part=NGRAM_PART,
                view=self.ngram_pool,
                indices={min(self.layer_map): 0} if self.layer_map else {},
            )

    # -- slot lifecycle --

    def fresh_request_slots(self, table) -> int:
        """Slots acquire allocates for a request that holds no state yet: its
        working slot and, with extra buffers, its initial tracking slots."""
        return 1 + (
            table.mamba_initial_tracking_slots if table.enable_mamba_extra_buffer else 0
        )

    def pending_request_slots(self, table, req: Req) -> int:
        """Slots acquire will allocate for this request's working/tracking state."""
        return int(not req.kv.holds_mamba) + (
            table.mamba_initial_tracking_slots
            if table.enable_mamba_extra_buffer
            and req.kv.mamba_ping_pong_track_buffer is None
            else 0
        )

    def held_request_slots(self, table, req: Req) -> int:
        """Slots the request holds before acquire: its working and tracking
        slots once taken, and the cached checkpoint its working copy is made
        from; with `pending_request_slots`, every slot it occupies once acquired."""
        return (
            int(req.kv.holds_mamba)
            + (
                table.mamba_initial_tracking_slots
                if table.enable_mamba_extra_buffer
                and req.kv.mamba_ping_pong_track_buffer is not None
                else 0
            )
            + int(req.kv.mamba_cow_src_index is not None)
        )

    def acquire(self, table, reqs: List[Req], rows: List[int]) -> None:
        """Gives each request of newly bound ``rows`` its state slot and, with
        the extra buffer, its ping-pong tracking slots, and writes them into
        the table's columns. A request that already holds a slot (a radix hit,
        a continuing chunk) keeps it, and one that holds tracking slots keeps
        those. Every slot taken comes from one allocation, so a shortfall
        hands out nothing."""
        assert len(rows) == len(reqs)
        tracking = table.enable_mamba_extra_buffer
        needing = [req for req in reqs if not req.kv.holds_mamba]
        untracked = (
            [req for req in reqs if req.kv.mamba_ping_pong_track_buffer is None]
            if tracking
            else []
        )
        per_request = table.mamba_initial_tracking_slots if untracked else 0
        count = len(needing) + per_request * len(untracked)
        mids = buffers = None
        if count:
            slots = self.slots.alloc(count)
            if slots is None:
                raise StateSlotsExhausted(
                    "Not enough space for mamba cache or its ping pong idx, try to "
                    "increase --mamba-full-memory-ratio or --max-mamba-cache-size. "
                    f"{self.pool.size=}, {self.slots.available_size()=}, "
                    f"{len(needing)=}, {len(untracked)=}, {per_request=}"
                )
            mids = slots[: len(needing)]
            for req, mid in zip(needing, mids):
                req.kv.mamba_pool_idx = mid
                req.kv.mamba_needs_clear = True
            # A zero write_pos empties the ReplaySSM ring;
            # decode then reads only the checkpoint written by prefill.
            if needing and self.pool.replayssm_write_pos is not None:
                self.pool.replayssm_write_pos.index_fill_(
                    0, self.slots.translate(mids).to(torch.int64), 0
                )
            if untracked:
                buffers = self._hand_out_tracking(
                    table, untracked, slots[len(needing) :]
                )
        # Device rows: indexing a device tensor with a Python list syncs the host.
        row_ids = async_h2d(rows, dtype=torch.int64, device=table.device)
        mamba_index_tensor = (
            mids
            if mids is not None and len(needing) == len(reqs)
            else torch.stack([req.kv.mamba_pool_idx for req in reqs])
        ).to(dtype=torch.int32)
        table.req_index_to_mamba_index_mapping[row_ids] = mamba_index_tensor
        if tracking:
            ping_pong_tensor = (
                buffers
                if buffers is not None and len(untracked) == len(reqs)
                else torch.stack([req.kv.mamba_ping_pong_track_buffer for req in reqs])
            )
            table.req_index_to_mamba_ping_pong_track_buffer_mapping[row_ids] = (
                ping_pong_tensor
            )

    def reset_request_rows(self, rows: List[int], *, device) -> None:
        """Restart the speculative ReplaySSM cursors of ``rows``, which start a
        new request; on the forward stream (`RowResets`). Static pools only: a
        unified pool resets them with the rest of its request memory."""
        write_pos = self.pool.replayssm_spec_write_pos
        if write_pos is None:
            return
        indices = async_h2d(rows, dtype=torch.int64, device=device)
        write_pos.index_fill_(0, indices, 0)
        self.pool.replayssm_cache_base.index_fill_(0, indices, 0)
        self.pool.replayssm_is_flush.index_fill_(0, indices, 0)

    def alloc_ping_pong_buffer(self, table, req: Req) -> bool:
        """Return False if slots are exhausted. Lazy mode starts with one slot
        and a -1 sentinel for the second; normal mode allocates all slots.
        """
        slots = self.slots.alloc(table.mamba_initial_tracking_slots)
        if slots is None:
            return False
        self._hand_out_tracking(table, [req], slots)
        return True

    def _hand_out_tracking(
        self, table, reqs: List[Req], slots: torch.Tensor
    ) -> torch.Tensor:
        """Give each of ``reqs`` its ping-pong buffer, the next
        `mamba_initial_tracking_slots` of ``slots`` in request order, and
        return the buffers as one ``[len(reqs), buffer_size]`` tensor whose
        rows the requests hold. Lazy mode leaves the rest of each buffer at
        the -1 sentinel."""
        n = table.mamba_initial_tracking_slots
        size = table.mamba_ping_pong_track_buffer_size
        buffers = slots.new_full((len(reqs), size), -1)
        buffers[:, :n] = slots.reshape(len(reqs), n)
        last_track_idx = (
            0
            if table.enable_mamba_extra_buffer_lazy
            else table.get_mamba_ping_pong_other_idx(0)
        )
        for req, buf in zip(reqs, buffers):
            req.kv.mamba_ping_pong_track_buffer = buf
            req.kv.mamba_ping_pong_empty = [False] * n + [True] * (size - n)
            req.kv.mamba_prev_track_seqlen = None
            req.kv.mamba_next_track_idx = 0
            req.kv.mamba_last_track_idx = last_track_idx
        return buffers

    def release(
        self,
        table,
        req: Req,
        mamba_ping_pong_track_buffer_to_keep: Optional[int] = None,
    ) -> None:
        """Gives back the request's state slot and its ping-pong slots, except
        the one kept for the radix cache. The request drops the slot tensors
        it frees here, so an open free group holds them without a copy."""
        mamba_index = req.kv.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.slots.free(mamba_index.unsqueeze(0), owned=True)
        req.kv.mamba_pool_idx = None

        if table.enable_mamba_extra_buffer:
            size = table.mamba_ping_pong_track_buffer_size
            keep = mamba_ping_pong_track_buffer_to_keep
            assert keep in (None, 0, 1) and (keep != 1 or size == 2), (
                f"cannot keep ping-pong entry {keep} of {size}"
            )
            empty = req.kv.mamba_ping_pong_empty
            positions = [p for p in range(size) if p != keep and not empty[p]]
            # The request's buffer, which its table row mirrors.
            buffer = req.kv.mamba_ping_pong_track_buffer
            if len(positions) == size:
                self.slots.free(buffer, owned=True)
            else:
                for p in positions:
                    self.slots.free(buffer[p : p + 1], owned=True)
            # Clear freed references so the next acquire allocates new slots.
            req.kv.mamba_ping_pong_track_buffer = None
            req.kv.mamba_ping_pong_empty = None
            req.kv.mamba_next_track_idx = None
            req.kv.mamba_last_track_idx = None
            req.kv.mamba_last_track_seqlen = None
            req.kv.mamba_prev_track_seqlen = None
            req.kv.mamba_cow_src_index = None
            req.kv.mamba_needs_clear = False

    def clear(self) -> None:
        self.slots.clear()
        self.short_conv_pool.clear()
        self.ngram_pool.clear()
        # Reset must also release the int8 pool's radix-cached slots;
        # dropping the tree alone leaves them allocated and unreachable.
        if self.ckpt_pool is not None:
            self.ckpt_pool.clear()

    # -- state operations --

    def copy(self, src_index: torch.Tensor, dst_index: torch.Tensor) -> None:
        if src_index.numel() == 0:
            return
        self.pool.copy_from(src_index, dst_index)
