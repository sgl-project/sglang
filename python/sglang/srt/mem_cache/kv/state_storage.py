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

from sglang.srt.mem_cache.kv.router import STATE_PART, KVLayerRouter
from sglang.srt.utils.common import async_h2d

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


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

    def acquire(
        self,
        table,
        reqs: List[Req],
        rows: List[int],
        fresh_rows: List[bool],
        *,
        reset_request_parts: bool = True,
    ) -> None:
        """Gives each request of newly bound ``rows`` its state slot and, with
        the extra buffer, its ping-pong tracking slots, and writes them into
        the table's columns. A request that already holds a slot (a radix hit,
        a continuing chunk) keeps it. ``fresh_rows`` marks rows the requests
        did not hold before: their speculative ReplaySSM cursors restart."""
        spec_write_pos = self.pool.replayssm_spec_write_pos
        if spec_write_pos is not None and reset_request_parts:
            fresh_indices = [idx for idx, fresh in zip(rows, fresh_rows) if fresh]
            if fresh_indices:
                fresh = async_h2d(fresh_indices, dtype=torch.int64, device=table.device)
                spec_write_pos.index_fill_(0, fresh, 0)
                self.pool.replayssm_cache_base.index_fill_(0, fresh, 0)
                self.pool.replayssm_is_flush.index_fill_(0, fresh, 0)

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        # What this call hands out, given back if a later request finds no slot.
        given_state: list[Req] = []
        given_tracking: list[Req] = []
        for req in reqs:
            if req.kv.holds_mamba:  # for radix cache / continuing chunked
                pass
            else:
                mid = self.slots.alloc(1)
                if mid is None:
                    self._give_back(given_state, given_tracking)
                    raise StateSlotsExhausted(
                        f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.pool.size=}, {self.slots.available_size()=}, {len(reqs)=}"
                    )
                req.kv.mamba_pool_idx = mid[0]
                given_state.append(req)
                req.kv.mamba_needs_clear = True
                # A zero write_pos empties the ReplaySSM ring;
                # decode then reads only the checkpoint written by prefill.
                if self.pool.replayssm_write_pos is not None:
                    self.pool.replayssm_write_pos.index_fill_(
                        0, self.slots.translate(mid).to(torch.int64), 0
                    )
            mamba_indices.append(req.kv.mamba_pool_idx)
            if table.enable_mamba_extra_buffer:
                if req.kv.mamba_ping_pong_track_buffer is None:
                    if not self.alloc_ping_pong_buffer(table, req):
                        self._give_back(given_state, given_tracking)
                        raise StateSlotsExhausted(
                            "Not enough space for mamba ping pong idx, "
                            "try to increase --mamba-full-memory-ratio."
                        )
                    given_tracking.append(req)
                mamba_ping_pong_track_buffers.append(
                    req.kv.mamba_ping_pong_track_buffer
                )
        assert len(rows) == len(mamba_indices), (
            "Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        )
        if table.enable_mamba_extra_buffer:
            assert len(rows) == len(mamba_ping_pong_track_buffers), (
                "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
            )
        # Device rows: indexing a device tensor with a Python list syncs the host.
        row_ids = async_h2d(rows, dtype=torch.int64, device=table.device)
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        table.req_index_to_mamba_index_mapping[row_ids] = mamba_index_tensor
        if table.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers)
            table.req_index_to_mamba_ping_pong_track_buffer_mapping[row_ids] = (
                ping_pong_tensor
            )

    def alloc_ping_pong_buffer(self, table, req: Req) -> bool:
        """Return False if slots are exhausted. Lazy mode starts with one slot
        and a -1 sentinel for the second; normal mode allocates all slots.
        """
        n = table.mamba_initial_tracking_slots
        slots = self.slots.alloc(n)
        if slots is None:
            return False
        buf = torch.full(
            (table.mamba_ping_pong_track_buffer_size,),
            -1,
            dtype=slots.dtype,
            device=slots.device,
        )
        buf[:n] = slots
        req.kv.mamba_ping_pong_track_buffer = buf
        req.kv.mamba_ping_pong_empty = [False] * n + [True] * (
            table.mamba_ping_pong_track_buffer_size - n
        )
        req.kv.mamba_prev_track_seqlen = None
        req.kv.mamba_next_track_idx = 0
        req.kv.mamba_last_track_idx = (
            0
            if table.enable_mamba_extra_buffer_lazy
            else table.get_mamba_ping_pong_other_idx(0)
        )
        return True

    def _give_back(self, given_state: List[Req], given_tracking: List[Req]) -> None:
        """Undo a failed `acquire`: the state and tracking slots it handed out."""
        for req in given_state:
            self.slots.free(req.kv.mamba_pool_idx.unsqueeze(0))
            req.kv.mamba_pool_idx = None
            req.kv.mamba_needs_clear = False
        for req in given_tracking:
            buf = req.kv.mamba_ping_pong_track_buffer
            for position, empty in enumerate(req.kv.mamba_ping_pong_empty):
                if not empty:
                    self.slots.free(buf[position : position + 1])
            req.kv.mamba_ping_pong_track_buffer = None
            req.kv.mamba_ping_pong_empty = None
            req.kv.mamba_next_track_idx = None
            req.kv.mamba_last_track_idx = None
            req.kv.mamba_prev_track_seqlen = None

    def release(
        self,
        table,
        req: Req,
        mamba_ping_pong_track_buffer_to_keep: Optional[int] = None,
    ) -> None:
        """Gives back the request's state slot and its ping-pong slots, except
        the one kept for the radix cache."""
        mamba_index = req.kv.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.slots.free(mamba_index.unsqueeze(0))
        req.kv.mamba_pool_idx = None

        if table.enable_mamba_extra_buffer:
            size = table.mamba_ping_pong_track_buffer_size
            keep = mamba_ping_pong_track_buffer_to_keep
            assert keep in (None, 0, 1) and (keep != 1 or size == 2), (
                f"cannot keep ping-pong entry {keep} of {size}"
            )
            empty = req.kv.mamba_ping_pong_empty
            positions = [p for p in range(size) if p != keep and not empty[p]]
            row = table.req_index_to_mamba_ping_pong_track_buffer_mapping[
                req.kv.req_pool_idx
            ]
            if len(positions) == size:
                self.slots.free(row)
            else:
                for p in positions:
                    self.slots.free(row[p : p + 1])
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
