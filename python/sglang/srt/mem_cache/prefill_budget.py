# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Memory reservations for one prefill pass.

The scheduler supplies token demand and its chunk/decode limits. These objects
account for admitted but not yet allocated work and query live cache capacity:
locking a prefix or preempting a request must affect the next admission check.
Selection checks do not mutate cache state. Shared-pool load preparation
realizes the selected reservation before a host transfer pins device rows.
"""

from contextlib import nullcontext
from typing import Dict, Mapping, Optional

from sglang.srt.layers.cp.base import get_cp_strategy, is_cp_sharded_padded
from sglang.srt.layers.cp.utils import (
    from_cp_sharded,
    get_cp_sharded_padded_allocation_tokens,
    get_cp_sharded_padded_available_tokens,
    get_cp_sharded_padded_chunk_tokens,
)
from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.base_prefix_cache import evictable_by_sub_pool
from sglang.srt.mem_cache.capacity import reclaim_for_load_back
from sglang.srt.mem_cache.kv.alloc.demand import (
    DemandPurpose,
    KVDemand,
    chunk_granule,
    demand_inputs,
    largest_fitting_chunk,
    window_retirement,
)


def estimate_swa_kv_tokens(
    extend_input_len: int,
    max_new_tokens: int,
    *,
    sliding_window_size: Optional[int],
    page_size: int,
    allocation_limit: Optional[int] = None,
    host_hit_length: int = 0,
) -> int:
    """Peak SWA reservation for one prefill/decode request."""
    if sliding_window_size is None or sliding_window_size <= 0:
        reserved = extend_input_len + max_new_tokens + page_size
    else:
        allocated = (
            extend_input_len
            if allocation_limit is None
            else min(extend_input_len, allocation_limit)
        )
        allocated_tail = max(allocated - sliding_window_size, 0)
        # With a roughly two-window SWA pool, a cached prefix can already lock
        # one window. Charging another whole window for a short resume can then
        # block admission forever on an idle pool. Reserve only its uncached
        # tail plus decode headroom, capped at the window.
        # Including the extension also keeps the reservation large enough for
        # this pass's prefill allocation.
        reserved = (
            allocated_tail
            + min(extend_input_len + max_new_tokens, sliding_window_size)
            + page_size
        )
    if host_hit_length > 0:
        reserved += -(-host_hit_length // page_size) * page_size
    return reserved


def _single_window_hit(
    window_host_hit_lengths: Optional[Mapping[str, int]],
) -> int:
    """Host hit tokens of a pool keeping one window, whatever its name."""
    return max((window_host_hit_lengths or {}).values(), default=0)


class PrefillBudget:
    """Fixed token pool. Offsets include pending allocations and decode headroom."""

    # When False, admission skips the FULL spec and passes the host hit length.
    load_back_needs_full_tokens = False
    # FULL-token conversion for fixed/split policies; joint admission preserves
    # the slot count. State capability is independent of this conversion.
    state_slot_cost = 0
    has_state_slots = False
    requires_final_shape_check = False
    # Admission prices the whole request and selects the chunk limit it fits
    # at (`check_prefill`); every admission path runs and commits that limit.
    admits_whole_requests = False
    # Set when `check_prefill` refuses a request no wait can admit;
    # the scheduler aborts it instead of keeping it queued.
    unschedulable = False

    def __init__(self, allocator, tree_cache, *, num_mixed_decode_tokens: int = 0):
        self.allocator = allocator
        self.tree_cache = tree_cache
        # Logical token alignment width: one allocator page, or a full CP group.
        self.page_size = allocator.page_size
        if is_cp_sharded_padded():
            self.page_size *= get_cp_strategy().cp_size
        self.total_offset = num_mixed_decode_tokens
        self.current_offset = num_mixed_decode_tokens
        self.swa_offset = 0

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

    def prepare_load_back(
        self,
        *,
        full_tokens: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int | None,
        window_host_hit_lengths: Optional[Mapping[str, int]] = None,
        state_slots: int = 0,
    ) -> bool:
        """Prepare pools whose admission depends on movable shared space."""
        return True

    def load_back_scope(self, **kwargs):
        return nullcontext()

    def _available_and_evictable(self):
        evictable = (
            self.tree_cache.full_evictable_size()
            if self.tree_cache.supports_mamba()
            else self.tree_cache.evictable_size()
        )
        global_available_size = self.allocator.available_size()
        if is_cp_sharded_padded():
            global_available_size = from_cp_sharded(global_available_size)
        return global_available_size + evictable

    @property
    def remaining_total(self):
        return self._available_and_evictable() - self.total_offset

    @property
    def remaining_current(self):
        return self._available_and_evictable() - self.current_offset

    @property
    def remaining_window_tokens(self):
        return 0

    def has_capacity(self) -> bool:
        if is_cp_sharded_padded():
            # CP can reuse padding in already reserved pages when fresh capacity is zero.
            return self.remaining_total >= 0 and self.remaining_current >= 0
        return self.remaining_total > 0 and self.remaining_current > 0

    def request_state_slots(self, req) -> int:
        return 0

    def request_held_state_slots(self, req) -> int:
        return 0

    def state_slots_exhausted(self) -> bool:
        return False

    def state_slots_fit(self, state_slots: int) -> bool:
        return True

    def check_prefill(
        self,
        *,
        extend_input_len: int,
        total_tokens: int,
        max_new_tokens: int,
        input_tokens: int,
        chunk_limit: int | None,
        window_host_hit_lengths: Optional[Mapping[str, int]] = None,
        state_slots: int = 0,
        chunk_align: int | None = None,
        host_hit_tokens: int = 0,
        held_state_slots: int = 0,
    ) -> tuple[bool, int | None]:
        """Return admission feasibility and a memory bound on the requested
        chunk. ``host_hit_tokens`` are the prompt tokens a host hit restores,
        which ``total_tokens`` charges and ``extend_input_len`` leaves out."""
        return (
            (True, chunk_limit)
            if total_tokens + state_slots * self.state_slot_cost < self.remaining_total
            else (False, None)
        )

    def can_allocate_prefill(
        self,
        *,
        paged_input: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int | None,
        state_slots: int = 0,
        window_host_hit_lengths: Optional[Mapping[str, int]] = None,
        admitted_chunk_limit: int | None = None,
        cached_tokens: int = 0,
    ) -> bool:
        """Whether the selected shape fits. ``admitted_chunk_limit`` (the
        chunk limit admission chose) and ``cached_tokens`` are for budgets
        that price a request's chunking; a fixed pool charges tokens."""
        return paged_input + state_slots * self.state_slot_cost <= min(
            self.remaining_current, self.remaining_total
        )

    def available_chunk_tokens(self, chunk_limit: int) -> int | None:
        available_tokens = int(self.remaining_total)
        if is_cp_sharded_padded():
            available_tokens = get_cp_sharded_padded_available_tokens(available_tokens)
        available = min(
            chunk_limit,
            available_tokens,
        )
        # Single-pool continuation must make progress to release its KV.
        return available if available > 0 else chunk_limit

    def fit_chunk(
        self,
        req,
        *,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int,
        state_slots: int = 0,
        chunk_align: int | None = None,
    ) -> int | None:
        return chunk_limit

    def reserve(
        self,
        extend_input_len: int,
        max_new_tokens: int,
        *,
        extra_tokens: int = 0,
        state_slots: int = 0,
        chunk_limit: int | None = None,
        is_chunked_continuation: bool = False,
        unfinished_input_len: int = 0,
        deferred_max_new_tokens: int = 0,
        admitted_chunk_limit: int | None = None,
        cached_tokens: int = 0,
    ) -> None:
        """Charge one admitted chunk or request. ``unfinished_input_len`` and
        ``deferred_max_new_tokens`` describe the rest of a request this chunk
        does not finish; a budget that commits whole requests charges them,
        at the chunk limit admission chose (``admitted_chunk_limit``)."""
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
        immediate = (
            extend_input_len
            + self.page_size
            + extra_tokens
            + state_slots * self.state_slot_cost
        )
        self.total_offset += immediate + max_new_tokens
        self.current_offset += immediate


class SWAPrefillBudget(PrefillBudget):
    """Separate FULL/SWA partitions, including per-request SWA rings."""

    def __init__(self, *args, all_swa=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_swa = all_swa
        self.req_ring = is_swa_req_ring(self.allocator)

    def _window_evictable(self) -> int:
        """Evictable tokens of the static pools' one window."""
        from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

        return self.tree_cache.window_evictable_size(BaseSWAKVPool.WINDOW_SUB_POOL)

    def _available_and_evictable(self):
        if self.all_swa:
            global_swa_available_size = self.allocator.swa_available_size()
            if is_cp_sharded_padded():
                global_swa_available_size = from_cp_sharded(global_swa_available_size)
            return global_swa_available_size + self._window_evictable()
        global_full_available_size = self.allocator.full_available_size()
        if is_cp_sharded_padded():
            global_full_available_size = from_cp_sharded(global_full_available_size)
        return global_full_available_size + self.tree_cache.full_evictable_size()

    @property
    def remaining_window_tokens(self):
        evictable = 0 if self.req_ring else self._window_evictable()
        global_swa_available_size = self.allocator.swa_available_size()
        if is_cp_sharded_padded():
            global_swa_available_size = from_cp_sharded(global_swa_available_size)
        return global_swa_available_size + evictable - self.swa_offset

    def swa_tokens(
        self,
        extend_input_len,
        max_new_tokens,
        *,
        chunk_limit=None,
        window_host_hit_lengths=None,
    ):
        if self.req_ring:
            return self.allocator.swa_ring_cost_tokens
        return estimate_swa_kv_tokens(
            extend_input_len,
            max_new_tokens,
            sliding_window_size=self.tree_cache.sliding_window_size,
            page_size=self.page_size,
            allocation_limit=chunk_limit,
            host_hit_length=_single_window_hit(window_host_hit_lengths),
        )

    def swa_never_fits(self, extend_input_len, max_new_tokens, **kwargs):
        needed = self.swa_tokens(extend_input_len, max_new_tokens, **kwargs)
        size_swa = self.allocator.size_swa
        if is_cp_sharded_padded():
            size_swa = from_cp_sharded(size_swa)
        return needed > size_swa if self.req_ring else needed >= size_swa

    def _chunk_cap(
        self,
        max_new_tokens,
        window_host_hit_lengths=None,
        *,
        prefix_tokens=None,
    ):
        # Only the sliding window stays locked between chunks, so a smaller
        # chunk can bound the transient SWA footprint of a longer prompt.
        headroom = self.swa_tokens(
            0, max_new_tokens, window_host_hit_lengths=window_host_hit_lengths
        )
        cap = int(self.remaining_window_tokens) - headroom
        cap = max(0, cap // self.page_size * self.page_size)
        if is_cp_sharded_padded() and cap > 0:
            # Keep SWA availability at the caller; the helper only aligns tokens.
            cap = get_cp_sharded_padded_chunk_tokens(
                cap + self.page_size - 1,
                prefix_tokens=prefix_tokens,
            )
        return cap

    def check_prefill(
        self,
        *,
        extend_input_len: int,
        total_tokens: int,
        max_new_tokens: int,
        input_tokens: int,
        chunk_limit: int | None,
        window_host_hit_lengths: Optional[Mapping[str, int]] = None,
        state_slots: int = 0,
        chunk_align: int | None = None,
        host_hit_tokens: int = 0,
        held_state_slots: int = 0,
    ) -> tuple[bool, int | None]:
        if total_tokens + state_slots * self.state_slot_cost >= self.remaining_total:
            return False, None
        prefix_tokens = input_tokens - extend_input_len
        if is_cp_sharded_padded():
            extend_input_len = get_cp_sharded_padded_allocation_tokens(
                extend_input_len, prefix_tokens=prefix_tokens
            )
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
        needed = self.swa_tokens(
            extend_input_len,
            max_new_tokens,
            chunk_limit=chunk_limit,
            window_host_hit_lengths=window_host_hit_lengths,
        )
        fits = (
            needed <= self.remaining_window_tokens
            if self.req_ring
            else needed < self.remaining_window_tokens
        )
        if fits:
            return True, chunk_limit
        # Only permanent shortfalls may shrink a chunk. Transient pressure waits
        # so a new prefill does not consume running decodes' window headroom.
        cap = 0
        if self.swa_never_fits(
            extend_input_len,
            max_new_tokens,
            chunk_limit=chunk_limit,
            window_host_hit_lengths=window_host_hit_lengths,
        ):
            cap = self._chunk_cap(
                max_new_tokens,
                window_host_hit_lengths,
                prefix_tokens=prefix_tokens,
            )
            if chunk_align:
                # A cap below the truncation alignment would cut the chunk
                # to nothing: such a request waits instead.
                granule = chunk_granule(self.page_size, chunk_align)
                cap = cap // granule * granule
        if chunk_limit is None or cap <= 0:
            return False, None
        return True, min(chunk_limit, cap)

    def has_capacity(self) -> bool:
        if is_cp_sharded_padded():
            # CP can reuse padding in already reserved SWA pages with zero fresh capacity.
            return super().has_capacity() and self.remaining_window_tokens >= 0
        return super().has_capacity() and self.remaining_window_tokens > 0

    def available_chunk_tokens(self, chunk_limit: int) -> int | None:
        available_tokens = int(self.remaining_total)
        if is_cp_sharded_padded():
            available_tokens = get_cp_sharded_padded_available_tokens(available_tokens)
        available = min(
            chunk_limit,
            available_tokens,
        )
        if not self.req_ring:
            available_tokens = int(self.remaining_window_tokens)
            if is_cp_sharded_padded():
                available_tokens = get_cp_sharded_padded_available_tokens(
                    available_tokens, reserved_tokens=self.page_size
                )
            available = min(
                available,
                available_tokens - self.page_size,
            )
        return available if available > 0 else None

    def can_allocate_prefill(
        self,
        *,
        paged_input: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int | None,
        state_slots: int = 0,
        window_host_hit_lengths: Optional[Mapping[str, int]] = None,
        admitted_chunk_limit: int | None = None,
        cached_tokens: int = 0,
    ) -> bool:
        # The window estimate reads the limit only through min(extend, limit),
        # and the extend is already cut to the admitted limit: keep ``chunk_limit``.
        return (
            super().can_allocate_prefill(
                paged_input=paged_input,
                extend_input_len=extend_input_len,
                max_new_tokens=max_new_tokens,
                chunk_limit=chunk_limit,
                state_slots=state_slots,
            )
            and self.swa_tokens(
                extend_input_len,
                max_new_tokens,
                chunk_limit=chunk_limit,
                window_host_hit_lengths=window_host_hit_lengths,
            )
            <= self.remaining_window_tokens
        )

    def reserve(
        self,
        extend_input_len: int,
        max_new_tokens: int,
        *,
        extra_tokens: int = 0,
        state_slots: int = 0,
        chunk_limit: int | None = None,
        is_chunked_continuation: bool = False,
        unfinished_input_len: int = 0,
        deferred_max_new_tokens: int = 0,
        admitted_chunk_limit: int | None = None,
        cached_tokens: int = 0,
    ) -> None:
        # Charges only this chunk, already cut to the admitted limit; the
        # static estimate keeps ``chunk_limit`` (see `can_allocate_prefill`).
        super().reserve(
            extend_input_len,
            max_new_tokens,
            extra_tokens=extra_tokens,
            state_slots=state_slots,
        )
        # A continuation already owns its ring slot.
        if not (self.req_ring and is_chunked_continuation):
            self.swa_offset += self.swa_tokens(
                self.ceil_paged_tokens(extend_input_len),
                max_new_tokens,
                chunk_limit=chunk_limit,
            )


class ChainPrefillBudget(PrefillBudget):
    """The prefill budget of every unified chain: one admission rule.

    A request is charged its FULL tokens, each window's retained tail and its
    state slots (`request_demand`, shared with the memory clip), and admitted when
    the allocator's `admits` finds a layout for it plus the pending work.
    A chunked request is admitted whole and recommits its rest before other
    requests of the pass, so each chunk needs only the rows it writes now.
    """

    load_back_needs_full_tokens = True
    requires_final_shape_check = True
    admits_whole_requests = True

    def __init__(self, allocator, tree_cache, *, num_mixed_decode_tokens: int = 0):
        super().__init__(
            allocator, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )
        self.has_state_slots = allocator.mamba_allocator is not None
        self.pending_state_slots = 0
        # (sub-pool, retained window) per window sub-pool.
        self.windows = tuple(
            (name, tree_cache.window_retain_size(name))
            for name in allocator.window_pool_names()
        )
        # When the scheduler retires window rows (`window_retirement`).
        self.retirement = window_retirement(
            shares_prefixes=tree_cache.supports_prefix_sharing()
        )
        # The window decode keeps behind the latest state checkpoint.
        self.checkpoint_interval = tree_cache.window_checkpoint_interval()
        # A mixed decode token writes one row in every window too.
        self.window_offsets = {
            name: num_mixed_decode_tokens for name, _ in self.windows
        }
        # The rows this pass's forward writes, without the commitments and
        # decode that come later (`current_offset` is the token sub-pool's).
        self.window_current_offsets = dict(self.window_offsets)

    def _available_and_evictable(self):
        # The scalar budget counts owner (FULL) tokens: admission prices every
        # sub-pool itself, and `available_size` understates small-window needs.
        available = self.allocator.full_available_size()
        if is_cp_sharded_padded():
            available = from_cp_sharded(available)
        return available + self.tree_cache.full_evictable_size()

    def request_demand(
        self,
        extend_input_len,
        max_new_tokens,
        *,
        purpose: DemandPurpose = DemandPurpose.ADMIT,
        chunk_limit=None,
        host_hit_lengths: Optional[Mapping[str, int]] = None,
        state_slots=0,
        extra_tokens=0,
        cached_tokens=0,
    ) -> KVDemand:
        """What one request still needs over this pool's windows, built for
        ``purpose`` as every site builds it (`demand.demand_inputs`)."""
        return demand_inputs(
            purpose,
            extend_input_len,
            max_new_tokens,
            windows=dict(self.windows),
            page_size=self.page_size,
            chunk_limit=chunk_limit,
            prefill_retire_lag=self.retirement.prefill_retire_lag,
            decode_retire_lag=self.retirement.decode_retire_lag,
            eviction_interval=self.retirement.eviction_interval,
            checkpoint_interval=self.checkpoint_interval,
            state_slots=state_slots,
            cached_tokens=cached_tokens,
            host_hit_lengths=host_hit_lengths,
            extra_tokens=extra_tokens,
        ).demand()

    def window_tokens(
        self,
        extend_input_len,
        max_new_tokens,
        *,
        purpose: DemandPurpose = DemandPurpose.ADMIT,
        chunk_limit=None,
        host_hit_lengths: Optional[Mapping[str, int]] = None,
        cached_tokens=0,
    ):
        """The tokens one request reserves in each window sub-pool; host hits
        are given per window sub-pool."""
        return dict(
            self.request_demand(
                extend_input_len,
                max_new_tokens,
                purpose=purpose,
                chunk_limit=chunk_limit,
                host_hit_lengths=host_hit_lengths,
                cached_tokens=cached_tokens,
            ).window_tokens
        )

    def request_state_slots(self, req) -> int:
        from sglang.srt.mem_cache.static_kv_storage import kv_storage_of

        # A chain is unified memory: its storage is the pool's own.
        return kv_storage_of(None, self.allocator).pending_state_slots(req)

    def request_held_state_slots(self, req) -> int:
        from sglang.srt.mem_cache.static_kv_storage import kv_storage_of

        return kv_storage_of(None, self.allocator).held_state_slots(req)

    def _others_hold_memory(self, req) -> bool:
        """Whether anything but ``req`` holds memory it gives back when it ends;
        asked of memory, since the running batch misses PD sends and microbatches."""
        from sglang.srt.mem_cache.static_kv_storage import kv_storage_of

        return kv_storage_of(None, self.allocator).others_hold_rows(req)

    def _waiting_frees_memory(self) -> bool:
        """Whether memory comes back to a request that waits: admitted work, a
        forward or host copy in flight, or a row other work holds (not a parked row)."""
        if self.total_offset > 0 or self.pending_state_slots > 0:
            return True
        steps = getattr(self.allocator, "forward_steps", None)
        if steps is not None and steps.in_flight():
            return True
        if self.tree_cache.transfers_in_flight():
            return True
        return self._others_hold_memory(None)

    def _refuse(self) -> tuple[bool, None]:
        """Refuse a request no chunk of which fits now: it waits while waiting
        frees memory, and is unschedulable when nothing will."""
        self.unschedulable = not self._waiting_frees_memory()
        return False, None

    def _no_window_tokens(self) -> Dict[str, int]:
        return {name: 0 for name, _ in self.windows}

    def _fits(
        self, full_tokens, window_tokens, state_slots=0, *, empty_pool=False, now=False
    ):
        """Whether the pending work plus this demand is admitted (the demand alone
        with ``empty_pool``); with ``now``, pending is only this pass's forward rows."""
        if empty_pool:
            full_pending, window_pending, state_pending = 0, None, 0
        elif now:
            full_pending = self.current_offset
            window_pending = self.window_current_offsets
            state_pending = self.pending_state_slots
        else:
            full_pending = self.total_offset
            window_pending = self.window_offsets
            state_pending = self.pending_state_slots
        demand = KVDemand(
            full_tokens + full_pending,
            {
                name: window_tokens[name]
                + (window_pending[name] if window_pending else 0)
                for name, _ in self.windows
            },
            state_slots + state_pending,
        )

        return self.allocator.admits(
            demand,
            evictable=(
                None
                if empty_pool
                else evictable_by_sub_pool(self.tree_cache, self.allocator)
            ),
            empty_pool=empty_pool,
        )

    def _admits(self, demand: KVDemand, **kwargs) -> bool:
        return self._fits(
            demand.full_tokens, demand.window_tokens, demand.state_slots, **kwargs
        )

    @property
    def remaining_window_tokens(self):
        # Token view for paths that size work by a token count (dLLM): the
        # fewest tokens any window can still take, less what is pending on it.
        if not self.windows:
            return 0
        return min(
            self.allocator.window_available_size(name)
            + self.tree_cache.window_evictable_size(name)
            - self.window_offsets[name]
            for name, _ in self.windows
        )

    def has_capacity(self) -> bool:
        return self._fits(0, self._no_window_tokens())

    def state_slots_fit(self, state_slots: int) -> bool:
        return not self.has_state_slots or self._fits(
            0, self._no_window_tokens(), state_slots
        )

    def state_slots_exhausted(self) -> bool:
        return not self.state_slots_fit(1)

    def check_prefill(
        self,
        *,
        extend_input_len: int,
        total_tokens: int,
        max_new_tokens: int,
        input_tokens: int,
        chunk_limit: int | None,
        window_host_hit_lengths: Optional[Mapping[str, int]] = None,
        state_slots: int = 0,
        chunk_align: int | None = None,
        host_hit_tokens: int = 0,
        held_state_slots: int = 0,
    ) -> tuple[bool, int | None]:
        """Admit the whole request if it fits; otherwise, with chunking, at the
        largest chunk limit at which all of it fits an empty pool, if it fits now.

        Transient pressure waits (a chunk cut to what is free now starves running
        decodes); when nothing will give memory back, the largest limit that fits
        now is taken, or the request is refused as `unschedulable`. The returned
        limit is what `can_allocate_prefill` and `reserve` price at, and lies on
        the truncation grid (``chunk_align``), so it never cuts an empty chunk.
        The empty-pool check counts the ``held_state_slots`` the request already
        holds, or it would wait on its own memory forever."""
        self.unschedulable = False
        host_hits = dict(window_host_hit_lengths or {})
        granule = chunk_granule(self.page_size, chunk_align)
        # Host-hit rows are charged page-ceiled with the extend, as the shape
        # check's ``paged_input`` charges them, so whatever is admitted passes it.
        host_rows = self.ceil_paged_tokens(
            extend_input_len + host_hit_tokens
        ) - self.ceil_paged_tokens(extend_input_len)
        extra_tokens = (
            total_tokens
            - extend_input_len
            - host_hit_tokens
            + host_rows
            - max_new_tokens
            - self.page_size
        )
        cached_tokens = input_tokens - extend_input_len

        def demand_at(limit):
            return self.request_demand(
                extend_input_len,
                max_new_tokens,
                chunk_limit=limit,
                host_hit_lengths=host_hits,
                state_slots=state_slots,
                extra_tokens=extra_tokens,
                cached_tokens=cached_tokens,
            )

        def whole_request_at(limit):
            # The rows a host hit restores are prompt rows here.
            return self.request_demand(
                input_tokens,
                max_new_tokens,
                chunk_limit=limit,
                state_slots=state_slots + held_state_slots,
                extra_tokens=extra_tokens - host_rows,
            )

        demand = demand_at(chunk_limit)
        if self._admits(demand):
            return True, chunk_limit
        if chunk_limit is None:
            return self._refuse()
        ceiling = (
            chunk_limit
            if self._admits(whole_request_at(chunk_limit), empty_pool=True)
            else self._whole_request_chunk_limit(
                whole_request_at, chunk_limit, empty_pool=True, granule=granule
            )
        )
        if ceiling:
            if ceiling < chunk_limit and self._admits(demand_at(ceiling)):
                return True, ceiling
            # Transient pressure waits while something will give memory back;
            # otherwise take the largest limit at which all of it fits now.
            if self._waiting_frees_memory():
                return False, None
            limit = self._whole_request_chunk_limit(demand_at, ceiling, granule=granule)
            return (True, limit) if limit else self._refuse()
        # No chunk lets all of it fit an empty pool (the clip prices no host
        # hit): take the largest chunk that fits now, at the caller's charge.
        cap = self._chunk_cap(
            max_chunk_tokens=min(chunk_limit, max(0, extend_input_len - 1)),
            chunk_limit=chunk_limit,
            host_hit_lengths=host_hits,
            state_slots=state_slots,
            extra_tokens=extra_tokens,
            granule=granule,
        )
        return (True, min(chunk_limit, cap)) if cap > 0 else self._refuse()

    def _whole_request_chunk_limit(
        self, demand_at, chunk_limit, *, empty_pool=False, granule=None
    ):
        """The largest chunk limit on ``granule`` (a page by default), at
        most ``chunk_limit``, at which the whole request
        (``demand_at(limit)``) is admitted now (or in an empty pool)."""
        return largest_fitting_chunk(
            lambda limit: self._admits(demand_at(limit), empty_pool=empty_pool),
            chunk_limit,
            granule or self.page_size,
        )

    def _chunk_fits(
        self,
        tokens,
        headroom,
        *,
        chunk_limit,
        host_hit_lengths,
        state_slots,
        extra_tokens=0,
        empty_pool=False,
        now=False,
    ):
        # A chunk alone: it spans no more than its chunk, so it keeps no
        # window beside the next one.
        return self._admits(
            self.request_demand(
                tokens,
                headroom,
                chunk_limit=chunk_limit,
                host_hit_lengths=host_hit_lengths,
                state_slots=state_slots,
                extra_tokens=extra_tokens,
            ),
            empty_pool=empty_pool,
            now=now,
        )

    def _chunk_cap(
        self,
        *,
        max_chunk_tokens,
        chunk_limit,
        host_hit_lengths,
        state_slots,
        extra_tokens=0,
        now=False,
        granule=None,
    ):
        """The largest chunk on ``granule`` (a page by default), without
        decode headroom, the budget admits (with ``now``, whose rows are
        there now)."""
        return largest_fitting_chunk(
            lambda tokens: self._chunk_fits(
                tokens,
                0,
                chunk_limit=chunk_limit,
                host_hit_lengths=host_hit_lengths,
                state_slots=state_slots,
                extra_tokens=extra_tokens,
                now=now,
            ),
            max_chunk_tokens,
            granule or self.page_size,
        )

    def available_chunk_tokens(self, chunk_limit: int) -> int | None:
        return chunk_limit if chunk_limit > 0 else None

    def fit_chunk(
        self,
        req,
        *,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int,
        state_slots: int = 0,
        chunk_align: int | None = None,
    ) -> int | None:
        """The next chunk of the admitted chunked request ``req``: it needs only
        the rows it writes now. Without them it waits while others hold rows,
        else shrinks onto the truncation grid, so a lone request always finishes."""
        candidate = min(extend_input_len, chunk_limit)
        fits = dict(
            chunk_limit=chunk_limit,
            host_hit_lengths=None,
            state_slots=state_slots,
            now=True,
        )
        if self._chunk_fits(candidate, 0, **fits):
            return chunk_limit
        if self._others_hold_memory(req):
            return None
        cap = self._chunk_cap(
            max_chunk_tokens=candidate,
            granule=chunk_granule(self.page_size, chunk_align),
            **fits,
        )
        return min(chunk_limit, cap) if cap else None

    def can_allocate_prefill(
        self,
        *,
        paged_input: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int | None,
        state_slots: int = 0,
        window_host_hit_lengths: Optional[Mapping[str, int]] = None,
        admitted_chunk_limit: int | None = None,
        cached_tokens: int = 0,
    ) -> bool:
        """Whether the selected shape fits, priced at ``admitted_chunk_limit`` as
        admission priced it, with ``paged_input`` as the token sub-pool's charge."""
        return self._admits(
            self.request_demand(
                extend_input_len,
                max_new_tokens,
                chunk_limit=(
                    chunk_limit
                    if admitted_chunk_limit is None
                    else admitted_chunk_limit
                ),
                host_hit_lengths=dict(window_host_hit_lengths or {}),
                state_slots=state_slots,
                extra_tokens=paged_input
                - self.ceil_paged_tokens(extend_input_len)
                - self.page_size,
                cached_tokens=cached_tokens,
            )
        )

    def reserve(
        self,
        extend_input_len: int,
        max_new_tokens: int,
        *,
        extra_tokens: int = 0,
        state_slots: int = 0,
        chunk_limit: int | None = None,
        is_chunked_continuation: bool = False,
        unfinished_input_len: int = 0,
        deferred_max_new_tokens: int = 0,
        admitted_chunk_limit: int | None = None,
        cached_tokens: int = 0,
    ) -> None:
        """Charge an admitted chunk now and commit its whole request (the rest of
        the prompt and its decode stay pending), priced as admission priced it."""
        if admitted_chunk_limit is not None:
            chunk_limit = admitted_chunk_limit
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
        committed = self.request_demand(
            extend_input_len + unfinished_input_len,
            max_new_tokens + deferred_max_new_tokens,
            chunk_limit=chunk_limit,
            extra_tokens=extra_tokens,
            cached_tokens=cached_tokens,
        )
        now = self.request_demand(
            extend_input_len,
            0,
            chunk_limit=chunk_limit,
            extra_tokens=extra_tokens,
        )
        self.total_offset += committed.full_tokens
        self.current_offset += now.full_tokens
        self.pending_state_slots += state_slots
        for name, _ in self.windows:
            self.window_offsets[name] += committed.window_tokens[name]
            self.window_current_offsets[name] += now.window_tokens[name]

    def _load_back_demand(
        self,
        *,
        full_tokens: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int | None,
        window_host_hit_lengths: Optional[Mapping[str, int]] = None,
        state_slots: int = 0,
    ):
        # H2D pins the restored rows. Realize the selected budget while the
        # chain can still be compacted, including pending prefill/decode demand.
        full_tokens = int(self.ceil_paged_tokens(full_tokens + self.total_offset))
        windows = {
            name: int(self.ceil_paged_tokens(tokens + self.window_offsets[name]))
            for name, tokens in self.window_tokens(
                extend_input_len,
                max_new_tokens,
                purpose=DemandPurpose.LOAD_BACK,
                chunk_limit=chunk_limit,
                host_hit_lengths=dict(window_host_hit_lengths or {}),
            ).items()
        }
        return KVDemand(
            full_tokens,
            windows,
            state_slots + self.pending_state_slots if self.has_state_slots else 0,
        )

    def prepare_load_back(self, **kwargs) -> bool:
        return (
            reclaim_for_load_back(
                self.tree_cache, self.allocator, self._load_back_demand(**kwargs)
            )
            is None
        )

    def load_back_scope(self, **kwargs):
        return self.allocator.scoped_physical_takes(self._load_back_demand(**kwargs))
