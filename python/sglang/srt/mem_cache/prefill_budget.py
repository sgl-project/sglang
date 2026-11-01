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
from sglang.srt.mem_cache.allocator.base import TokenDemand
from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


def _single_window_hit(
    window_host_hit_lengths: Optional[Mapping[ComponentType, int]],
) -> int:
    """Host hit tokens of a pool keeping one window, whichever slot id the
    tree gives it."""
    return max((window_host_hit_lengths or {}).values(), default=0)


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


def new_window_peak(
    peak: int,
    input_tokens: int,
    *,
    sliding_window_size: Optional[int],
    page_size: int,
    allocation_limit: Optional[int],
    host_hit_length: int = 0,
) -> int:
    """``peak`` (`estimate_swa_kv_tokens`) for a request that holds none of its
    window yet (no cached prefix): between its chunks it keeps a window of
    rows it took itself, beside the next chunk, so a request of more than one
    chunk peaks at that window plus a chunk."""
    if (
        sliding_window_size is None
        or sliding_window_size <= 0
        or allocation_limit is None
        or input_tokens <= allocation_limit
    ):
        return peak
    hit = -(-host_hit_length // page_size) * page_size if host_hit_length > 0 else 0
    held_between_chunks = min(sliding_window_size, input_tokens - allocation_limit)
    return max(peak, held_between_chunks + allocation_limit + page_size + hit)


class PrefillBudget:
    """Fixed token pool. Offsets include pending allocations and decode headroom."""

    # When False, admission skips the FULL spec and passes the host hit length.
    load_back_needs_full_tokens = False
    # FULL-token conversion for fixed/split policies; joint admission preserves
    # the slot count. State capability is independent of this conversion.
    state_slot_cost = 0
    has_state_slots = False
    requires_final_shape_check = False

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
        window_host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
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
    def remaining_swa(self):
        return 0

    def has_capacity(self) -> bool:
        if is_cp_sharded_padded():
            # CP can reuse padding in already reserved pages when fresh capacity is zero.
            return self.remaining_total >= 0 and self.remaining_current >= 0
        return self.remaining_total > 0 and self.remaining_current > 0

    def request_state_slots(self, req) -> int:
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
        window_host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
        state_slots: int = 0,
    ) -> tuple[bool, int | None]:
        """Return admission feasibility and a memory bound on the requested chunk."""
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
        window_host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
    ) -> bool:
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
    ) -> None:
        """Charge one admitted chunk or request. ``unfinished_input_len`` and
        ``deferred_max_new_tokens`` describe the rest of a request this chunk
        does not finish; a budget that commits whole requests charges them."""
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

    def _available_and_evictable(self):
        if self.all_swa:
            global_swa_available_size = self.allocator.swa_available_size()
            if is_cp_sharded_padded():
                global_swa_available_size = from_cp_sharded(global_swa_available_size)
            return global_swa_available_size + self.tree_cache.swa_evictable_size()
        global_full_available_size = self.allocator.full_available_size()
        if is_cp_sharded_padded():
            global_full_available_size = from_cp_sharded(global_full_available_size)
        return global_full_available_size + self.tree_cache.full_evictable_size()

    @property
    def remaining_swa(self):
        evictable = 0 if self.req_ring else self.tree_cache.swa_evictable_size()
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
        cap = int(self.remaining_swa) - headroom
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
        window_host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
        state_slots: int = 0,
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
            needed <= self.remaining_swa
            if self.req_ring
            else needed < self.remaining_swa
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
        if chunk_limit is None or cap <= 0:
            return False, None
        return True, min(chunk_limit, cap)

    def has_capacity(self) -> bool:
        if is_cp_sharded_padded():
            # CP can reuse padding in already reserved SWA pages with zero fresh capacity.
            return super().has_capacity() and self.remaining_swa >= 0
        return super().has_capacity() and self.remaining_swa > 0

    def available_chunk_tokens(self, chunk_limit: int) -> int | None:
        available_tokens = int(self.remaining_total)
        if is_cp_sharded_padded():
            available_tokens = get_cp_sharded_padded_available_tokens(available_tokens)
        available = min(
            chunk_limit,
            available_tokens,
        )
        if not self.req_ring:
            available_tokens = int(self.remaining_swa)
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
        window_host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
    ) -> bool:
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
            <= self.remaining_swa
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
    ) -> None:
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

    A request is charged its FULL tokens (input plus expected output), for each
    window sub-pool the window it retains at most (plus this pass's transient
    tail beyond it), and its state slots in full (`request_demand`, the
    pricing the memory clip uses too); the admitted work, and the expected
    decode of running requests, accumulate as offsets. A demand is admitted
    when the allocator's `admits` finds a layout for all of it after the
    cached entries the tree could evict.

    A chunked request is admitted once, for all of it: its first chunk
    commits the rest, and each later chunk commits what is left again before
    any other request of its pass, so a chunk needs only the rows it writes
    now (`fit_chunk`).
    """

    load_back_needs_full_tokens = True
    requires_final_shape_check = True

    def __init__(self, allocator, tree_cache, *, num_mixed_decode_tokens: int = 0):
        super().__init__(
            allocator, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )
        self.has_state_slots = allocator.mamba_allocator is not None
        self.pending_state_slots = 0
        # (sub-pool, tree window slot, retained window) per window sub-pool.
        self.windows = tuple(
            (name, component, window)
            for name, component, window, _ in allocator.admission_windows(tree_cache)
        )
        # A mixed decode token writes one row in every window too.
        self.window_offsets = {
            name: num_mixed_decode_tokens for name, _, _ in self.windows
        }
        # The rows this pass's forward writes, without the commitments and
        # decode that come later (`current_offset` is the token sub-pool's).
        self.window_current_offsets = dict(self.window_offsets)

    def request_demand(
        self,
        extend_input_len,
        max_new_tokens,
        *,
        chunk_limit=None,
        host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
        state_slots=0,
        extra_tokens=0,
        new_window=False,
    ):
        """What one request still needs to place ``extend_input_len`` prompt
        tokens and then decode ``max_new_tokens``, as (token sub-pool tokens,
        tokens per window sub-pool, state slots): the page-ceiled prompt, its
        decode and one page of allocation rounding (plus ``extra_tokens`` its
        caller charges on top), the peak each window holds, and its state
        slots. With ``new_window`` the request holds none of its window yet
        (`new_window_peak`). One pricing serves the memory clip, admission
        and the commitment a chunked request keeps until its last chunk: a
        request the clip lets through is one its chunks can finish."""
        windows = self.window_tokens(
            extend_input_len,
            max_new_tokens,
            chunk_limit=chunk_limit,
            host_hit_lengths=host_hit_lengths,
        )
        if new_window:
            hits = host_hit_lengths or {}
            windows = {
                name: new_window_peak(
                    windows[name],
                    extend_input_len,
                    sliding_window_size=window,
                    page_size=self.page_size,
                    allocation_limit=chunk_limit,
                    host_hit_length=hits.get(component, 0),
                )
                for name, component, window in self.windows
            }
        return (
            self.ceil_paged_tokens(extend_input_len)
            + max_new_tokens
            + self.page_size
            + extra_tokens,
            windows,
            state_slots,
        )

    def window_tokens(
        self,
        extend_input_len,
        max_new_tokens,
        *,
        chunk_limit=None,
        host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
    ):
        """The tokens one request reserves in each window sub-pool; host hits
        are given per tree window slot."""
        host_hit_lengths = host_hit_lengths or {}
        return {
            name: estimate_swa_kv_tokens(
                extend_input_len,
                max_new_tokens,
                sliding_window_size=window,
                page_size=self.page_size,
                allocation_limit=chunk_limit,
                host_hit_length=host_hit_lengths.get(component, 0),
            )
            for name, component, window in self.windows
        }

    def request_state_slots(self, req) -> int:
        return self.allocator.storage.pending_state_slots(req)

    def _others_hold_memory(self, req) -> bool:
        """Whether memory is held by anything but ``req`` that gives it back
        when it ends: memory's answer, not the running batch's, which misses
        a PD prefill's sent requests and other pipeline microbatches."""
        return self.allocator.storage.others_hold_rows(req)

    def _only_parked_rows_hold_memory(self) -> bool:
        """Whether the memory held by others is only parked rows' (idle
        streaming sessions'), which no work in flight gives back."""
        return self.allocator.storage.only_parked_rows_held()

    def _no_window_tokens(self) -> Dict[str, int]:
        return {name: 0 for name, _, _ in self.windows}

    def _fits(
        self, full_tokens, window_tokens, state_slots=0, *, empty_pool=False, now=False
    ):
        """Whether the pending work plus this demand is admitted (or this demand
        alone, in an empty pool). With ``now``, the pending work is only what
        this pass's forward writes: whether the demand's rows are there now."""
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
        return self.allocator.admits(
            full_tokens + full_pending,
            {
                name: window_tokens[name]
                + (window_pending[name] if window_pending else 0)
                for name, _, _ in self.windows
            },
            state_slots + state_pending,
            evictable=(
                None
                if empty_pool
                else self.allocator.admission_evictable(self.tree_cache)
            ),
            empty_pool=empty_pool,
        )

    def _admits(self, demand, **kwargs) -> bool:
        full_tokens, window_tokens, state_slots = demand
        return self._fits(full_tokens, window_tokens, state_slots, **kwargs)

    @property
    def remaining_swa(self):
        # Token views for paths that size work by a token count (dLLM): the
        # target window's reachable tokens less what is pending on it.
        if not self.windows:
            return 0
        name = self.allocator.swa_attn_allocator.sub_pool_name
        return (
            self.allocator.swa_available_size()
            + self.tree_cache.swa_evictable_size()
            - self.window_offsets[name]
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
        window_host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
        state_slots: int = 0,
    ) -> tuple[bool, int | None]:
        """Admit the whole request if it fits; otherwise, with chunking and
        only when the request would not fit even an empty pool, the largest
        page-aligned chunk that fits now. Transient pressure waits: a chunk cut
        to what is free now takes the room running decodes grow into, and a
        stream of tiny prefills starves them. Memory held only by parked rows
        (idle streaming sessions) is not transient: then the whole request is
        admitted with the largest chunk limit at which all of it fits now."""
        host_hits = dict(window_host_hit_lengths or {})

        def demand_at(limit):
            # ``total_tokens`` is the caller's charge: the prompt, its decode and
            # the caller's own terms (rounding pages, rows a host hit restores).
            # Priced as the memory clip prices it, the prompt page-ceiled.
            return self.request_demand(
                extend_input_len,
                max_new_tokens,
                chunk_limit=limit,
                host_hit_lengths=host_hits,
                state_slots=state_slots,
                extra_tokens=total_tokens
                - extend_input_len
                - max_new_tokens
                - self.page_size,
                # With nothing cached, the window it keeps between chunks is new.
                new_window=input_tokens == extend_input_len,
            )

        demand = demand_at(chunk_limit)
        if self._admits(demand):
            return True, chunk_limit
        if chunk_limit is None:
            return False, None
        if self._admits(demand, empty_pool=True):
            if not self._only_parked_rows_hold_memory():
                return False, None
            limit = self._whole_request_chunk_limit(demand_at, chunk_limit)
            return (True, limit) if limit else (False, None)
        cap = self._chunk_cap(
            max_chunk_tokens=min(chunk_limit, max(0, extend_input_len - 1)),
            chunk_limit=chunk_limit,
            host_hit_lengths=host_hits,
            state_slots=state_slots,
        )
        return (True, min(chunk_limit, cap)) if cap > 0 else (False, None)

    def _whole_request_chunk_limit(self, demand_at, chunk_limit):
        """The largest page-aligned chunk limit, at most ``chunk_limit``, at
        which the whole request (``demand_at(limit)``) is admitted now; a
        smaller limit only lowers its window peak, so the answer is monotone."""
        low, high = 0, chunk_limit // self.page_size
        while low < high:
            mid = (low + high + 1) // 2
            if self._admits(demand_at(mid * self.page_size)):
                low = mid
            else:
                high = mid - 1
        return low * self.page_size

    def _chunk_fits(
        self,
        tokens,
        headroom,
        *,
        chunk_limit,
        host_hit_lengths,
        state_slots,
        empty_pool=False,
        now=False,
    ):
        return self._admits(
            self.request_demand(
                tokens,
                headroom,
                chunk_limit=chunk_limit,
                host_hit_lengths=host_hit_lengths,
                state_slots=state_slots,
            ),
            empty_pool=empty_pool,
            now=now,
        )

    def _chunk_cap(
        self, *, max_chunk_tokens, chunk_limit, host_hit_lengths, state_slots, now=False
    ):
        """The largest page-aligned chunk, without decode headroom, the
        budget admits (with ``now``, whose rows are there now); a larger
        chunk only adds demand, so the answer is monotone."""
        low, high = 0, max(0, max_chunk_tokens) // self.page_size
        while low < high:
            mid = (low + high + 1) // 2
            if self._chunk_fits(
                mid * self.page_size,
                0,
                chunk_limit=chunk_limit,
                host_hit_lengths=host_hit_lengths,
                state_slots=state_slots,
                now=now,
            ):
                low = mid
            else:
                high = mid - 1
        return low * self.page_size

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
    ) -> int | None:
        """The next chunk of the unfinished chunked request ``req``. The request was
        admitted whole, its decode included, and commits what is left of it
        again before any other request of the pass (`reserve`): requests
        admitted beside it were priced after its commitment, so their decode
        does not grow into its room. Its chunk then needs only the rows it
        writes now. When they are not there, it waits while memory has other
        holders that will give rows back (running requests, transfers in
        flight, other pipeline microbatches: whatever holds a request row);
        with no other holder it shrinks to what is there, so a lone request
        always reaches its last chunk."""
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
        cap = self._chunk_cap(max_chunk_tokens=candidate, **fits)
        return min(chunk_limit, cap) if cap else None

    def can_allocate_prefill(
        self,
        *,
        paged_input: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int | None,
        state_slots: int = 0,
        window_host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
    ) -> bool:
        return self._fits(
            paged_input + max_new_tokens,
            self.window_tokens(
                extend_input_len,
                max_new_tokens,
                chunk_limit=chunk_limit,
                host_hit_lengths=dict(window_host_hit_lengths or {}),
            ),
            state_slots,
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
    ) -> None:
        """Charge an admitted chunk now, and commit its whole request: the
        rest of the prompt (``unfinished_input_len``) and the decode after
        the last chunk (``deferred_max_new_tokens``) stay pending for every
        later admission, though this pass takes only the chunk's rows."""
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
        committed_full, committed_windows, _ = self.request_demand(
            extend_input_len + unfinished_input_len,
            max_new_tokens + deferred_max_new_tokens,
            chunk_limit=chunk_limit,
            extra_tokens=extra_tokens,
            # A first chunk commits the window its request keeps between
            # chunks; a later chunk's request already holds it. Charged
            # whether or not a prefix was cached: more pending, never less.
            new_window=unfinished_input_len > 0 and not is_chunked_continuation,
        )
        now_full, now_windows, _ = self.request_demand(
            extend_input_len, 0, chunk_limit=chunk_limit, extra_tokens=extra_tokens
        )
        self.total_offset += committed_full
        self.current_offset += now_full
        self.pending_state_slots += state_slots
        for name, _, _ in self.windows:
            self.window_offsets[name] += committed_windows[name]
            self.window_current_offsets[name] += now_windows[name]

    def _load_back_demand(
        self,
        *,
        full_tokens: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int | None,
        window_host_hit_lengths: Optional[Mapping[ComponentType, int]] = None,
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
                chunk_limit=chunk_limit,
                host_hit_lengths=dict(window_host_hit_lengths or {}),
            ).items()
        }
        return TokenDemand(
            full_tokens,
            windows,
            state_slots + self.pending_state_slots if self.has_state_slots else 0,
        )

    def prepare_load_back(self, **kwargs) -> bool:
        return (
            self.allocator.reclaim_for_load_back(
                self.tree_cache, self._load_back_demand(**kwargs)
            )
            is None
        )

    def load_back_scope(self, **kwargs):
        return self.allocator.scoped_physical_takes(self._load_back_demand(**kwargs))
