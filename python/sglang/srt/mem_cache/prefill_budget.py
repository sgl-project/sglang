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
from typing import Mapping, Optional

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
        self.page_size = allocator.page_size
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
        return self.allocator.available_size() + evictable

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
        available = min(chunk_limit, int(self.remaining_total))
        # Single-pool continuation must make progress to release its KV.
        return available if available > 0 else chunk_limit

    def fit_chunk(
        self,
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
    ) -> None:
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
            return (
                self.allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size()
            )
        return (
            self.allocator.full_available_size() + self.tree_cache.full_evictable_size()
        )

    @property
    def remaining_swa(self):
        evictable = 0 if self.req_ring else self.tree_cache.swa_evictable_size()
        return self.allocator.swa_available_size() + evictable - self.swa_offset

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
        return (
            needed > self.allocator.size_swa
            if self.req_ring
            else needed >= self.allocator.size_swa
        )

    def _chunk_cap(self, max_new_tokens, window_host_hit_lengths=None):
        # Only the sliding window stays locked between chunks, so a smaller
        # chunk can bound the transient SWA footprint of a longer prompt.
        headroom = self.swa_tokens(
            0, max_new_tokens, window_host_hit_lengths=window_host_hit_lengths
        )
        cap = int(self.remaining_swa) - headroom
        return max(0, cap // self.page_size * self.page_size)

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
            cap = self._chunk_cap(max_new_tokens, window_host_hit_lengths)
        if chunk_limit is None or cap <= 0:
            return False, None
        return True, min(chunk_limit, cap)

    def has_capacity(self) -> bool:
        return super().has_capacity() and self.remaining_swa > 0

    def available_chunk_tokens(self, chunk_limit: int) -> int | None:
        available = min(chunk_limit, int(self.remaining_total))
        if not self.req_ring:
            available = min(available, int(self.remaining_swa) - self.page_size)
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
    tail beyond it), and its state slots in full; the admitted work, and the
    expected decode of running requests, accumulate as offsets. A demand is
    admitted when the allocator's `admits` finds a layout for all of it after
    the cached entries the tree could evict.
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

    def _fits(self, full_tokens, window_tokens, state_slots=0, *, empty_pool=False):
        """Whether the pending work plus this demand is admitted (or this demand
        alone, in an empty pool)."""
        pending = not empty_pool
        return self.allocator.admits(
            full_tokens + (self.total_offset if pending else 0),
            {
                name: window_tokens.get(name, 0)
                + (self.window_offsets[name] if pending else 0)
                for name, _, _ in self.windows
            },
            state_slots + (self.pending_state_slots if pending else 0),
            evictable=(
                None
                if empty_pool
                else self.allocator.admission_evictable(self.tree_cache)
            ),
            empty_pool=empty_pool,
        )

    @property
    def remaining_swa(self):
        # Token views for paths that size work by a token count (dLLM): the
        # target window's reachable tokens less what is pending on it.
        if not self.windows:
            return 0
        name = next(name for name, component, _ in self.windows if component.is_swa)
        return (
            self.allocator.swa_available_size()
            + self.tree_cache.swa_evictable_size()
            - self.window_offsets[name]
        )

    def has_capacity(self) -> bool:
        return self._fits(0, {})

    def state_slots_fit(self, state_slots: int) -> bool:
        return not self.has_state_slots or self._fits(0, {}, state_slots)

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
        """Admit the whole request if it fits; otherwise, with chunking, the
        largest page-aligned chunk that fits now."""
        host_hits = dict(window_host_hit_lengths or {})
        if self._fits(
            total_tokens,
            self.window_tokens(
                extend_input_len,
                max_new_tokens,
                chunk_limit=chunk_limit,
                host_hit_lengths=host_hits,
            ),
            state_slots,
        ):
            return True, chunk_limit
        if chunk_limit is None:
            return False, None
        cap = self._chunk_cap(
            max_chunk_tokens=min(chunk_limit, max(0, extend_input_len - 1)),
            chunk_limit=chunk_limit,
            host_hit_lengths=host_hits,
            state_slots=state_slots,
        )
        return (True, min(chunk_limit, cap)) if cap > 0 else (False, None)

    def _chunk_fits(
        self, tokens, headroom, *, chunk_limit, host_hit_lengths, state_slots
    ):
        return self._fits(
            self.ceil_paged_tokens(tokens) + headroom + self.page_size,
            self.window_tokens(
                tokens,
                headroom,
                chunk_limit=chunk_limit,
                host_hit_lengths=host_hit_lengths,
            ),
            state_slots,
        )

    def _chunk_cap(
        self, *, max_chunk_tokens, chunk_limit, host_hit_lengths, state_slots
    ):
        """The largest page-aligned chunk (no decode headroom: it does not
        finish the request) the budget admits; a larger chunk only adds
        demand, so the answer is monotone."""
        low, high = 0, max(0, max_chunk_tokens) // self.page_size
        while low < high:
            mid = (low + high + 1) // 2
            if self._chunk_fits(
                mid * self.page_size,
                0,
                chunk_limit=chunk_limit,
                host_hit_lengths=host_hit_lengths,
                state_slots=state_slots,
            ):
                low = mid
            else:
                high = mid - 1
        return low * self.page_size

    def available_chunk_tokens(self, chunk_limit: int) -> int | None:
        return chunk_limit if chunk_limit > 0 else None

    def fit_chunk(
        self,
        *,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int,
        state_slots: int = 0,
    ) -> int | None:
        """Choose a bounded chunk with decode headroom only on its final pass."""
        candidate = min(extend_input_len, chunk_limit)
        finishes = candidate >= extend_input_len
        fits = dict(
            chunk_limit=chunk_limit, host_hit_lengths=None, state_slots=state_slots
        )
        if self._chunk_fits(candidate, max_new_tokens if finishes else 0, **fits):
            return chunk_limit
        cap = self._chunk_cap(
            max_chunk_tokens=max(0, candidate - 1) if finishes else candidate,
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
    ) -> None:
        super().reserve(
            extend_input_len,
            max_new_tokens,
            extra_tokens=extra_tokens,
        )
        self.pending_state_slots += state_slots
        for name, tokens in self.window_tokens(
            self.ceil_paged_tokens(extend_input_len),
            max_new_tokens,
            chunk_limit=chunk_limit,
        ).items():
            self.window_offsets[name] += tokens

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
        demand = {"window_tokens": windows}
        if self.has_state_slots:
            demand["state_slots"] = state_slots + self.pending_state_slots
        swa_tokens = next(
            (windows[name] for name, component, _ in self.windows if component.is_swa),
            0,
        )
        return full_tokens, swa_tokens, demand

    def prepare_load_back(self, **kwargs) -> bool:
        full_tokens, swa_tokens, demand = self._load_back_demand(**kwargs)
        return (
            self.allocator.reclaim_for_load_back(
                self.tree_cache, full_tokens, swa_tokens, **demand
            )
            is None
        )

    def load_back_scope(self, **kwargs):
        full_tokens, swa_tokens, demand = self._load_back_demand(**kwargs)
        return self.allocator.scoped_physical_takes(full_tokens, swa_tokens, **demand)
