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
They neither select requests nor mutate the prefix cache or allocator.
"""

from typing import Optional


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
        # A full window would double-charge short cached-prefix resumes;
        # including extend keeps the reservation above the prefill allocation.
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

    def __init__(self, allocator, tree_cache, *, num_mixed_decode_tokens: int = 0):
        self.allocator = allocator
        self.tree_cache = tree_cache
        self.page_size = allocator.page_size
        self.total_offset = num_mixed_decode_tokens
        self.current_offset = num_mixed_decode_tokens
        self.swa_offset = 0

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

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

    def check_prefill(
        self,
        *,
        extend_input_len: int,
        total_tokens: int,
        max_new_tokens: int,
        input_tokens: int,
        swa_host_hit_length: int,
        chunk_limit: int | None,
    ) -> tuple[bool, int | None]:
        """Return admission feasibility and a memory bound on the requested chunk."""
        return (
            (True, chunk_limit)
            if total_tokens < self.remaining_total
            else (False, None)
        )

    def can_allocate_prefill(
        self,
        *,
        paged_input: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int | None,
    ) -> bool:
        return paged_input <= min(self.remaining_current, self.remaining_total)

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
    ) -> int | None:
        return chunk_limit

    def reserve(
        self,
        extend_input_len: int,
        max_new_tokens: int,
        *,
        extra_tokens: int = 0,
        chunk_limit: int | None = None,
        is_chunked_continuation: bool = False,
    ) -> None:
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
        immediate = extend_input_len + self.page_size + extra_tokens
        self.total_offset += immediate + max_new_tokens
        self.current_offset += immediate


class SWAPrefillBudget(PrefillBudget):
    """Separate FULL/SWA partitions, including per-request SWA rings."""

    def __init__(self, *args, all_swa=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_swa = all_swa
        self.req_ring = getattr(self.allocator, "swa_req_ring", False) is True

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
        swa_host_hit_length=0,
    ):
        if self.req_ring:
            return self.allocator.swa_ring_cost_tokens
        return estimate_swa_kv_tokens(
            extend_input_len,
            max_new_tokens,
            sliding_window_size=self.tree_cache.sliding_window_size,
            page_size=self.page_size,
            allocation_limit=chunk_limit,
            host_hit_length=swa_host_hit_length,
        )

    def swa_never_fits(self, extend_input_len, max_new_tokens, **kwargs):
        needed = self.swa_tokens(extend_input_len, max_new_tokens, **kwargs)
        return (
            needed > self.allocator.size_swa
            if self.req_ring
            else needed >= self.allocator.size_swa
        )

    def _chunk_cap(self, max_new_tokens, swa_host_hit_length=0):
        headroom = self.swa_tokens(
            0, max_new_tokens, swa_host_hit_length=swa_host_hit_length
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
        swa_host_hit_length: int,
        chunk_limit: int | None,
    ) -> tuple[bool, int | None]:
        if total_tokens >= self.remaining_total:
            return False, None
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
        needed = self.swa_tokens(
            extend_input_len,
            max_new_tokens,
            chunk_limit=chunk_limit,
            swa_host_hit_length=swa_host_hit_length,
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
            swa_host_hit_length=swa_host_hit_length,
        ):
            cap = self._chunk_cap(max_new_tokens, swa_host_hit_length)
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
    ) -> bool:
        return (
            super().can_allocate_prefill(
                paged_input=paged_input,
                extend_input_len=extend_input_len,
                max_new_tokens=max_new_tokens,
                chunk_limit=chunk_limit,
            )
            and self.swa_tokens(
                extend_input_len, max_new_tokens, chunk_limit=chunk_limit
            )
            <= self.remaining_swa
        )

    def reserve(
        self,
        extend_input_len: int,
        max_new_tokens: int,
        *,
        extra_tokens: int = 0,
        chunk_limit: int | None = None,
        is_chunked_continuation: bool = False,
    ) -> None:
        super().reserve(extend_input_len, max_new_tokens, extra_tokens=extra_tokens)
        # A continuation already owns its ring slot.
        if not (self.req_ring and is_chunked_continuation):
            self.swa_offset += self.swa_tokens(
                self.ceil_paged_tokens(extend_input_len),
                max_new_tokens,
                chunk_limit=chunk_limit,
            )


class SharedSWAPrefillBudget(SWAPrefillBudget):
    """FULL and SWA reservations compete for the same physical byte budget."""

    def __init__(self, *args, num_mixed_decode_tokens=0, **kwargs):
        super().__init__(
            *args, num_mixed_decode_tokens=num_mixed_decode_tokens, **kwargs
        )
        self.swa_offset = num_mixed_decode_tokens

    def _fits(self, full_tokens, swa_tokens, *, empty_pool=False):
        return self.allocator.can_reserve(
            full_tokens + (0 if empty_pool else self.total_offset),
            swa_tokens + (0 if empty_pool else self.swa_offset),
            full_evictable_tokens=0
            if empty_pool
            else self.tree_cache.full_evictable_size(),
            swa_evictable_tokens=0
            if empty_pool
            else self.tree_cache.swa_evictable_size(),
            empty_pool=empty_pool,
            require_token_slack=empty_pool,
        )

    def _joint_chunk_cap(self, *, max_chunk_tokens, chunk_limit, swa_host_hit_length=0):
        lo, hi = 0, max(0, max_chunk_tokens) // self.page_size
        while lo < hi:
            mid = (lo + hi + 1) // 2
            tokens = mid * self.page_size
            if self._fits(
                tokens + self.page_size,
                self.swa_tokens(
                    tokens,
                    0,
                    chunk_limit=chunk_limit,
                    swa_host_hit_length=swa_host_hit_length,
                ),
            ):
                lo = mid
            else:
                hi = mid - 1
        return lo * self.page_size

    def check_prefill(
        self,
        *,
        extend_input_len: int,
        total_tokens: int,
        max_new_tokens: int,
        input_tokens: int,
        swa_host_hit_length: int,
        chunk_limit: int | None,
    ) -> tuple[bool, int | None]:
        needed = self.swa_tokens(
            extend_input_len,
            max_new_tokens,
            chunk_limit=chunk_limit,
            swa_host_hit_length=swa_host_hit_length,
        )
        if self._fits(total_tokens, needed):
            return True, chunk_limit
        if chunk_limit is None or self._fits(
            input_tokens + max_new_tokens + self.page_size,
            self.swa_tokens(input_tokens, max_new_tokens, chunk_limit=chunk_limit),
            empty_pool=True,
        ):
            return False, None
        cap = self._joint_chunk_cap(
            max_chunk_tokens=min(chunk_limit, max(0, extend_input_len - 1)),
            chunk_limit=chunk_limit,
            swa_host_hit_length=swa_host_hit_length,
        )
        return (True, min(chunk_limit, cap)) if cap > 0 else (False, None)

    def has_capacity(self) -> bool:
        return self._fits(0, 0)

    def available_chunk_tokens(self, chunk_limit: int) -> int | None:
        return chunk_limit

    def fit_chunk(
        self,
        *,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int,
    ) -> int | None:
        candidate = min(extend_input_len, chunk_limit)
        finishes = candidate >= extend_input_len
        headroom = max_new_tokens if finishes else 0
        if self._fits(
            candidate + headroom + self.page_size,
            self.swa_tokens(candidate, headroom, chunk_limit=chunk_limit),
        ):
            return chunk_limit
        cap = self._joint_chunk_cap(
            max_chunk_tokens=max(0, candidate - 1) if finishes else candidate,
            chunk_limit=chunk_limit,
        )
        return min(chunk_limit, cap) if cap > 0 else None

    def can_allocate_prefill(
        self,
        *,
        paged_input: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: int | None,
    ) -> bool:
        return self._fits(
            extend_input_len + max_new_tokens + self.page_size,
            self.swa_tokens(extend_input_len, max_new_tokens, chunk_limit=chunk_limit),
        )
