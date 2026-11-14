"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple

import torch

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv.alloc.forward_steps import ForwardSteps
    from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
    from sglang.srt.mem_cache.kv.state_storage import StateStorage
    from sglang.srt.mem_cache.kv.storage import KVStorage
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    # Whether every pool takes its rows from one shared memory, so the prefix
    # cache reclaims an allocation's demand as one; else each pool is its own.
    pools_share_memory = False
    # Ids handed out index a virtual space the allocator translates to physical
    # rows (a unified pool's chain), not the pool's rows themselves.
    hands_out_virtual_ids = False
    # A request's recurrent state and its slots, when the model has them; the
    # KV cache configurator binds the request table's.
    state: Optional[StateStorage] = None
    # A unified pool's sub-pools as host and peer transfers read them.
    storage: Optional[KVStorage] = None
    # The steps a lazily compacting unified pool's freed pages wait for.
    forward_steps: Optional[ForwardSteps] = None

    @property
    def state_slots(self):
        """The state slot allocator, or None without recurrent state."""
        return None if self.state is None else self.state.slots

    # Whether freed pages can wait for in-flight KV writes before PD reuse.
    supports_reuse_guard = False

    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        # None: free right away. A list: hold frees until free_group_end().
        self.free_group: list[torch.Tensor] | None = None

    @property
    def size_full(self):
        return self.size

    @property
    def has_window_pool(self) -> bool:
        """Whether a sliding-window pool sits beside the full-attention one,
        with the window allocation surface (`free_swa`, `swa_available_size`, ...)."""
        return False

    def state_id_spaces(self) -> Dict[str, IdSpace]:
        """The id space of each recurrent-state sub-pool whose slots this
        allocator hands out, keyed by sub-pool name. Empty when the slot ids the
        request table holds index the state pool as they are."""
        return {}

    def state_ids(self, sub_pool: str) -> IdSpace:
        """How the state sub-pool ``sub_pool`` indexes the slot ids the request
        table holds, for a consumer outside a forward (the tree, PD transfers)."""
        from sglang.srt.mem_cache.kv.alloc.id_space import state_id_space

        return state_id_space(self.state_id_spaces(), sub_pool)

    # -- scheduler-facing capacity hooks --
    # The scheduler calls these unconditionally, with no allocator-type branches
    # on its side; the byte-accounted chain allocator overrides the token-count
    # defaults.

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        from sglang.srt.mem_cache.prefill_budget import PrefillBudget

        return PrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def request_slot_reserve(
        self, *, has_req_pool_slot: bool, prefill_tokens: int = 0
    ) -> int:
        """Extra KV slots beyond prefill, including request-owned workspace.

        ``prefill_tokens`` includes the cached prefix and selected extension.
        """
        return 0

    def max_new_tokens_for_memory(
        self,
        input_tokens: int,
        max_new_tokens: int,
        *,
        token_capacity: int,
        sliding_window_size: int | None,
        chunk_size: int | None,
        prefill_retire_lag: int,
        eviction_interval: int,
        checkpoint_interval: int = 0,
        decode_retire_lag: int = 0,
        chunk_align: int | None = None,
    ) -> int | None:
        """Clip generation to the empty-pool budget; None means prompt cannot fit.

        token_capacity is the scheduler's configured capacity, including its
        distributed token scaling. Shared pools use their physical byte layout.
        The window retirement inputs (`demand.window_retirement`, and the
        cache's `window_checkpoint_interval`) matter only to pools that keep
        windows.
        """
        paged_input = -(-input_tokens // self.page_size) * self.page_size
        return max(
            0, min(max_new_tokens, token_capacity - paged_input - self.page_size - 1)
        )

    def window_pool_names(self) -> Tuple[str, ...]:
        """The window pools this allocator keeps, by the names a
        `KVDemand` uses for them; none for a pool without windows."""
        return ()

    def target_window_pool(self) -> Optional[str]:
        """The window pool a reader that sizes one window reads (the scheduler's
        window budget, a running request's retention): the target model's own
        window, or None without windows."""
        names = self.window_pool_names()
        return names[0] if names else None

    def prealloc_ceiling_fits(self, demand: KVDemand) -> bool | None:
        """Whether ``demand`` could EVER be preallocated, or None when this
        pool has no ceiling of its own and the caller's token capacity is the
        only bound.
        """
        return None

    def window_capacity_and_available(self, sub_pool: str) -> Tuple[int, int]:
        """Window sub-pool ``sub_pool``'s token capacity and what a take from it
        can serve now, as the scheduler's window budget reads them."""
        if sub_pool not in self.window_pool_names():
            raise KeyError(f"no window sub-pool {sub_pool!r}")
        _, window = self.swa_capacity_and_available(
            full_capacity=self.size_full, swa_capacity=self.size_swa
        )
        return window

    def prealloc_fits(
        self,
        demand: KVDemand,
        *,
        evictable: Mapping[str, int],
        full_budget_tokens: int,
        window_budget_tokens: Optional[Mapping[str, int]] = None,
    ) -> bool:
        """Whether a decode-node preallocation of this size fits.

        The budgets are what the owner and each window (by sub-pool name) have
        left once decode headroom and retraction are reserved. Separate buffers
        check each side against its own budget, which already holds what
        ``evictable`` could free; state slots live in a pool of their own. A
        pool cutting every side from one buffer overrides this.
        """
        demand = demand.checked(self.window_pool_names())
        budgets = window_budget_tokens or {}
        return demand.full_tokens <= full_budget_tokens and all(
            tokens <= budgets[name]
            for name, tokens in demand.window_tokens.items()
            if name in budgets
        )

    def admit_preallocation(
        self,
        tree_cache,
        admission: KVDemand,
        take: KVDemand,
        *,
        evictable: Mapping[str, int],
        full_budget_tokens: int,
        window_budget_tokens: Optional[Mapping[str, int]] = None,
    ) -> bool:
        """Whether a decode-node preallocation is admitted (``admission``,
        with its decode reservation) and its rows (``take``) can be taken
        now. Separate buffers hold the room their budgets count, so nothing
        is reclaimed here: admission is the whole answer."""
        return self.prealloc_fits(
            admission,
            evictable=evictable,
            full_budget_tokens=full_budget_tokens,
            window_budget_tokens=window_budget_tokens,
        )

    def get_extend_allocation_demand(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        *,
        conservative_num_tokens: int,
        shard_size: int,
    ) -> int:
        """Token demand for eviction before paged extend."""
        return conservative_num_tokens

    def reclaim_for_load_back(self, tree_cache, demand: KVDemand) -> Optional[str]:
        """Make room before a host load takes ``demand``'s rows; None means
        ready. Separate buffers reclaim nothing here."""
        return None

    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> bool | None:
        """Evict unlocked prefix-cache entries until this allocator can serve
        ``num_tokens`` or nothing evictable remains.

        Return whether capacity was realized, or None if it still needs checking.
        """
        from sglang.srt.mem_cache.common import _evict_until_allocatable
        from sglang.srt.mem_cache.kv.alloc.demand import KVDemand

        if tree_cache is None or not tree_cache.supports_prefix_sharing():
            return
        if self.available_size() < num_tokens:
            tree_cache.evict_for_alloc(KVDemand(full_tokens=num_tokens))
            _evict_until_allocatable(tree_cache, self, num_tokens)

    def check_decode_capacity(
        self,
        *,
        num_tokens: int,
        tree_cache,
        requests=None,
        spec_algorithm=None,
    ) -> bool:
        """Whether the next decode step's ``num_tokens`` allocation fits after
        evicting reclaimable cache. The retract loop converges on this same
        check, so a shortfall here retracts instead of failing in alloc.
        ``requests`` and ``spec_algorithm`` provide optional request-level context for allocators
        whose demand cannot be represented by a single token count."""
        self.evict_to_free_tokens(tree_cache, num_tokens)
        return self.available_size() >= num_tokens

    def verify_byte_accounting(self) -> list:
        """Idle-time diagnostic: recompute byte/slot accounting and return
        violation strings, empty when healthy. Static pools have no byte model."""
        return []

    def window_available_size(self, sub_pool: str) -> int:
        """What a take from window pool ``sub_pool`` can serve now; a pool
        keeping one window serves it under any name."""
        return self.swa_available_size()

    def sub_pool_roles(self) -> Tuple[str, Tuple[str, ...], Optional[str]]:
        """The token pool's name, every window pool's, and the state pool's
        (None when state slots are not taken from this allocator's memory)."""
        return ("kv", self.window_pool_names(), None)

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def owner_available_size(self):
        """What the token pool can serve now; an allocator keeping a window
        pool beside it answers for its full-attention side."""
        return self.available_size()

    def get_kvcache(self):
        return self._kvcache

    @property
    def cpu_copy_carries_mamba(self) -> bool:
        """Whether `get_cpu_copy` carries a request's recurrent state; else the
        scheduler backs the state up on its own."""
        return self._kvcache.cpu_copy_carries_mamba

    def get_all_free_pages(self):
        # Debug / invariant census; None when the pool has no page free list.
        if self.free_pages is None:
            return None
        if self.release_pages is None or len(self.release_pages) == 0:
            return self.free_pages
        return torch.cat((self.free_pages, self.release_pages))

    def free_group_begin(self):
        assert self.free_group is None, "free groups cannot be nested"
        self.free_group = []

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self.free(torch.cat(pending))

    @staticmethod
    def _copy_for_free_group(free_index: torch.Tensor) -> torch.Tensor:
        """Take ownership before a caller can mutate a deferred tensor view."""
        return free_index.clone()

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Token ids as the PD transfer engine addresses them. Identity here
        because a static pool's ids index its registered buffers directly;
        virtual-id pools must override."""
        return kv_indices

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError()

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        if self.page_size > 1:
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()

    def free_full(self, free_index: torch.Tensor):
        """Free full-attention slots whose paired SWA slots the caller already
        released. A single pool has no SWA peer, so this is a plain free()."""
        self.free(free_index)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """Free ``kv_row[start_pos : start_pos + n]`` of one request.

        In page units the segment is ``[start_pos // ps, ceil(end / ps))``:
        ``start_pos`` sits on a page boundary, the end may fall mid-page, and
        the whole last page is released."""
        assert start_pos % self.page_size == 0, (
            f"segment start {start_pos} is not page-aligned"
        )
        self.free(free_index)

    def free_segments(self, segments):
        """Free several ``(free_index, start_pos)`` segments of one request's
        kv row. Each covers pages ``[start_pos // ps, ceil(end / ps))``; starts
        are page-aligned and consecutive page ranges do not overlap, so every
        page is released exactly once."""
        for free_index, start_pos in self._page_disjoint(segments):
            self.free_segment(free_index, start_pos=start_pos)

    def free_full_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """free_full() for a kv-row segment; same start-alignment contract as
        free_segment()."""
        assert start_pos % self.page_size == 0, (
            f"segment start {start_pos} is not page-aligned"
        )
        self.free_full(free_index)

    def free_full_segments(self, segments):
        """free_segments() for the full side alone; see free_full()."""
        for free_index, start_pos in self._page_disjoint(segments):
            self.free_full_segment(free_index, start_pos=start_pos)

    def _page_disjoint(self, segments):
        ps = self.page_size
        prev_end = None
        for free_index, start_pos in segments:
            n = free_index.numel()
            if n == 0:
                continue
            assert prev_end is None or start_pos // ps > (prev_end - 1) // ps, (
                f"segment at {start_pos} shares a page with the one ending at {prev_end}"
            )
            prev_end = start_pos + n
            yield free_index, start_pos
