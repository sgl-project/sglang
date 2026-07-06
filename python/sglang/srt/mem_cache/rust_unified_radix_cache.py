# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Python orchestrator over the Rust radix cache."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

import torch
from sglang.srt.mem_cache._mem_cache_core import (
    ComponentType,
    RadixCacheInfraPyError,
    RustBigramRadixCacheWrapper,
    RustPageRadixCacheWrapper,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.rust_unified_cache_components import build_components
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

_DEFAULT_INIT_NODE_CAPACITY = 1024


class RustUnifiedRadixCache(BasePrefixCache):
    """Orchestration layer: route tree ops to the Rust radix cache and process
    the emitted deferred actions on the Python side; Python owns the pools.
    """

    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable_finished_insert = params.disable_finished_insert
        self.sliding_window_size = params.sliding_window_size
        self.is_eagle = params.is_eagle
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        server_args = get_global_server_args()
        if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            self.mamba_cache_chunk_size: Optional[int] = (
                server_args.mamba_cache_chunk_size
            )
        else:
            self.mamba_cache_chunk_size = None
        self._reject_unsupported(params, server_args)

        if params.enable_metrics:
            self.init_metrics_collector()
        if self.token_to_kv_pool_allocator is not None:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")
        device_str = self._device_to_rust_str(self.device)
        wrapper_cls = (
            RustBigramRadixCacheWrapper if self.is_eagle else RustPageRadixCacheWrapper
        )
        self._rust_radix: Any = wrapper_cls(
            device=device_str,
            page_size=self.page_size,
            init_node_capacity=_DEFAULT_INIT_NODE_CAPACITY,
            sliding_window_size=self.sliding_window_size,
            mamba_cache_chunk_size=self.mamba_cache_chunk_size,
        )
        # Shared empty tensor for fast return; must not be mutated.
        self._empty_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        # Pool-side per-component handlers; dispatch to these instead of
        # branching per component inline.
        self.components = build_components(self)

    def _reject_unsupported(self, params: CacheInitParams, server_args: Any) -> None:
        if params.eviction_policy.lower() != "lru":
            raise RadixCacheInfraPyError(
                f"RustUnifiedRadixCache: eviction_policy={params.eviction_policy!r} "
                "not supported, only 'lru'"
            )
        if params.cache_ttl_seconds is not None:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: cache_ttl_seconds not supported"
            )
        if params.enable_kv_cache_events:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: enable_kv_cache_events=True not supported"
            )
        if params.tree_components is not None:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: tree_components not supported"
            )
        if server_args.enable_hierarchical_cache:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: hierarchical cache (HiCache) not supported"
            )

    @staticmethod
    def _device_to_rust_str(device: Any) -> str:
        # Resolve bare "cuda" to the current device so TP>1 ranks pick the right index.
        def _resolve_cuda_index() -> str:
            try:
                return f"cuda:{torch.cuda.current_device()}"
            except Exception:
                return "cuda"

        if isinstance(device, torch.device):
            if device.type == "cpu":
                return "cpu"
            if device.type == "cuda":
                return (
                    f"cuda:{device.index}"
                    if device.index is not None
                    else _resolve_cuda_index()
                )
            raise RadixCacheInfraPyError(
                f"RustUnifiedRadixCache: device {device!r} not supported"
            )
        if isinstance(device, str) and device == "cuda":
            return _resolve_cuda_index()
        return str(device)

    def reset(self) -> None:
        # Clears tree state only; not the allocator.
        self._rust_radix.reset()

    def supports_fast_match_prefix(self) -> bool:
        return True

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if self.disable:
            return self._empty_match_result()

        token_ids = params.key.raw_token_ids()
        rust_result = self._rust_radix.match_prefix(token_ids, params.key.extra_key)

        # Device-only cache: host / best-match nodes collapse onto the device node.
        last_device_node = rust_result.last_device_node_idx
        result = MatchResult(
            device_indices=rust_result.device_indices,
            last_device_node=last_device_node,
            last_host_node=last_device_node,
            best_match_node=last_device_node,
            mamba_branching_seqlen=rust_result.mamba_branching_seqlen,
        )
        return self._finalize_match_result(params, rust_result, result)

    def _finalize_match_result(
        self,
        params: MatchPrefixParams,
        rust_result: Any,
        result: MatchResult,
    ) -> MatchResult:
        """Component post-processing that requires Python-owned resources."""
        for comp in self.components.values():
            comp.finalize_match_result(params, rust_result)
        return result

    def _empty_match_result(self) -> MatchResult:
        # `last_device_node=None` makes inc/dec_lock_ref short-circuit.
        return MatchResult(
            device_indices=self._empty_indices,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if params.priority != 0:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: InsertParams.priority != 0 not supported (LRU only)"
            )

        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64, device=self.device)

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)

        aligned_key = key.page_aligned(self.page_size)
        atom_count = len(aligned_key)
        token_ids = aligned_key.token_ids
        value = value[:atom_count] if atom_count > 0 else value

        rust_result = self._rust_radix.insert(
            token_ids,
            value,
            aligned_key.extra_key,
            params.prev_prefix_len,
            params.swa_evicted_seqlen,
            params.mamba_value,
        )
        self._process_insert_actions(rust_result.deferred_actions)
        return InsertResult(
            prefix_len=rust_result.prefix_len,
            mamba_exist=rust_result.mamba_value_exists,
        )

    def _process_insert_actions(self, deferred_actions: list[tuple]) -> None:
        """Route each action to its component by ComponentType, then commit."""
        if not deferred_actions or self.token_to_kv_pool_allocator is None:
            return
        for action in deferred_actions:
            self.components[action[0]].stage_insert_action(action)
        # Commit once per insert: each component flushes its staged actions in
        # a single batch (e.g. SWA emits one apply_swa_writes call).
        for comp in self.components.values():
            comp.commit_insert_actions()

    def evict(self, params: EvictParams) -> EvictResult:
        if params.mamba_num != 0 and not self.supports_mamba():
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: EvictParams.mamba_num != 0 requires "
                "Mamba configuration (HybridReqToTokenPool)"
            )

        full_budget = max(0, params.num_tokens)
        swa_budget = (
            max(0, params.swa_num_tokens) if self.sliding_window_size is not None else 0
        )
        mamba_budget = max(0, params.mamba_num) if self.supports_mamba() else 0
        if (
            self.disable
            or self.token_to_kv_pool_allocator is None
            or (full_budget == 0 and swa_budget == 0 and mamba_budget == 0)
        ):
            return EvictResult(num_tokens_evicted=0)

        # FULL eviction can cross-bump freed[Swa], so release every freed bin
        # below even when that component's budget is 0.
        start_time = time.perf_counter()
        result = self._rust_radix.evict([full_budget, swa_budget, mamba_budget])

        for idx, comp in self.components.items():
            comp.free_evicted(result.freed[idx])

        self.update_eviction_metrics(sum(result.evicted), start_time)
        return EvictResult(
            num_tokens_evicted=result.evicted[ComponentType.Full],
            swa_num_tokens_evicted=result.evicted[ComponentType.Swa],
            mamba_num_evicted=result.evicted[ComponentType.Mamba],
        )

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if self.disable or node is None:
            return IncLockRefResult(delta=0)
        delta, swa_uuid_for_lock = self._rust_radix.inc_lock_ref(node)
        return IncLockRefResult(delta=delta, swa_uuid_for_lock=swa_uuid_for_lock)

    def dec_lock_ref(
        self,
        node: Any,
        params: Optional[DecLockRefParams] = None,
    ) -> DecLockRefResult:
        if self.disable or node is None:
            return DecLockRefResult()
        # `swa_uuid_for_lock` stops SWA's release walk at the matching boundary.
        swa_uuid_for_lock = params.swa_uuid_for_lock if params is not None else None
        self._rust_radix.dec_lock_ref(node, swa_uuid_for_lock)
        return DecLockRefResult()

    def evictable_size(self) -> int:
        return self.components[ComponentType.Full].evictable_size()

    def protected_size(self) -> int:
        return self.components[ComponentType.Full].protected_size()

    def total_size(self):
        return self._rust_radix.total_size()

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        comp = self.components.get(ComponentType.Swa)
        return comp.evictable_size() if comp is not None else 0

    def swa_protected_size(self) -> int:
        comp = self.components.get(ComponentType.Swa)
        return comp.protected_size() if comp is not None else 0

    def mamba_evictable_size(self) -> int:
        comp = self.components.get(ComponentType.Mamba)
        return comp.evictable_size() if comp is not None else 0

    def mamba_protected_size(self) -> int:
        comp = self.components.get(ComponentType.Mamba)
        return comp.protected_size() if comp is not None else 0

    def mamba_total_size(self) -> int:
        comp = self.components.get(ComponentType.Mamba)
        return comp.total_size() if comp is not None else 0

    def sanity_check(self) -> None:
        # TODO(Jialin): no-op stub — wire up a Rust-side LRU/tree consistency
        # check (the scheduler calls it on idle ticks).
        return None

    def supports_swa(self) -> bool:
        return ComponentType.Swa in self.components

    def supports_mamba(self) -> bool:
        return ComponentType.Mamba in self.components

    # TODO(Jialin): expose Rust-side iteration; leak-diagnostic only.
    def all_values_flatten(self) -> torch.Tensor:
        return self._empty_indices

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._empty_indices

    def pretty_print(self):
        raise RadixCacheInfraPyError(
            "RustUnifiedRadixCache: pretty_print() not supported"
        )

    def take_events(self):
        return []

    def cache_finished_req(self, req: "Req", is_insert: bool = True, **kwargs) -> None:
        """Cache the prefix of a finished request and free its tail."""
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            for comp in self.components.values():
                comp.cleanup_after_caching_req(req, is_finished=True, disabled=True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        insert_params = InsertParams(
            prev_prefix_len=req.cache_protected_len,
            swa_evicted_seqlen=req.swa_evicted_seqlen,
        )
        # Components fill their insert value and may shorten the cached length.
        cache_len = len(token_ids)
        for comp in self.components.values():
            cl = comp.prepare_for_caching_req(
                req, insert_params, len(token_ids), is_finished=True
            )
            if cl is not None:
                cache_len = min(cache_len, cl)
        if cache_len != len(token_ids):
            cache_end_idx = max(cache_len, req.cache_protected_len)
            self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
            token_ids = token_ids[:cache_len]
            kv_indices = kv_indices[:cache_len]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        atom_len = len(radix_key)
        insert_params.key = radix_key
        insert_params.value = kv_indices[:atom_len].to(dtype=torch.int64, copy=True)

        insert_result = None
        if is_insert:
            insert_result = self.insert(insert_params)
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : atom_len]
            )

        # Free everything past the aligned atom prefix.
        self.token_to_kv_pool_allocator.free(kv_indices[atom_len:])

        for comp in self.components.values():
            comp.cleanup_after_caching_req(
                req,
                is_finished=True,
                inserted=is_insert,
                insert_result=insert_result,
                insert_params=insert_params,
            )

        # Release the prefill lock.
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
        )

    def cache_unfinished_req(self, req: "Req", chunked: bool = False, **kwargs) -> None:
        if self.disable:
            return

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        insert_params = InsertParams(
            chunked=chunked,
            prev_prefix_len=req.cache_protected_len,
        )
        # Components fill their insert value; a None return means skip caching.
        cache_len = len(token_ids)
        for comp in self.components.values():
            cl = comp.prepare_for_caching_req(
                req, insert_params, len(token_ids), is_finished=False
            )
            if cl is None:
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
                return
            cache_len = min(cache_len, cl)
        token_ids = token_ids[:cache_len]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        atom_len = len(radix_key)
        insert_params.key = radix_key
        insert_params.value = kv_indices[:atom_len].to(dtype=torch.int64, copy=True)

        insert_result = self.insert(insert_params)

        for comp in self.components.values():
            comp.cleanup_after_caching_req(
                req,
                is_finished=False,
                inserted=True,
                insert_result=insert_result,
                insert_params=insert_params,
            )

        # Re-read the canonical tree-owned indices; insert may have de-duplicated.
        # With SWA a rematch can return fewer indices than atom_len.
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        # The `+ page_size - 1` slack tolerates a trailing partial page.
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ) and len(new_indices) <= atom_len, (
            f"cache_unfinished_req post-insert rematch out of bounds: "
            f"{req.cache_protected_len=}, {len(new_indices)=}, "
            f"{atom_len=}, {self.page_size=}"
        )

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # Lock-ref handoff: dec the old node before inc the new one.
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
        )
        inc_result = self.inc_lock_ref(new_last_node)
        req.swa_uuid_for_lock = inc_result.swa_uuid_for_lock

        # Extend back kv indices past the aligned boundary.
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node
        req.cache_protected_len = len(new_indices)
