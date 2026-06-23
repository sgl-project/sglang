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
    RadixCacheRuntimePyError,
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
        # Shared disabled-path tensor; must not be mutated.
        self._empty_indices = torch.empty((0,), dtype=torch.int64, device=self.device)

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
        """Per-component post-processing that requires Python-owned resources."""
        # Mamba CoW: allocate a request-local slot from the matched state.
        if (
            self.supports_mamba()
            and params.cow_mamba
            and rust_result.mamba_value is not None
        ):
            self._copy_on_write_mamba(
                params.req, rust_result.last_device_node_idx, rust_result.mamba_value
            )
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
        """Apply the insert-path emitted actions."""
        if not deferred_actions or self.token_to_kv_pool_allocator is None:
            return

        swa_node_indices: list[int] = []
        swa_values: list[torch.Tensor] = []
        for action in deferred_actions:
            tag = action[0]
            if tag == "FullFree":
                _, full_to_free = action
                self.token_to_kv_pool_allocator.free(full_to_free)
            elif tag == "SwaRecover":
                _, node_idx, old_full_to_free, new_full_value = action
                self.token_to_kv_pool_allocator.free(old_full_to_free)
                swa_node_indices.append(node_idx)
                swa_values.append(
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        new_full_value
                    )
                )
            elif tag == "SwaStamp":
                _, node_idx, full_value = action
                swa_node_indices.append(node_idx)
                swa_values.append(
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        full_value
                    )
                )
            else:
                raise RadixCacheRuntimePyError(
                    f"_process_insert_actions: unsupported insert action {tag!r}"
                )

        if swa_node_indices:
            self._rust_radix.apply_swa_writes(swa_node_indices, swa_values)

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
        if self.disable or (full_budget == 0 and swa_budget == 0 and mamba_budget == 0):
            return EvictResult(num_tokens_evicted=0)

        # FULL eviction can cross-bump freed[Swa], so release every freed bin
        # below even when that component's budget is 0.
        start_time = time.perf_counter()
        result = self._rust_radix.evict([full_budget, swa_budget, mamba_budget])

        full_idx = int(ComponentType.Full)
        swa_idx = int(ComponentType.Swa)
        mamba_idx = int(ComponentType.Mamba)
        if self.token_to_kv_pool_allocator is not None:
            for freed in result.freed[full_idx]:
                self.token_to_kv_pool_allocator.free(freed)
            for freed in result.freed[swa_idx]:
                self.token_to_kv_pool_allocator.free_swa(freed)
        if self.supports_mamba():
            for freed in result.freed[mamba_idx]:
                self.req_to_token_pool.mamba_allocator.free(freed)

        self.update_eviction_metrics(sum(result.evicted), start_time)
        return EvictResult(
            num_tokens_evicted=result.evicted[full_idx],
            swa_num_tokens_evicted=result.evicted[swa_idx],
            mamba_num_evicted=result.evicted[mamba_idx],
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
        return self._rust_radix.evictable_token_size()

    def protected_size(self) -> int:
        return self._rust_radix.protected_token_size()

    def total_size(self):
        return self._rust_radix.total_size()

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        if self.sliding_window_size is None:
            return 0
        return self._rust_radix.swa_evictable_token_size()

    def swa_protected_size(self) -> int:
        if self.sliding_window_size is None:
            return 0
        return self._rust_radix.swa_protected_token_size()

    def mamba_evictable_size(self) -> int:
        return (
            self._rust_radix.mamba_evictable_token_size()
            if self.supports_mamba()
            else 0
        )

    def mamba_protected_size(self) -> int:
        return (
            self._rust_radix.mamba_protected_token_size()
            if self.supports_mamba()
            else 0
        )

    def mamba_total_size(self) -> int:
        # Mamba's unit is slots, not tokens.
        return self._rust_radix.mamba_total_size() if self.supports_mamba() else 0

    def sanity_check(self) -> None:
        # TODO(Jialin): no-op stub — wire up a Rust-side LRU/tree consistency
        # check (the scheduler calls it on idle ticks).
        return None

    def supports_swa(self) -> bool:
        return self.sliding_window_size is not None

    def supports_mamba(self) -> bool:
        return self.mamba_cache_chunk_size is not None

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
            if self.supports_mamba():
                self.req_to_token_pool.free_mamba_cache(req)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # extra_buffer mode: truncate to Mamba chunk aligned.
        if self.enable_mamba_extra_buffer:
            cache_len = req.mamba_last_track_seqlen or 0
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        atom_len = len(radix_key)
        values = kv_indices[:atom_len].to(dtype=torch.int64, copy=True)

        mamba_value, mamba_ping_pong_track_buffer_to_keep = self._extract_mamba_value(
            req
        )

        mamba_exist = False
        if is_insert:
            insert_result = self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    prev_prefix_len=req.cache_protected_len,
                    swa_evicted_seqlen=req.swa_evicted_seqlen,
                    mamba_value=mamba_value,
                )
            )
            mamba_exist = insert_result.mamba_exist
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : atom_len]
            )
            # Skipped insert: caller still owns the Mamba slot.
            mamba_exist = mamba_value is not None

        # Free everything past the aligned atom prefix.
        self.token_to_kv_pool_allocator.free(kv_indices[atom_len:])

        # Mamba slot release. extra_buffer always frees the orphaned primary
        # (keeping the surviving ping-pong slot); no_buffer frees only on reject.
        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None
        free_mamba_cache = self.enable_mamba_extra_buffer or mamba_exist
        if self.supports_mamba() and free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
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

        # extra_buffer mode: truncate to Mamba chunk aligned.
        if self.enable_mamba_extra_buffer:
            cache_len = req.mamba_last_track_seqlen
            # No chunk-aligned boundary yet, skip caching.
            if cache_len is None:
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
                return
            token_ids = token_ids[:cache_len]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        atom_len = len(radix_key)
        values = kv_indices[:atom_len].to(dtype=torch.int64, copy=True)

        # Fork so decode mutations don't alias the cached state.
        mamba_value_forked = None
        if self.supports_mamba() and req.mamba_pool_idx is not None:
            mamba_value_src, _ = self._extract_mamba_value(req)
            assert mamba_value_src is not None, (
                "mamba_value_src must be present when supports_mamba() and "
                "req.mamba_pool_idx is not None"
            )
            mamba_value_forked = self._mamba_fork_from(mamba_value_src)

        insert_result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                prev_prefix_len=req.cache_protected_len,
                mamba_value=mamba_value_forked,
            )
        )

        # Release the forked slot when the cache didn't consume it.
        if mamba_value_forked is not None and insert_result.mamba_exist:
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

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
        # Clear the chunk-aligned marker so the next call recomputes it.
        if self.supports_mamba():
            req.mamba_last_track_seqlen = None

    # ----- Utils -----
    def _extract_mamba_value(
        self, req: "Req"
    ) -> tuple[Optional[torch.Tensor], Optional[int]]:
        """Build the `mamba_value` tensor to insert into the radix tree."""
        if not self.supports_mamba() or req.mamba_pool_idx is None:
            return None, None
        if not self.enable_mamba_extra_buffer:
            return req.mamba_pool_idx.unsqueeze(-1).clone(), None
        # extra_buffer mode: also return the ping-pong index to release later.
        track_buffer_to_keep = self.req_to_token_pool.get_mamba_ping_pong_other_idx(
            req.mamba_next_track_idx
        )
        mamba_value = (
            req.mamba_ping_pong_track_buffer[track_buffer_to_keep].unsqueeze(-1).clone()
        )
        return mamba_value, track_buffer_to_keep

    def _mamba_fork_from(
        self,
        mamba_value: torch.Tensor,
        protect_node_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Fork `mamba_value` in the pool, evicting only if alloc fails.

        TODO(Jialin): port this retry-with-alloc wrapper into `MambaPool`.
        """
        mamba_pool = self.req_to_token_pool.mamba_pool
        dst = mamba_pool.alloc(1)
        if dst is None:
            if protect_node_idx is not None:
                self.inc_lock_ref(protect_node_idx)
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            dst = mamba_pool.alloc(1)
            if protect_node_idx is not None:
                self.dec_lock_ref(protect_node_idx)
            assert dst is not None, "Can not alloc mamba cache"
        mamba_pool.copy_from(mamba_value, dst)
        return dst

    def _copy_on_write_mamba(
        self, req: "Req", last_node_idx: int, src_index: torch.Tensor
    ) -> None:
        """Copy the matched node's Mamba SSM state into req-local space."""
        if req.mamba_pool_idx is None:
            forked = self._mamba_fork_from(src_index, protect_node_idx=last_node_idx)
            req.mamba_pool_idx = forked[0]
        else:
            self.req_to_token_pool.mamba_pool.copy_from(
                src_index, req.mamba_pool_idx.unsqueeze(0)
            )
