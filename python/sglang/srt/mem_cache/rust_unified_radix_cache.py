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
"""RustUnifiedRadixCache: Python orchestrator over the Rust radix cache.

Coordinates three pieces:
    1. Rust RadixCache (sglang.srt.mem_cache._mem_cache_core.RustPageRadixCacheWrapper) — owns
       tree state and lock_ref accounting. Handles all page sizes
       (`page_size >= 1`); `page_size=1` uses one-element page keys.
    2. Python ReqToTokenPool — owns per-request kv-index storage; unchanged.
    3. Python {Token,Paged}TokenToKVPoolAllocator — owns slot allocation and
       per-token KV cache references; unchanged.

Drop-in replacement for `sglang.srt.mem_cache.radix_cache.RadixCache` for the
v1 supported configuration:
    * Full attention; SWA when `sliding_window_size` is set (no Mamba, no HiCache).
    * page_size >= 1.
    * LRU eviction only.
    * No EAGLE bigram, no `enable_kv_cache_events`, no TTL eviction.
    * No insert priority (LRU ignores it).

Unsupported features raise `RadixCacheInfraPyError`. Construction-time
rejections fail fast at process start; per-call rejections fail the call
without corrupting cache state.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import torch
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.registry import (
    get_radix_cache_factory,
    register_radix_cache_backend,
)
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

ComponentType: Any = None
RadixCacheInfraPyError: Any = None
RadixCacheRuntimePyError: Any = None
RustBigramRadixCacheWrapper: Any = None
RustPageRadixCacheWrapper: Any = None
_NATIVE_SYMBOLS_LOADED = False


def _load_native_symbols() -> None:
    """Load the PyO3 extension only when the Rust backend is selected."""
    global ComponentType
    global RadixCacheInfraPyError
    global RadixCacheRuntimePyError
    global RustBigramRadixCacheWrapper
    global RustPageRadixCacheWrapper
    global _NATIVE_SYMBOLS_LOADED

    if _NATIVE_SYMBOLS_LOADED:
        return

    try:
        from sglang.srt.mem_cache._mem_cache_core import (
            ComponentType as NativeComponentType,
            RadixCacheInfraPyError as NativeRadixCacheInfraPyError,
            RadixCacheRuntimePyError as NativeRadixCacheRuntimePyError,
            RustBigramRadixCacheWrapper as NativeRustBigramRadixCacheWrapper,
            RustPageRadixCacheWrapper as NativeRustPageRadixCacheWrapper,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "sglang.srt.mem_cache._mem_cache_core":
            raise ModuleNotFoundError(
                "RustUnifiedRadixCache requires native extension "
                "sglang.srt.mem_cache._mem_cache_core. Install SGLang with "
                "`python -m pip install -e python` or build the package before "
                "using `--radix-cache-backend rust_unified`."
            ) from exc
        raise

    ComponentType = NativeComponentType
    RadixCacheInfraPyError = NativeRadixCacheInfraPyError
    RadixCacheRuntimePyError = NativeRadixCacheRuntimePyError
    RustBigramRadixCacheWrapper = NativeRustBigramRadixCacheWrapper
    RustPageRadixCacheWrapper = NativeRustPageRadixCacheWrapper
    _NATIVE_SYMBOLS_LOADED = True


# Initial capacity hint for the Rust tree node pool. The pool grows on demand;
# this just avoids early reallocations during warmup.
_DEFAULT_INIT_NODE_CAPACITY = 1024


class RustUnifiedRadixCache(BasePrefixCache):
    """Python orchestrator routing tree ops to the Rust radix cache while
    keeping Python ownership of `req_to_token_pool` and the allocator.

    The `req.last_node` field carries an opaque integer (the Rust NodeIdx)
    instead of a Python `TreeNode`. External code that previously read
    `req.last_node.X` attributes will break — only HiCache / Mamba / LMCache
    paths do this today, all out of v1 scope.
    """

    def __init__(self, params: CacheInitParams):
        _load_native_symbols()

        # Required fields per `PrefixCacheTrait`. External code reads these
        # directly (e.g. observability `available_and_evictable_str`).
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable_finished_insert = params.disable_finished_insert
        self.sliding_window_size = params.sliding_window_size
        self.is_eagle = params.is_eagle
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        server_args = get_global_server_args()
        # Enable Mamba if the scheduler passed in a HybridReqToTokenPool.
        if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            self.mamba_cache_chunk_size: Optional[int] = (
                server_args.mamba_cache_chunk_size
            )
        else:
            self.mamba_cache_chunk_size = None
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        # Confirm the radix cache related setups are supported.
        self._reject_unsupported(params)
        if self.enable_hierarchical_cache:
            self._reject_unsupported_hicache(server_args)

        if params.enable_metrics:
            self.init_metrics_collector()
        if self.token_to_kv_pool_allocator is not None:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")
        device_str = self._device_to_rust_str(self.device)
        self.cache_controller: Any = None
        # Pick the concrete wrapper class at construction. The two share an
        # identical Python surface (constructor signature + every method
        # signature), so no per-method dispatch is needed downstream — the
        # bigram wrapper internally builds `(t[i], t[i+1])` overlap pairs
        # from the raw 1-D `int64` keys it receives. Pre-call trimming in
        # Python (page-align in raw-token space) would silently corrupt
        # the bigram count for the EAGLE path; we drop all such trims
        # below and let Rust own page-alignment in atom units (= bigram
        # pairs when `is_eagle`).
        wrapper_cls = (
            RustBigramRadixCacheWrapper if self.is_eagle else RustPageRadixCacheWrapper
        )
        self._rust_radix: Any = wrapper_cls(
            device=device_str,
            page_size=self.page_size,
            init_node_capacity=_DEFAULT_INIT_NODE_CAPACITY,
            sliding_window_size=self.sliding_window_size,
            mamba_cache_chunk_size=self.mamba_cache_chunk_size,
            enable_hicache=self.enable_hierarchical_cache,
        )
        # Cache a single empty tensor for the disabled-cache match-result
        # path so we don't allocate per call. Callers must not mutate it
        # (empty tensors aren't typically mutated, so this is safe by
        # convention).
        self._empty_indices = torch.empty((0,), dtype=torch.int64, device=self.device)

    def _reject_unsupported(self, params: CacheInitParams) -> None:
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

    # HiCache (host tier) restrictions — OSS has no equivalent gate.
    def _reject_unsupported_hicache(self, server_args: Any) -> None:
        # No Mamba host tier yet.
        if self.mamba_cache_chunk_size is not None:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache HiCache: Mamba host tier is not supported yet"
            )
        # No SWA host tier yet.
        if self.sliding_window_size is not None:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache HiCache: SWA host tier is not supported yet"
            )
        # Device <-> host only (no L3 storage backend yet).
        if server_args.hicache_storage_backend is not None:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache HiCache: storage backend (L3) is not supported yet"
            )
        # Write-through only (no write-back yet).
        if server_args.hicache_write_policy not in (
            "write_through",
            "write_through_selective",
        ):
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache HiCache: write_policy="
                f"{server_args.hicache_write_policy!r} is not supported yet "
                "(only write_through / write_through_selective)"
            )

    @staticmethod
    def _device_to_rust_str(device: Any) -> str:
        # Resolve unindexed cuda → "cuda:<current_device>" so per-rank
        # processes (TP > 1) get the right index. Rust's `parse_device`
        # treats bare "cuda" as `Cuda(0)`, which silently puts the rank-N
        # cache on cuda:0 even though incoming KV-index tensors are on
        # cuda:N → `InsertValueWrongDevice` at first insert. Allocator /
        # model_runner pass `device='cuda'` (the global server arg) on
        # all ranks, so the resolution must happen here at the boundary.
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

    # ----- HiCache (host tier) -----

    def _write_backup(
        self, node_indices: list[int], device_values: list[torch.Tensor]
    ) -> None:
        """Kick off the backup against `cache_controller` and reflect the
        successful backups in the tree."""
        backed_nodes: list[int] = []
        host_values: list[torch.Tensor] = []
        for node_idx, device_value in zip(node_indices, device_values):
            host_indices = self.cache_controller.write(
                device_indices=device_value, node_id=node_idx
            )
            if host_indices is None:
                self.evict_host(len(device_value))
                host_indices = self.cache_controller.write(
                    device_indices=device_value, node_id=node_idx
                )
            if host_indices is None:
                # Stop if any node failed to back up — preserves host-value
                # contiguity (the backed-up set stays a gapless prefix).
                break
            backed_nodes.append(node_idx)
            host_values.append(host_indices)
        if not backed_nodes:
            return
        self._rust_radix.set_host_full_values_and_lock_device(backed_nodes, host_values)
        for node_idx in backed_nodes:
            self.ongoing_write_through[node_idx] = node_idx

    def evict_host(self, num_tokens: int) -> None:
        """Best effort to free up at least `num_tokens` host-tier KV in LRU order."""
        if self.cache_controller is None or num_tokens <= 0:
            return
        result = self._rust_radix.evict_host(num_tokens)
        self._process_evict_actions(result.deferred_actions)

    def _process_evict_actions(self, deferred_actions: list[tuple]) -> None:
        """Process evict actions generated by the Rust radix tree."""
        for action in deferred_actions:
            tag = action[0]
            if tag == "FullDeviceEvictOnBackedUp":
                # Free the device value on an already backed-up node.
                _, _node_idx, device_value = action
                if self.token_to_kv_pool_allocator is not None:
                    self.token_to_kv_pool_allocator.free(device_value)
            elif tag == "FullHostEvict":
                # Free host value.
                _, _node_idx, host_value = action
                self.token_to_kv_pool_host.free(host_value)
            else:
                raise RadixCacheRuntimePyError(
                    f"_process_evict_actions: unsupported evict action {tag!r}"
                )

    # ----- BasePrefixCache contract: lifecycle -----

    def reset(self) -> None:
        # Mirrors `RadixCache.reset` — clears tree state only. The allocator
        # is NOT cleared here; callers that need to release in-tree slots back
        # to the allocator must call `token_to_kv_pool_allocator.clear()`
        # separately. In practice, `reset()` runs at startup or warmup when
        # the allocator is already empty.
        self._rust_radix.reset()

    def supports_fast_match_prefix(self) -> bool:
        return True

    # ----- BasePrefixCache contract: lookup / insert / evict -----

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if params.cow_mamba and not self.supports_mamba():
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: MatchPrefixParams.cow_mamba=True requires "
                "Mamba configuration (HybridReqToTokenPool)"
            )

        if self.disable:
            # Disabled cache: skip Rust entirely. `None` signals "nothing to
            # lock" so inc/dec_lock_ref short-circuit. This is the only path
            # that produces a `None` last-node — every other path passes
            # through Rust's idx (which may be a namespace root for empty
            # keys / no-match cases, where lock-ref ops are documented no-ops).
            return self._empty_match_result()

        token_ids = params.key.token_ids
        rust_result = self._rust_radix.match_prefix(token_ids, params.key.extra_key)

        # Mamba CoW: copy the SSM state of matched node to callers so
        # they could directly manipulate it freely.
        if params.cow_mamba and rust_result.mamba_value is not None:
            self._copy_on_write_mamba(
                params.req, rust_result.last_device_node_idx, rust_result.mamba_value
            )

        best_match_node = (
            rust_result.last_host_node_idx
            if rust_result.host_only_length > 0
            else rust_result.last_device_node_idx
        )
        return MatchResult(
            device_indices=rust_result.device_indices,
            last_device_node=rust_result.last_device_node_idx,
            last_host_node=rust_result.last_host_node_idx,
            best_match_node=best_match_node,
            host_hit_length=rust_result.host_only_length,
            mamba_branching_seqlen=rust_result.mamba_branching_seqlen,
        )

    def _extract_mamba_value(
        self, req: "Req"
    ) -> tuple[Optional[torch.Tensor], Optional[int]]:
        """Build the `mamba_value` tensor to insert into the radix tree."""
        if not self.supports_mamba() or req.mamba_pool_idx is None:
            return None, None
        if not self.enable_mamba_extra_buffer:
            return req.mamba_pool_idx.unsqueeze(-1).clone(), None
        # extra_buffer mode: include the ping-pong index of the
        # returned value so the mamba pool can release the right slot later
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
        """Fork `mamba_value` in the pool; only try to evict if direct
        allocation failed.

        TODO(Jialin): port this retry-with-alloc wrapper into OSS
        `MambaPool` so callers don't reimplement it.
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

    def _empty_match_result(self) -> MatchResult:
        # Disabled-cache sentinel. Reuses the cached `_empty_indices` to
        # avoid per-call tensor allocation. `last_device_node=None` makes
        # inc/dec_lock_ref short-circuit so callers don't need to branch.
        return MatchResult(
            device_indices=self._empty_indices,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if params.mamba_value is not None and not self.supports_mamba():
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: InsertParams.mamba_value requires Mamba "
                "configuration (HybridReqToTokenPool)"
            )
        if params.priority != 0:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: InsertParams.priority != 0 not supported (LRU only)"
            )
        # `params.chunked` only affects Python's hit_count, which LRU never
        # reads — silently ignored.

        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64, device=self.device)

        # Normalize the key's `is_bigram` to match `self.is_eagle` at the
        # orchestrator boundary, mirroring OSS `RadixCache.insert` /
        # `SWARadixCache.insert`. Defensive against an external caller
        # passing `RadixKey(is_bigram=False)` to an `is_eagle=True`
        # orchestrator (or vice versa) — without this, downstream
        # `page_aligned(...)` + `len(aligned_key)` math would silently
        # disagree with the Rust wrapper's bigram-pair-count and corrupt
        # `prefix_len` accounting. Idempotent when the caller already
        # set `is_bigram=self.is_eagle` (the `cache_*_req` path).
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)

        # Orchestrator owns page-alignment in atom units. Delegate to
        # `RadixKey.page_aligned`, which is bigram-aware via the key's
        # (now-normalized) `is_bigram` flag.
        # `len(aligned_key)` is the atom count (= N-1 for is_bigram=True,
        # N otherwise); we slice `value` to that length so the cache's
        # value-length invariant holds at the atom granularity. The trim
        # is idempotent: callers that pre-align hit a no-op here.
        #
        # TODO(future PR): make the Rust wrapper / cache layer reject
        # non-aligned keys with a typed error instead of silently
        # trimming. Today `PageAlignedQueryKey::new` does an internal
        # `key.len() / ps * ps` trim — a contract-violation safety net
        # rather than an explicit invariant.
        aligned_key = key.page_aligned(self.page_size)
        atom_count = len(aligned_key)
        token_ids = aligned_key.token_ids
        # Trim value to atom_count. If the caller passed a shorter value
        # the slice returns the original (still-too-short) tensor; the
        # Rust cache layer catches it via `validate_insert_value` →
        # `RadixCacheRuntimePyError::InsertValueTooShort`, so we don't
        # duplicate the check here.
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
        """Apply the insert-path emitted actions in the orchestration layer."""
        if not deferred_actions or self.token_to_kv_pool_allocator is None:
            return

        swa_node_indices: list[int] = []
        swa_values: list[torch.Tensor] = []
        write_through_nodes: list[int] = []
        write_through_values: list[torch.Tensor] = []
        for action in deferred_actions:
            tag = action[0]
            if tag == "FullDupFreed":
                _, freed_indices = action
                self.token_to_kv_pool_allocator.free(freed_indices)
            elif tag == "SwaRecover":
                _, node_idx, freed_full, source_value = action
                self.token_to_kv_pool_allocator.free(freed_full)
                swa_node_indices.append(node_idx)
                swa_values.append(
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        source_value
                    )
                )
            elif tag == "SwaStamp":
                _, node_idx, source_value = action
                swa_node_indices.append(node_idx)
                swa_values.append(
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        source_value
                    )
                )
            elif tag == "FullWriteThroughBackup":
                _, node_idx, device_value = action
                write_through_nodes.append(node_idx)
                write_through_values.append(device_value)
            else:
                raise RadixCacheRuntimePyError(
                    f"_process_insert_actions: unsupported insert action {tag!r}"
                )

        if write_through_nodes:
            # Back up FULL values from device to host (write-through).
            self._write_backup(write_through_nodes, write_through_values)
        if swa_node_indices:
            # Single batched apply_swa_writes call — stamps SWA values on
            # all affected nodes, splices into SWA's LRU, credits
            # evictable_size. Mirrors OSS's per-action insert_mru pattern
            # collapsed into one call.
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

        # Single Rust call: the dispatcher iterates configured components
        # forward (FULL → SWA → Mamba) with the per-component budget.
        # FULL leaf-evict cross-bumps `result.freed[Swa]` via the
        # `free_swa(full_value)` cookie cascade — iterating both bins
        # below is what releases those cascaded handles back to the SWA
        # allocator. Iterating adapters here would re-trigger the
        # dispatcher per call and (a) double-count budget, (b) drop the
        # cascaded SWA handles from the FULL call's result.
        start_time = time.perf_counter()
        result = self._rust_radix.evict([full_budget, swa_budget, mamba_budget])

        # When a component isn't configured, the Rust dispatcher
        # doesn't iterate it, so its `evicted[ct] == 0` and
        # `freed[ct] == []` — the SWA/MAMBA branches below are
        # safe-by-shape and don't need a `sliding_window_size`
        # gate. (FULL leaf-evict CAN cross-bump `freed[Swa]` via the
        # `free_swa(full_value)` cookie cascade even when
        # `swa_budget == 0`, so we always iterate the SWA bin when
        # SWA is configured — empty otherwise.)
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
                self.req_to_token_pool.mamba_pool.free(freed)

        self._process_evict_actions(result.deferred_actions)

        self.update_eviction_metrics(sum(result.evicted), start_time)
        return EvictResult(
            num_tokens_evicted=result.evicted[full_idx],
            swa_num_tokens_evicted=result.evicted[swa_idx],
            mamba_num_evicted=result.evicted[mamba_idx],
        )

    # ----- BasePrefixCache contract: lock_ref -----

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        # `None` covers both: empty match (no node to lock) and disabled cache.
        if self.disable or node is None:
            return IncLockRefResult(delta=0)
        # Rust's `inc_lock_ref` is a per-cache dispatcher: forward iter
        # over all configured components (FULL always; SWA when
        # configured). Returns `(delta, swa_uuid_for_lock)` where delta
        # is the signed change to evictable_token_size aggregated across
        # components, and swa_uuid_for_lock is `Some(uuid)` when SWA
        # stamped a window boundary (for symmetric release later).
        delta, swa_uuid_for_lock = self._rust_radix.inc_lock_ref(node)
        return IncLockRefResult(delta=delta, swa_uuid_for_lock=swa_uuid_for_lock)

    def dec_lock_ref(
        self,
        node: Any,
        params: Optional[DecLockRefParams] = None,
    ) -> DecLockRefResult:
        if self.disable or node is None:
            return DecLockRefResult()
        # Rust's `dec_lock_ref` is the symmetric per-cache dispatcher
        # (reverse iter — SWA then FULL). `swa_uuid_for_lock` gates
        # SWA's release walk to stop at the matching boundary; FULL's
        # walk is unconditional. FULL-only configs pass `None` and the
        # param is ignored Rust-side.
        swa_uuid_for_lock = params.swa_uuid_for_lock if params is not None else None
        self._rust_radix.dec_lock_ref(node, swa_uuid_for_lock)
        return DecLockRefResult()

    # ----- BasePrefixCache contract: size accessors -----

    def evictable_size(self) -> int:
        return self._rust_radix.evictable_token_size()

    def protected_size(self) -> int:
        return self._rust_radix.protected_token_size()

    def total_size(self) -> int:
        # Total tokens (evictable + protected) across FULL and SWA components.
        return self._rust_radix.total_token_size()

    # Per-component aliases (mirror OSS `UnifiedRadixCache.full_*` /
    # `swa_*`). Scheduler reads these directly when `is_hybrid_swa`
    # (e.g. `schedule_policy.py rem_total_tokens`); without the
    # overrides the inherited `BasePrefixCache` defaults would silently
    # return 0 and starve the hybrid capacity calculation.

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
        # Total Mamba slots (evictable + protected); separate from `total_size()` because Mamba's unit is slots, not tokens.
        return self._rust_radix.mamba_total_size() if self.supports_mamba() else 0

    # ----- BasePrefixCache contract: idle invariant check -----

    def sanity_check(self) -> None:
        # Called by `scheduler_runtime_checker_mixin._check_tree_cache`
        # on idle ticks for hybrid-SWA / hybrid-SSM caches (gated by our
        # `supports_swa()` returning True). Currently a no-op stub
        # added during SWA bring-up to keep the scheduler's hot-path
        # call from raising AttributeError.
        #
        # TODO: implement a real check. Add a Rust-side
        # `sanity_check_aggregates()` walker that rebuilds the LRU
        # lists from the tree and asserts heap-ordered consistency
        # (mirrors `swa_radix_cache.py::sanity_check`). The Rust side
        # already debug_asserts at every mutation site, but a
        # Python-callable orchestrator-level check gives the scheduler
        # a cheap idle-tick invariant that catches drift between the
        # tree and the LRU bookkeeping (e.g. evictable_size /
        # protected_size aggregate skew).
        return None

    # ----- BasePrefixCache contract: SWA capability flag -----

    def supports_swa(self) -> bool:
        # Gates `Scheduler.maybe_evict_swa()` and the schedule-policy
        # paths that preserve `swa_uuid_for_lock` across decode steps.
        # Without `True` here, decode-time SWA evictions never fire and
        # `dec_lock_ref` calls land at the Rust dispatcher with
        # `swa_uuid_for_lock=None`, walking past the SWA boundary and
        # underflowing `swa_lock_ref`.
        return self.sliding_window_size is not None

    def supports_mamba(self) -> bool:
        return self.mamba_cache_chunk_size is not None

    # TODO(Jialin): expose Rust-side iteration; leak-diagnostic only.
    def all_values_flatten(self) -> torch.Tensor:
        return self._empty_indices

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._empty_indices

    # ----- BasePrefixCache contract: features rejected in v1 -----

    def pretty_print(self):
        raise RadixCacheInfraPyError(
            "RustUnifiedRadixCache: pretty_print() not supported"
        )

    def take_events(self):
        # `enable_kv_cache_events=True` is rejected at __init__, so the queue
        # is always empty.
        return []

    # ----- Per-request orchestration -----

    def cache_finished_req(self, req: "Req", is_insert: bool = True, **kwargs) -> None:
        """Mirrors `sglang.srt.mem_cache.radix_cache.RadixCache.cache_finished_req`.

        Cache the prefix of a finished request and free its tail. The disabled
        path frees everything; the inserting path inserts the page-aligned
        prefix and frees only the duplicate slots that the tree already owned.
        """
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

        # Mamba extra_buffer mode: truncate the cache range to Mamba chunk aligned.
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

        # Free everything past the aligned atom prefix. For bigram,
        # this includes the trailing boundary token of the last cached
        # pair as well as any unaligned bigram positions.
        self.token_to_kv_pool_allocator.free(kv_indices[atom_len:])

        # Mamba slot release.
        #
        #              | tree takes ownership of | req-owned slots to free
        # -------------|-------------------------|--------------------------------
        # extra_buffer | ping_pong[keep_idx]     | primary + ping_pong[other_idx]
        # extra_buffer | nothing (mamba_exist)   | primary + ping_pong[0,1]
        # no_buffer    | primary                 | nothing
        # no_buffer    | nothing (mamba_exist)   | primary
        #
        # extra_buffer: primary is ALWAYS orphaned (the tree took a ping-pong slot,
        # not the primary), so always invoke free_mamba_cache. The
        # `ping_pong_track_buffer_to_keep` arg tells the pool which ping-pong slot
        # to spare; set to None on mamba_exist so all three slots are freed.
        # no_buffer: primary IS the slot handed to the tree, so free only when the
        # tree rejected (mamba_exist=True).
        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None
        free_mamba_cache = self.enable_mamba_extra_buffer or mamba_exist
        if self.supports_mamba() and free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
            )

        # Release the lock taken when this req was scheduled for prefill.
        # Pass through `swa_uuid_for_lock` so SWA's release walk stops at
        # the right boundary node. FULL-only configs always have it as
        # None (DecLockRefParams default).
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
        )

    def cache_unfinished_req(self, req: "Req", chunked: bool = False, **kwargs) -> None:
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Mamba extra_buffer mode: truncate the cache range to Mamba chunk aligned.
        if self.enable_mamba_extra_buffer:
            cache_len = req.mamba_last_track_seqlen
            # No Mamba chunk-aligned boundary reached yet, skip caching.
            if cache_len is None:
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
                return
            token_ids = token_ids[:cache_len]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        atom_len = len(radix_key)
        values = kv_indices[:atom_len].to(dtype=torch.int64, copy=True)

        # Fork into a tree-owned slot so decode mutations don't alias the cached state.
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

        # Mamba: release the forked slot when the cache didn't consume
        # it (target already had a Mamba value).
        if mamba_value_forked is not None and insert_result.mamba_exist:
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

        # Re-match: the tree may have de-duplicated against an existing branch
        # during insert, so the canonical tree-owned indices for this prefix
        # may differ from `values` we just inserted. We need those canonical
        # indices in `req_to_token_pool` so subsequent reads see the survivor.
        #
        # SWA caveat: `match_prefix` can legitimately return FEWER indices
        # than the inserted atom count when the path crosses an SWA
        # tombstone and the contiguous post-tombstone run hasn't refilled
        # `sliding_window_size` yet. So the tight `len(new_indices) ==
        # atom_len` check would crash valid SWA states. Mirrors baseline
        # `swa_radix_cache.py`'s `assert old_prefix_len <= new_prefix_len
        # <= len(keys_np)` (and `unified_radix_cache.py`'s equivalent
        # partial-rematch tolerance). The bookkeeping below already uses
        # `len(new_indices)` everywhere, so a short rematch slots in
        # cleanly: `req.prefix_indices` keeps the unmatched tail,
        # `cache_protected_len` records what the tree actually owns.
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        # Ensure `cache_protected_len` does not extend beyond the
        # last aligned position the tree owns. The `+ page_size - 1`
        # slack tolerates a trailing partial page (page-aligned for
        # FULL/SWA; chunk-aligned for Mamba extra_buffer).
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

        # Lock-ref handoff. dec first, inc second so the brief moment between
        # the two doesn't hold a redundant +2 along ancestors that overlap
        # between old and new last_node (this method is synchronous wrt
        # eviction, so the brief drop is safe).
        # Pass `swa_uuid_for_lock` to dec so SWA's release walk stops at
        # the boundary node stamped at acquire time. After the new
        # acquire, store the new uuid back on req for the next call.
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock),
        )
        inc_result = self.inc_lock_ref(new_last_node)
        req.swa_uuid_for_lock = inc_result.swa_uuid_for_lock

        # Extend back kv indices after the last Mamba chunk or page-aligned boundary.
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node
        req.cache_protected_len = len(new_indices)
        # Clear the chunk-aligned marker so the next call to
        # `cache_unfinished_req` recomputes from the request's current
        # progress instead of reusing a stale boundary.
        if self.supports_mamba():
            req.mamba_last_track_seqlen = None

    # ----- HiCache: OSS-identical bodies -----
    # TODO(Jialin): introduce HiCacheMixin in OSS for consolidation.

    def init_hicache(self, server_args: Any, params: CacheInitParams) -> None:
        """Second-phase setup: build the host pool + `HiCacheController`.

        The factory calls this after construction when
        `enable_hierarchical_cache` is set. Mirrors OSS
        `HiRadixCache.__init__`'s host setup, restricted to the supported
        config: FULL-only, write-through, device<->host only (no
        storage/prefetch).
        """
        self.kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
        from sglang.srt.mem_cache.memory_pool_host import (
            MHATokenToKVPoolHost,
            MLATokenToKVPoolHost,
        )

        if isinstance(self.kv_cache, MHATokenToKVPool):
            host_cls: Any = MHATokenToKVPoolHost
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            host_cls = MLATokenToKVPoolHost
        else:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache HiCache: kv_cache "
                f"{type(self.kv_cache).__name__} is not supported yet (only MHA / MLA)"
            )
        self.token_to_kv_pool_host = host_cls(
            self.kv_cache,
            server_args.hicache_ratio,
            server_args.hicache_size,
            self.page_size,
            server_args.hicache_mem_layout,
            allocator_type=server_args.hicache_storage_backend,
        )

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.load_cache_event = threading.Event()

        from sglang.srt.managers.cache_controller import HiCacheController

        self.cache_controller = HiCacheController(
            self.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            attn_cp_group=self.attn_cp_group,
            attn_tp_group=self.attn_tp_group,
            pp_group=params.pp_cache_group,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=None,
        )

        # Nodes with an in-flight write-through; the host stamp is deferred to
        # the controller ack (`writing_check`). Keyed by Rust NodeIdx.
        self.ongoing_write_through: dict[int, Any] = {}
        # Nodes with an in-flight load-back; the device lock handed off by
        # `postprocess_load_back` is released on the ack (`loading_check`).
        self.ongoing_load_back: dict[int, int] = {}
        # L3 storage tier not supported yet
        self.enable_storage = False
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

    def writing_check(self) -> None:
        """Release the device lock on nodes whose host copy has completed."""
        if not self.ongoing_write_through:
            return
        finish_count = 0
        if self.pp_rank == 0:
            for ack in self.cache_controller.ack_write_queue:
                if not ack.finish_event.query():
                    break
                finish_count += 1
        finish_count = self._hicache_min_ready(finish_count)
        while finish_count > 0:
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for node_id in ack.node_ids:
                self.dec_lock_ref(node_id)
                self.ongoing_write_through.pop(node_id, None)
            finish_count -= 1

    def _hicache_min_ready(self, finish_count: int) -> int:
        # All ranks must drain the same number of acks to keep the queue in
        # lockstep; MIN-reduce the locally-ready count over the attn / TP groups.
        if (
            self.tp_world_size <= 1
            and self.attn_cp_group is None
            and self.attn_tp_group is None
        ):
            return finish_count
        tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(
                    tensor, op=torch.distributed.ReduceOp.MIN, group=group
                )
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        return int(tensor.item())

    def prepare_load_back(self, node_idx: int) -> Any:
        """Compute host-only chain to loadback (starting from node closer to root)."""
        return self._rust_radix.prepare_load_back(node_idx)

    def postprocess_load_back(
        self,
        chain: list[int],
        ancestor_node_idx: int,
        device_values: Optional[torch.Tensor] = None,
    ) -> None:
        """Commit a load-back: write `device_values` onto `chain` + hand off the
        device lock, or release prepare's locks when `device_values` is None."""
        self._rust_radix.postprocess_load_back(chain, ancestor_node_idx, device_values)

    def init_load_back(self, params: InitLoadBackParams) -> tuple[torch.Tensor, int]:
        """If needed, restore host-backed prefix to device, up to `best_match_node`."""
        node_idx = params.best_match_node
        mem_quota = params.mem_quota
        plan = self.prepare_load_back(node_idx)
        if not plan.chain:
            # Already device-present: nothing to restore and no locks taken, so
            # skip postprocess (its unconditional ancestor-unlock would underflow).
            return self._empty_indices, node_idx
        host_indices = plan.host_indices
        # Skip tiny loads / those over mem_quota.
        if len(host_indices) < self.load_back_threshold or (
            mem_quota is not None and len(host_indices) > mem_quota
        ):
            self.postprocess_load_back(plan.chain, plan.ancestor_node_idx, None)
            return self._empty_indices, plan.ancestor_node_idx
        # Loadback with retry
        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=node_idx
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=node_idx
            )
        if device_indices is None:
            self.postprocess_load_back(plan.chain, plan.ancestor_node_idx, None)
            return self._empty_indices, plan.ancestor_node_idx
        self.postprocess_load_back(plan.chain, plan.ancestor_node_idx, device_indices)
        # Save the ongoing loadback for lock ref reverts.
        self.ongoing_load_back[node_idx] = node_idx
        return device_indices, node_idx

    def loading_check(self) -> None:
        """Release the device lock handed off to each loaded prefix once its
        host->device copy has completed."""
        if not self.ongoing_load_back:
            return
        finish_count = 0
        if self.pp_rank == 0:
            for ack in self.cache_controller.ack_load_queue:
                if not ack.finish_event.query():
                    break
                finish_count += 1
        finish_count = self._hicache_min_ready(finish_count)
        while finish_count > 0:
            ack = self.cache_controller.ack_load_queue.pop(0)
            ack.finish_event.synchronize()
            for node_id in ack.node_ids:
                end_node = self.ongoing_load_back.pop(node_id, None)
                if end_node is not None:
                    self.dec_lock_ref(end_node)
            finish_count -= 1

    def ready_to_load_host_cache(self) -> int:
        """Kick off the queued host->device loads; return the consumer index the
        scheduler tracks (-1 when the load queue is empty)."""
        return self.cache_controller.start_loading()

    def check_hicache_events(self):
        """Revert locks for already finished loadback and backup."""
        if self.cache_controller is None:
            return
        self.writing_check()
        self.loading_check()


def install_rust_radix_cache() -> None:
    """Register Rust Unified Radix Cache.

    The native extension is loaded by the factory, not during registration, so
    importing the default registry path remains safe before the extension is
    built.
    """
    if get_radix_cache_factory("rust_unified") is not None:
        return

    def factory(ctx):
        _load_native_symbols()
        if ctx.enable_hierarchical_cache:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: hierarchical cache (HiCache) not supported"
            )
        if ctx.is_hybrid_ssm:
            raise RadixCacheInfraPyError(
                "RustUnifiedRadixCache: hybrid SSM / Mamba models not supported"
            )
        return RustUnifiedRadixCache(ctx.params)

    register_radix_cache_backend("rust_unified", factory)
