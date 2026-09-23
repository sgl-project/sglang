"""FlexKV adapter for SGLang's component-based hybrid radix cache."""

from __future__ import annotations

import logging
import os
import threading
from array import array
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch
from flexkv.integration.sglang.connector import (
    FlexKVConnector,
    FlexKVHostReleaseShim,
)

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    CacheRequestHandle,
    DecLockRefParams,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.flexkv.flexkv_cache_lifecycle import (
    FlexKVCacheLifecycleMixin,
    _request_key,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class _LoadMarker:
    device_length: int


@dataclass
class _StoreRequest:
    store_key: str
    sglang_req_id: str
    token_ids: list[int]
    kv_indices: torch.Tensor


class FlexKVHybridRadixCache(FlexKVCacheLifecycleMixin, BasePrefixCache):
    """Compose FlexKV I/O with an existing hybrid radix implementation.

    The inner cache remains the sole owner of radix/SWA bookkeeping. FlexKV
    only restores request-owned slots; the normal cache_finished_req path then
    inserts those slots with the same component semantics as fresh prefill.
    """

    # FlexKV mounts one SWA slot per radix node, so a turn contributes a single
    # snapshot at its end. A later request whose prefix stops mid-turn finds no
    # SWA there, and a full-KV hit without one is dropped whole. Storing every N
    # pages of prefill puts snapshots on a grid instead; 0 keeps turn-end only.
    # Declared here so __getattr__ never forwards it to the inner cache.
    _swa_grid_tokens: int = 0

    def __init__(
        self,
        *,
        params: CacheInitParams,
        inner_cache: BasePrefixCache,
        model_config: Optional[ModelConfig],
        server_args: ServerArgs,
        tp_rank: int,
        dp_rank: Optional[int],
        pp_rank: int,
        attn_cp_rank: int,
        tp_group: Any = None,
        pp_group: Any = None,
        attn_tp_group: Any = None,
        attn_cp_group: Any = None,
    ) -> None:
        self._inner_cache = inner_cache
        self.req_to_token_pool = inner_cache.req_to_token_pool
        self.token_to_kv_pool_allocator = inner_cache.token_to_kv_pool_allocator
        self.page_size = inner_cache.page_size
        self.disable = inner_cache.disable
        self.device = inner_cache.device

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(
            self.token_to_kv_pool_allocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ):
            raise NotImplementedError(
                "FlexKV does not support the independent DSv4 HiSparse "
                "device-page mapping yet"
            )
        self.flexkv_connector = FlexKVConnector(
            sgl_model_config=model_config,
            server_args=server_args,
            page_size=self.page_size,
            kvcache=kvcache,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            pp_group=pp_group,
            attn_tp_group=attn_tp_group if attn_tp_group is not None else tp_group,
            attn_cp_group=attn_cp_group,
        )
        if self.flexkv_connector.enable_layerwise:
            self.flexkv_connector.register_layer_transfer_counter(kvcache)

        # Same hook HiCache uses: scheduler.release_host_resources → destroy().
        self.token_to_kv_pool_host = FlexKVHostReleaseShim(self.flexkv_connector)

        self._load_markers: dict[str, _LoadMarker] = {}
        self._init_restore_state()
        self._inflight_store_nodes: dict[str, tuple[Any, DecLockRefParams]] = {}
        self._store_generation = 0
        self._profile_store_stages = os.getenv(
            "FLEXKV_PROFILE_STORE_STAGES", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # SWA requires GPU-side FULL-to-SWA translation in the connector.
        # Deferred leader-only CPU mappings are supported only by the ordinary
        # adapter; keep this path synchronous until SWA has a sideband protocol.
        self._node_lock = threading.Lock()

        self._swa_grid_tokens = envs.SGLANG_FLEXKV_SWA_GRID_PAGES.get() * self.page_size

    def reset(self) -> None:
        # FlexKV still owns references to GPU source/destination slots while an
        # asynchronous store or layerwise load is in flight. Drain those tasks
        # before the inner cache releases the slots.
        self.flexkv_connector.reset()
        self._free_uncommitted_restores()
        self._inner_cache.reset()
        self._load_markers.clear()
        with self._node_lock:
            self._inflight_store_nodes.clear()

    def release_host_resources(self) -> None:
        self.token_to_kv_pool_host.destroy()

    def shutdown(self) -> None:
        # Prefer token_to_kv_pool_host.destroy() (HiCache path); keep this alias.
        self.token_to_kv_pool_host.destroy()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if params.req is not None and self.has_uncommitted_restore(params.req):
            raise RuntimeError(
                f"FlexKV prefix rematch before restore commit: rid={_request_key(params.req.cache_request_handle)}"
            )
        result = self._inner_cache.match_prefix(params)
        if self.disable or params.req is None:
            return result

        key = params.key.page_aligned(self.page_size)
        if key.extra_key is not None or key.cache_salt is not None:
            return result
        token_ids = key.raw_token_ids()
        device_length = int(result.device_indices.numel())
        if not token_ids or device_length >= len(token_ids):
            return result

        token_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        token_mask[device_length:] = True
        _, hit_length = self.flexkv_connector.lookup_kv(
            token_ids,
            token_mask,
            rid=_request_key(params.req.cache_request_handle),
            sglang_req_id=params.req.rid,
        )
        if hit_length <= 0:
            return result

        self._load_markers[_request_key(params.req.cache_request_handle)] = _LoadMarker(
            device_length=device_length,
        )
        return result._replace(
            last_host_node=result.last_device_node,
            best_match_node=result.last_device_node,
            host_hit_length=hit_length,
            cache_protected_len=device_length,
        )

    def init_load_back(self, params: InitLoadBackParams) -> tuple[torch.Tensor, Any]:
        req = params.req
        if _request_key(req.cache_request_handle) in self._restore_leases:
            raise RuntimeError(
                f"FlexKV load-back before restore commit: rid={_request_key(req.cache_request_handle)}"
            )

        marker = self._load_markers.pop(_request_key(req.cache_request_handle), None)
        if marker is None or params.host_hit_length <= 0:
            self.flexkv_connector.release_pending(
                _request_key(req.cache_request_handle)
            )
            return self._empty_indices(), req.last_node

        device_indices = self._alloc_restore_slots(req, params.host_hit_length)
        if device_indices is None:
            self.flexkv_connector.release_pending(
                _request_key(req.cache_request_handle)
            )
            return self._empty_indices(), req.last_node

        # Register ownership before launching the connector. If launch raises,
        # its write status may be unknown; reset must drain the connector before
        # these slots can be freed safely.
        lease = self._register_restore_lease(req, device_indices)

        if self.flexkv_connector.enable_layerwise:
            loaded, _ = self.flexkv_connector.start_load_kv_layerwise(
                _request_key(req.cache_request_handle), device_indices
            )
        else:
            loaded = self.flexkv_connector.retrieve_kv(
                _request_key(req.cache_request_handle), device_indices
            )

        # Admission planning uses the lookup hit length. Keep load-back
        # all-or-nothing so a successful restore cannot invalidate the checks
        # that deliberately ran before this allocation. Both connector paths
        # normally return either the entire mapping or zero.
        if loaded != device_indices.numel():
            if self.flexkv_connector.enable_layerwise and loaded > 0:
                # The count is not a DMA completion fence. A partial result
                # cannot be freed or used safely. This violates the connector
                # contract and is engine-fatal in the scheduler; the retained
                # lease is diagnostic ownership, not a promise of RPC recovery.
                raise RuntimeError(
                    "Unexpected layerwise restore length: "
                    f"rid={_request_key(req.cache_request_handle)}, retrieved={loaded}, "
                    f"requested={device_indices.numel()}; slots cannot be proven idle"
                )
            self.token_to_kv_pool_allocator.free(lease.device_indices)
            self._commit_restore(req)
            return self._empty_indices(), req.last_node

        if (
            self.supports_swa()
            and self.page_size > 1
            and hasattr(self.token_to_kv_pool_allocator, "alloc_extend_swa_tail")
            and device_indices.numel() > 0
        ):
            restored_end = int(req.prefix_indices.numel() + device_indices.numel())
            swa_tail_length = min(self.page_size, int(device_indices.numel()))
            req._flexkv_swa_evicted_seqlen = restored_end - swa_tail_length

        # The restored tail is request-owned until the normal cache completion
        # path inserts it. Preserve the pre-restore protection boundary so the
        # inner cache can deduplicate or free every restored slot correctly.
        req.kv.cache_protected_len = marker.device_length
        req._flexkv_uncached_restore = True
        return device_indices, req.last_node

    def _alloc_restore_slots(
        self, req: Req, host_hit_length: int
    ) -> Optional[torch.Tensor]:
        allocator = self.token_to_kv_pool_allocator
        if self.page_size == 1:
            slots = allocator.alloc(host_hit_length)
        else:
            prefix_length = int(req.prefix_indices.numel())
            sequence_length = prefix_length + host_hit_length
            prefix_lengths = torch.tensor(
                [prefix_length], dtype=torch.int64, device=self.device
            )
            prefix_lengths_cpu = torch.tensor([prefix_length], dtype=torch.int64)
            sequence_lengths = torch.tensor(
                [sequence_length], dtype=torch.int64, device=self.device
            )
            sequence_lengths_cpu = torch.tensor([sequence_length], dtype=torch.int64)
            last_location = (
                req.prefix_indices[-1:].to(device=self.device, dtype=torch.int64)
                if prefix_length > 0
                else torch.tensor([-1], dtype=torch.int64, device=self.device)
            )
            if hasattr(allocator, "alloc_extend_swa_tail") and self.supports_swa():
                # FlexKV stores one page of SWA/state sidecars for a full-prefix hit.
                swa_tail_length = min(self.page_size, host_hit_length)
                slots = allocator.alloc_extend_swa_tail(
                    prefix_lengths,
                    prefix_lengths_cpu,
                    sequence_lengths,
                    sequence_lengths_cpu,
                    last_location,
                    host_hit_length,
                    swa_tail_length,
                )
            else:
                slots = allocator.alloc_extend(
                    prefix_lengths,
                    prefix_lengths_cpu,
                    sequence_lengths,
                    sequence_lengths_cpu,
                    last_location,
                    host_hit_length,
                )

        if slots is not None:
            return slots

        from sglang.srt.mem_cache.common import evict_from_tree_cache

        swa_need = None
        if self.supports_swa():
            swa_need = (
                host_hit_length
                if self.page_size == 1
                else min(self.page_size, host_hit_length)
            )
        evict_from_tree_cache(self, host_hit_length, swa_num_tokens=swa_need)
        if self.page_size == 1:
            return allocator.alloc(host_hit_length)
        return self._alloc_restore_slots_once(req, host_hit_length)

    def _alloc_restore_slots_once(
        self, req: Req, host_hit_length: int
    ) -> Optional[torch.Tensor]:
        """Retry the paged allocation once after eviction."""
        allocator = self.token_to_kv_pool_allocator
        prefix_length = int(req.prefix_indices.numel())
        sequence_length = prefix_length + host_hit_length
        prefix_lengths = torch.tensor(
            [prefix_length], dtype=torch.int64, device=self.device
        )
        prefix_lengths_cpu = torch.tensor([prefix_length], dtype=torch.int64)
        sequence_lengths = torch.tensor(
            [sequence_length], dtype=torch.int64, device=self.device
        )
        sequence_lengths_cpu = torch.tensor([sequence_length], dtype=torch.int64)
        last_location = (
            req.prefix_indices[-1:].to(device=self.device, dtype=torch.int64)
            if prefix_length > 0
            else torch.tensor([-1], dtype=torch.int64, device=self.device)
        )
        if hasattr(allocator, "alloc_extend_swa_tail") and self.supports_swa():
            return allocator.alloc_extend_swa_tail(
                prefix_lengths,
                prefix_lengths_cpu,
                sequence_lengths,
                sequence_lengths_cpu,
                last_location,
                host_hit_length,
                min(self.page_size, host_hit_length),
            )
        return allocator.alloc_extend(
            prefix_lengths,
            prefix_lengths_cpu,
            sequence_lengths,
            sequence_lengths_cpu,
            last_location,
            host_hit_length,
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs) -> None:
        self._validate_restore_lease(req)
        self._apply_restore_swa_boundary(req)
        kv_length = int(kwargs.get("owned_kv_len", req.kv.kv_committed_len))
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_length]
        self._inner_cache.cache_finished_req(req, is_insert=is_insert, **kwargs)
        self._commit_restore(req)
        if not is_insert:
            return

        self._store_prefix(req, token_ids)

    def cache_unfinished_req(self, req: Req, **kwargs) -> None:
        self._validate_restore_lease(req)
        self._apply_restore_swa_boundary(req)
        chunked = kwargs.get("chunked", False)
        grid_tokens = self._swa_grid_tokens
        # Prefix length as of the previous chunk boundary; the inner insert
        # below advances it to this chunk's end.
        prev_protected_len = (
            req.kv.cache_protected_len if chunked and grid_tokens > 0 else 0
        )
        self._inner_cache.cache_unfinished_req(req, **kwargs)
        self._commit_restore(req)

        if chunked:
            if grid_tokens > 0:
                self._store_grid_point(req, prev_protected_len, grid_tokens)
            return
        self._store_prefix(req, list(req.get_fill_ids()))

    def _store_grid_point(
        self, req: Req, prev_protected_len: int, grid_tokens: int
    ) -> None:
        """Store at a chunk boundary when the prefix crosses a grid multiple.

        The store must follow the inner insert: only then is this chunk's SWA
        page under ``cache_protected_len``, which ``Req.kv.swa_dead_lo`` makes a
        floor for window eviction, so the page the D2H reads cannot be freed
        beneath it.
        """
        fill_ids = list(req.get_fill_ids())
        aligned_length = len(fill_ids) // self.page_size * self.page_size
        if aligned_length // grid_tokens == prev_protected_len // grid_tokens:
            return
        self._store_prefix(req, fill_ids)

    def _store_prefix(self, req: Req, token_ids: Sequence[int]) -> None:
        """Store a page-aligned prefix and its exact SWA/state snapshot."""
        # The connector has no namespace/salt parameter. Keep these requests
        # in the inner GPU cache rather than publishing unscoped host entries.
        if req.extra_key is not None or req.cache_salt is not None:
            return
        aligned_length = len(token_ids) // self.page_size * self.page_size
        if aligned_length <= 0:
            return
        token_ids = list(token_ids[:aligned_length])
        key = RadixKey(
            array("q", token_ids),
            req.extra_key,
            is_bigram=bool(getattr(self._inner_cache, "is_eagle", False)),
        )
        match = self._inner_cache.match_prefix(MatchPrefixParams(key=key))
        node = match.last_device_node
        indices = match.device_indices
        if self._inner_cache.is_root(node) or indices.numel() == 0:
            return
        if indices.numel() < len(token_ids):
            token_ids = token_ids[: indices.numel()]
        if not token_ids or len(token_ids) != indices.numel():
            return

        lock_result = self._inner_cache.inc_lock_ref(node)
        with self._node_lock:
            store_key = f"{_request_key(req.cache_request_handle)}:flexkv-store:{self._store_generation}"
            self._store_generation += 1
        pending = _StoreRequest(
            store_key=store_key,
            sglang_req_id=req.rid,
            token_ids=token_ids,
            kv_indices=indices,
        )
        try:
            task_id = self._launch_store(pending)
        except Exception:
            self._inner_cache.dec_lock_ref(node, lock_result.to_dec_params())
            raise
        if task_id < 0:
            self._inner_cache.dec_lock_ref(node, lock_result.to_dec_params())
            return
        with self._node_lock:
            self._inflight_store_nodes[store_key] = (
                node,
                lock_result.to_dec_params(),
            )

    def _launch_store(self, pending: _StoreRequest) -> int:
        indices = pending.kv_indices
        store_stream = getattr(self.flexkv_connector, "store_stream", None)
        if store_stream is None:
            store_stream = torch.cuda.current_stream()
        with self._store_profile_scope("flexkv.store.wait_producer_stream"):
            store_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(store_stream):
            if self.page_size > 1:
                with self._store_profile_scope("flexkv.store.slot_mapping_to_cpu"):
                    page_reps = indices[:: self.page_size]
                    page_reps = page_reps.to(device="cpu", dtype=torch.int64)
                with self._store_profile_scope("flexkv.store.slot_mapping_validate"):
                    page_ids = page_reps // self.page_size
                    unique_pages = torch.unique(page_ids)
                    aligned = bool((page_reps % self.page_size == 0).all())
                if not aligned or unique_pages.numel() != page_ids.numel():
                    raise RuntimeError(
                        "FlexKV D2H received an invalid GPU slot mapping: "
                        f"rid={pending.store_key}, pages={page_ids.numel()}, "
                        f"unique_pages={unique_pages.numel()}, aligned={aligned}"
                    )
            with self._store_profile_scope("flexkv.store.connector_store_kv"):
                return self.flexkv_connector.store_kv(
                    pending.store_key,
                    pending.token_ids,
                    indices,
                    sglang_req_id=pending.sglang_req_id,
                )

    def _store_profile_scope(self, name: str):
        if not self.__dict__.get("_profile_store_stages", False):
            return nullcontext()
        return torch.profiler.record_function(name)

    @staticmethod
    def _apply_restore_swa_boundary(req: Req) -> None:
        boundary = getattr(req, "_flexkv_swa_evicted_seqlen", None)
        if boundary is None or req.kv is None:
            return
        req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, boundary)
        del req._flexkv_swa_evicted_seqlen

    def evict(self, params: EvictParams) -> EvictResult:
        # Local memory pressure can make eviction asymmetric across TP/CP
        # ranks. Do not poll FlexKV here: completion uses a cross-rank scatter
        # and belongs to the synchronized scheduler hook below. The inner
        # cache cannot evict active store nodes because their lock refs remain.
        return self._inner_cache.evict(params)

    def check_hicache_events(self) -> None:
        self._drain_completed_stores()
        self.flexkv_connector.drain_launched_loads()

    def _drain_completed_stores(self) -> None:
        completed = self.flexkv_connector.check_completed_stores()
        if not completed:
            return
        with self._node_lock:
            for rid in completed:
                tracked = self._inflight_store_nodes.pop(rid, None)
                if tracked is not None:
                    node, dec_params = tracked
                    self._inner_cache.dec_lock_ref(node, dec_params)

    def release_aborted_request(self, handle: CacheRequestHandle) -> None:
        rid = _request_key(handle)
        self._load_markers.pop(rid, None)
        self.flexkv_connector.release_pending(rid)
        self.flexkv_connector.cancel_prefetch(rid)
        # Queue-limit/timeout aborts can finish without cache_finished_req.
        # Remove the scheduling guard, but keep a separate allocation ledger:
        # dropping the only slot record would leak such pre-admission restores.
        lease = self._restore_leases.pop(rid, None)
        if lease is not None:
            self._aborted_restore_leases[lease.generation] = lease
        # Preserve request cleanup flags/boundaries. Scheduled requests still
        # release through cache_finished_req; orphaned allocations await an idle
        # flush, whose connector reset fences H2D before freeing them.

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        return self._inner_cache.inc_lock_ref(node)

    def dec_lock_ref(self, node: Any, params: Optional[DecLockRefParams] = None) -> Any:
        return self._inner_cache.dec_lock_ref(node, params)

    def resolve_node_handle(self, node_handle):
        return self._inner_cache.resolve_node_handle(node_handle)

    def root_node_handle(self, extra_key: Optional[str] = None):
        return self._inner_cache.root_node_handle(extra_key)

    def is_root(self, node):
        return self._inner_cache.is_root(node)

    def dfs_weight_order(self, node_handles):
        return self._inner_cache.dfs_weight_order(node_handles)

    def rotation_base_of(self, node):
        return self._inner_cache.rotation_base_of(node)

    def evict_for_alloc(self, params: EvictParams):
        return self._inner_cache.evict_for_alloc(params)

    def swa_retain_floor(self, req):
        return self._inner_cache.swa_retain_floor(req)

    def swa_transient_size(self):
        return self._inner_cache.swa_transient_size()

    def supports_swa(self) -> bool:
        return self._inner_cache.supports_swa()

    def supports_mamba(self) -> bool:
        return self._inner_cache.supports_mamba()

    def supports_fast_match_prefix(self) -> bool:
        return self._inner_cache.supports_fast_match_prefix()

    # BasePrefixCache provides default implementations for these methods, so
    # __getattr__ cannot forward them. Delegate them explicitly; otherwise the
    # scheduler sees zero evictable/protected tokens and reports the inner
    # UnifiedRadixCache's live pages as a pool leak.
    def evictable_size(self) -> int:
        return self._inner_cache.evictable_size()

    def full_evictable_size(self) -> int:
        return self._inner_cache.full_evictable_size()

    def swa_evictable_size(self) -> int:
        return self._inner_cache.swa_evictable_size()

    def protected_size(self) -> int:
        return self._inner_cache.protected_size()

    def full_protected_size(self) -> int:
        return self._inner_cache.full_protected_size()

    def swa_protected_size(self) -> int:
        return self._inner_cache.swa_protected_size()

    def total_size(self) -> int:
        return self._inner_cache.total_size()

    def pretty_print(self) -> None:
        return self._inner_cache.pretty_print()

    def ready_to_load_host_cache(self) -> Any:
        return self._inner_cache.ready_to_load_host_cache()

    def take_events(self) -> list[Any]:
        return self._inner_cache.take_events()

    def swa_reprefill_tail_tokens(self) -> int:
        return self._inner_cache.swa_reprefill_tail_tokens()

    def supports_streaming_session(self) -> bool:
        return self._inner_cache.supports_streaming_session()

    def release_session(self, session_id: str) -> None:
        self._inner_cache.release_session(session_id)

    def release_radix_session(self, session_id: str) -> None:
        self._inner_cache.release_radix_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self._inner_cache.session_held_tokens(active_pool_idxs)

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self._inner_cache.session_held_full_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self._inner_cache.session_held_swa_tokens(active_pool_idxs)

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return self._inner_cache.session_held_req_count(active_pool_idxs)

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return self._inner_cache.session_held_mamba_slots(active_pool_idxs)

    def is_chunk_cache(self) -> bool:
        return self._inner_cache.is_chunk_cache()

    def is_tree_cache(self) -> bool:
        return self._inner_cache.is_tree_cache()

    def available_and_evictable_str(self) -> str:
        return self._inner_cache.available_and_evictable_str()

    def init_metrics_collector(self) -> None:
        self._inner_cache.init_metrics_collector()

    def _empty_indices(self) -> torch.Tensor:
        return torch.empty((0,), dtype=torch.int64, device=self.device)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        inner = self.__dict__.get("_inner_cache")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)
