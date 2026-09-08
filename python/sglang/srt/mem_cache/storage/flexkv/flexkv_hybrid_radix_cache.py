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

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey

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
class _RestoreLease:
    generation: int
    rid: str
    req: Req
    device_indices: torch.Tensor


@dataclass
class _PendingStoreLaunch:
    store_key: str
    sglang_req_id: str
    node: Any
    dec_params: DecLockRefParams
    token_ids: list[int]
    kv_indices: torch.Tensor


@dataclass
class _PendingStoreCopy:
    launch: _PendingStoreLaunch
    cpu_indices: Optional[torch.Tensor]
    ready_event: Optional[torch.cuda.Event]


class FlexKVHybridRadixCache(BasePrefixCache):
    """Compose FlexKV I/O with an existing hybrid radix implementation.

    The inner cache remains the sole owner of radix/SWA bookkeeping. FlexKV
    only restores request-owned slots; the normal cache_finished_req path then
    inserts those slots with the same component semantics as fresh prefill.
    """

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
        self._restore_leases: dict[str, _RestoreLease] = {}
        # Aborted requests no longer block rid reuse, but their allocations
        # still need an owner until normal cleanup or a drained idle flush.
        self._aborted_restore_leases: dict[int, _RestoreLease] = {}
        self._restore_generation = 0
        self._inflight_store_nodes: dict[str, tuple[Any, DecLockRefParams]] = {}
        self._store_generation = 0
        self._profile_store_stages = os.getenv(
            "FLEXKV_PROFILE_STORE_STAGES", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._async_store_slot_mapping = bool(
            getattr(
                self.flexkv_connector,
                "supports_async_store_slot_mapping",
                False,
            )
        )
        logger.info(
            "[FlexKV] hybrid store slot-mapping mode: %s",
            "async" if self._async_store_slot_mapping else "sync",
        )
        self._pending_store_launches: dict[str, _PendingStoreLaunch] = {}
        self._pending_store_copies: dict[str, _PendingStoreCopy] = {}
        self._node_lock = threading.Lock()

    def reset(self) -> None:
        # Mapping copies may still be staged outside the connector. Wait for
        # their own events, including the current-stream fallback, before
        # discarding CPU buffers or allowing source slots to be reused.
        for pending in self.__dict__.get("_pending_store_copies", {}).values():
            if pending.ready_event is not None:
                pending.ready_event.synchronize()
        # FlexKV still owns references to GPU source/destination slots while an
        # asynchronous store or layerwise load is in flight. Drain those tasks
        # before the inner cache releases the slots.
        self.flexkv_connector.reset()
        self._free_uncommitted_restores()
        self._inner_cache.reset()
        self._load_markers.clear()
        with self._node_lock:
            self._inflight_store_nodes.clear()
            if hasattr(self, "_pending_store_launches"):
                self._pending_store_launches.clear()
            if hasattr(self, "_pending_store_copies"):
                self._pending_store_copies.clear()

    def shutdown(self) -> None:
        # Prefer token_to_kv_pool_host.destroy() (HiCache path); keep this alias.
        self.token_to_kv_pool_host.destroy()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if params.req is not None and self.has_uncommitted_restore(params.req):
            raise RuntimeError(
                f"FlexKV prefix rematch before restore commit: rid={params.req.rid}"
            )
        result = self._inner_cache.match_prefix(params)
        if self.disable or params.req is None:
            return result

        key = params.key.page_aligned(self.page_size)
        token_ids = key.raw_token_ids()
        device_length = int(result.device_indices.numel())
        if not token_ids or device_length >= len(token_ids):
            return result

        token_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        token_mask[device_length:] = True
        _, hit_length = self.flexkv_connector.lookup_kv(
            token_ids,
            token_mask,
            rid=params.req.rid,
            sglang_req_id=params.req.rid,
        )
        if hit_length <= 0:
            return result

        self._load_markers[params.req.rid] = _LoadMarker(
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
        if req.rid in self._restore_leases:
            raise RuntimeError(f"FlexKV load-back before restore commit: rid={req.rid}")

        marker = self._load_markers.pop(req.rid, None)
        if marker is None or params.host_hit_length <= 0:
            self.flexkv_connector.release_pending(req.rid)
            return self._empty_indices(), req.last_node

        device_indices = self._alloc_restore_slots(req, params.host_hit_length)
        if device_indices is None:
            self.flexkv_connector.release_pending(req.rid)
            return self._empty_indices(), req.last_node

        # Register ownership before launching the connector. If launch raises,
        # its write status may be unknown; reset must drain the connector before
        # these slots can be freed safely.
        generation = self._restore_generation
        self._restore_generation += 1
        lease = _RestoreLease(
            generation=generation,
            rid=req.rid,
            req=req,
            device_indices=device_indices,
        )
        self._restore_leases[req.rid] = lease
        req.pending_restore_generation = generation
        req.pending_restore_slots = device_indices

        if self.flexkv_connector.enable_layerwise:
            loaded, _ = self.flexkv_connector.start_load_kv_layerwise(
                req.rid, device_indices
            )
        else:
            loaded = self.flexkv_connector.retrieve_kv(req.rid, device_indices)

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
                    f"rid={req.rid}, retrieved={loaded}, "
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

    def has_uncommitted_restore(self, req: Req) -> bool:
        return req.rid in self._restore_leases

    @staticmethod
    def _restore_lease_matches_req(req: Req, lease: _RestoreLease) -> bool:
        return (
            lease.req is req
            and getattr(req, "pending_restore_generation", None) == lease.generation
            and getattr(req, "pending_restore_slots", None) is lease.device_indices
        )

    def _validate_restore_lease(self, req: Req) -> Optional[_RestoreLease]:
        lease = self._restore_leases.get(req.rid)
        if lease is None or lease.req is not req:
            # An older aborted Req may finish after a new Req reused its rid.
            # Find by object identity, never commit the successor's lease.
            lease = next(
                (
                    item
                    for item in self._aborted_restore_leases.values()
                    if item.req is req
                ),
                lease,
            )
        if lease is not None and not self._restore_lease_matches_req(req, lease):
            # Ordinary completion mutates/frees KV. Continuing on a mismatch
            # could free a different owner's slots and free them again at reset.
            raise RuntimeError(f"FlexKV restore lease mismatch: rid={req.rid}")
        return lease

    def _forget_restore_lease(self, lease: _RestoreLease) -> None:
        if self._restore_leases.get(lease.rid) is lease:
            self._restore_leases.pop(lease.rid)
        if self._aborted_restore_leases.get(lease.generation) is lease:
            self._aborted_restore_leases.pop(lease.generation)
        # Reset trusts the allocation ledger, not mutable request fields. Do
        # not overwrite fields belonging to another generation of the Req.
        if self._restore_lease_matches_req(lease.req, lease):
            lease.req.pending_restore_generation = None
            lease.req.pending_restore_slots = None
            lease.req._flexkv_uncached_restore = False

    def _commit_restore(self, req: Req) -> None:
        lease = self._validate_restore_lease(req)
        if lease is not None:
            self._forget_restore_lease(lease)
        else:
            req._flexkv_uncached_restore = False

    def _free_uncommitted_restores(self) -> None:
        # Only call after connector.reset has fenced all DMA. Request metadata
        # may be stale; each ledger entry still identifies the allocation to free.
        failed = []
        leases = list(self._restore_leases.values()) + list(
            self._aborted_restore_leases.values()
        )
        for lease in leases:
            try:
                self.token_to_kv_pool_allocator.free(lease.device_indices)
            except Exception:
                logger.exception(
                    "FlexKV failed to free restore slots rid=%s", lease.rid
                )
                failed.append(lease.rid)
                continue
            self._forget_restore_lease(lease)
        if failed:
            # Attempt every allocation, retain failures for diagnosis/retry, and
            # do not report a successful reset or discard the remaining ledger.
            raise RuntimeError(f"FlexKV failed to free restore allocations: {failed}")

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
        kv_length = int(kwargs.get("kv_len_to_handle", req.kv.kv_committed_len))
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_length]
        self._inner_cache.cache_finished_req(req, is_insert=is_insert, **kwargs)
        self._commit_restore(req)
        if not is_insert:
            return

        self._store_prefix(req, token_ids)

    def cache_unfinished_req(self, req: Req, **kwargs) -> None:
        self._validate_restore_lease(req)
        self._apply_restore_swa_boundary(req)
        self._inner_cache.cache_unfinished_req(req, **kwargs)
        self._commit_restore(req)

        # A chunk boundary is not a reusable request boundary and its state may
        # still be changing. The non-chunked call marks prefill completion, when
        # DSv4's SWA/compress state exactly describes the prompt prefix.
        if kwargs.get("chunked", False):
            return
        self._store_prefix(req, list(req.get_fill_ids()))

    def _store_prefix(self, req: Req, token_ids: Sequence[int]) -> None:
        """Store a page-aligned prefix and its exact SWA/state snapshot."""

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
        if node is self._inner_cache.root_node or indices.numel() == 0:
            return
        if indices.numel() < len(token_ids):
            token_ids = token_ids[: indices.numel()]
        if not token_ids or len(token_ids) != indices.numel():
            return

        lock_result = self._inner_cache.inc_lock_ref(node)
        with self._node_lock:
            store_key = f"{req.rid}:flexkv-store:{self._store_generation}"
            self._store_generation += 1
        pending = _PendingStoreLaunch(
            store_key=store_key,
            sglang_req_id=req.rid,
            node=node,
            dec_params=lock_result.to_dec_params(),
            token_ids=token_ids,
            kv_indices=indices,
        )
        if self.__dict__.get("_async_store_slot_mapping", False):
            with self._node_lock:
                self._pending_store_launches[store_key] = pending
            return
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

    def _launch_store(
        self,
        pending: _PendingStoreLaunch,
        *,
        kv_indices: Optional[torch.Tensor] = None,
        mapping_already_on_cpu: bool = False,
        skip_mapping_validation: bool = False,
    ) -> int:
        indices = pending.kv_indices if kv_indices is None else kv_indices
        store_stream = getattr(self.flexkv_connector, "store_stream", None)
        if store_stream is None:
            store_stream = torch.cuda.current_stream()
        if not mapping_already_on_cpu and not skip_mapping_validation:
            producer_stream = torch.cuda.current_stream()
            with self._store_profile_scope("flexkv.store.wait_producer_stream"):
                store_stream.wait_stream(producer_stream)
        with torch.cuda.stream(store_stream):
            if self.page_size > 1 and not skip_mapping_validation:
                with self._store_profile_scope("flexkv.store.slot_mapping_to_cpu"):
                    page_reps = indices[:: self.page_size]
                    if not mapping_already_on_cpu:
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

    def _stage_store_copy(self, pending: _PendingStoreLaunch) -> None:
        cpu_indices: Optional[torch.Tensor] = None
        ready_event: Optional[torch.cuda.Event] = None
        if bool(getattr(self.flexkv_connector, "is_store_sync_leader", True)):
            store_stream = getattr(self.flexkv_connector, "store_stream", None)
            if store_stream is None:
                store_stream = torch.cuda.current_stream()
            store_stream.wait_stream(torch.cuda.current_stream())
            cpu_indices = torch.empty(
                pending.kv_indices.shape,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            ready_event = torch.cuda.Event()
            with torch.cuda.stream(store_stream):
                cpu_indices.copy_(pending.kv_indices, non_blocking=True)
                ready_event.record(store_stream)
        self._pending_store_copies[pending.store_key] = _PendingStoreCopy(
            launch=pending,
            cpu_indices=cpu_indices,
            ready_event=ready_event,
        )

    def _launch_ready_store_copies(self) -> None:
        local_ready: list[str] = []
        if bool(getattr(self.flexkv_connector, "is_store_sync_leader", True)):
            for store_key, pending in self._pending_store_copies.items():
                if pending.ready_event is None or not pending.ready_event.query():
                    break
                local_ready.append(store_key)
        ready_keys = self.flexkv_connector.sync_ready_store_rids(local_ready)
        for store_key in ready_keys:
            pending_copy = self._pending_store_copies.pop(store_key, None)
            if pending_copy is None:
                raise RuntimeError(
                    f"FlexKV async store-ready key is not locally pending: {store_key}"
                )
            pending = pending_copy.launch
            indices = (
                pending_copy.cpu_indices
                if pending_copy.cpu_indices is not None
                else pending.kv_indices
            )
            try:
                task_id = self._launch_store(
                    pending,
                    kv_indices=indices,
                    mapping_already_on_cpu=pending_copy.cpu_indices is not None,
                    skip_mapping_validation=pending_copy.cpu_indices is None,
                )
            except Exception:
                self._inner_cache.dec_lock_ref(pending.node, pending.dec_params)
                raise
            if task_id < 0:
                self._inner_cache.dec_lock_ref(pending.node, pending.dec_params)
                continue
            with self._node_lock:
                self._inflight_store_nodes[store_key] = (
                    pending.node,
                    pending.dec_params,
                )

    def _launch_pending_stores(self) -> None:
        if not hasattr(self, "_pending_store_launches"):
            return
        while True:
            with self._node_lock:
                if not self._pending_store_launches:
                    break
                store_key = next(iter(self._pending_store_launches))
                pending = self._pending_store_launches.pop(store_key)
            try:
                self._stage_store_copy(pending)
            except Exception:
                self._inner_cache.dec_lock_ref(pending.node, pending.dec_params)
                raise
        self._launch_ready_store_copies()

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
        self._launch_pending_stores()

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

    def release_aborted_request(self, rid: str) -> None:
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

    def prefetch_request(self, req: Req) -> None:
        """Start queued prefetch without a foreground lookup or H2D allocation."""
        # Foreground lookup runs after stop-and-drain, otherwise it could fetch
        # the whole remote prefix before the prefetch policy gets a chance to stop.
        req.init_next_round_input(tree_cache=None, cow_mamba=False)
        fill_ids = req.full_untruncated_fill_ids
        if not fill_ids:
            return
        match_end = req._compute_max_prefix_len(len(fill_ids))
        self.prefetch_from_storage(
            req.rid,
            None,
            fill_ids[:match_end],
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
        )

    def prefetch_from_storage(
        self,
        rid: str,
        last_host_node=None,
        token_ids=None,
        last_hash=None,
        prefix_keys=None,
        *,
        matched_prefix_tokens=None,
        extra_key=None,
        cache_salt=None,
    ) -> None:
        """Pass the complete token hash chain and the candidate's absolute offset."""
        del last_host_node, last_hash, prefix_keys
        # The foreground adapter does not yet propagate namespace/salt.
        # Skip this optional path until both lookup and prefetch use the same key.
        if extra_key is not None or cache_salt is not None or not token_ids:
            return
        prefix = [] if matched_prefix_tokens is None else list(matched_prefix_tokens)
        ids = prefix + list(token_ids)
        ids = ids[: len(ids) // self.page_size * self.page_size]
        if len(ids) <= len(prefix):
            return
        if getattr(self.flexkv_connector, "_chunked_prefetch", False):
            self.flexkv_connector.prefetch_async(
                rid, ids, sglang_req_id=rid, candidate_start_token=len(prefix)
            )
        else:
            self.flexkv_connector.prefetch_async(rid, ids, sglang_req_id=rid)

    def check_prefetch_progress(self, rid: str) -> bool:
        return self.flexkv_connector.check_prefetch_progress(rid)

    def terminate_prefetch(self, rid: str) -> None:
        self.flexkv_connector.cancel_prefetch(rid)

    def pop_prefetch_loaded_span(self, rid: str) -> tuple[int, Optional[int]]:
        if getattr(self.flexkv_connector, "_chunked_prefetch", False):
            return self.flexkv_connector.pop_prefetch_loaded_span(rid)
        return self.pop_prefetch_loaded_tokens(rid), None

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        pop = getattr(self.flexkv_connector, "pop_prefetch_loaded_tokens", None)
        if callable(pop):
            return int(pop(rid))
        del rid
        return 0

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        return self._inner_cache.inc_lock_ref(node)

    def dec_lock_ref(self, node: Any, params: Optional[DecLockRefParams] = None) -> Any:
        return self._inner_cache.dec_lock_ref(node, params)

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
