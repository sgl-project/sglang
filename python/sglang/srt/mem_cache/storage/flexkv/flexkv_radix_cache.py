"""FlexKV-backed RadixCache for sglang.

This module exposes :class:`FlexKVRadixCache`, a subclass of
:class:`sglang.srt.mem_cache.radix_cache.RadixCache` that delegates
host-side prefix storage to a FlexKV ``KVManager``. The design mirrors
``LMCRadixCache`` (the LMCache integration) so the scheduler-side
contract is identical:

* MP (synchronous) mode — the default.
  ``match_prefix`` fires only a FlexKV LOOKUP and returns ``host_hit_length``;
  the scheduler then calls :meth:`init_load_back` at dispatch time which
  allocates slots and fires the FlexKV RETRIEVE.

* IP (layerwise) mode — enabled with ``FLEXKV_ENABLE_LAYERWISE_TRANSFER=1``.
  ``match_prefix`` allocates uncached slots and kicks off a layerwise
  load; the per-layer hook registered via
  ``register_layer_transfer_counter`` then waits on each layer's
  eventfd inside the model's forward pass.

Selection: ``--enable-flexkv`` on the sglang CLI routes the default
RadixCache factory here. See ``__init__.py`` in this package for the
``register_radix_cache_backend("flexkv", ...)`` entry-point that backs
the explicit ``--radix-cache-backend=flexkv`` form.
"""

from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheFinishedReqResult,
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.storage.flexkv.flexkv_connector import (
    FlexKVConnector,
    _FlexKVRetrieveResult,
    is_flexkv_layerwise_transfer_enabled,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class FlexKVMode(enum.Enum):
    MP = enum.auto()  # synchronous lookup → retrieve in two phases
    IP = enum.auto()  # in-process layerwise transfer


@dataclass
class _LoadBackMarker:
    """State carried from a hit-producing ``match_prefix`` to its
    matching ``init_load_back``. The detached ``RadixKey`` is a snapshot
    of the matched key at lookup time (the live request key aliases
    ``req.fill_ids`` which keeps growing)."""

    key: RadixKey
    value_numel: int  # device tokens already present at lookup time
    lookup_task_id: int


@dataclass
class _PendingFlexKVMPLease:
    lease_id: int
    rid: str
    lookup_task_id: int
    slots: torch.Tensor


class FlexKVRadixCache(RadixCache):
    """RadixCache extended with FlexKV host-tier IO."""

    def __init__(
        self,
        params: CacheInitParams,
        model_config: Optional[ModelConfig],
        server_args: ServerArgs,
        tp_rank: int,
        tp_size: int,
        dp_rank: Optional[int],
        pp_rank: int,
        attn_cp_rank: int,
        tp_group=None,
        pp_group=None,
        attn_tp_group=None,
        attn_cp_group=None,
    ) -> None:
        allocator_page_size = int(params.token_to_kv_pool_allocator.page_size)
        if allocator_page_size > 1 and is_flexkv_layerwise_transfer_enabled():
            raise ValueError(
                "FlexKV layerwise transfer currently requires allocator page "
                "size 1; disable layerwise transfer or use MP"
            )

        super().__init__(params)
        self._allocator_page_size = allocator_page_size

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        # ``tp_group`` and ``attn_tp_group`` are sometimes passed
        # interchangeably by sglang's factory; prefer the explicit
        # ``attn_tp_group`` when given.
        attn_tp_group_eff = attn_tp_group if attn_tp_group is not None else tp_group

        self.flexkv_connector = FlexKVConnector(
            sgl_model_config=model_config,
            server_args=server_args,
            page_size=params.page_size,
            allocator_page_size=self._allocator_page_size,
            kvcache=kvcache,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            pp_group=pp_group,
            attn_tp_group=attn_tp_group_eff,
            attn_cp_group=attn_cp_group,
        )

        self._mode = (
            FlexKVMode.IP if self.flexkv_connector.enable_layerwise else FlexKVMode.MP
        )
        if self._mode is FlexKVMode.IP:
            # Register the eventfd counter onto sglang's KV pool so each
            # forward layer blocks on its own eventfd.
            self.flexkv_connector.register_layer_transfer_counter(kvcache)

        # CUDA streams (mirroring LMCRadixCache).
        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        # Two-phase MP load: stash marker between ``match_prefix`` and
        # ``init_load_back``.
        self._load_markers: dict[str, _LoadBackMarker] = {}
        self._pending_mp_leases: dict[int, _PendingFlexKVMPLease] = {}
        self._next_mp_lease_id = 0
        # ``store_kv`` is async — we keep a lock on the source node
        # until FlexKV signals completion, draining in ``evict`` /
        # ``check_hicache_events``.
        self._inflight_store_nodes: dict[str, TreeNode] = {}
        self._node_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:  # type: ignore[override]
        if hasattr(self, "_pending_mp_leases") and self._pending_mp_leases:
            raise RuntimeError(
                "Cannot reset FlexKV while MP slot leases lack terminal proof"
            )
        if hasattr(self, "flexkv_connector"):
            self.flexkv_connector.reset()
        super().reset()
        if hasattr(self, "_load_markers"):
            self._load_markers.clear()
        if hasattr(self, "_inflight_store_nodes"):
            with self._node_lock:
                self._inflight_store_nodes.clear()

    def shutdown(self) -> None:
        if hasattr(self, "_pending_mp_leases") and self._pending_mp_leases:
            raise RuntimeError(
                "Cannot shut down FlexKV while MP slot leases lack terminal proof"
            )
        if hasattr(self, "flexkv_connector"):
            self.flexkv_connector.shutdown()

    # ------------------------------------------------------------------
    # match_prefix
    # ------------------------------------------------------------------

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # type: ignore[override]
        """Look up the longest cached prefix on host KV (FlexKV).

        Dispatches to :meth:`_mp_match_prefix` or :meth:`_ip_match_prefix`
        depending on whether layerwise transfer is enabled.
        """
        key = params.key
        if self.disable or not key:
            return super().match_prefix(params)

        # FlexKV operates at page granularity — round the lookup query
        # down to a multiple of ``page_size`` so the hit count we report
        # back to sglang matches what FlexKV can actually serve.
        if self._allocator_page_size != 1:
            aligned_len = (
                len(key) // self._allocator_page_size * self._allocator_page_size
            )
            key = key[:aligned_len]

        base_res = super().match_prefix(params)
        if len(key) == 0:
            return base_res

        device_value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        if self._mode is FlexKVMode.MP:
            if params.req is None:
                return base_res
            return self._mp_match_prefix(
                key, base_res, device_value, last_node, params.req
            )
        return self._ip_match_prefix(key, base_res, device_value, last_node)

    def _mp_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        device_value: torch.Tensor,
        last_node: TreeNode,
        req: Req,
    ) -> MatchResult:
        """LOOKUP-only path. Sets ``host_hit_length`` on the result so
        the scheduler later invokes :meth:`init_load_back`."""
        token_ids = key.raw_token_ids()
        device_len = int(device_value.numel())
        if device_len >= len(token_ids):
            return base_res

        # token_mask=True for tokens NOT on device — FlexKV decides
        # which of those it can serve.
        token_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        token_mask[device_len:] = True

        lookup_task_id, hit = self.flexkv_connector.lookup_kv(
            token_ids=token_ids, token_mask=token_mask, rid=req.rid
        )
        if hit <= 0:
            return base_res
        if device_len % self._allocator_page_size != 0:
            self.flexkv_connector.release_pending(req.rid)
            raise RuntimeError(
                "FlexKV MP load-back requires an allocator-page-aligned "
                "device prefix"
            )

        # Snapshot the matched key (the live key aliases ``req.fill_ids``).
        if token_ids is key.token_ids:
            token_ids_snap = token_ids[:]
        else:
            token_ids_snap = token_ids
        self._load_markers[req.rid] = _LoadBackMarker(
            key=RadixKey(token_ids_snap, key.extra_key, key.is_bigram),
            value_numel=device_len,
            lookup_task_id=lookup_task_id,
        )
        return MatchResult(
            device_indices=device_value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            host_hit_length=hit,
        )

    def _ip_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        device_value: torch.Tensor,
        last_node: TreeNode,
    ) -> MatchResult:
        """Layerwise path: allocate slots and fire ``start_load_kv_layerwise``
        immediately. Per-layer hook waits during forward."""
        token_ids = key.raw_token_ids()
        device_len = int(device_value.numel())
        if device_len >= len(token_ids):
            return base_res

        # Quick LOOKUP first to discover how many slots we'd need.
        token_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        token_mask[device_len:] = True
        # No rid here — IP mode self-pops; pass a synthetic stable key.
        synthetic_rid = f"_ip_{id(key)}"
        _, hit = self.flexkv_connector.lookup_kv(
            token_ids=token_ids, token_mask=token_mask, rid=synthetic_rid
        )
        if hit <= 0:
            return base_res

        result = self._allocate_and_load_layerwise(
            key=key,
            value_numel=device_len,
            uncached_len=hit,
            last_node=last_node,
            load_fn=lambda slot_mapping: self.flexkv_connector.start_load_kv_layerwise(
                synthetic_rid, slot_mapping
            )[0],
        )
        if result is None:
            return base_res
        new_slots, new_node = result
        return MatchResult(
            device_indices=torch.cat([device_value, new_slots]),
            last_device_node=new_node,
            last_host_node=new_node,
            best_match_node=new_node,
        )

    # ------------------------------------------------------------------
    # init_load_back (MP RETRIEVE)
    # ------------------------------------------------------------------

    def init_load_back(  # type: ignore[override]
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:
        """MP RETRIEVE. Allocates uncached slots and fires the FlexKV
        load; inserts the resulting TreeNode."""
        req = params.req
        last_node: TreeNode = params.best_match_node
        marker = self._load_markers.pop(req.rid, None)
        if marker is None:
            # ``match_prefix`` decided there was no work to do, but the
            # scheduler still called us. Release any held task and
            # return an empty load.
            self.flexkv_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )

        result = self._allocate_and_load_mp(
            rid=req.rid,
            lookup_task_id=marker.lookup_task_id,
            key=marker.key,
            value_numel=marker.value_numel,
            uncached_len=params.host_hit_length,
            last_node=last_node,
        )
        if result is None:
            # Allocation failed or load returned zero. ``retrieve_kv``
            # already cancels/cleans up on failure paths; release_pending
            # is idempotent for the case where allocation failed before
            # we even popped the held task.
            self.flexkv_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        return result

    def _allocate_and_load_mp(
        self,
        *,
        rid: str,
        lookup_task_id: int,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
        last_node: TreeNode,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        local_manifest_error: Optional[str] = None
        published_length = (
            uncached_len // self._allocator_page_size * self._allocator_page_size
        )
        if uncached_len <= 0:
            local_manifest_error = "FlexKV MP load-back requires a positive hit length"
        elif value_numel % self._allocator_page_size != 0:
            local_manifest_error = (
                "FlexKV MP load-back requires an allocator-page-aligned radix prefix"
            )
        elif value_numel + published_length > len(key):
            local_manifest_error = (
                "FlexKV MP retrieve exceeds the matched radix key length"
            )

        allocation_size = (
            (
                (uncached_len + self._allocator_page_size - 1)
                // self._allocator_page_size
                * self._allocator_page_size
            )
            if uncached_len > 0
            else 0
        )
        local_prelaunch_error: Optional[str] = None
        local_has_capacity = local_manifest_error is not None
        if local_manifest_error is None:
            try:
                local_has_capacity = (
                    self.token_to_kv_pool_allocator.available_size() >= allocation_size
                )
            except Exception as exc:  # noqa: BLE001
                local_prelaunch_error = f"allocator capacity query failed: {exc}"

        requires_eviction = self.flexkv_connector.requires_mp_eviction(
            local_has_capacity=local_has_capacity
        )
        if requires_eviction:
            try:
                self.evict(EvictParams(num_tokens=allocation_size))
            except Exception as exc:  # noqa: BLE001
                if local_prelaunch_error is None:
                    local_prelaunch_error = f"allocator eviction failed: {exc}"

        token_slots: Optional[torch.Tensor] = None
        if local_manifest_error is None and local_prelaunch_error is None:
            try:
                token_slots = self.token_to_kv_pool_allocator.alloc(allocation_size)
            except Exception as exc:  # noqa: BLE001
                local_prelaunch_error = f"allocator allocation failed: {exc}"
        lease: Optional[_PendingFlexKVMPLease] = None
        if token_slots is not None:
            lease_id = self._next_mp_lease_id
            self._next_mp_lease_id += 1
            lease = _PendingFlexKVMPLease(
                lease_id=lease_id,
                rid=rid,
                lookup_task_id=lookup_task_id,
                slots=token_slots,
            )
            self._pending_mp_leases[lease_id] = lease

        retrieve_result: _FlexKVRetrieveResult = self.flexkv_connector.retrieve_kv(
            rid=rid,
            slot_mapping=(
                token_slots[:uncached_len].to(torch.int64)
                if token_slots is not None
                else None
            ),
            expected_lookup_task_id=lookup_task_id,
            local_manifest_error=local_manifest_error,
            local_prelaunch_error=local_prelaunch_error,
        )
        if retrieve_result.prelaunch_miss:
            if lease is not None:
                self._release_mp_lease(lease.lease_id)
            if retrieve_result.prelaunch_contract_error:
                raise RuntimeError("FlexKV MP prelaunch manifest validation failed")
            return None
        if token_slots is None or lease is None:
            raise RuntimeError("FlexKV launched without a local allocator slot lease")
        if retrieve_result.terminal_proof and not retrieve_result.terminal_success:
            self._release_mp_lease(lease.lease_id)
            return None

        valid_result = (
            retrieve_result.lookup_task_id == lookup_task_id
            and retrieve_result.requested_slots == uncached_len
            and retrieve_result.terminal_proof
            and retrieve_result.terminal_success
            and len(retrieve_result.terminal_task_ids) > 0
        )
        if not valid_result:
            if retrieve_result.terminal_proof:
                self._release_mp_lease(lease.lease_id)
            raise RuntimeError(
                "FlexKV MP retrieve returned an inconsistent terminal manifest"
            )

        if published_length == 0:
            self._release_mp_lease(lease.lease_id)
            return None
        if value_numel + published_length > len(key):
            self._release_mp_lease(lease.lease_id)
            raise RuntimeError(
                "FlexKV MP retrieve exceeds the matched radix key length"
            )

        fetched_slots = token_slots[:published_length]
        new_node = TreeNode(priority=last_node.priority)
        new_node.key = key[value_numel : value_numel + published_length]
        new_node.value = fetched_slots
        new_node.parent = last_node

        lease.slots = fetched_slots
        if published_length < allocation_size:
            self.token_to_kv_pool_allocator.free(token_slots[published_length:])

        last_node.children[new_node.key.child_key(self.page_size)] = new_node
        self.evictable_size_ += published_length
        self._update_leaf_status(last_node)
        self._update_leaf_status(new_node)
        self._record_store_event(new_node.parent)
        self._record_store_event(new_node)
        self._pending_mp_leases.pop(lease.lease_id)

        return fetched_slots, new_node

    def _release_mp_lease(self, lease_id: int) -> None:
        lease = self._pending_mp_leases.get(lease_id)
        if lease is None:
            raise RuntimeError(f"Unknown FlexKV MP lease {lease_id}")
        self.token_to_kv_pool_allocator.free(lease.slots)
        self._pending_mp_leases.pop(lease_id)

    def _allocate_and_load_layerwise(
        self,
        *,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
        last_node: TreeNode,
        load_fn,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Legacy allocator + post-load bookkeeping for layerwise transfer.

        Returns ``(token_slots[:fetched], new_node)`` on success.
        ``None`` on either allocation failure or zero retrieved (in
        which case all slots are freed).
        """
        if uncached_len <= 0:
            return None

        # Evict to make room when needed.
        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))
        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return None

        # The FlexKV ``launch`` interface takes the slot indices for the
        # tokens it will write — no leading ``-1`` padding (FlexKV has
        # no concept of "skip these device slots, they're already
        # cached"; we pass it exactly the destinations for the
        # uncached tail).
        num_retrieved = load_fn(token_slots.to(torch.int64))

        if num_retrieved <= 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return None

        # Free the tail of the over-allocation when FlexKV returned
        # fewer than expected.
        if num_retrieved < uncached_len:
            self.token_to_kv_pool_allocator.free(token_slots[num_retrieved:])
            fetched_slots = token_slots[:num_retrieved]
        else:
            fetched_slots = token_slots

        new_node = TreeNode(priority=last_node.priority)
        start = value_numel
        end = start + num_retrieved
        new_node.key = key[start:end]
        new_node.value = fetched_slots
        new_node.parent = last_node
        last_node.children[new_node.key.child_key(self.page_size)] = new_node
        self.evictable_size_ += num_retrieved
        self._update_leaf_status(last_node)
        self._update_leaf_status(new_node)

        self._record_store_event(new_node.parent)
        self._record_store_event(new_node)

        return fetched_slots, new_node

    # ------------------------------------------------------------------
    # cache_finished_req (STORE)
    # ------------------------------------------------------------------

    def cache_finished_req(  # type: ignore[override]
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> CacheFinishedReqResult:
        """Base cache_finished_req then fire an async FlexKV store."""
        result = super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        if not is_insert:
            self._load_markers.pop(req.rid, None)
            return result

        # Compute the committed prefix mirroring LMCRadixCache's logic.
        from sglang.srt.runtime_context import get_server_args

        global_server_args = get_server_args()
        topk = global_server_args.speculative_eagle_topk
        enable_kv_committed_len = topk is None or topk == 1
        if enable_kv_committed_len:
            kv_committed_len = req.kv_committed_len
        else:
            kv_committed_len = len(req.origin_input_ids) + max(
                len(req.output_ids) - 1, 0
            )

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        if not token_ids:
            return result
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        # Anchor on the new last_device_node so FlexKV's lock matches
        # the node we'll later unlock when the store completes.
        match_result = super().match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids, req.extra_key))
        )
        new_last_node = match_result.last_device_node
        if new_last_node is None:
            return result

        self.inc_lock_ref(new_last_node)
        try:
            with torch.cuda.stream(self.store_stream):
                fkv_task_id = self.flexkv_connector.store_kv(
                    rid=req.rid,
                    token_ids=list(token_ids),
                    kv_indices=kv_indices,
                )
        except Exception:  # noqa: BLE001
            self.dec_lock_ref(new_last_node)
            raise

        if fkv_task_id < 0:
            # Nothing to write back (either everything already in
            # FlexKV, or put_match failed / returned None).
            self.dec_lock_ref(new_last_node)
            return result

        with self._node_lock:
            self._inflight_store_nodes[req.rid] = new_last_node

        return result

    # ------------------------------------------------------------------
    # evict + completion draining
    # ------------------------------------------------------------------

    def evict(self, params: EvictParams) -> EvictResult:  # type: ignore[override]
        """Drain completed stores before letting the base evict touch
        the source nodes."""
        if self.disable:
            return EvictResult()
        self._drain_completed_stores()
        # Make sure the store stream's GPU work is observed before any
        # eviction frees the source slots.
        self.store_stream.synchronize()
        return super().evict(params)

    def check_hicache_events(self) -> None:  # type: ignore[override]
        """Periodic non-blocking sweep called by the scheduler tick.

        Drains both store completions (so source nodes get unlocked
        quickly) and the launched-load tail (so the FlexKV pipe
        doesn't accumulate)."""
        self._drain_completed_stores()
        self.flexkv_connector.drain_launched_loads()

    def _drain_completed_stores(self) -> None:
        completed_rids = self.flexkv_connector.check_completed_stores()
        if not completed_rids:
            return
        with self._node_lock:
            for rid in completed_rids:
                node = self._inflight_store_nodes.pop(rid, None)
                if node is not None:
                    self.dec_lock_ref(node)

    # ------------------------------------------------------------------
    # Optional pass-throughs used by the scheduler
    # ------------------------------------------------------------------

    def release_aborted_request(self, rid: str) -> None:
        """Clean up tracking for an aborted request without invoking FlexKV."""
        if any(lease.rid == rid for lease in self._pending_mp_leases.values()):
            raise RuntimeError(
                "Cannot abort a FlexKV request while its MP slot lease lacks "
                "terminal proof"
            )
        self._load_markers.pop(rid, None)
        with self._node_lock:
            node = self._inflight_store_nodes.pop(rid, None)
        if node is not None:
            self.dec_lock_ref(node)
        self.flexkv_connector.release_pending(rid)
        self.flexkv_connector.cancel_prefetch(rid)

    def prefetch_from_storage(
        self, rid: str, last_host_node: TreeNode, token_ids
    ) -> None:
        """Kick off an opportunistic prefetch (SSD/Remote → CPU)."""
        try:
            self.flexkv_connector.prefetch_async(rid, list(token_ids))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[FlexKV] prefetch_from_storage: %s", exc)

    def check_prefetch_progress(self, rid: str) -> bool:
        return self.flexkv_connector.check_prefetch_progress(rid)

    def terminate_prefetch(self, rid: str) -> None:
        self.flexkv_connector.cancel_prefetch(rid)

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        # FlexKV doesn't expose per-rid prefetched token counts yet.
        return 0

    @property
    def hicache_storage_pass_prefix_keys(self) -> bool:
        # We pass token ids, not opaque key strings, so no prefix-key
        # accounting in the scheduler.
        return False
