"""FlexKV-backed RadixCache for sglang.

This module exposes :class:`FlexKVRadixCache`, a subclass of
:class:`sglang.srt.mem_cache.radix_cache.RadixCache` that delegates
host-side prefix storage to a FlexKV ``KVManager``. The design mirrors
``LMCRadixCache`` (the LMCache integration) so the scheduler-side
contract is identical:

* MP (synchronous) mode — the default.
  ``match_prefix`` fires only a FlexKV LOOKUP and returns ``host_hit_length``;
  the scheduler then calls :meth:`init_load_back` at dispatch time which
  allocates slots and fires the FlexKV RETRIEVE. With ``--enable-flexkv``,
  the scheduler also runs enqueue-time :meth:`prefetch_from_storage` and
  waits via :meth:`check_prefetch_progress` so Remote/Mooncake blocks are
  on CPU before lookup/retrieve (compute GET no longer issues REMOTE2H).

* IP (layerwise) mode — enabled with ``FLEXKV_ENABLE_LAYERWISE_TRANSFER=1``.
  ``match_prefix`` records the host hit and the scheduler starts the layerwise
  load after admission; the per-layer hook registered via
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
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from flexkv.integration.sglang.connector import (
    FlexKVConnector,
    FlexKVHostReleaseShim,
)

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.runtime_context import get_spec

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


@dataclass
class _PendingStoreLaunch:
    """Radix-owned source retained until the deferred D2H launch runs."""

    node: TreeNode
    token_ids: list[int]
    kv_indices: torch.Tensor


@dataclass
class _PendingStoreCopy:
    """Deferred store whose leader-side GPU-to-pinned-CPU copy is in flight."""

    node: TreeNode
    token_ids: list[int]
    kv_indices: torch.Tensor
    cpu_indices: Optional[torch.Tensor]
    ready_event: Optional[torch.cuda.Event]


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
        super().__init__(params)

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        # ``tp_group`` and ``attn_tp_group`` are sometimes passed
        # interchangeably by sglang's factory; prefer the explicit
        # ``attn_tp_group`` when given.
        attn_tp_group_eff = attn_tp_group if attn_tp_group is not None else tp_group

        self.flexkv_connector = FlexKVConnector(
            sgl_model_config=model_config,
            server_args=server_args,
            page_size=params.page_size,
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

        # Same hook HiCache uses: scheduler.release_host_resources → destroy().
        self.token_to_kv_pool_host = FlexKVHostReleaseShim(self.flexkv_connector)

        # CUDA streams (mirroring LMCRadixCache).
        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        # Two-phase MP load: stash marker between ``match_prefix`` and
        # ``init_load_back``.
        self._load_markers: dict[str, _LoadBackMarker] = {}
        # ``store_kv`` is async — we keep a lock on the source node until
        # FlexKV signals completion at the scheduler's synchronized
        # ``check_hicache_events`` point.
        self._inflight_store_nodes: dict[str, TreeNode] = {}
        # Overlap scheduling can enqueue the next forward before the previous
        # request reaches cache_finished_req().  The synchronous GPU->CPU slot
        # mapping below then waits behind that whole forward and delays the
        # already-computed response.  Keep this experimental and opt-in while
        # we validate the scheduler-lifecycle boundary in production traces.
        self._defer_store_launch = os.getenv(
            "FLEXKV_DEFER_STORE_LAUNCH", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._profile_store_stages = os.getenv(
            "FLEXKV_PROFILE_STORE_STAGES", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        async_store_requested = os.getenv(
            "FLEXKV_ASYNC_STORE_SLOT_MAPPING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        async_store_supported = bool(
            getattr(
                self.flexkv_connector,
                "supports_async_store_slot_mapping",
                False,
            )
        )
        self._async_store_slot_mapping = (
            async_store_requested and self._defer_store_launch and async_store_supported
        )
        if async_store_requested and not self._async_store_slot_mapping:
            logger.warning(
                "[FlexKV] FLEXKV_ASYNC_STORE_SLOT_MAPPING requested but disabled: "
                "deferred=%s connector_supported=%s",
                self._defer_store_launch,
                async_store_supported,
            )
        self._pending_store_launches: dict[str, _PendingStoreLaunch] = {}
        self._pending_store_copies: dict[str, _PendingStoreCopy] = {}
        self._node_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        if hasattr(self, "_load_markers"):
            self._load_markers.clear()
        if hasattr(self, "_inflight_store_nodes"):
            with self._node_lock:
                self._inflight_store_nodes.clear()
                self._pending_store_launches.clear()
                self._pending_store_copies.clear()
        if hasattr(self, "flexkv_connector"):
            self.flexkv_connector.reset()

    def shutdown(self) -> None:
        if hasattr(self, "token_to_kv_pool_host"):
            self.token_to_kv_pool_host.destroy()
        elif hasattr(self, "flexkv_connector"):
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
        if self.page_size != 1:
            aligned_len = (len(key) // self.page_size) * self.page_size
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
        if params.req is None:
            return base_res
        return self._ip_match_prefix(key, base_res, device_value, last_node, params.req)

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

        fkv_task_id, hit = self.flexkv_connector.lookup_kv(
            token_ids=token_ids,
            token_mask=token_mask,
            rid=req.rid,
            sglang_req_id=req.rid,
        )
        if hit <= 0:
            return base_res

        # Snapshot the matched key (the live key aliases ``req.fill_ids``).
        if token_ids is key.token_ids:
            token_ids_snap = token_ids[:]
        else:
            token_ids_snap = token_ids
        self._load_markers[req.rid] = _LoadBackMarker(
            key=RadixKey(
                token_ids_snap,
                key.extra_key,
                key.is_bigram,
                cache_salt=key.cache_salt,
            ),
            value_numel=device_len,
        )
        return MatchResult(
            device_indices=device_value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            host_hit_length=hit,
            cache_protected_len=device_len,
        )

    def _ip_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        device_value: torch.Tensor,
        last_node: TreeNode,
        req: Req,
    ) -> MatchResult:
        """Layerwise LOOKUP phase.

        Prefix matching is also used for waiting-queue priority calculation,
        before admission has committed the request. Allocating and attaching a
        page here gives it two owners: the radix tree and request cleanup.
        ``init_load_back`` performs the allocation after admission instead.
        """
        return self._mp_match_prefix(key, base_res, device_value, last_node, req)

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

        request_owned_req = req if self._mode is FlexKVMode.IP else None
        load_fn = (
            (
                lambda slot_mapping: self.flexkv_connector.start_load_kv_layerwise(
                    req.rid, slot_mapping
                )[0]
            )
            if self._mode is FlexKVMode.IP
            else (
                lambda slot_mapping: self.flexkv_connector.retrieve_kv(
                    req.rid, slot_mapping
                )
            )
        )
        result = self._allocate_and_load(
            key=marker.key,
            value_numel=marker.value_numel,
            uncached_len=params.host_hit_length,
            last_node=last_node,
            tracking_rid=req.rid,
            sglang_req_id=req.rid,
            load_fn=load_fn,
            request_owned_req=request_owned_req,
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

    def _allocate_and_load(
        self,
        *,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
        last_node: TreeNode,
        tracking_rid: str,
        sglang_req_id: Optional[str],
        load_fn,
        request_owned_req: Optional[Req] = None,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Shared allocator + post-load bookkeeping for MP/IP.

        Prefix matches are computed before admission, so another request can
        populate part or all of the same prefix before this method runs. Refresh
        the device match before launching H2D to avoid overwriting a live child
        while leaving the old node in ``evictable_leaves``.
        """
        if uncached_len <= 0:
            return None

        original_value_numel = value_numel
        target_end = min(value_numel + uncached_len, len(key))
        uncached_len = target_end - value_numel
        target_key = key[:target_end]
        refreshed = super().match_prefix(MatchPrefixParams(key=target_key))
        refreshed_indices = refreshed.device_indices
        refreshed_len = int(refreshed_indices.numel())

        if refreshed_len < value_numel:
            self.flexkv_connector.release_pending(tracking_rid)
            raise RuntimeError(
                "FlexKV load-back anchor disappeared while it was protected: "
                f"expected_at_least={value_numel}, matched={refreshed_len}, "
                f"rid={tracking_rid}"
            )

        reused_indices = refreshed_indices[original_value_numel:refreshed_len]
        if refreshed_len >= target_end:
            self.flexkv_connector.release_pending(tracking_rid)
            logger.debug(
                "FlexKV load-back reused an already restored prefix: rid=%s "
                "original_device=%d refreshed_device=%d target=%d",
                tracking_rid,
                original_value_numel,
                refreshed_len,
                target_end,
            )
            return reused_indices, refreshed.last_device_node

        if refreshed_len > value_numel:
            self.flexkv_connector.release_pending(tracking_rid)
            token_ids = target_key.raw_token_ids()
            token_mask = torch.zeros(len(token_ids), dtype=torch.bool)
            token_mask[refreshed_len:] = True
            _, uncached_len = self.flexkv_connector.lookup_kv(
                token_ids=token_ids,
                token_mask=token_mask,
                rid=tracking_rid,
                sglang_req_id=sglang_req_id,
            )
            uncached_len = min(uncached_len, target_end - refreshed_len)
            value_numel = refreshed_len
            last_node = refreshed.last_device_node
            if uncached_len <= 0:
                return reused_indices, last_node
        else:
            last_node = refreshed.last_device_node

        # Evict to make room when needed.
        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))
        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return (reused_indices, last_node) if reused_indices.numel() > 0 else None

        # The FlexKV ``launch`` interface takes the slot indices for the
        # tokens it will write — no leading ``-1`` padding (FlexKV has
        # no concept of "skip these device slots, they're already
        # cached"; we pass it exactly the destinations for the
        # uncached tail).
        num_retrieved = load_fn(token_slots.to(torch.int64))

        if num_retrieved <= 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return (reused_indices, last_node) if reused_indices.numel() > 0 else None

        # Free the tail of the over-allocation when FlexKV returned
        # fewer than expected.
        if num_retrieved < uncached_len:
            self.token_to_kv_pool_allocator.free(token_slots[num_retrieved:])
            fetched_slots = token_slots[:num_retrieved]
        else:
            fetched_slots = token_slots

        if request_owned_req is not None:
            # Normal request completion inserts these slots into the radix tree.
            # Until then they have exactly one owner: the request cleanup path.
            request_owned_req.kv.cache_protected_len = value_numel
            request_owned_req._flexkv_uncached_restore = True
            # SchedulePolicy sets cache_protected_len to the complete restored
            # prefix after init_load_back.  That is correct for accounting while
            # the request is active, but these newly allocated slots are not in
            # the radix tree yet.  Preserve the genuinely tree-owned length so
            # cache_finished_req/cache_unfinished_req can free duplicate restores
            # when several concurrent requests load the same host prefix.
            request_owned_req._flexkv_restore_tree_owned_len = value_numel
            if reused_indices.numel() > 0:
                fetched_slots = torch.cat([reused_indices, fetched_slots])
            return fetched_slots, last_node

        new_node = TreeNode(priority=last_node.priority)
        start = value_numel
        end = start + num_retrieved
        new_node.key = key[start:end]
        new_node.value = fetched_slots
        new_node.parent = last_node
        child_key = new_node.key.child_key(self.page_size)
        if child_key in last_node.children:
            raise RuntimeError(
                "FlexKV load-back child appeared after prefix refresh: "
                f"rid={tracking_rid}, child_key={child_key}"
            )
        last_node.children[child_key] = new_node
        self.evictable_size_ += num_retrieved
        self._update_leaf_status(last_node)
        self._update_leaf_status(new_node)

        self.kv_events.record_store(new_node.parent)
        self.kv_events.record_store(new_node)

        if reused_indices.numel() > 0:
            fetched_slots = torch.cat([reused_indices, fetched_slots])
        return fetched_slots, new_node

    # ------------------------------------------------------------------
    # cache_finished_req (STORE)
    # ------------------------------------------------------------------

    def cache_finished_req(  # type: ignore[override]
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> None:
        """Base cache_finished_req then fire an async FlexKV store."""
        if getattr(req, "_flexkv_uncached_restore", False):
            # Restored IP/layerwise slots are request-owned until this insertion.
            # SchedulePolicy temporarily counts them as protected, so restore the
            # pre-load tree-owned boundary before RadixCache handles duplicates.
            req.kv.cache_protected_len = getattr(
                req, "_flexkv_restore_tree_owned_len", req.kv.cache_protected_len
            )
        super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        req._flexkv_uncached_restore = False
        if hasattr(req, "_flexkv_restore_tree_owned_len"):
            del req._flexkv_restore_tree_owned_len
        if not is_insert:
            self._load_markers.pop(req.rid, None)
            return

        # Compute the committed prefix mirroring LMCRadixCache's logic.
        topk = get_spec().speculative_eagle_topk
        enable_kv_committed_len = topk is None or topk == 1
        if enable_kv_committed_len:
            kv_committed_len = req.kv.kv_committed_len
        else:
            kv_committed_len = len(req.origin_input_ids) + max(
                len(req.output_ids) - 1, 0
            )

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        if not token_ids:
            return
        # Anchor on the new last_device_node so FlexKV's lock matches
        # the node we'll later unlock when the store completes.  Read the KV
        # slots back from the radix result, not from req_to_token: the base
        # cache_finished_req has already transferred ownership to the radix
        # tree, and the request row may be cleared/reused before an async
        # FlexKV store consumes it.
        match_result = super().match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids,
                    req.extra_key,
                    cache_salt=req.cache_salt,
                )
            )
        )
        new_last_node = match_result.last_device_node
        if new_last_node is None:
            return
        kv_indices = match_result.device_indices
        if kv_indices.numel() != len(token_ids):
            logger.warning(
                "[FlexKV] store prefix length aligned from %d to %d tokens for rid=%s",
                len(token_ids),
                kv_indices.numel(),
                req.rid,
            )
            token_ids = token_ids[: kv_indices.numel()]
        if kv_indices.numel() == 0:
            return

        self.inc_lock_ref(new_last_node)
        if self._defer_store_launch:
            with self._node_lock:
                if (
                    req.rid in self._pending_store_launches
                    or req.rid in self._inflight_store_nodes
                ):
                    self.dec_lock_ref(new_last_node)
                    raise RuntimeError(f"FlexKV duplicate pending store rid={req.rid}")
                self._pending_store_launches[req.rid] = _PendingStoreLaunch(
                    node=new_last_node,
                    token_ids=list(token_ids),
                    kv_indices=kv_indices,
                )
            return

        try:
            fkv_task_id = self._launch_store(req.rid, list(token_ids), kv_indices)
        except Exception:  # noqa: BLE001
            self.dec_lock_ref(new_last_node)
            raise

        if fkv_task_id < 0:
            # Nothing to write back (either everything already in
            # FlexKV, or put_match failed / returned None).
            self.dec_lock_ref(new_last_node)
            return

        with self._node_lock:
            self._inflight_store_nodes[req.rid] = new_last_node

    def _launch_store(
        self,
        rid: str,
        token_ids: list[int],
        kv_indices: torch.Tensor,
        *,
        mapping_already_on_cpu: bool = False,
        skip_mapping_validation: bool = False,
    ) -> int:
        # ``kv_indices`` is produced on the scheduler/model stream (the radix
        # copy above is asynchronous on GPU).  The FlexKV D2H path consumes it
        # from ``store_stream``; without this edge the new stream can observe
        # zero-initialized slots and collapse every source page to block 0.
        if not mapping_already_on_cpu and not skip_mapping_validation:
            producer_stream = torch.cuda.current_stream()
            with self._store_profile_scope("flexkv.store.wait_producer_stream"):
                self.store_stream.wait_stream(producer_stream)
        with torch.cuda.stream(self.store_stream):
            if self.page_size > 1 and not skip_mapping_validation:
                with self._store_profile_scope("flexkv.store.slot_mapping_to_cpu"):
                    page_reps = kv_indices[:: self.page_size]
                    if not mapping_already_on_cpu:
                        page_reps = page_reps.to(device="cpu", dtype=torch.int64)
                with self._store_profile_scope("flexkv.store.slot_mapping_validate"):
                    page_ids = page_reps // self.page_size
                    unique_pages = torch.unique(page_ids)
                    aligned = bool((page_reps % self.page_size == 0).all())
                logger.info(
                    "[FlexKV] D2H slot mapping rid=%s tokens=%d pages=%d "
                    "slot_first=%d slot_last=%d block_min=%d block_max=%d "
                    "unique_blocks=%d aligned=%s",
                    rid,
                    kv_indices.numel(),
                    page_reps.numel(),
                    int(page_reps[0]),
                    int(page_reps[-1]),
                    int(page_ids.min()),
                    int(page_ids.max()),
                    unique_pages.numel(),
                    aligned,
                )
                if not aligned or unique_pages.numel() != page_ids.numel():
                    raise RuntimeError(
                        "FlexKV D2H received an invalid GPU slot mapping: "
                        f"rid={rid}, pages={page_ids.numel()}, "
                        f"unique_pages={unique_pages.numel()}, aligned={aligned}"
                    )
            with self._store_profile_scope("flexkv.store.connector_store_kv"):
                return self.flexkv_connector.store_kv(
                    rid=rid,
                    token_ids=token_ids,
                    kv_indices=kv_indices,
                    sglang_req_id=rid,
                )

    def _store_profile_scope(self, name: str):
        if not getattr(self, "_profile_store_stages", False):
            return nullcontext()
        return torch.profiler.record_function(name)

    def _stage_store_copy(self, rid: str, pending: _PendingStoreLaunch) -> None:
        """Enqueue the leader's full slot mapping into pinned host memory."""
        cpu_indices: Optional[torch.Tensor] = None
        ready_event: Optional[torch.cuda.Event] = None
        if bool(getattr(self.flexkv_connector, "is_store_sync_leader", True)):
            producer_stream = torch.cuda.current_stream()
            self.store_stream.wait_stream(producer_stream)
            cpu_indices = torch.empty(
                pending.kv_indices.shape,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            ready_event = torch.cuda.Event()
            with torch.cuda.stream(self.store_stream):
                cpu_indices.copy_(pending.kv_indices, non_blocking=True)
                ready_event.record(self.store_stream)
        self._pending_store_copies[rid] = _PendingStoreCopy(
            node=pending.node,
            token_ids=pending.token_ids,
            kv_indices=pending.kv_indices,
            cpu_indices=cpu_indices,
            ready_event=ready_event,
        )

    def _launch_ready_store_copies(self) -> None:
        """Poll pinned copies and launch the same ready prefix on every rank."""
        local_ready: list[str] = []
        if bool(getattr(self.flexkv_connector, "is_store_sync_leader", True)):
            for rid, pending in self._pending_store_copies.items():
                if pending.ready_event is None or not pending.ready_event.query():
                    break
                local_ready.append(rid)
        ready_rids = self.flexkv_connector.sync_ready_store_rids(local_ready)
        for rid in ready_rids:
            pending = self._pending_store_copies.pop(rid, None)
            if pending is None:
                raise RuntimeError(
                    f"FlexKV async store-ready rid is not locally pending: {rid}"
                )
            store_indices = (
                pending.cpu_indices
                if pending.cpu_indices is not None
                else pending.kv_indices
            )
            try:
                with self._store_profile_scope("flexkv.store.launch_one_ready_copy"):
                    fkv_task_id = self._launch_store(
                        rid,
                        pending.token_ids,
                        store_indices,
                        mapping_already_on_cpu=pending.cpu_indices is not None,
                        skip_mapping_validation=pending.cpu_indices is None,
                    )
            except Exception:  # noqa: BLE001
                self.dec_lock_ref(pending.node)
                raise
            if fkv_task_id < 0:
                self.dec_lock_ref(pending.node)
                continue
            with self._node_lock:
                self._inflight_store_nodes[rid] = pending.node

    def _launch_pending_stores(self) -> None:
        with self._store_profile_scope("flexkv.store.drain_pending"):
            while True:
                with self._node_lock:
                    if not self._pending_store_launches:
                        break
                    rid = next(iter(self._pending_store_launches))
                    pending = self._pending_store_launches.pop(rid)
                if self._async_store_slot_mapping:
                    try:
                        self._stage_store_copy(rid, pending)
                    except Exception:  # noqa: BLE001
                        self.dec_lock_ref(pending.node)
                        raise
                    continue
                try:
                    with self._store_profile_scope("flexkv.store.launch_one_pending"):
                        fkv_task_id = self._launch_store(
                            rid, pending.token_ids, pending.kv_indices
                        )
                except Exception:  # noqa: BLE001
                    self.dec_lock_ref(pending.node)
                    raise
                if fkv_task_id < 0:
                    self.dec_lock_ref(pending.node)
                    continue
                with self._node_lock:
                    self._inflight_store_nodes[rid] = pending.node
            if self._async_store_slot_mapping:
                self._launch_ready_store_copies()

    def cache_unfinished_req(  # type: ignore[override]
        self, req: Req, chunked=False
    ) -> None:
        if getattr(req, "_flexkv_uncached_restore", False):
            req.kv.cache_protected_len = getattr(
                req, "_flexkv_restore_tree_owned_len", req.kv.cache_protected_len
            )
        super().cache_unfinished_req(req, chunked=chunked)
        req._flexkv_uncached_restore = False
        if hasattr(req, "_flexkv_restore_tree_owned_len"):
            del req._flexkv_restore_tree_owned_len

    # ------------------------------------------------------------------
    # evict + completion draining
    # ------------------------------------------------------------------

    def evict(self, params: EvictParams) -> EvictResult:  # type: ignore[override]
        """Evict unlocked nodes without entering a cross-rank protocol."""
        if self.disable:
            return EvictResult()
        # Eviction is conditional on local allocator pressure, so not every
        # TP/CP rank calls it. Store completion is a scatter protocol and must
        # only run from the synchronized scheduler hook below. In-flight store
        # nodes remain protected by their lock refs here.
        # Make sure the store stream's GPU work is observed before any
        # eviction frees the source slots.
        self.store_stream.synchronize()
        if self._async_store_slot_mapping:
            self._launch_ready_store_copies()
            self._drain_completed_stores()
        return super().evict(params)

    def check_hicache_events(self) -> None:  # type: ignore[override]
        """Periodic non-blocking sweep called by the scheduler tick.

        Drains both store completions (so source nodes get unlocked
        quickly) and the launched-load tail (so the FlexKV pipe
        doesn't accumulate)."""
        self._drain_completed_stores()
        self.flexkv_connector.drain_launched_loads()
        self._launch_pending_stores()

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
        self._load_markers.pop(rid, None)
        with self._node_lock:
            pending = self._pending_store_launches.pop(rid, None)
            pending_copy = self._pending_store_copies.pop(rid, None)
            node = self._inflight_store_nodes.pop(rid, None)
        if pending is not None:
            self.dec_lock_ref(pending.node)
        if pending_copy is not None:
            self.dec_lock_ref(pending_copy.node)
        if node is not None:
            self.dec_lock_ref(node)
        self.flexkv_connector.release_pending(rid)
        self.flexkv_connector.cancel_prefetch(rid)

    def prefetch_request(self, req: Req) -> None:
        """Wait-complete FlexKV prefetch for a queued request.

        Owns fill-id refresh / page alignment so the scheduler only needs
        ``tree_cache.prefetch_request(req)``. Does not call FlexKV lookup
        (that happens at admission after prefetch completes).
        """
        req.init_next_round_input(tree_cache=None, cow_mamba=False)
        fill_ids = req.full_untruncated_fill_ids
        if not fill_ids:
            return
        match_end = req._compute_max_prefix_len(len(fill_ids))
        tokens = fill_ids[:match_end]
        self.prefetch_from_storage(req.rid, None, tokens)

    def prefetch_from_storage(
        self,
        rid: str,
        last_host_node=None,
        token_ids=None,
        last_hash=None,
        prefix_keys=None,
    ) -> None:
        """Kick off FlexKV prefetch (SSD/Remote/Mooncake → CPU).

        Extra HiCache-style args (``last_host_node`` / hashes) are ignored;
        FlexKV addresses blocks by token ids.
        """
        del last_host_node, last_hash, prefix_keys
        if not token_ids:
            return
        ids = list(token_ids)
        if self.page_size > 1:
            aligned = (len(ids) // self.page_size) * self.page_size
            ids = ids[:aligned]
        if not ids:
            return
        try:
            self.flexkv_connector.prefetch_async(rid, ids, sglang_req_id=rid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[FlexKV] prefetch_from_storage: %s", exc)

    def check_prefetch_progress(self, rid: str) -> bool:
        return self.flexkv_connector.check_prefetch_progress(rid)

    def terminate_prefetch(self, rid: str) -> None:
        self.flexkv_connector.cancel_prefetch(rid)

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        pop = getattr(self.flexkv_connector, "pop_prefetch_loaded_tokens", None)
        if callable(pop):
            return int(pop(rid))
        # Fallback until connector exposes actual prefetch hit length (M1).
        del rid
        return 0

    @property
    def hicache_storage_pass_prefix_keys(self) -> bool:
        # We pass token ids, not opaque key strings, so no prefix-key
        # accounting in the scheduler.
        return False
