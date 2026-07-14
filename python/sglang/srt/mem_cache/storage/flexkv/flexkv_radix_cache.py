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
import os
import threading
from dataclasses import dataclass, replace
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
    FlexKVAmbiguousLoadError,
    FlexKVConnector,
    FlexKVRetrieveStatus,
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
    expected_slots: int


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
        storage_page_size = int(params.page_size)
        allocator_page_size = int(params.token_to_kv_pool_allocator.page_size)
        if storage_page_size <= 0 or allocator_page_size <= 0:
            raise ValueError("FlexKV page sizes must be positive")
        if allocator_page_size % storage_page_size != 0:
            raise ValueError(
                "FlexKV requires the storage page size to divide the allocator "
                "page size"
            )
        enable_layerwise = bool(
            int(os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0"))
        )
        if enable_layerwise and allocator_page_size > 1:
            raise ValueError("FlexKV layerwise transfer requires allocator page size 1")

        super().__init__(replace(params, page_size=allocator_page_size))
        self._storage_page_size = storage_page_size
        self._allocator_page_size = allocator_page_size

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        # ``tp_group`` and ``attn_tp_group`` are sometimes passed
        # interchangeably by sglang's factory; prefer the explicit
        # ``attn_tp_group`` when given.
        attn_tp_group_eff = attn_tp_group if attn_tp_group is not None else tp_group

        self.flexkv_connector = FlexKVConnector(
            sgl_model_config=model_config,
            server_args=server_args,
            page_size=self._storage_page_size,
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
        self._quarantined_load_slots: list[torch.Tensor] = []
        # ``store_kv`` is async — we keep a lock on the source node
        # until FlexKV signals completion, draining in ``evict`` /
        # ``check_hicache_events``.
        self._inflight_store_nodes: dict[str, TreeNode] = {}
        self._node_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:  # type: ignore[override]
        if hasattr(self, "flexkv_connector"):
            with self._node_lock:
                has_active_store_nodes = bool(self._inflight_store_nodes)
            self.flexkv_connector.ensure_reset_safe(
                has_active_store_nodes=has_active_store_nodes,
                has_quarantined_load_slots=bool(self._quarantined_load_slots),
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
        self.flexkv_connector.ensure_load_back_safe()
        key = params.key
        if self.disable or not key:
            return super().match_prefix(params)

        # FlexKV load-back ownership uses allocator pages, so the query
        # must not expose a partial allocator page to the connector.
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
        return self._ip_match_prefix(key, base_res, device_value)

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

        _, hit = self.flexkv_connector.lookup_kv(
            token_ids=token_ids, token_mask=token_mask, rid=req.rid
        )
        if hit <= 0:
            return base_res

        # Snapshot the matched key (the live key aliases ``req.fill_ids``).
        if token_ids is key.token_ids:
            token_ids_snap = token_ids[:]
        else:
            token_ids_snap = token_ids
        self._load_markers[req.rid] = _LoadBackMarker(
            key=RadixKey(token_ids_snap, key.extra_key, key.is_bigram),
            value_numel=device_len,
            expected_slots=hit,
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
            rid=synthetic_rid,
            key=key,
            value_numel=device_len,
            uncached_len=hit,
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
            key=marker.key,
            value_numel=marker.value_numel,
            uncached_len=params.host_hit_length,
            expected_slots=marker.expected_slots,
        )
        if result is None:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        return result

    def _allocate_and_load_mp(
        self,
        *,
        rid: str,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
        expected_slots: int,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        valid_manifest = (
            uncached_len > 0
            and uncached_len == expected_slots
            and uncached_len % self._allocator_page_size == 0
            and value_numel % self._allocator_page_size == 0
            and value_numel + uncached_len <= len(key)
        )
        token_slots = (
            self._allocate_load_slots(num_slots=uncached_len)
            if valid_manifest
            else None
        )
        try:
            retrieve_result = self.flexkv_connector.retrieve_kv(
                rid=rid,
                slot_mapping=(
                    token_slots.to(torch.int64) if token_slots is not None else None
                ),
            )
        except Exception:  # noqa: BLE001
            self._quarantine_load_slots(token_slots)
            raise

        if retrieve_result.status is FlexKVRetrieveStatus.DEFINITE_TERMINAL_FAILURE:
            if token_slots is not None:
                self._release_load_slots(token_slots)
            return None
        if retrieve_result.status is FlexKVRetrieveStatus.AMBIGUOUS:
            self._quarantine_load_slots(token_slots)
            raise RuntimeError(
                f"FlexKV MP load outcome is ambiguous: {retrieve_result.reason}"
            )
        if (
            retrieve_result.status is not FlexKVRetrieveStatus.SUCCESS
            or token_slots is None
            or retrieve_result.num_slots != uncached_len
        ):
            self._quarantine_load_slots(token_slots)
            raise RuntimeError("FlexKV MP load returned an inconsistent success result")

        return self._publish_loaded_slots(
            key=key,
            value_numel=value_numel,
            loaded_slots=token_slots,
        )

    def _allocate_and_load_layerwise(
        self,
        *,
        rid: str,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        valid_manifest = (
            uncached_len > 0
            and uncached_len % self._allocator_page_size == 0
            and value_numel % self._allocator_page_size == 0
            and value_numel + uncached_len <= len(key)
        )
        token_slots = (
            self._allocate_load_slots(num_slots=uncached_len)
            if valid_manifest
            else None
        )
        try:
            num_retrieved, _ = self.flexkv_connector.start_load_kv_layerwise(
                rid=rid,
                slot_mapping=(
                    token_slots.to(torch.int64) if token_slots is not None else None
                ),
            )
        except FlexKVAmbiguousLoadError:
            self._quarantine_load_slots(token_slots)
            raise
        except Exception:  # noqa: BLE001
            self._quarantine_load_slots(token_slots)
            raise

        if num_retrieved <= 0:
            if token_slots is not None:
                self._release_load_slots(token_slots)
            return None
        if token_slots is None or num_retrieved != uncached_len:
            self._quarantine_load_slots(token_slots)
            raise RuntimeError(
                "FlexKV layerwise load returned an inconsistent success result"
            )

        return self._publish_loaded_slots(
            key=key,
            value_numel=value_numel,
            loaded_slots=token_slots,
        )

    def _allocate_load_slots(self, *, num_slots: int) -> Optional[torch.Tensor]:
        token_slots: Optional[torch.Tensor] = None
        try:
            local_shortage = max(
                0,
                num_slots - self.token_to_kv_pool_allocator.available_size(),
            )
        except Exception as exc:  # noqa: BLE001
            local_shortage = num_slots
            logger.warning(
                "[FlexKV] load slot capacity query failed: %s",
                exc,
                exc_info=True,
            )

        try:
            self.evict(EvictParams(num_tokens=local_shortage))
            token_slots = self.token_to_kv_pool_allocator.alloc(num_slots)
            if token_slots is None:
                return None
            if token_slots.numel() != num_slots:
                self.token_to_kv_pool_allocator.free(token_slots)
                logger.warning(
                    "[FlexKV] allocator returned %d slots for a %d-slot request",
                    token_slots.numel(),
                    num_slots,
                )
                return None
            return token_slots
        except Exception as exc:  # noqa: BLE001
            self._quarantine_load_slots(token_slots)
            logger.warning(
                "[FlexKV] load slot allocation failed: %s",
                exc,
                exc_info=True,
            )
            return None

    def _release_load_slots(self, token_slots: torch.Tensor) -> None:
        try:
            self.token_to_kv_pool_allocator.free(token_slots)
        except Exception:  # noqa: BLE001
            self._quarantine_load_slots(token_slots)
            raise

    def _quarantine_load_slots(self, token_slots: Optional[torch.Tensor]) -> None:
        if token_slots is not None:
            self._quarantined_load_slots.append(token_slots)

    def _publish_loaded_slots(
        self,
        *,
        key: RadixKey,
        value_numel: int,
        loaded_slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, TreeNode]:
        loaded_length = int(loaded_slots.numel())
        target_key = key[: value_numel + loaded_length]
        current_match = super().match_prefix(MatchPrefixParams(key=target_key))
        current_length = int(current_match.device_indices.numel())
        current_parent = current_match.last_device_node

        if current_length == len(target_key):
            if self._mode is FlexKVMode.IP:
                self._quarantine_load_slots(loaded_slots)
                raise RuntimeError(
                    "FlexKV layerwise load collided with an existing Radix entry"
                )
            current_loaded_slots = current_match.device_indices[value_numel:]
            if int(current_loaded_slots.numel()) != loaded_length:
                self._release_load_slots(loaded_slots)
                raise RuntimeError(
                    "FlexKV MP load collided with a non-exact Radix suffix"
                )
            self._release_load_slots(loaded_slots)
            return current_loaded_slots, current_parent

        if current_length != value_numel:
            if self._mode is FlexKVMode.IP:
                self._quarantine_load_slots(loaded_slots)
            else:
                self._release_load_slots(loaded_slots)
            raise RuntimeError(
                "FlexKV load publication found a stale or partial Radix prefix"
            )

        new_node = TreeNode(priority=current_parent.priority)
        new_node.key = target_key[value_numel:]
        new_node.value = loaded_slots
        new_node.parent = current_parent
        child_key = new_node.key.child_key(self.page_size)
        if current_parent.children.get(child_key) is not None:
            if self._mode is FlexKVMode.IP:
                self._quarantine_load_slots(loaded_slots)
            else:
                self._release_load_slots(loaded_slots)
            raise RuntimeError("FlexKV load publication collided with a Radix child")

        parent_was_evictable_leaf = current_parent in self.evictable_leaves
        child_attached = False
        size_added = False
        try:
            current_parent.children[child_key] = new_node
            child_attached = True
            self.evictable_size_ += loaded_length
            size_added = True
            self._update_leaf_status(current_parent)
            self._update_leaf_status(new_node)
        except Exception:  # noqa: BLE001
            if child_attached and current_parent.children.get(child_key) is new_node:
                current_parent.children.pop(child_key)
            if size_added:
                self.evictable_size_ -= loaded_length
            self.evictable_leaves.discard(new_node)
            if parent_was_evictable_leaf:
                self.evictable_leaves.add(current_parent)
            else:
                self.evictable_leaves.discard(current_parent)
            if self._mode is FlexKVMode.IP:
                self._quarantine_load_slots(loaded_slots)
            else:
                self._release_load_slots(loaded_slots)
            raise

        for event_node in (current_parent, new_node):
            try:
                self._record_store_event(event_node)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[FlexKV] failed to record loaded Radix node: %s",
                    exc,
                    exc_info=True,
                )

        return loaded_slots, new_node

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
        kv_committed_len = (
            kv_committed_len // self._allocator_page_size * self._allocator_page_size
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
        if self._mode is FlexKVMode.IP:
            self.flexkv_connector.ensure_layerwise_evict_safe()
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
        if self._mode is FlexKVMode.IP:
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
