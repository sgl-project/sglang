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
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.storage.flexkv.flexkv_connector import FlexKVConnector

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

        # CUDA streams (mirroring LMCRadixCache).
        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        # Two-phase MP load: stash marker between ``match_prefix`` and
        # ``init_load_back``.
        self._load_markers: dict[str, _LoadBackMarker] = {}
        # ``store_kv`` is async — we keep a lock on the source node
        # until FlexKV signals completion, draining in ``evict`` /
        # ``check_hicache_events``.
        self._inflight_store_nodes: dict[str, TreeNode] = {}
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
        if hasattr(self, "flexkv_connector"):
            self.flexkv_connector.reset()

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

        fkv_task_id, hit = self.flexkv_connector.lookup_kv(
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

        result = self._allocate_and_load(
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

        result = self._allocate_and_load(
            key=marker.key,
            value_numel=marker.value_numel,
            uncached_len=params.host_hit_length,
            last_node=last_node,
            load_fn=lambda slot_mapping: self.flexkv_connector.retrieve_kv(
                req.rid, slot_mapping
            ),
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
        load_fn,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Shared allocator + post-load bookkeeping for MP/IP.

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
    ) -> None:
        """Base cache_finished_req then fire an async FlexKV store."""
        super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        if not is_insert:
            self._load_markers.pop(req.rid, None)
            return

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
            return
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        # Anchor on the new last_device_node so FlexKV's lock matches
        # the node we'll later unlock when the store completes.
        match_result = super().match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids,
                    req.extra_key,
                    cache_salt=getattr(req, "cache_salt", None),
                )
            )
        )
        new_last_node = match_result.last_device_node
        if new_last_node is None:
            return

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
            return

        with self._node_lock:
            self._inflight_store_nodes[req.rid] = new_last_node

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
