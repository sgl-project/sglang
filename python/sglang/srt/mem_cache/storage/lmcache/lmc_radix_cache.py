from __future__ import annotations

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

try:
    from lmcache.integration.sglang.multi_process_adapter import LMCacheMPConnector
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )
except ImportError as e:
    raise RuntimeError(
        "LMCache is not installed. Please install it by running `pip install lmcache`"
    ) from e


if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


@dataclass
class _LMCacheLoadMarker:
    """Carries the data ``init_load_back`` needs from the
    ``match_prefix`` call that decided a load is warranted.
    """

    key: RadixKey  # page-aligned key the scheduler matched on
    value_numel: int  # number of tokens already in radix at match time


class LayerTransferCounter:
    """Minimal adapter that lets the memory pool notify LMCache per-layer.

    The KV pool calls `wait_until(layer_id)` after finishing a layer, which we
    translate into a `load_kv_layerwise(layer_id)` call on the LMCache connector
    within the provided CUDA stream.
    """

    def __init__(
        self,
        num_layers: int,
        load_stream: torch.cuda.Stream,
        lmc_connector: LMCacheLayerwiseConnector,
        printable: bool = False,
    ):
        self.num_layers = num_layers
        self.load_stream = load_stream
        self.lmc_connector = lmc_connector

    def wait_until(self, layer_id: int):
        # Ensure ordering of the async loads wrt compute stream(s).
        self.load_stream.synchronize()
        with self.load_stream:
            self.lmc_connector.load_kv_layerwise(layer_id)


class LMCRadixCache(RadixCache):
    """RadixCache + LMCache IO.

    In-process mode (no MP host) keeps the existing layerwise connector and
    its per-layer transfer hook: ``match_prefix`` kicks off the load via
    ``start_load_kv`` and SGLang's per-layer KV-pool hook drives subsequent
    layers during forward.

    MP mode uses ``LMCacheMPConnector`` with a HiCache-style two-phase
    load: ``match_prefix`` fires LOOKUP only (``connector.lookup_kv``) and
    returns ``host_hit_length`` on the ``MatchResult``; the SGLang
    scheduler then calls our :meth:`init_load_back` at dispatch time,
    which fires the actual RETRIEVE (``connector.retrieve_kv``) into
    pre-allocated GPU slots. No per-layer hook is registered for the MP
    path — a single ``multi_layer_block_kv_transfer`` kernel covers all
    layers.
    """

    def __init__(
        self,
        params: CacheInitParams,
        model_config: Optional["ModelConfig"] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(params)

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        connector_kwargs = dict(
            sgl_config=model_config,
            tp_size=tp_size,
            rank=rank,
            # NOTE: The original implementation accessed private buffers via
            # `_kvcache.k_buffer` / `.v_buffer`. We prefer public accessors when
            # available; fall back to private fields if needed.
            k_pool=getattr(
                kvcache,
                "k_buffer",
                getattr(self.token_to_kv_pool_allocator._kvcache, "k_buffer"),
            ),
            v_pool=getattr(
                kvcache,
                "v_buffer",
                getattr(self.token_to_kv_pool_allocator._kvcache, "v_buffer"),
            ),
            tp_group=tp_group.device_group if tp_group is not None else None,
        )
        from sglang.srt.server_args import get_global_server_args

        global_server_args = get_global_server_args()
        self._mp_mode = global_server_args.lmcache_mp_host is not None
        if self._mp_mode:
            self.lmcache_connector = LMCacheMPConnector(
                page_size=params.page_size,
                host=global_server_args.lmcache_mp_host,
                port=global_server_args.lmcache_mp_port,
                **connector_kwargs,
            )
        else:
            self.lmcache_connector = LMCacheLayerwiseConnector(**connector_kwargs)

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        # Per-layer hook only matters for the in-process layerwise path.
        if not self._mp_mode:
            self.layer_done_executor = LayerTransferCounter(
                num_layers=(
                    model_config.num_hidden_layers if model_config is not None else 0
                ),
                load_stream=self.load_stream,
                lmc_connector=self.lmcache_connector,
            )
            kvcache.register_layer_transfer_counter(self.layer_done_executor)

        self._in_flight_nodes: list[TreeNode] = []
        self._node_lock = threading.Lock()
        self._mp_load_markers: dict[str, _LMCacheLoadMarker] = {}

    def reset(self):  # type: ignore[override]
        super().reset()
        if hasattr(self, "_in_flight_nodes"):
            with self._node_lock:
                self._in_flight_nodes.clear()
        if hasattr(self, "_mp_load_markers"):
            self._mp_load_markers.clear()

    def _mp_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        value: torch.Tensor,
        last_node: TreeNode,
        req: Req,
    ) -> MatchResult:
        """Phase 1 of the MP two-phase load: fire LOOKUP only.

        Returns a ``MatchResult`` with ``host_hit_length`` set when
        LMCache has tokens beyond radix — the scheduler then calls
        ``init_load_back`` to fire the RETRIEVE. Otherwise releases
        the held read locks and returns the radix-only result.
        """
        matched = self.lmcache_connector.lookup_kv(key.token_ids, req.rid)
        if matched <= value.numel():
            # Release the read locks; keep the pending session for end_session.
            self.lmcache_connector.release_pending(req.rid)
            return base_res

        self._mp_load_markers[req.rid] = _LMCacheLoadMarker(
            key=key,
            value_numel=int(value.numel()),
        )
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            host_hit_length=matched - int(value.numel()),
        )

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # type: ignore[override]
        """Match cached prefix; if there's a tail miss, prefetch from LMCache.

        Reuses the base matching logic to obtain (value, last_node). If there
        remains a *page-aligned* uncached suffix and there is room (or after
        eviction), we allocate token slots and trigger an async LMCache load
        into those slots, then materialize a new child node for the retrieved
        chunk.
        """
        key = params.key
        if self.disable or not key:
            return super().match_prefix(params)

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]

        base_res = super().match_prefix(params)
        value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        # MP mode: phase 1 LOOKUP here, phase 2 RETRIEVE in init_load_back.
        # Peek callers without a Req (e.g. schedule_policy) see radix only.
        if self._mp_mode:
            if params.req is None:
                return base_res
            return self._mp_match_prefix(key, base_res, value, last_node, params.req)

        # In-process mode: single-shot start_load_kv + per-layer transfer hook.
        if value.numel() == len(key):
            return base_res

        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        def _load(slot_mapping: torch.Tensor, prefix_pad: int) -> int:
            with torch.cuda.stream(self.load_stream):
                return self.lmcache_connector.start_load_kv(
                    LoadMetadata(
                        token_ids=key.token_ids,
                        slot_mapping=slot_mapping,
                        offset=value.numel() - prefix_pad,
                    )
                )

        result = self._alloc_and_load_chunk(
            key=key,
            value_numel=int(value.numel()),
            uncached_len=uncached_len,
            last_node=last_node,
            load_fn=_load,
        )
        if result is None:
            return base_res
        new_slots, new_node = result
        return MatchResult(
            device_indices=torch.cat([value, new_slots]),
            last_device_node=new_node,
            last_host_node=new_node,
            best_match_node=new_node,
        )

    def _alloc_and_load_chunk(
        self,
        *,
        key: RadixKey,
        value_numel: int,
        uncached_len: int,
        last_node: TreeNode,
        load_fn,  # Callable[[torch.Tensor, int], int] — (slot_mapping, prefix_pad) -> num_retrieved
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Shared body for in-process ``match_prefix`` and MP ``init_load_back``.

        Allocates ``uncached_len`` GPU slots (evicting if needed), builds
        the slot_mapping with leading sentinels, calls ``load_fn`` with
        ``(slot_mapping, prefix_pad)``, frees unused slots, and builds a
        new ``TreeNode`` under ``last_node`` for the fetched range.

        Returns ``(slots[:fetched], new_node)`` on success, ``None`` on
        alloc failure or zero-token retrieve (in either case all
        allocated slots have been freed).
        """
        chunk_size = self.lmcache_connector.chunk_size()
        prefix_pad = value_numel % chunk_size

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return None

        slot_mapping = torch.cat(
            [
                torch.full((value_numel,), -1, dtype=torch.int64, device=self.device),
                token_slots.detach().clone().to(torch.int64).to(self.device),
            ]
        )

        num_retrieved = load_fn(slot_mapping, prefix_pad)
        logger.debug("num_retrieved_tokens: %s", num_retrieved)

        if num_retrieved > 0:
            self.token_to_kv_pool_allocator.free(
                token_slots[(num_retrieved - prefix_pad) :]
            )
        else:
            self.token_to_kv_pool_allocator.free(token_slots)

        if num_retrieved > 0:
            fetched = num_retrieved - prefix_pad
            new_node = TreeNode(priority=last_node.priority)
            start = value_numel
            end = start + fetched
            new_node.key = key[start:end]
            new_node.value = token_slots[:fetched]
            new_node.parent = last_node
            last_node.children[new_node.key.child_key(self.page_size)] = new_node
            self.evictable_size_ += fetched

            self._record_store_event(new_node.parent)
            self._record_store_event(new_node)

            return token_slots[:fetched], new_node

        return None

    def init_load_back(  # type: ignore[override]
        self, params: InitLoadBackParams
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:
        """Phase 2 of the MP two-phase load: fire RETRIEVE.

        Called by the scheduler when ``match_prefix`` returned
        ``host_hit_length > 0``. Uses the cached LOOKUP result to
        allocate slots and fire RETRIEVE, inserts the resulting
        TreeNode into the radix tree, and returns
        ``(new_indices, new_last_node)``.
        """
        req = params.req
        marker = self._mp_load_markers.pop(req.rid)
        last_node: TreeNode = params.last_host_node

        def _load(slot_mapping: torch.Tensor, prefix_pad: int) -> int:
            with torch.cuda.stream(self.load_stream):
                n = self.lmcache_connector.retrieve_kv(
                    LoadMetadata(
                        token_ids=marker.key.token_ids,
                        slot_mapping=slot_mapping,
                        offset=marker.value_numel - prefix_pad,
                        prefix_pad=prefix_pad,
                        request_id=req.rid,
                    )
                )
            torch.cuda.current_stream().wait_stream(self.load_stream)
            return n

        result = self._alloc_and_load_chunk(
            key=marker.key,
            value_numel=marker.value_numel,
            uncached_len=params.host_hit_length,
            last_node=last_node,
            load_fn=_load,
        )
        if result is None:
            # Either alloc failed (locks still held by lookup_kv) or
            # retrieve returned nothing (locks already released by
            # retrieve_kv). release_pending is idempotent on locks_held.
            self.lmcache_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        return result

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:  # type: ignore[override]
        """On request completion, insert device KV into radix and store to LMCache."""

        super().cache_finished_req(req, is_insert=is_insert)
        if not is_insert:
            if self._mp_mode:
                self._mp_load_markers.pop(req.rid, None)
                self.lmcache_connector.end_session(req.rid)
            return

        from sglang.srt.server_args import get_global_server_args

        global_server_args = get_global_server_args()
        topk = global_server_args.speculative_eagle_topk
        enable_kv_committed_len = topk is None or topk == 1
        if enable_kv_committed_len:
            kv_committed_len = req.kv_committed_len
        else:
            kv_committed_len = len(req.origin_input_ids) + max(
                len(req.output_ids) - 1, 0
            )

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        # Use super() to avoid a redundant LOOKUP — we only need new_last_node from radix.
        match_result = super().match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids, req.extra_key))
        )
        new_last_node = match_result.last_device_node
        assert new_last_node is not None

        self.inc_lock_ref(new_last_node)
        store_md = StoreMetadata(
            last_node=new_last_node,
            token_ids=token_ids,
            kv_indices=kv_indices,
            offset=0,
            request_id=req.rid,
        )
        with torch.cuda.stream(self.store_stream):
            self.lmcache_connector.store_kv(store_md)
        if self._mp_mode:
            # MP store_kv blocks until the daemon's signal event fires, so the slots are safe to evict immediately.
            self._mp_load_markers.pop(req.rid, None)
            self.dec_lock_ref(new_last_node)
            self.lmcache_connector.end_session(req.rid)
        else:
            # Layerwise store is async on store_stream; defer the unlock to evict()'s store_stream.synchronize().
            with self._node_lock:
                self._in_flight_nodes.append(new_last_node)

    def evict(self, params: EvictParams) -> EvictResult:
        """Before base eviction, wait for any outstanding stores and release locks."""
        if self.disable:
            return EvictResult()

        self.store_stream.synchronize()
        with self._node_lock:
            for node in self._in_flight_nodes:
                self.dec_lock_ref(node)
            self._in_flight_nodes.clear()

        return super().evict(params)

    def pretty_print(self):  # type: ignore[override]
        super().pretty_print()
        try:
            logger.debug(
                "evictable=%d protected=%d", self.evictable_size_, self.protected_size_
            )
        except Exception:  # pragma: no cover
            pass
