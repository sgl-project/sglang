from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

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

    Stuffed into ``MatchResult.last_host_node`` for the MP path; the
    base ``BasePrefixCache.last_host_node`` is typed ``Any`` so this
    is fine. ``init_load_back`` reads it back from
    ``InitLoadBackParams.last_host_node`` to know what to allocate
    and retrieve.
    """

    token_ids: Any  # the engine's token id sequence (list-like)
    extra_key: Any  # RadixKey extra-key (model-supplied disambiguator)
    value_numel: int  # number of tokens already in radix at match time
    matched: int  # daemon-reported chunk-aligned match count from LOOKUP
    last_device_node: Any  # last on-device TreeNode at match time


class LayerTransferCounter:
    """Per-layer hook the in-process layerwise connector uses to interleave
    LMCache loads with model execution.

    The KV pool calls `wait_until(layer_id)` after finishing a layer, which
    we translate into a `load_kv_layerwise(layer_id)` call on the LMCache
    connector within the provided CUDA stream.
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
    its per-layer transfer hook. MP mode uses the non-layerwise
    `LMCacheMPConnector`: a single blocking retrieve completes in
    `match_prefix` before the forward pass, so no per-layer hook is needed.
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
        from sglang.srt.server_args import get_global_server_args

        global_server_args = get_global_server_args()

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        connector_kwargs = dict(
            sgl_config=model_config,
            tp_size=tp_size,
            rank=rank,
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

    def reset(self):  # type: ignore[override]
        super().reset()
        if hasattr(self, "_in_flight_nodes"):
            with self._node_lock:
                self._in_flight_nodes.clear()

    def _load_into_slots(self, load_metadata: LoadMetadata) -> int:
        with torch.cuda.stream(self.load_stream):
            num = self.lmcache_connector.start_load_kv(load_metadata)
        if self._mp_mode:
            # MP non-layerwise: start_load_kv host-blocked until every layer
            # was queued on load_stream. Make compute streams wait for those
            # writes before reading the slots.
            torch.cuda.current_stream().wait_stream(self.load_stream)
        return num

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # type: ignore[override]
        """Match cached prefix; if there's a tail miss, prefetch from LMCache."""
        key = params.key
        if self.disable or not key:
            return super().match_prefix(params)

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]

        base_res = super().match_prefix(params)
        value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        # In MP mode the daemon-side session is keyed by ``req.rid`` so
        # the same id flows from LOOKUP through STORE/END_SESSION. Peek
        # callers in ``schedule_policy`` don't pass a Req (the field is
        # Mamba-flavored optional in SGLang); fall through to the radix
        # result for those — the real ``schedule_batch`` pass passes
        # ``req`` and our two-phase load lands then.
        is_mp_load = self._mp_mode and params.req is not None

        if is_mp_load:
            return self._mp_match_prefix(key, base_res, value, last_node, params.req)

        # In-process mode below: existing single-shot pattern that runs
        # LOOKUP + first-layer retrieve inside ``start_load_kv`` and lets
        # the per-layer transfer hook drive the rest during forward.
        if value.numel() == len(key):
            return base_res
        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        chunk_size = self.lmcache_connector.chunk_size()
        prefix_pad = value.numel() % chunk_size

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return base_res

        slot_mapping = torch.cat(
            [
                torch.full((value.numel(),), -1, dtype=torch.int64, device=self.device),
                token_slots.detach().clone().to(torch.int64).to(self.device),
            ]
        )

        num_retrieved = self._load_into_slots(
            LoadMetadata(
                token_ids=key.token_ids,
                slot_mapping=slot_mapping,
                offset=value.numel() - prefix_pad,
                prefix_pad=prefix_pad,
                request_id="",
            )
        )
        logger.debug("num_retrieved_tokens: %s", num_retrieved)

        if num_retrieved > 0:
            self.token_to_kv_pool_allocator.free(
                token_slots[(num_retrieved - prefix_pad) :]
            )
        else:
            self.token_to_kv_pool_allocator.free(token_slots)
            return base_res

        fetched = num_retrieved - prefix_pad
        new_node = TreeNode(priority=last_node.priority)
        start = value.numel()
        end = start + fetched
        new_node.key = key[start:end]
        new_node.value = token_slots[:fetched]
        new_node.parent = last_node
        last_node.children[new_node.key.child_key(self.page_size)] = new_node
        last_node = new_node

        value = torch.cat([value, token_slots[:fetched]])
        self.evictable_size_ += fetched

        self._record_store_event(new_node.parent)
        self._record_store_event(new_node)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def _mp_match_prefix(
        self,
        key: RadixKey,
        base_res: MatchResult,
        value: torch.Tensor,
        last_node: TreeNode,
        req: Req,
    ) -> MatchResult:
        """Phase-1 of the MP-mode two-phase load.

        Fires LOOKUP via ``connector.lookup_kv`` (no GPU copy). If the
        daemon's matched-token count goes beyond what radix already
        has, return a ``MatchResult`` with ``host_hit_length`` set so
        the SGLang scheduler will call our ``init_load_back`` at
        dispatch time to fire the actual RETRIEVE. Otherwise, release
        the held read locks and return the radix-only result.

        In both cases the daemon now has a session keyed by ``req.rid``
        with ``lookup_ipc_key`` set, so the eventual STORE+END_SESSION
        in ``cache_finished_req`` reuses it without warnings.
        """
        matched = self.lmcache_connector.lookup_kv(key.token_ids, req.rid)
        if matched <= value.numel():
            # LMCache had nothing fresh beyond what's already in radix.
            # Release the read locks; pending entry stays so end_session
            # still sends END_SESSION wire (clean session lifecycle).
            self.lmcache_connector.release_pending(req.rid)
            return base_res

        marker = _LMCacheLoadMarker(
            token_ids=key.token_ids,
            extra_key=key.extra_key,
            value_numel=int(value.numel()),
            matched=matched,
            last_device_node=last_node,
        )
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=marker,
            host_hit_length=matched - int(value.numel()),
        )

    def init_load_back(  # type: ignore[override]
        self, params: InitLoadBackParams
    ) -> Tuple[torch.Tensor, Any]:
        """Phase-2 of the MP-mode two-phase load.

        Called by the scheduler at dispatch time when ``match_prefix``
        returned ``host_hit_length > 0``. Allocates the new GPU slots,
        builds the slot_mapping, fires RETRIEVE via
        ``connector.retrieve_kv`` (which uses the cached LOOKUP result
        from ``_mp_match_prefix`` — no second LOOKUP wire), inserts a
        new ``TreeNode`` into the radix tree, and returns
        ``(new_indices, new_last_node)`` for the scheduler to splice
        onto ``req.prefix_indices``.
        """
        marker = params.last_host_node
        if not isinstance(marker, _LMCacheLoadMarker) or params.req is None:
            # Shouldn't happen if our match_prefix produced this hit,
            # but guard anyway.
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                (
                    marker
                    if isinstance(marker, TreeNode)
                    else (params.req.last_node if params.req is not None else None)
                ),
            )

        req = params.req
        last_node: TreeNode = marker.last_device_node
        new_count = marker.matched - marker.value_numel

        if self.token_to_kv_pool_allocator.available_size() < new_count:
            self.evict(EvictParams(num_tokens=new_count))

        token_slots = self.token_to_kv_pool_allocator.alloc(new_count)
        if token_slots is None:
            # Allocation failed; release held locks and report no load.
            self.lmcache_connector.release_pending(req.rid)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )

        chunk_size = self.lmcache_connector.chunk_size()
        prefix_pad = marker.value_numel % chunk_size

        slot_mapping = torch.cat(
            [
                torch.full(
                    (marker.value_numel,),
                    -1,
                    dtype=torch.int64,
                    device=self.device,
                ),
                token_slots.detach().clone().to(torch.int64).to(self.device),
            ]
        )

        with torch.cuda.stream(self.load_stream):
            retrieved = self.lmcache_connector.retrieve_kv(
                LoadMetadata(
                    token_ids=marker.token_ids,
                    slot_mapping=slot_mapping,
                    offset=marker.value_numel - prefix_pad,
                    prefix_pad=prefix_pad,
                    request_id=req.rid,
                )
            )
        torch.cuda.current_stream().wait_stream(self.load_stream)

        if retrieved <= 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )

        fetched = retrieved - prefix_pad
        if fetched < new_count:
            self.token_to_kv_pool_allocator.free(token_slots[fetched:])

        new_node = TreeNode(priority=last_node.priority)
        new_node.key = RadixKey(
            marker.token_ids[marker.value_numel : marker.value_numel + fetched],
            marker.extra_key,
        )
        new_node.value = token_slots[:fetched]
        new_node.parent = last_node
        last_node.children[new_node.key.child_key(self.page_size)] = new_node
        self.evictable_size_ += fetched

        self._record_store_event(new_node.parent)
        self._record_store_event(new_node)

        return token_slots[:fetched], new_node

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:  # type: ignore[override]
        """On request completion, insert device KV into radix and store to LMCache."""

        super().cache_finished_req(req, is_insert=is_insert)
        if not is_insert:
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

        # Bypass the LMCache-aware override here — we just need
        # ``new_last_node`` from the radix tree, which now contains the
        # tokens we inserted via ``super().cache_finished_req`` above.
        # Calling ``self.match_prefix`` would trigger a redundant
        # ``ensure_session`` LOOKUP on the MP path (the scheduler-time
        # match_prefix already created the daemon session for this rid).
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
        with self._node_lock:
            self._in_flight_nodes.append(new_last_node)
        # MP-mode single per-request cleanup hook — fires once regardless of
        # whether store_kv ran or early-returned, mirroring vLLM's
        # request_finished hook. Releases the daemon-side session created
        # by LOOKUP and/or STORE. The in-process connector has no such
        # session concept and exposes no end_session method.
        if self._mp_mode:
            self.lmcache_connector.end_session(req.rid)

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
