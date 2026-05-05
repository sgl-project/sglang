from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
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

        # MP mode keys the daemon-side session by ``req.rid`` so the same
        # id flows from LOOKUP through STORE/END_SESSION (mirroring vLLM /
        # TRT-LLM). Some scheduler paths call ``match_prefix`` as a peek
        # without a Req (e.g. in-batch prefix-caching priority in
        # schedule_policy); for those, skip the LMCache fetch — the
        # actual prefill pass through schedule_batch passes ``req`` and
        # picks up the cached prefix at that time.
        is_mp_load = self._mp_mode and params.req is not None

        # Full-radix-hit / no-fresh-tokens cases: ``_load_into_slots`` won't
        # fire, so ``start_load_kv`` won't issue a LOOKUP. Send a session-
        # only LOOKUP via ``ensure_session`` so the eventual STORE+END_SESSION
        # at request finish lands on a session that has ``lookup_ipc_key``
        # set, matching vLLM/TRT-LLM's per-request session lifecycle.
        uncached_len = len(key) - value.numel()
        if value.numel() == len(key) or uncached_len == 0:
            if is_mp_load:
                self.lmcache_connector.ensure_session(
                    key.token_ids, params.req.rid
                )
            return base_res

        if not is_mp_load:
            return base_res

        chunk_size = self.lmcache_connector.chunk_size()
        prefix_pad = value.numel() % chunk_size

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            # Allocation failed — _load_into_slots won't fire, so
            # start_load_kv won't issue a LOOKUP. Still create the daemon
            # session so the eventual STORE+END_SESSION pairs cleanly.
            self.lmcache_connector.ensure_session(
                key.token_ids, params.req.rid
            )
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
                request_id=params.req.rid,
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

        # Pass req so MP-mode match_prefix can derive the wire request_id
        # from req.rid. Falls through the early-return path inside
        # match_prefix because the just-inserted tokens are already in radix.
        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids, req.extra_key), req=req)
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
