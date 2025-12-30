from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

try:
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )
    HAS_LMCACHE_INPROC = True
except ImportError:
    HAS_LMCACHE_INPROC = False
    LMCacheLayerwiseConnector = None
    LoadMetadata = None
    StoreMetadata = None

# Try importing MP connector
try:
    from sglang.srt.mem_cache.storage.lmcache.multi_process_adapter import (
        LMCacheMPConnector,
    )
    HAS_LMCACHE_MP = True
except ImportError:
    HAS_LMCACHE_MP = False
    LMCacheMPConnector = None


if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


def _check_lmcache_available(mp_mode: bool) -> None:
    """Check if the required LMCache connector is available."""
    if mp_mode:
        if not HAS_LMCACHE_MP:
            raise RuntimeError(
                "LMCache MP mode requires the multi_process_adapter module. "
                "Please ensure LMCache is properly installed."
            )
    else:
        if not HAS_LMCACHE_INPROC:
            raise RuntimeError(
                "LMCache is not installed. Please install it by running "
                "`pip install lmcache`"
            )


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
        lmc_connector: Union["LMCacheLayerwiseConnector", "LMCacheMPConnector"],
        printable: bool = False,
    ):
        self.num_layers = num_layers
        self.load_stream = load_stream
        self.lmc_connector = lmc_connector

    def wait_until(self, layer_id: int):
        # Ensure ordering of the async loads wrt compute stream(s).
        self.load_stream.synchronize()
        with self.load_stream:
            # Both connectors support load_kv_layerwise (MP connector has a stub)
            if hasattr(self.lmc_connector, 'load_kv_layerwise'):
                self.lmc_connector.load_kv_layerwise(layer_id)


class LMCRadixCache(RadixCache):
    """RadixCache + LMCache IO.

    This subclass adds:
      - LMCache connector setup (device/host buffers, TP rank/size)
      - Two CUDA streams for async load/store
      - Layer-wise transfer executor wiring to the KV cache
      - Overridden `match_prefix` to fetch missing prefix chunks from LMCache
      - Extended cache_finalization paths to store back into LMCache
      - Eviction barrier that respects any in-flight host->device stores
    
    Supports two modes:
      - In-process mode (default): Uses LMCacheLayerwiseConnector
      - Multi-process mode: Uses LMCacheMPConnector to connect to external server
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

        # Get server args to check MP mode
        from sglang.srt.server_args import get_global_server_args
        server_args = get_global_server_args()
        
        # Determine if we should use MP mode
        self._use_mp_mode = (
            server_args is not None and 
            getattr(server_args, 'lmcache_mp_enable', False)
        )
        
        # Check if required connector is available
        _check_lmcache_available(self._use_mp_mode)

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        
        # Get KV cache buffers
        k_pool = getattr(
            kvcache,
            "k_buffer",
            getattr(self.token_to_kv_pool_allocator._kvcache, "k_buffer"),
        )
        v_pool = getattr(
            kvcache,
            "v_buffer",
            getattr(self.token_to_kv_pool_allocator._kvcache, "v_buffer"),
        )
        
        if self._use_mp_mode:
            # Use Multi-Process mode connector
            logger.info(
                f"Initializing LMCache in MP mode: "
                f"host={server_args.lmcache_mp_host}, "
                f"port={server_args.lmcache_mp_port}"
            )
            self.lmcache_connector = LMCacheMPConnector(
                model_config=model_config,
                tp_size=tp_size,
                rank=rank,
                k_pool=k_pool,
                v_pool=v_pool,
                mp_host=server_args.lmcache_mp_host,
                mp_port=server_args.lmcache_mp_port,
                tp_group=tp_group.device_group if tp_group is not None else None,
            )
        else:
            # Use in-process connector
            logger.info("Initializing LMCache in in-process mode")
            self.lmcache_connector = LMCacheLayerwiseConnector(
                sgl_config=model_config,
                tp_size=tp_size,
                rank=rank,
                k_pool=k_pool,
                v_pool=v_pool,
                tp_group=tp_group.device_group if tp_group is not None else None,
            )

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

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

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:  # type: ignore[override]
        """Match cached prefix; if there's a tail miss, prefetch from LMCache.

        Reuses the base matching logic to obtain (value, last_node). If there
        remains a *page-aligned* uncached suffix and there is room (or after
        eviction), we allocate token slots and trigger an async LMCache load
        into those slots, then materialize a new child node for the retrieved
        chunk.
        """
        if self.disable or not key:
            return super().match_prefix(key, **kwargs)

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]

        base_res = super().match_prefix(key, **kwargs)
        value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        if value.numel() == len(key):
            return base_res

        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        chunk_size = self.lmcache_connector.chunk_size()
        prefix_pad = value.numel() % chunk_size

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(uncached_len)

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return base_res

        slot_mapping = torch.cat(
            [
                torch.full((value.numel(),), -1, dtype=torch.int64, device=self.device),
                token_slots.detach().clone().to(torch.int64).to(self.device),
            ]
        )

        with torch.cuda.stream(self.load_stream):
            offset = value.numel() - prefix_pad
            if self._use_mp_mode:
                # MP mode uses direct arguments
                num_retrieved = self.lmcache_connector.start_load_kv(
                    token_ids=key.token_ids,
                    slot_mapping=slot_mapping,
                    offset=offset,
                )
            else:
                # In-process mode uses LoadMetadata
                num_retrieved = self.lmcache_connector.start_load_kv(
                    LoadMetadata(
                        token_ids=key.token_ids,  # full page-aligned key
                        slot_mapping=slot_mapping,
                        offset=offset,  # LMCache offset convention
                    )
                )
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
            start = value.numel()
            end = start + fetched
            new_node.key = key[start:end]
            new_node.value = token_slots[:fetched]
            new_node.parent = last_node
            last_node.children[self.get_child_key_fn(new_node.key)] = new_node
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

        return base_res

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

        match_result = self.match_prefix(RadixKey(token_ids, req.extra_key))
        new_last_node = match_result.last_device_node
        assert new_last_node is not None

        self.inc_lock_ref(new_last_node)
        
        with torch.cuda.stream(self.store_stream):
            if self._use_mp_mode:
                # MP mode uses direct arguments
                self.lmcache_connector.store_kv(
                    token_ids=token_ids,
                    kv_indices=kv_indices,
                    offset=0,
                )
            else:
                # In-process mode uses StoreMetadata
                store_md = StoreMetadata(
                    last_node=new_last_node,
                    token_ids=token_ids,
                    kv_indices=kv_indices,
                    offset=0,
                )
                self.lmcache_connector.store_kv(store_md)
        
        with self._node_lock:
            self._in_flight_nodes.append(new_last_node)

    def evict(self, num_tokens: int) -> None:  # type: ignore[override]
        """Before base eviction, wait for any outstanding stores and release locks."""
        if self.disable:
            return

        self.store_stream.synchronize()
        with self._node_lock:
            for node in self._in_flight_nodes:
                self.dec_lock_ref(node)
            self._in_flight_nodes.clear()

        super().evict(num_tokens)

    def pretty_print(self):  # type: ignore[override]
        super().pretty_print()
        try:
            logger.debug(
                "evictable=%d protected=%d", self.evictable_size_, self.protected_size_
            )
        except Exception:  # pragma: no cover
            pass
