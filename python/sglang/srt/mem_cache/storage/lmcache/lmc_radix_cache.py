from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.utils import convert_to_bigram_key

try:
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

    This subclass adds:
      - LMCache connector setup (device/host buffers, TP rank/size)
      - Two CUDA streams for async load/store
      - Layer-wise transfer executor wiring to the KV cache
      - Overridden `match_prefix` to fetch missing prefix chunks from LMCache
      - Extended cache_finalization paths to store back into LMCache
      - Eviction barrier that respects any in-flight host->device stores
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
        self.lmcache_connector = LMCacheLayerwiseConnector(
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
            num_retrieved = self.lmcache_connector.start_load_kv(
                LoadMetadata(
                    token_ids=key.token_ids,  # full page-aligned key
                    slot_mapping=slot_mapping,
                    offset=value.numel() - prefix_pad,  # LMCache offset convention
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

    def cache_finished_req(self, req: "Req", is_insert: bool = True) -> None:  # type: ignore[override]
        """Cache finished request and write committed KV to LMCache."""
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        keys = convert_to_bigram_key(req.fill_ids) if self.is_eagle else req.fill_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            new_prefix_len = self.insert(radix_key, values, priority=priority)
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        self._store_finished_request(
            req=req,
            is_insert=is_insert,
            kv_committed_len=kv_committed_len,
            token_ids=token_ids,
            kv_indices=kv_indices,
        )

        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def _store_finished_request(
        self,
        *,
        req: Req,
        is_insert: bool,
        kv_committed_len: int,
        token_ids: list[int],
        kv_indices: torch.Tensor,
    ) -> None:
        """Store finished KV to LMCache while pool resources are still valid."""

        should_store = (
            is_insert
            and not self.disable
            and kv_committed_len > 0
            and req.req_pool_idx is not None
        )
        if not should_store:
            return

        committed_indices = kv_indices[:kv_committed_len].clone()

        match_result = self.match_prefix(RadixKey(token_ids, req.extra_key))
        new_last_node = match_result.last_device_node
        if new_last_node is None:
            return

        self.inc_lock_ref(new_last_node)
        store_md = StoreMetadata(
            last_node=new_last_node,
            token_ids=token_ids,
            kv_indices=committed_indices,
            offset=0,
        )
        with torch.cuda.stream(self.store_stream):
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


if __name__ == "__main__":
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    params = CacheInitParams(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        disable=False,
        enable_kv_cache_events=False,
    )
    cache = LMCRadixCache(
        params=params,
        model_config=None,
        tp_size=1,
        rank=0,
        tp_group=None,
    )
    cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 11, 12], dtype=torch.int64))
    cache.insert(
        RadixKey([1, 2, 3, 4]), torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    )
    cache.pretty_print()
