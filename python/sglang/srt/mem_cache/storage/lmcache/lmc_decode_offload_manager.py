from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.base.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool

try:
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheConnector,
        StoreMetadata,
    )
except ImportError as e:
    raise RuntimeError(
        "LMCache is not installed. Please install it by running `pip install lmcache`"
    ) from e

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LMCCacheDecodeKVCacheOffloadManager(DecodeKVCacheOffloadManager):
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_counter = 0
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            pass
        elif isinstance(kv_cache, MLATokenToKVPool):
            raise ValueError(
                "MLA is not supported yet in LMCache decode offload manager"
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.lmcache_connector = LMCacheConnector(
            sgl_config=self.model_config,
            tp_size=self.tp_size,
            rank=self.tp_rank,
            k_pool=kv_cache.k_buffer,
            v_pool=kv_cache.v_buffer,
        )
        self.chunk_size = self.lmcache_connector.chunk_size()
        logger.info("Enable offload kv cache for decode side")

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""
        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.chunk_size * self.chunk_size
        )
        incremental_len = len(all_tokens) - prefill_offloaded_len
        incremental_aligned_len = incremental_len // self.chunk_size * self.chunk_size

        if incremental_aligned_len == 0:
            return False

        # prefill offload tokens is also needed for hash calculation
        decode_offload_len = prefill_offloaded_len + incremental_aligned_len
        decode_offload_indices = token_indices[:decode_offload_len]
        store_md = StoreMetadata(
            last_node=None,
            token_ids=all_tokens[:decode_offload_len],
            kv_indices=decode_offload_indices,
            offset=0,
        )
        # offload incremental KV cache from device to host
        self.lmcache_connector.store_kv(store_md)
        # offload to cpu is already done. let scheduler free gpu memory so return False
        return False

    def check_offload_progress(self):
        pass
