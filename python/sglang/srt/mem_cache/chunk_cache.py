from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class ChunkCacheEntry:
    def __init__(self, rid, value):
        self.rid = rid
        self.value = value


class ChunkCache(BasePrefixCache):
    def __init__(
        self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: BaseTokenToKVPool
    ):
        self.disable = True
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

        self.reset()

    def reset(self):
        self.entries = {}

    def match_prefix(self, rid: int, key: List[int]) -> Tuple[List[int], int]:
        if rid not in self.entries:
            return [], None

        entry = self.entries[rid]
        max_prefix_len = len(key)
        return entry.value[:max_prefix_len], entry

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if token_ids is None:
            token_id_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        else:
            token_id_len = len(token_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool.free(kv_indices)

        if req.rid in self.entries:
            del self.entries[req.rid]

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        if token_ids is None:
            token_id_len = len(req.fill_ids)
        else:
            token_id_len = len(token_ids)

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :token_id_len
        ]

        if req.rid not in self.entries:
            self.entries[req.rid] = ChunkCacheEntry(req.rid, kv_indices)

        entry = self.entries[req.rid]
        entry.value = kv_indices
        req.prefix_indices = kv_indices
        req.last_node = entry

    def insert(self):
        raise NotImplementedError()

    def evict(self, num_tokens: int, evict_callback: Callable):
        pass

    def inc_lock_ref(self, node):
        return 0

    def dec_lock_ref(self, node):
        return 0

    def evictable_size(self):
        return 0

    def protected_size(self):
        return 0
