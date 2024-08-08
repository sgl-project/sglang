"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class ChunkCacheEntry:
    def __init__(self, rid, value):
        self.rid = rid
        self.value = value


class ChunkCache(BasePrefixCache):
    def __init__(self, req_to_token_pool, token_to_kv_pool):
        self.disable = True
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

        self.reset()

    def reset(self):
        self.entries = {}

    def match_prefix(self, rid, **kwargs):
        if rid not in self.entries:
            return [], None

        entry = self.entries[rid]
        return entry.value, entry

    def cache_finished_req(self, req: "Req", token_ids=None):
        if token_ids is None:
            token_ids = (req.input_ids + req.output_ids)[:-1]

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        assert req.rid in self.entries
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool.free(kv_indices)

    def cache_unfinished_req(self, req: "Req", token_ids=None):
        if token_ids is None:
            token_ids = req.input_ids

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if req.rid not in self.entries:
            self.entries[req.rid] = ChunkCacheEntry(req.rid, kv_indices)

        entry = self.entries[req.rid]
        entry.value = kv_indices
        req.prefix_indices = kv_indices
        req.last_node = entry

    def insert(self):
        raise NotImplementedError

    def evict(self, num_tokens, evict_callback):
        pass

    def inc_lock_ref(self, node):
        return 0

    def dec_lock_ref(self, node):
        return 0

    def evictable_size(self):
        return 0
