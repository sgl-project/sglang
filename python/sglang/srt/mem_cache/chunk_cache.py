"""Cache for chunked prefill, used when RadixCache is disabled."""

from sglang.srt.mem_cache.base_cache import BasePrefixCache


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

    def cache_req(
        self, rid, token_ids, req_pool_idx, del_in_memory_pool=True, **kwargs
    ):
        indices = self.req_to_token_pool.req_to_token[req_pool_idx, : len(token_ids)]
        if del_in_memory_pool:
            assert rid in self.entries
            self.req_to_token_pool.free(req_pool_idx)
            self.token_to_kv_pool.free(indices)
            return

        if rid not in self.entries:
            self.entries[rid] = ChunkCacheEntry(rid, indices)

        entry = self.entries[rid]
        entry.value = indices
        return indices, entry

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
