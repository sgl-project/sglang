import unittest
from array import array

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Allocator:
    page_size = 1
    device = "cpu"

    def free_segment(self, value, start_pos=0):
        return None


class RadixLazyEvictionTest(unittest.TestCase):
    def test_queue_persists_and_refreshes_mru_between_calls(self):
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=_Allocator(),
                page_size=1,
                eviction_policy="mru",
            )
        )
        for token in (1, 2, 3):
            cache.insert(InsertParams(key=RadixKey(array("q", [token]))))
        queue = cache._eviction_heap

        cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(
            {node.key.token_ids[0] for node in cache.evictable_leaves}, {1, 2}
        )
        self.assertIs(cache._eviction_heap, queue)

        cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [1]))))
        cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(
            {node.key.token_ids[0] for node in cache.evictable_leaves}, {2}
        )

        cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [2]))))
        cache.evict(EvictParams(num_tokens=1))
        self.assertFalse(cache.evictable_leaves)

        result = cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(len(queue), 0)


if __name__ == "__main__":
    unittest.main()
