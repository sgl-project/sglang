"""
Accounting-contract tests for the classic RadixCache.

The running evictable_/protected_size_ counters must always agree with a
recomputation from the tree (see RadixCache.sanity_check). This guards the
class of drift reported in sgl-project/sglang#35270, where the idle
pool-invariant check tripped on an over-counted evictable size.

All tests run on CPU via create_simulated: no model, no GPU, deterministic
seeds. Usage:
    python -m pytest test/registered/unit/mem_cache/test_radix_cache_accounting.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import random
import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


def _make_cache(page_size=32):
    mock_allocator = MagicMock()
    mock_allocator.device = torch.device("cpu")
    return RadixCache.create_simulated(
        mock_allocator=mock_allocator, page_size=page_size
    )


def _key(ids):
    return RadixKey(array("q", ids))


class TestAccountingContract(unittest.TestCase):
    def test_empty_tree_is_balanced(self):
        cache = _make_cache()
        cache.sanity_check()
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)

    def test_insert_match_lock_unlock_evict_sequence(self):
        # One shared page-aligned prefix, a divergent tail, then release
        # everything and drain the tree: counters must return to zero.
        cache = _make_cache()
        shared = list(range(1024))
        cache.insert(InsertParams(key=_key(shared + list(range(1000, 1064)))))
        res = cache.match_prefix(MatchPrefixParams(key=_key(shared)))
        cache.inc_lock_ref(res.last_device_node)
        cache.sanity_check()
        cache.dec_lock_ref(res.last_device_node)
        cache.sanity_check()
        cache.evict(EvictParams(num_tokens=10**9))
        cache.sanity_check()
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)

    def test_split_during_hold_then_scrambled_release(self):
        # Lock a node, force a split inside the held region via a partial
        # match, then release and evict in a different order. The inherited
        # lock_refs on split halves must still balance exactly.
        cache = _make_cache()
        base = list(range(2048))
        cache.insert(InsertParams(key=_key(base)))
        res = cache.match_prefix(MatchPrefixParams(key=_key(base)))
        cache.inc_lock_ref(res.last_device_node)
        # partial match ends mid-node -> split
        cache.match_prefix(MatchPrefixParams(key=_key(base[:1024] + [9999] * 32)))
        cache.sanity_check()
        cache.dec_lock_ref(res.last_device_node)
        cache.sanity_check()
        cache.evict(EvictParams(num_tokens=10**9))
        cache.sanity_check()
        self.assertEqual(cache.evictable_size(), 0)

    def test_remask_style_duplicate_inserts(self):
        # Re-inserting identical page-aligned keys (e.g. diffusion remask
        # rewrites) must be accounting-neutral.
        cache = _make_cache()
        ids = list(range(4096))
        for _ in range(3):
            cache.insert(InsertParams(key=_key(ids)))
            cache.sanity_check()
        self.assertEqual(cache.evictable_size(), 4096)

    def test_sanity_check_catches_drift(self):
        # Prove the detector detects: corrupt the counter by one page and
        # the check must raise instead of staying green.
        cache = _make_cache()
        cache.insert(InsertParams(key=_key(list(range(128)))))
        cache.sanity_check()
        cache.evictable_size_ += 32
        with self.assertRaises(AssertionError):
            cache.sanity_check()

    def test_model_based_random_ops(self):
        # Paired lock/unlock, splits, evictions and duplicate inserts over
        # shared page-aligned prefixes. Deterministic seeds.
        for seed in (11, 22, 33):
            rng = random.Random(seed)
            cache = _make_cache()
            base = [rng.randrange(0, 60) for _ in range(4096)]
            held = []
            for _ in range(300):
                r = rng.random()
                if r < 0.35:
                    plen = rng.choice([0, 1024, 2048, 3072])
                    tail = [rng.randrange(0, 200) for _ in range(64)]
                    cache.insert(InsertParams(key=_key(base[:plen] + tail)))
                elif r < 0.55:
                    plen = rng.choice([0, 1024, 2048, 3072, 4096])
                    res = cache.match_prefix(
                        MatchPrefixParams(key=_key(base[:plen]))
                    )
                    node = res.last_device_node
                    if node is not None and node is not cache.root_node:
                        cache.inc_lock_ref(node)
                        held.append(node)
                elif r < 0.70 and held:
                    node = held.pop(rng.randrange(len(held)))
                    cache.dec_lock_ref(node)
                elif r < 0.85:
                    cache.evict(EvictParams(num_tokens=rng.choice([32, 128, 512])))
                else:
                    plen = rng.choice([1024, 2048])
                    cache.insert(InsertParams(key=_key(base[:plen])))
                cache.sanity_check()
            for node in held:
                cache.dec_lock_ref(node)
            cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
