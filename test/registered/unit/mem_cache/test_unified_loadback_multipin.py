"""Deterministic repro + fix validation for the load-back multi-pin race.

Scenario (matches the production crash `node X pinned by load-back Y, new anchor X`):
a first load-back anchored at a descendant pins an evicted ancestor via its Full-KV
chain; before that load-back acks, a second load-back anchors AT the ancestor to
restore its independently-evicted mamba state. The single-valued
`load_back_pending_id` cannot represent both live pins and asserts.

Run from test/registered/unit/mem_cache/:
  python3 -m unittest test_unified_loadback_multipin -v
UNPATCHED: the test errors with AssertionError("... pinned by load-back ...").
PATCHED (load_back_pending_ids set): the test passes, pins drain to empty.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_unified_radix_cache_unittest as U


# Inherit the full suite for its fixture helpers and setUp state; invoke ONLY
# test_overlapping_load_back_pins (the inherited suite tests run elsewhere):
#   python3 -m unittest \
#     test_unified_loadback_multipin.TestConcurrentLoadBackPins.test_overlapping_load_back_pins
class TestConcurrentLoadBackPins(U.UnifiedRadixCacheSuite, U.CustomTestCase):
    cfg = U.CacheConfig(
        page_size=1, components=(U.ComponentType.FULL, U.ComponentType.MAMBA)
    )

    def test_overlapping_load_back_pins(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(cache, allocator, req_to_token_pool, 4)
        self.assertGreaterEqual(len(chain), 2, "need a >=2 node chain")

        self._backup_tree(cache)
        total = sum(len(n.key) for n in chain)
        cache.evict(U.EvictParams(num_tokens=total + 16, mamba_num=64))

        leaf = chain[-1]
        self.assertTrue(leaf.evicted, "leaf must be device-evicted")

        # An ancestor that is FULL-evicted (so the leaf anchor's KV chain pins
        # it) and mamba host-only (so a second anchor can target it).
        MAMBA = U.ComponentType.MAMBA
        ancestor = None
        for cand in chain[:-1]:
            cd = cand.component_data[MAMBA]
            if cand.evicted and cd.value is None and cd.host_value is not None:
                ancestor = cand
                break
        self.assertIsNotNone(
            ancestor,
            "fixture produced no FULL-evicted + mamba-host-only ancestor; "
            f"chain state: {[(n.id, n.evicted, n.component_data[MAMBA].value is not None, n.component_data[MAMBA].host_value is not None) for n in chain]}",
        )

        # Load-back 1: anchored at the leaf; its Full-KV chain covers the
        # ancestor. Do NOT process acks — the pin stays live.
        self.assertTrue(cache.load_back(leaf.id))

        # Load-back 2: anchored at the still-pinned ancestor for its mamba
        # state. Unpatched code dies here with
        # AssertionError: node <ancestor> pinned by load-back <leaf>, new anchor <ancestor>
        self.assertTrue(cache.load_back(ancestor.id))

        # Both DMAs ack; every pin must drain.
        self._finish_pending_loads(cache)
        for node in chain:
            self.assertEqual(
                node.load_back_pending_ids,
                set(),
                f"node {node.id} kept stale pins {node.load_back_pending_ids}",
            )
        self._release_ongoing_load_back_locks(cache)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
