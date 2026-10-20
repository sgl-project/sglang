from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    node_children,
    node_device_value,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures


class UnifiedRadixCacheHiCacheFixtures(UnifiedRadixCacheFixtures):
    def _build_internal_mamba_fixture(self, write_policy):
        """HiCache fixture where seq_a's node is INTERNAL (seq_b splits a
        suffix off it) and holds its own mamba state."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy=write_policy)
        seq_a = self._make_seq(1, 2)
        seq_b = seq_a + self._make_seq(1000, 1)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_b)
        return cache, req_to_token_pool, seq_a

    def _build_internal_swa_fixture(self, write_policy):
        """HiCache fixture where seq_a's node is INTERNAL and its child spans
        the sliding window, so the leaf backup walk cannot cover it."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy=write_policy)
        ps = self.cfg.page_size
        tail_pages = max(1, -(-self.cfg.sliding_window_size // ps))
        seq_a = self._make_seq(1, 2)
        seq_b = seq_a + self._make_seq(1000, tail_pages)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_b)
        return cache, req_to_token_pool, seq_a, seq_b

    def _demote_subtree(self, cache, node):
        """Demote every descendant of ``node`` D->H, children before parents.

        Each descendant is backed up by simulation: the real backup needs a host
        pool, which the HiCache fixture lacks for SWA + Mamba stacks.
        """

        def descendants(node_id):
            out = []
            for child_id in node_children(cache, node_id):
                out.extend(descendants(child_id))
                out.append(child_id)
            return out

        nodes = descendants(node)
        self.assertGreater(len(nodes), 0)
        for desc in nodes:
            self._simulate_backup(cache, desc)
            self.assertTrue(
                cache.tree_core.is_backuped(desc), "desc must be backuped before demote"
            )
            result = cache.tree_core.demote(desc)
            cache._free_values(result.device_frees, result.host_frees)
            self.assertTrue(
                cache.tree_core.is_full_device_evicted(desc),
                "desc should be D->H demoted",
            )
            self.assertIsNone(node_device_value(cache, desc, ComponentType.FULL))
