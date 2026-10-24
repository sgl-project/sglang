from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
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
