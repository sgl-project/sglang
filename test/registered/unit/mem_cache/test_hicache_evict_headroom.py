"""SGLANG_HICACHE_EVICT_HEADROOM_TOKENS raises small write_back eviction requests."""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Captured(Exception):
    def __init__(self, params):
        super().__init__()
        self.params = params


class TestHiCacheEvictHeadroom(CustomTestCase):
    def _evict_params_seen(self, *, headroom, is_write_back=True, **params):
        cache = SimpleNamespace(disable=False, is_write_back=is_write_back)

        def capture(evict_params):
            raise _Captured(evict_params)

        # Stop right after the headroom decision, before any allocator access.
        cache._evict_request_by_type = capture
        with envs.SGLANG_HICACHE_EVICT_HEADROOM_TOKENS.override(headroom):
            with self.assertRaises(_Captured) as ctx:
                UnifiedRadixCache.evict_for_alloc(cache, EvictParams(**params))
        return ctx.exception.params

    def test_off_by_default(self):
        seen = self._evict_params_seen(headroom=0, num_tokens=100)
        self.assertEqual(seen.num_tokens, 100)

    def test_small_request_is_raised(self):
        seen = self._evict_params_seen(headroom=4096, num_tokens=100, swa_num_tokens=7)
        self.assertEqual(seen.num_tokens, 4096)
        self.assertEqual(seen.swa_num_tokens, 7)

    def test_large_request_is_kept(self):
        seen = self._evict_params_seen(headroom=4096, num_tokens=10000)
        self.assertEqual(seen.num_tokens, 10000)

    def test_no_full_token_request_is_kept(self):
        seen = self._evict_params_seen(headroom=4096, num_tokens=0, mamba_num=3)
        self.assertEqual(seen.num_tokens, 0)

    def test_only_with_write_back(self):
        seen = self._evict_params_seen(
            headroom=4096, is_write_back=False, num_tokens=100
        )
        self.assertEqual(seen.num_tokens, 100)


if __name__ == "__main__":
    unittest.main()
