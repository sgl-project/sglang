"""match_kv_cache records how deep the Full KV was cached on the request.

The prefill adder counts the tokens between that depth and the reusable
prefix as state misses (sglang:prefill_state_miss_tokens_total).
"""

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StubReq:
    def __init__(self, num_tokens):
        self.origin_input_ids = array("q", range(num_tokens))
        self.output_ids = array("q")
        self.extra_key = None
        self.cache_salt = None
        self.prefix_len = 0
        self.last_node = None
        self.last_host_node = None
        self.best_match_node = None
        self.host_hit_length = None
        self.num_matched_prefix_tokens = 0
        self.full_kv_hit_length = 0
        self.swa_branching_seqlen = None
        self.kv = SimpleNamespace(cache_protected_len=None)

    def _compute_max_prefix_len(self, input_len):
        return max(input_len - 1, 0)


class _StubTreeCache:
    """Full KV is cached 256 tokens deep; every component only 64 deep."""

    def swa_reprefill_tail_tokens(self, req=None):
        return 0

    def match_prefix(self, params):
        return MatchResult(
            device_prefix_len=64,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            host_hit_length=0,
            full_kv_hit_length=256,
        )


class TestMatchKvCacheFullKvHit(CustomTestCase):
    def test_full_kv_hit_length_is_recorded(self):
        req = _StubReq(num_tokens=1000)
        match_kv_cache(req, _StubTreeCache())
        self.assertEqual(req.num_matched_prefix_tokens, 64)
        self.assertEqual(req.full_kv_hit_length, 256)

    def test_full_kv_hit_length_is_capped_like_the_matched_prefix(self):
        # 100 tokens: at most 99 can come from the cache.
        req = _StubReq(num_tokens=100)
        match_kv_cache(req, _StubTreeCache())
        self.assertEqual(req.full_kv_hit_length, 99)


if __name__ == "__main__":
    unittest.main()
