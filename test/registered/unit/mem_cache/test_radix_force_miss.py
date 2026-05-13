"""Unit tests for SGLANG_RADIX_FORCE_MISS.

The flag is gated at the scheduler boundary, so we test the helper directly
plus an end-to-end check of `match_prefix_for_req` driving a populated
RadixCache.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest
import unittest.mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
    MatchResult,
    zero_match_result,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


class _StubReq:
    def __init__(self, token_ids):
        self.origin_input_ids = list(token_ids)
        self.output_ids = []
        self.extra_key = None
        self.prefix_indices = None
        self.last_node = None
        self.last_host_node = None
        self.host_hit_length = None
        self.mamba_branching_seqlen = None
        self.cache_protected_len = None


class TestZeroMatchResult(unittest.TestCase):
    def test_zero_replaces_indices_and_nodes(self):
        tree = RadixCache.create_simulated()
        tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3, 4, 5])))
        match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 9]))
        )
        self.assertGreater(len(match.device_indices), 0)
        zeroed = zero_match_result(tree, match)
        self.assertEqual(int(zeroed.device_indices.numel()), 0)
        self.assertIs(zeroed.last_device_node, tree.root_node)
        self.assertIs(zeroed.last_host_node, tree.root_node)
        self.assertEqual(zeroed.host_hit_length, 0)
        # dtype/device preserved (slice-not-allocate).
        self.assertEqual(zeroed.device_indices.dtype, match.device_indices.dtype)
        self.assertEqual(zeroed.device_indices.device, match.device_indices.device)

    def test_chunk_cache_is_passthrough(self):
        class _StubChunkCache:
            def is_chunk_cache(self) -> bool:
                return True

        original = MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            host_hit_length=0,
        )
        self.assertIs(zero_match_result(_StubChunkCache(), original), original)


class TestMatchPrefixForReqForceMiss(unittest.TestCase):
    def test_force_miss_zeros_req_prefix(self):
        tree = RadixCache.create_simulated()
        tree.insert(
            InsertParams(key=RadixKey(token_ids=[10, 11, 12, 13, 14, 15, 16, 17]))
        )

        # Sanity: without the flag, the same lookup hits.
        baseline_req = _StubReq([10, 11, 12, 13, 99, 100])
        with envs.SGLANG_RADIX_FORCE_MISS.override(False):
            match_prefix_for_req(tree, baseline_req)
        self.assertGreater(int(baseline_req.prefix_indices.numel()), 0)
        self.assertIsNot(baseline_req.last_node, tree.root_node)

        # With the flag, the same lookup is forced to miss.
        forced_req = _StubReq([10, 11, 12, 13, 99, 100])
        with envs.SGLANG_RADIX_FORCE_MISS.override(True):
            match_prefix_for_req(tree, forced_req)
        self.assertEqual(int(forced_req.prefix_indices.numel()), 0)
        self.assertIs(forced_req.last_node, tree.root_node)
        self.assertIs(forced_req.last_host_node, tree.root_node)
        self.assertEqual(forced_req.host_hit_length, 0)


if __name__ == "__main__":
    unittest.main()
