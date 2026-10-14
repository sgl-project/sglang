"""Unit tests for SGLANG_RADIX_FORCE_MISS.

The flag is gated at the scheduler boundary, so we test the helper directly
plus an end-to-end check of `match_kv_cache` driving a populated
RadixCache.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
    MatchResult,
    zero_match_result,
)
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams


class _StubReq:
    def __init__(self, token_ids):
        self.origin_input_ids = array("q", token_ids)
        self.output_ids = array("q")
        self.extra_key = None
        self.cache_salt = None
        self.prefix_len = 0
        self.last_node = None
        self.last_host_node = None
        self.best_match_node = None
        self.host_hit_length = None
        self.num_matched_prefix_tokens = 0
        self.swa_branching_seqlen = None
        self.kv = SimpleNamespace(cache_protected_len=None)

    def _compute_max_prefix_len(self, input_len):
        return max(input_len - 1, 0)


class TestZeroMatchResult(unittest.TestCase):
    def test_zero_replaces_indices_and_nodes(self):
        tree = RadixCache.create_simulated()
        tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 4, 5]))))
        match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 9])))
        )
        self.assertGreater(match.device_prefix_len, 0)
        zeroed = zero_match_result(tree, match)
        self.assertEqual(int(zeroed.device_prefix_len), 0)
        self.assertIs(zeroed.last_device_node, tree.root_node)
        self.assertIs(zeroed.last_host_node, tree.root_node)
        self.assertIs(zeroed.best_match_node, tree.root_node)
        self.assertEqual(zeroed.host_hit_length, 0)

    def test_non_sharing_cache_is_passthrough(self):
        class _StubNonSharingCache:
            def supports_prefix_sharing(self) -> bool:
                return False

        original = MatchResult(
            device_prefix_len=0,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            host_hit_length=0,
        )
        self.assertIs(zero_match_result(_StubNonSharingCache(), original), original)


class TestMatchKvCacheForceMiss(unittest.TestCase):
    def test_swa_branching_seqlen_is_cleared_without_new_branch(self):
        class _StubTreeCache:
            def swa_reprefill_tail_tokens(self):
                return 0

            def match_prefix(self, params):
                return MatchResult(
                    device_prefix_len=0,
                    last_device_node=None,
                    last_host_node=None,
                    best_match_node=None,
                    host_hit_length=0,
                    swa_branching_seqlen=None,
                )

        req = _StubReq([1, 2, 3, 4])
        req.swa_branching_seqlen = 8

        match_kv_cache(req, _StubTreeCache())

        self.assertIsNone(req.swa_branching_seqlen)

    def test_force_miss_zeros_req_prefix(self):
        tree = RadixCache.create_simulated()
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", [10, 11, 12, 13, 14, 15, 16, 17]))
            )
        )

        # Sanity: without the flag, the same lookup hits.
        baseline_req = _StubReq([10, 11, 12, 13, 99, 100])
        with envs.SGLANG_RADIX_FORCE_MISS.override(False):
            match_kv_cache(baseline_req, tree)
        self.assertGreater(baseline_req.prefix_len, 0)
        self.assertIsNot(baseline_req.last_node, tree.root_node)

        # With the flag, the same lookup is forced to miss.
        forced_req = _StubReq([10, 11, 12, 13, 99, 100])
        with envs.SGLANG_RADIX_FORCE_MISS.override(True):
            match_kv_cache(forced_req, tree)
        self.assertEqual(forced_req.prefix_len, 0)
        self.assertIs(forced_req.last_node, tree.root_node)
        self.assertIs(forced_req.last_host_node, tree.root_node)
        self.assertEqual(forced_req.host_hit_length, 0)
        self.assertEqual(forced_req.full_kv_hit_length, 0)
        self.assertEqual(forced_req.prefix_swa_truncated_tokens, 0)


class TestSWAPrefixMetrics(unittest.TestCase):
    def make_req(self):
        return _StubReq(range(100))

    def run_match(self, tree, req):
        match_kv_cache(req, tree)

    def match(
        self,
        *,
        full=90,
        device=40,
        host=20,
        swa=True,
        mamba=False,
        ring=0,
        limit=99,
        req=None,
    ):
        tree = unittest.mock.Mock()
        tree.swa_reprefill_tail_tokens.return_value = ring
        tree.supports_swa.return_value = swa
        tree.supports_mamba.return_value = mamba
        tree.match_prefix.return_value = MatchResult(
            device_prefix_len=device,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            host_hit_length=host,
            full_kv_hit_length=full,
        )
        req = req or self.make_req()
        req._compute_max_prefix_len = lambda _: limit
        with envs.SGLANG_RADIX_FORCE_MISS.override(False):
            self.run_match(tree, req)
        return req

    def test_counts_gpu_and_host_coverage(self):
        req = self.match()
        self.assertEqual(req.full_kv_hit_length, 90)
        self.assertEqual(req.prefix_swa_truncated_tokens, 30)
        self.assertEqual(self.match(host=50).prefix_swa_truncated_tokens, 0)

    def test_same_cap_for_full_and_swa_matches(self):
        req = self.match(limit=70)
        self.assertEqual(req.full_kv_hit_length, 70)
        self.assertEqual(req.prefix_swa_truncated_tokens, 10)
        self.assertEqual(self.match(limit=30).prefix_swa_truncated_tokens, 0)

    def test_excludes_other_constraints(self):
        for kwargs in ({"swa": False}, {"mamba": True}, {"ring": 512}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.match(**kwargs).prefix_swa_truncated_tokens, 0)

    def test_refresh_before_admission_freeze_after_first_chunk(self):
        req = self.match()
        self.match(full=80, req=req)
        self.assertEqual(req.full_kv_hit_length, 80)
        self.assertEqual(req.prefix_swa_truncated_tokens, 20)
        req._cache_breakdown_computed = True
        self.match(full=99, host=0, req=req)
        self.assertEqual(req.full_kv_hit_length, 80)
        self.assertEqual(req.prefix_swa_truncated_tokens, 20)


class TestAdmissionSWAPrefixMetrics(TestSWAPrefixMetrics):
    """FCFS admission must collect metrics without a scheduling-policy match."""

    def make_req(self):
        return Req("metrics", "", array("q", range(100)), SamplingParams())

    def run_match(self, tree, req):
        req.init_next_round_input(tree)

    def test_admission_snapshot_reaches_counter_and_log(self):
        req = self.match()
        collector = unittest.mock.Mock()
        req.record_prefix_swa_metrics(60, collector)
        collector.increment_prefix_swa_truncated_tokens.assert_called_once_with(30)
        with unittest.mock.patch(
            "sglang.srt.managers.schedule_batch.logger.info"
        ) as log:
            req.log_time_stats()
        self.assertIn("full_kv_hit_len=90", log.call_args.args[0])
        self.assertIn("swa_truncated_tokens=30", log.call_args.args[0])

    def test_retraction_does_not_replace_snapshot(self):
        req = self.match()
        req.retracted_stain = True
        self.match(full=99, host=0, req=req)
        self.assertEqual(req.full_kv_hit_length, 90)
        self.assertEqual(req.prefix_swa_truncated_tokens, 30)

    def test_force_miss_clears_admission_snapshot(self):
        tree = RadixCache.create_simulated()
        tree.insert(InsertParams(key=RadixKey(array("q", range(80)))))
        req = self.make_req()
        req.full_kv_hit_length = 90
        req.prefix_swa_truncated_tokens = 30
        with envs.SGLANG_RADIX_FORCE_MISS.override(True):
            req.init_next_round_input(tree)
        self.assertEqual(req.prefix_len, 0)
        self.assertEqual(req.full_kv_hit_length, 0)
        self.assertEqual(req.prefix_swa_truncated_tokens, 0)


if __name__ == "__main__":
    unittest.main()
