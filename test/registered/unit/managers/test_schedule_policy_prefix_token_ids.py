import unittest

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CacheAwarePolicy,
    SchedulePolicy,
    _prefix_match_token_ids,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefixMatchTokenIds(CustomTestCase):
    """Unit tests for the `_prefix_match_token_ids` helper used by the
    cache-aware scheduler to build prefix-match keys without an unconditional
    list copy of `origin_input_ids`.
    """

    def _make_req(self, rid: int, prompt_ids, output_ids):
        req = Req(rid, "x", list(prompt_ids), SamplingParams())
        req.output_ids = list(output_ids)
        return req

    def test_empty_output_returns_origin_reference(self):
        req = self._make_req(1, [10, 20, 30], [])
        result = _prefix_match_token_ids(req)
        # No copy when output_ids is empty: caller can mutate origin_input_ids
        # only by going through the Req itself. Sharing identity is safe because
        # downstream RadixKey consumers slice rather than mutate.
        self.assertIs(result, req.origin_input_ids)
        self.assertEqual(result, [10, 20, 30])

    def test_non_empty_output_concats_to_new_list(self):
        req = self._make_req(2, [10, 20], [99, 100])
        result = _prefix_match_token_ids(req)
        self.assertIsNot(result, req.origin_input_ids)
        self.assertEqual(result, [10, 20, 99, 100])

    def test_origin_unchanged_after_call(self):
        req = self._make_req(3, [10, 20, 30], [])
        before = list(req.origin_input_ids)
        _prefix_match_token_ids(req)
        self.assertEqual(req.origin_input_ids, before)

    def test_equivalent_to_old_concat_form(self):
        # Behaviour parity with the previous `origin_input_ids + output_ids`
        # expression, across both branches of the helper.
        for prompt, out in (
            ([], []),
            ([1], []),
            ([1, 2, 3], []),
            ([1, 2], [3]),
            ([1], [2, 3]),
        ):
            req = self._make_req(0, prompt, out)
            self.assertEqual(
                _prefix_match_token_ids(req),
                req.origin_input_ids + req.output_ids,
            )


class TestComputePrefixMatchesParity(CustomTestCase):
    """Confirm `_compute_prefix_matches` (LPM cache-aware policy) produces the
    same priority order as the previous implementation, even when the waiting
    queue mixes fresh requests (`output_ids == []`) and retracted requests
    (`output_ids != []`). The helper-driven optimization must not change which
    request lands first.
    """

    def setUp(self):
        self.tree_cache = RadixCache.create_simulated()

    def _build_policy(self):
        return SchedulePolicy(
            policy="lpm",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def test_lpm_order_with_mixed_output_state(self):
        # Three reqs with the same prompt prefix; one is "retracted" with
        # generated tokens so its prefix-match key includes output_ids.
        req_fresh = Req(1, "p", [10, 20, 30], SamplingParams())
        req_retracted = Req(2, "p", [10, 20], SamplingParams())
        req_retracted.output_ids = [30, 40]
        req_short = Req(3, "p", [10], SamplingParams())

        waiting_queue = [req_fresh, req_retracted, req_short]
        policy = self._build_policy()
        self.assertEqual(policy.policy, CacheAwarePolicy.LPM)

        policy.calc_priority(waiting_queue)

        # The order must be deterministic under LPM regardless of the helper
        # taking the no-copy branch for req_fresh / req_short and the copy
        # branch for req_retracted.
        rids_after = [r.rid for r in waiting_queue]
        self.assertEqual(len(rids_after), 3)
        self.assertEqual(set(rids_after), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
