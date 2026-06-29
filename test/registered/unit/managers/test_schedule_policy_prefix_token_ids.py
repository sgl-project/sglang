import unittest

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CacheAwarePolicy,
    SchedulePolicy,
    _prefix_match_token_ids,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
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
    expected priority order across the helper's two branches:

    - fresh requests (``output_ids == []``) take the no-copy branch and the
      cache lookup uses ``origin_input_ids`` directly;
    - retracted requests (``output_ids != []``) take the concat branch and
      the cache lookup must include the generated tokens.
    """

    def setUp(self):
        # Pre-populate the radix cache with [100, 200, 300, 400] so the three
        # requests below land at distinct match lengths and the LPM order is
        # determined by `len(prefix_indices)` rather than stable-sort fallback.
        self.tree_cache = RadixCache.create_simulated()
        self.tree_cache.insert(
            InsertParams(
                key=RadixKey(token_ids=[100, 200, 300, 400]),
                value=torch.arange(4, dtype=torch.int64),
            )
        )

    def _build_policy(self):
        return SchedulePolicy(
            policy="lpm",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def test_lpm_order_with_mixed_output_state(self):
        # rid=1 (req_long): no-copy branch; origin alone yields a 2-token match.
        req_long = Req(1, "p", [100, 200, 999], SamplingParams())
        # rid=2 (req_retracted): concat branch; the 4-token match only exists
        # because output_ids contributes the trailing 400, proving output_ids
        # really is included in the prefix-match key.
        req_retracted = Req(2, "p", [100, 200, 300], SamplingParams())
        req_retracted.output_ids = [400, 999]
        # rid=3 (req_short): no-copy branch; matches a 1-token prefix only.
        req_short = Req(3, "p", [100], SamplingParams())

        waiting_queue = [req_long, req_retracted, req_short]
        policy = self._build_policy()
        self.assertEqual(policy.policy, CacheAwarePolicy.LPM)

        policy.calc_priority(waiting_queue)

        # Match lengths witnessed during _compute_prefix_matches.
        self.assertEqual(req_retracted.prefix_indices.numel(), 4)
        self.assertEqual(req_long.prefix_indices.numel(), 2)
        self.assertEqual(req_short.prefix_indices.numel(), 1)

        # LPM sorts by descending match length, so the queue order is the
        # exact list below -- not just a set membership check.
        self.assertEqual(
            [r.rid for r in waiting_queue],
            [req_retracted.rid, req_long.rid, req_short.rid],
        )


if __name__ == "__main__":
    unittest.main()
