import unittest
from array import array
from unittest.mock import Mock, patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    WAITING_PREFIX_REFRESH_INTERVAL,
    SchedulePolicy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_requests(n):
    return [
        Req(str(i), "", array("q", [i + 100, 9]), SamplingParams()) for i in range(n)
    ]


class TestBoundedLPMV2(CustomTestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        self.p = SchedulePolicy(
            "lpm", RadixCache.create_simulated(), False, False, False
        )
        self.p._should_start_bounded_batch = lambda waiting: True
        self.match = patch.object(
            self.p, "_compute_prefix_matches", return_value=set()
        ).start()
        self.addCleanup(patch.stopall)

    def ordered(self, q):
        return self.p.calc_priority(q)

    def pass_with_skips(self, q, skipped):
        admitted = []
        for req in self.p.admission_candidates(q):
            self.assertTrue(self.p.allows_admission(req))
            if req in skipped:
                self.p.mark_admission_skipped(req)
                continue
            admitted.append(req)
        self.p.retire_waiting_members(admitted)
        return admitted

    def test_no_skip_never_opens_second_batch(self):
        q = self.ordered(make_requests(200))
        self.assertEqual(self.pass_with_skips(q, set()), q[:64])
        self.assertFalse(self.p._bounded_next_selected)
        self.assertEqual(self.match.call_count, 1)

    def test_skip_opens_only_one_fixed_batch_and_retraction_cannot_refill(self):
        q = self.ordered(make_requests(200))
        blocked = q[0]
        admitted = self.pass_with_skips(q, {blocked})
        self.assertEqual(admitted, q[1:128])
        retracted = q[64]
        remaining = self.ordered([blocked] + q[128:] + [retracted])
        self.assertEqual(self.pass_with_skips(remaining, {blocked}), [])
        self.assertFalse(self.p.allows_admission(retracted))
        self.assertTrue(self.p._bounded_next_selected)
        self.assertEqual(self.p._bounded_next_members, [])

    def test_break_before_boundary_does_not_match_or_open_next_batch(self):
        q = self.ordered(make_requests(200))
        candidates = self.p.admission_candidates(q)
        self.p.mark_admission_skipped(next(candidates))
        next(candidates)  # this candidate does not fit; scheduler breaks here
        candidates.close()
        self.assertFalse(self.p._bounded_next_selected)
        self.assertEqual(self.match.call_count, 1)

    def test_ready_member_precedes_next_batch_and_resets_skip_state(self):
        q = self.ordered(make_requests(130))
        candidates = self.p.admission_candidates(q)
        for _ in range(64):
            self.p.mark_admission_skipped(next(candidates))
        self.assertIs(next(candidates), q[64])
        candidates.close()
        q = self.ordered(q)
        self.assertEqual(self.pass_with_skips(q, set()), q[:64])
        self.assertFalse(self.p.allows_admission(q[64]))

    def test_next_batch_survivors_promote_without_retracted_members(self):
        q = self.ordered(make_requests(200))
        candidates = self.p.admission_candidates(q)
        for _ in range(64):
            self.p.mark_admission_skipped(next(candidates))
        next(candidates)
        candidates.close()
        self.p.retire_waiting_members(q[64:100])
        self.p.retire_waiting_members(q[:64])
        self.ordered(q[100:] + q[64:100])
        self.assertEqual(self.p._bounded_members, q[100:128])
        self.assertFalse(self.p._bounded_next_selected)
        self.assertFalse(self.p.allows_admission(q[64]))

    def test_next_batch_cancel_does_not_replenish(self):
        q = self.ordered(make_requests(200))
        candidates = self.p.admission_candidates(q)
        for _ in range(64):
            self.p.mark_admission_skipped(next(candidates))
        next(candidates)
        candidates.close()
        self.p.retire_waiting_members(q[64:128])
        remaining = self.ordered(q[:64] + q[128:])
        self.assertEqual(self.pass_with_skips(remaining, set(q[:64])), [])

    def test_tail_refresh_cap_cadence_and_draining_batch(self):
        q = make_requests(200)
        with patch("sglang.srt.managers.schedule_policy.touch_waiting_prefix") as touch:
            self.ordered(q)
            self.assertEqual(
                [c.args[0] for c in touch.call_args_list], list(reversed(q[64:192]))
            )
            touch.reset_mock()
            for _ in range(WAITING_PREFIX_REFRESH_INTERVAL - 1):
                self.ordered(q)
            touch.assert_not_called()
            self.p.retire_waiting_members(q[:60])
            self.ordered(q[60:])
            self.assertEqual(
                [c.args[0] for c in touch.call_args_list], list(reversed(q[64:192]))
            )
            self.assertEqual(len(self.match.call_args.args[0]), 4)

    def test_tail_refresh_100_requests_and_no_double_match(self):
        q = make_requests(200)
        with patch("sglang.srt.managers.schedule_policy.touch_waiting_prefix") as touch:
            self.ordered(q)
            self.assertEqual(
                [c.args[0] for c in touch.call_args_list], list(reversed(q[64:192]))
            )
            self.assertEqual(self.match.call_count, 1)
            self.assertEqual(self.match.call_args.args[0], q[:64])

    def test_refresh_disabled_and_non_lru_keep_existing_behavior(self):
        for override in (
            {"radix_eviction_policy": "mru"},
            {"disaggregation_mode": "decode"},
        ):
            with (
                get_context().override_server_args(**override),
                patch(
                    "sglang.srt.managers.schedule_policy.touch_waiting_prefix"
                ) as touch,
            ):
                self.ordered(make_requests(200))
                touch.assert_not_called()

    def test_fast_cache_matches_tail_without_touch(self):
        q = make_requests(200)
        for req in q:
            req.init_next_round_input = Mock()
        with (
            patch.object(
                self.p.tree_cache, "supports_fast_match_prefix", return_value=True
            ),
            patch("sglang.srt.managers.schedule_policy.touch_waiting_prefix") as touch,
        ):
            self.ordered(q)
            touch.assert_not_called()
        for req in q[:64]:
            req.init_next_round_input.assert_not_called()
        for req in q[64:]:
            req.init_next_round_input.assert_called_once_with(self.p.tree_cache)

    def test_stock_candidates_unchanged(self):
        q = make_requests(200)
        for name in ("lpm", "fcfs", "hrrn"):
            p = SchedulePolicy(name, RadixCache.create_simulated(), False, False, False)
            p.mark_admission_skipped(q[0])
            self.assertEqual(list(p.admission_candidates(q)), q)


if __name__ == "__main__":
    unittest.main()
