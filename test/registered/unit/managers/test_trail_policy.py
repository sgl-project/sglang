"""Unit tests for TRAIL scheduling policies and preemption logic."""
import dataclasses
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_batch import Req, TrailState
from sglang.srt.managers.schedule_policy import SchedulePolicy


def make_req(rid, output_ids=None, trail_state=None):
    """Create a minimal Req-like object for testing TRAIL policies."""
    req = MagicMock(spec=Req)
    req.rid = rid
    req.output_ids = output_ids or []
    req.trail_state = trail_state
    return req


def make_trail_state(initial=100.0, remaining=50.0, count=1):
    return TrailState(
        initial_predicted_len=initial,
        current_predicted_remaining=remaining,
        prediction_count=count,
    )


class TestTrailSorting(unittest.TestCase):
    """Test TRAIL policy sort methods produce correct ordering."""

    def test_sjf_sorts_by_initial_predicted_len(self):
        r1 = make_req("a", trail_state=make_trail_state(initial=200))
        r2 = make_req("b", trail_state=make_trail_state(initial=50))
        r3 = make_req("c", trail_state=make_trail_state(initial=100))
        queue = [r1, r2, r3]
        SchedulePolicy._sort_by_trail_sjf(queue)
        self.assertEqual([r.rid for r in queue], ["b", "c", "a"])

    def test_sjf_no_prediction_goes_last(self):
        r1 = make_req("a", trail_state=make_trail_state(initial=100))
        r2 = make_req("b")  # No trail_state
        queue = [r2, r1]
        SchedulePolicy._sort_by_trail_sjf(queue)
        self.assertEqual([r.rid for r in queue], ["a", "b"])

    def test_sprpt_sorts_by_current_remaining(self):
        r1 = make_req("a", trail_state=make_trail_state(remaining=80))
        r2 = make_req("b", trail_state=make_trail_state(remaining=20))
        r3 = make_req("c", trail_state=make_trail_state(remaining=50))
        queue = [r1, r2, r3]
        SchedulePolicy._sort_by_trail_sprpt(queue)
        self.assertEqual([r.rid for r in queue], ["b", "c", "a"])

    def test_lrpsprpt_unpreemptable_requests_first(self):
        # r1: generated 90/100 tokens (90% > 80% threshold) → un-preemptable
        r1 = make_req("a", output_ids=[0]*90,
                       trail_state=make_trail_state(initial=100, remaining=10))
        # r2: generated 10/100 tokens → preemptable, remaining=90
        r2 = make_req("b", output_ids=[0]*10,
                       trail_state=make_trail_state(initial=100, remaining=90))
        # r3: generated 50/100 tokens → preemptable, remaining=50
        r3 = make_req("c", output_ids=[0]*50,
                       trail_state=make_trail_state(initial=100, remaining=50))
        queue = [r2, r3, r1]
        SchedulePolicy._sort_by_trail_lrpsprpt(queue, preemption_threshold=0.8)
        # Un-preemptable (r1) first, then by remaining: r3(50) < r2(90)
        self.assertEqual([r.rid for r in queue], ["a", "c", "b"])

    def test_lrpsprpt_threshold_boundary(self):
        # Exactly at 80% boundary — should be un-preemptable
        r1 = make_req("at_boundary", output_ids=[0]*80,
                       trail_state=make_trail_state(initial=100, remaining=20))
        # Just below 80% — still preemptable
        r2 = make_req("below", output_ids=[0]*79,
                       trail_state=make_trail_state(initial=100, remaining=21))
        queue = [r2, r1]
        SchedulePolicy._sort_by_trail_lrpsprpt(queue, preemption_threshold=0.8)
        # At boundary (80/100 = 0.8, 0.8 > 0.8 is False) → preemptable
        # Actually 80 > 80 is False, so at_boundary is preemptable too
        # Need > threshold, not >=
        # Both preemptable, sort by remaining: at_boundary(20) < below(21)
        self.assertEqual([r.rid for r in queue], ["at_boundary", "below"])

    def test_lrpsprpt_past_threshold_is_unpreemptable(self):
        # 81 > 0.8*100=80 → un-preemptable
        r1 = make_req("past", output_ids=[0]*81,
                       trail_state=make_trail_state(initial=100, remaining=19))
        r2 = make_req("early", output_ids=[0]*10,
                       trail_state=make_trail_state(initial=100, remaining=5))
        queue = [r2, r1]
        SchedulePolicy._sort_by_trail_lrpsprpt(queue, preemption_threshold=0.8)
        # r1 un-preemptable (sorted first), r2 preemptable
        self.assertEqual([r.rid for r in queue], ["past", "early"])

    def test_lsprpt_uses_50_percent_threshold(self):
        # 51 > 0.5*100=50 → un-preemptable
        r1 = make_req("past50", output_ids=[0]*51,
                       trail_state=make_trail_state(initial=100, remaining=49))
        r2 = make_req("early", output_ids=[0]*10,
                       trail_state=make_trail_state(initial=100, remaining=90))
        queue = [r2, r1]
        SchedulePolicy._sort_by_trail_lsprpt(queue, preemption_threshold=0.5)
        self.assertEqual([r.rid for r in queue], ["past50", "early"])

    def test_rpsprpt_same_as_sprpt(self):
        # RPSPRPT is refined SPRPT without preemption limit — just sorts by remaining
        r1 = make_req("a", trail_state=make_trail_state(remaining=100))
        r2 = make_req("b", trail_state=make_trail_state(remaining=10))
        queue = [r1, r2]
        SchedulePolicy._sort_by_trail_rpsprpt(queue)
        self.assertEqual([r.rid for r in queue], ["b", "a"])


class TestTrailCompare(unittest.TestCase):
    """Test trail_compare for preemption decisions."""

    def test_preempt_when_waiting_has_shorter_remaining(self):
        waiting = make_req("w", output_ids=[0]*5,
                           trail_state=make_trail_state(initial=50, remaining=20))
        running = make_req("r", output_ids=[0]*5,
                           trail_state=make_trail_state(initial=200, remaining=150))
        result = SchedulePolicy.trail_compare(waiting, running)
        self.assertGreater(result, 0, "Should preempt: waiting has shorter remaining")

    def test_no_preempt_when_running_has_shorter_remaining(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=150))
        running = make_req("r", output_ids=[0]*5,
                           trail_state=make_trail_state(initial=200, remaining=20))
        result = SchedulePolicy.trail_compare(waiting, running)
        self.assertLessEqual(result, 0, "Should not preempt: running has shorter remaining")

    def test_no_preempt_when_running_is_unpreemptable(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=5))
        # Running has generated 85/100 = 85% > 80% threshold
        running = make_req("r", output_ids=[0]*85,
                           trail_state=make_trail_state(initial=100, remaining=15))
        result = SchedulePolicy.trail_compare(waiting, running, preemption_threshold=0.8)
        self.assertLessEqual(result, 0, "Should not preempt: running past 80% threshold")

    def test_no_preempt_when_waiting_has_no_prediction_short_running(self):
        waiting = make_req("w")  # No trail_state
        running = make_req("r", output_ids=[0]*5,
                           trail_state=make_trail_state(initial=200, remaining=150))
        result = SchedulePolicy.trail_compare(waiting, running)
        # With remaining=150 < 256 threshold, don't preempt
        self.assertLessEqual(result, 0, "Should not preempt: running remaining not clearly long")

    def test_preempt_when_waiting_has_no_prediction_long_running(self):
        waiting = make_req("w")  # No trail_state
        running = make_req("r", output_ids=[0]*5,
                           trail_state=make_trail_state(initial=500, remaining=400))
        result = SchedulePolicy.trail_compare(waiting, running)
        # With remaining=400 > 256, allow preemption for new request
        self.assertGreater(result, 0, "Should preempt: running has clearly long remaining")

    def test_no_preempt_when_running_has_no_prediction(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=10))
        running = make_req("r")  # No trail_state (just started)
        result = SchedulePolicy.trail_compare(waiting, running)
        self.assertLessEqual(result, 0, "Should not preempt: running has no prediction yet")

    def test_no_preempt_when_equal_remaining(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=50))
        running = make_req("r", output_ids=[0]*5,
                           trail_state=make_trail_state(initial=200, remaining=50))
        result = SchedulePolicy.trail_compare(waiting, running)
        self.assertLessEqual(result, 0, "Should not preempt when equal remaining")

    def test_custom_threshold(self):
        waiting = make_req("w", trail_state=make_trail_state(remaining=5))
        # Running at 55% of initial — below 0.5 threshold but above 0.8
        running = make_req("r", output_ids=[0]*55,
                           trail_state=make_trail_state(initial=100, remaining=45))
        # With threshold=0.5, 55 > 50 → un-preemptable
        result = SchedulePolicy.trail_compare(waiting, running, preemption_threshold=0.5)
        self.assertLessEqual(result, 0, "Should not preempt: past 50% threshold")
        # With threshold=0.8, 55 < 80 → preemptable
        result = SchedulePolicy.trail_compare(waiting, running, preemption_threshold=0.8)
        self.assertGreater(result, 0, "Should preempt: below 80% threshold, waiting shorter")


class TestTrailState(unittest.TestCase):
    """Test TrailState dataclass behavior."""

    def test_trail_state_defaults(self):
        ts = TrailState()
        self.assertEqual(ts.initial_predicted_len, 0.0)
        self.assertEqual(ts.current_predicted_remaining, 0.0)
        self.assertEqual(ts.prediction_count, 0)

    def test_trail_state_update(self):
        ts = TrailState(initial_predicted_len=100, current_predicted_remaining=100, prediction_count=1)
        ts.current_predicted_remaining = 50
        ts.prediction_count += 1
        self.assertEqual(ts.current_predicted_remaining, 50)
        self.assertEqual(ts.prediction_count, 2)
        # Initial should not change
        self.assertEqual(ts.initial_predicted_len, 100)

    def test_trail_state_survives_dataclass_copy(self):
        ts = TrailState(initial_predicted_len=100, current_predicted_remaining=75, prediction_count=3)
        ts2 = dataclasses.replace(ts)
        self.assertEqual(ts2.initial_predicted_len, 100)
        self.assertEqual(ts2.current_predicted_remaining, 75)
        self.assertEqual(ts2.prediction_count, 3)


if __name__ == "__main__":
    unittest.main()
