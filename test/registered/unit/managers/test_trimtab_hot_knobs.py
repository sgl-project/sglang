"""Hot scheduler knobs applied through set_internal_state.

The knobs are plain scheduler state that the event loop reads on every step, so
these tests exercise the decision logic directly: which values are accepted,
what the scheduler is left holding afterwards, and what a rejection leaves
untouched. No server, no GPU, no model.
"""

import unittest

from sglang.srt.managers.scheduler import _trimtab_apply_hot_knobs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Scheduler:
    """The scheduler attributes the knob code reads and writes, and nothing else."""

    def __init__(self, max_running_requests=64, max_total_num_tokens=100_000):
        self.max_running_requests = max_running_requests
        self.max_queued_requests = 0
        self.chunked_prefill_size = 8192
        self.max_prefill_tokens = 16384
        self.schedule_policy = "fcfs"
        self.max_total_num_tokens = max_total_num_tokens


class TestTrimtabHotKnobs(CustomTestCase):
    def test_applies_every_knob_it_claims(self):
        s = _Scheduler()
        args = {
            "max_running_requests": 8,
            "max_queued_requests": 32,
            "chunked_prefill_size": 4096,
            "max_prefill_tokens": 2048,
        }
        ok, msgs = _trimtab_apply_hot_knobs(s, args)
        self.assertTrue(ok, msgs)
        self.assertEqual(s.max_running_requests, 8)
        self.assertEqual(s.max_queued_requests, 32)
        self.assertEqual(s.chunked_prefill_size, 4096)
        self.assertEqual(s.max_prefill_tokens, 2048)
        # handled keys are consumed, so the existing allowlist below never sees them
        self.assertEqual(args, {})

    def test_unknown_keys_fall_through_untouched(self):
        """The five upstream keys must still reach the existing handler."""
        s = _Scheduler()
        args = {"pp_max_micro_batch_size": 4, "max_running_requests": 2}
        ok, _ = _trimtab_apply_hot_knobs(s, args)
        self.assertTrue(ok)
        self.assertEqual(args, {"pp_max_micro_batch_size": 4})

    def test_rejects_values_outside_the_boot_allocation(self):
        s = _Scheduler(max_running_requests=64)
        for value in (0, -1, 65, "many", None):
            with self.subTest(value=value):
                s2 = _Scheduler(max_running_requests=64)
                ok, msgs = _trimtab_apply_hot_knobs(s2, {"max_running_requests": value})
                self.assertFalse(ok)
                self.assertTrue(any("rejected" in m for m in msgs), msgs)
                self.assertEqual(s2.max_running_requests, 64)
        self.assertEqual(s.max_running_requests, 64)

    def test_rejects_a_prefill_budget_larger_than_the_kv_pool(self):
        s = _Scheduler(max_total_num_tokens=100_000)
        ok, msgs = _trimtab_apply_hot_knobs(s, {"max_prefill_tokens": 100_001})
        self.assertFalse(ok, msgs)
        self.assertEqual(s.max_prefill_tokens, 16384)

    def test_rejects_a_non_positive_prefill_chunk(self):
        s = _Scheduler()
        ok, _ = _trimtab_apply_hot_knobs(s, {"chunked_prefill_size": 0})
        self.assertFalse(ok)
        self.assertEqual(s.chunked_prefill_size, 8192)

    def test_rejects_an_unknown_schedule_policy(self):
        s = _Scheduler()
        ok, _ = _trimtab_apply_hot_knobs(s, {"schedule_policy": "whatever"})
        self.assertFalse(ok)
        self.assertEqual(s.schedule_policy, "fcfs")

    def test_log_level_is_case_insensitive_and_validated(self):
        s = _Scheduler()
        ok, _ = _trimtab_apply_hot_knobs(s, {"log_level": "debug"})
        self.assertTrue(ok)
        self.assertEqual(s._trimtab_log_level, "DEBUG")
        ok, _ = _trimtab_apply_hot_knobs(_Scheduler(), {"log_level": "LOUD"})
        self.assertFalse(ok)

    def test_the_ceiling_is_the_boot_value_not_the_latest_one(self):
        """Lowering then raising must not let a knob climb past what was allocated."""
        s = _Scheduler(max_running_requests=64)
        self.assertTrue(_trimtab_apply_hot_knobs(s, {"max_running_requests": 4})[0])
        ok, _ = _trimtab_apply_hot_knobs(s, {"max_running_requests": 64})
        self.assertTrue(ok)
        self.assertEqual(s.max_running_requests, 64)
        self.assertFalse(_trimtab_apply_hot_knobs(s, {"max_running_requests": 65})[0])


if __name__ == "__main__":
    unittest.main()
