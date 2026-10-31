"""--log-schedule-decisions stays quiet on repeated blocked passes and non-zero ranks."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOGGER = "sglang.srt.managers.scheduler"


def _scheduler(free_tokens):
    s = Scheduler.__new__(Scheduler)
    s.log_schedule_decisions = True
    s.waiting_queue = [object()]
    s.pool_stats_observer = SimpleNamespace(
        get_pool_stats=lambda: SimpleNamespace(
            full_available_size=free_tokens[0], full_evictable_size=0
        )
    )
    return s


def _log_pass(s, picked):
    s._maybe_log_schedule_decision(
        can_run_list=[SimpleNamespace(rid=f"r{i}") for i in range(picked)],
        running_batch=SimpleNamespace(reqs=[]),
        adder=SimpleNamespace(preempt_list=[]),
    )


def _rank_scope(attn_tp_rank):
    return get_parallel().override(
        tp_size=2,
        attn_tp_size=2,
        attn_dp_size=1,
        attn_cp_size=1,
        tp_rank=attn_tp_rank,
        attn_tp_rank=attn_tp_rank,
        attn_dp_rank=0,
        attn_cp_rank=0,
    )


class TestScheduleDecisionLog(CustomTestCase):
    def test_blocked_queue_logs_each_state_once(self):
        """A queue blocked without batch_is_full retries every iteration; only
        a change in the pass's counters, or an admitting pass in between,
        should produce another empty-pass line."""
        free_tokens = [100]
        s = _scheduler(free_tokens)
        with _rank_scope(0), self.assertLogs(LOGGER, level="INFO") as logs:
            _log_pass(s, picked=0)
            _log_pass(s, picked=0)
            free_tokens[0] = 50
            _log_pass(s, picked=0)
            _log_pass(s, picked=1)
            _log_pass(s, picked=0)
        self.assertEqual(
            [line.split("#picked=")[1][0] for line in logs.output],
            ["0", "0", "1", "0"],
        )

    def test_long_pick_list_is_capped(self):
        s = _scheduler([100])
        with _rank_scope(0), self.assertLogs(LOGGER, level="INFO") as logs:
            _log_pass(s, picked=20)
        (line,) = logs.output
        self.assertIn("'r15']", line)
        self.assertNotIn("'r16'", line)
        self.assertTrue(line.endswith("(+4 more)"))

    def test_only_attention_rank_zero_logs(self):
        s = _scheduler([100])
        with _rank_scope(1), self.assertNoLogs(LOGGER, level="INFO"):
            _log_pass(s, picked=1)


if __name__ == "__main__":
    unittest.main()
