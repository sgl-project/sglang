import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(
    mode: ForwardMode,
    forward_iter: int,
    launch_ts: float,
    *,
    after_idle_gap: bool = False,
    batch_size: int = 1,
    extend_num_tokens: int = 0,
):
    return SimpleNamespace(
        forward_mode=mode,
        reqs=[SimpleNamespace(rid=f"req-{i}") for i in range(batch_size)],
        forward_iter=forward_iter,
        launch_ts=launch_ts,
        after_idle_gap=after_idle_gap,
        extend_num_tokens=extend_num_tokens,
    )


class TestSchedulerStepCounters(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.total_prefill_uncached_tokens = 0
        self.scheduler.total_prefill_busy_us = 0
        self.scheduler.decode_moment_totals = [0.0] * 6
        self.scheduler._prev_step = None
        self.result = SimpleNamespace(num_correct_drafts=0)

    def record(self, batch) -> None:
        Scheduler._record_step_counters(self.scheduler, batch, self.result)

    def test_prefill_uses_launch_to_launch_cadence(self):
        self.record(_batch(ForwardMode.EXTEND, 1, 10.0, extend_num_tokens=32))
        self.record(_batch(ForwardMode.EXTEND, 2, 10.25, extend_num_tokens=64))

        self.assertEqual(self.scheduler.total_prefill_busy_us, 250_000)
        self.assertEqual(self.scheduler.total_prefill_uncached_tokens, 64)

    def test_idle_gap_and_mode_transition_drop_intervals(self):
        self.record(_batch(ForwardMode.EXTEND, 1, 10.0, extend_num_tokens=32))
        self.record(
            _batch(
                ForwardMode.EXTEND,
                2,
                20.0,
                after_idle_gap=True,
                extend_num_tokens=64,
            )
        )
        self.record(_batch(ForwardMode.EXTEND, 3, 20.125, extend_num_tokens=96))
        self.record(_batch(ForwardMode.DECODE, 4, 20.25, batch_size=2))
        self.record(_batch(ForwardMode.DECODE, 5, 20.5, batch_size=2))

        self.assertEqual(self.scheduler.total_prefill_busy_us, 125_000)
        self.assertEqual(self.scheduler.total_prefill_uncached_tokens, 96)
        self.assertEqual(
            self.scheduler.decode_moment_totals,
            [1.0, 2.0, 250_000.0, 4.0, 500_000.0, 2.0],
        )


if __name__ == "__main__":
    unittest.main()
