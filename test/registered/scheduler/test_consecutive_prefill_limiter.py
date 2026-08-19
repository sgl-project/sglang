import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.consecutive_prefill_limiter import (
    ConsecutivePrefillLimiter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestConsecutivePrefillLimiter(unittest.TestCase):
    def test_default_preserves_prefill_first_scheduling(self):
        limiter = ConsecutivePrefillLimiter(0)

        for _ in range(4):
            limiter.on_prefill()

        self.assertFalse(limiter.should_force_decode(has_runnable_decode=True))

    def test_reserves_decode_after_limit(self):
        limiter = ConsecutivePrefillLimiter(2)

        limiter.on_prefill()
        self.assertFalse(limiter.should_force_decode(has_runnable_decode=True))
        limiter.on_prefill()
        self.assertTrue(limiter.should_force_decode(has_runnable_decode=True))

    def test_does_not_force_decode_when_none_is_runnable(self):
        limiter = ConsecutivePrefillLimiter(1)
        limiter.on_prefill()

        self.assertFalse(limiter.should_force_decode(has_runnable_decode=False))
        self.assertTrue(limiter.should_force_decode(has_runnable_decode=True))

    def test_decode_resets_the_prefill_run(self):
        limiter = ConsecutivePrefillLimiter(1)
        limiter.on_prefill()
        self.assertTrue(limiter.should_force_decode(has_runnable_decode=True))

        limiter.on_decode()

        self.assertFalse(limiter.should_force_decode(has_runnable_decode=True))

    def test_prefill_count_saturates_at_limit(self):
        limiter = ConsecutivePrefillLimiter(2)

        for _ in range(4):
            limiter.on_prefill()

        self.assertEqual(limiter._consecutive_prefill_batches, 2)

    def test_rejects_negative_limit(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            ConsecutivePrefillLimiter(-1)


class _Batch:
    def __init__(self, *, empty=False, prefill_only=False):
        self._empty = empty
        self.is_prefill_only = prefill_only
        self.batch_is_full = False

    def is_empty(self):
        return self._empty


def _make_scheduler(limiter, prefill_plan):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.process_pending_chunked_abort = Mock()
    scheduler.enable_fpm = False
    scheduler._abort_on_waiting_timeout = Mock()
    scheduler._abort_on_running_timeout = Mock()
    scheduler.dllm_config = None
    scheduler.chunked_req = None
    scheduler.enable_hisparse = False
    scheduler.get_new_batch_prefill = Mock(return_value=prefill_plan)
    scheduler.require_mlp_sync = False
    scheduler.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda batch, need_sync: batch
    )
    scheduler.ngram_embedding_manager = SimpleNamespace(
        prepare_for_forward=lambda batch, chunked_req: batch
    )
    scheduler.update_running_batch = Mock()
    scheduler.consecutive_prefill_limiter = limiter
    return scheduler


class TestSchedulerConsecutivePrefillLimiter(unittest.TestCase):
    @patch("sglang.srt.managers.scheduler.set_schedule_time_batch")
    def test_default_still_selects_prefill(self, _set_schedule_time_batch):
        running_batch = _Batch()
        prefill_batch = _Batch()
        scheduler = _make_scheduler(
            ConsecutivePrefillLimiter(0),
            SimpleNamespace(batch_to_run=prefill_batch, running_batch=running_batch),
        )

        plan = Scheduler.get_next_batch_to_run(scheduler, running_batch, None)

        self.assertIs(plan.batch_to_run, prefill_batch)
        scheduler.get_new_batch_prefill.assert_called_once_with(running_batch)
        scheduler.update_running_batch.assert_not_called()

    @patch("sglang.srt.managers.scheduler.set_schedule_time_batch")
    def test_limit_skips_new_prefill_for_a_decode_turn(self, _set_schedule_time_batch):
        running_batch = _Batch()
        limiter = ConsecutivePrefillLimiter(1)
        limiter.on_prefill()
        scheduler = _make_scheduler(
            limiter,
            SimpleNamespace(batch_to_run=_Batch(), running_batch=running_batch),
        )
        scheduler.update_running_batch.return_value = running_batch

        plan = Scheduler.get_next_batch_to_run(scheduler, running_batch, None)

        self.assertIs(plan.batch_to_run, running_batch)
        scheduler.get_new_batch_prefill.assert_not_called()
        scheduler.update_running_batch.assert_called_once_with(running_batch)
        self.assertFalse(limiter.should_force_decode(has_runnable_decode=True))

    @patch(
        "sglang.srt.managers.scheduler.get_spec",
        return_value=SimpleNamespace(speculative_skip_dp_mlp_sync=False),
    )
    @patch("sglang.srt.managers.scheduler.set_schedule_time_batch")
    def test_dp_sync_idle_batch_does_not_count_as_prefill(
        self, _set_schedule_time_batch, _get_spec
    ):
        running_batch = _Batch()
        limiter = ConsecutivePrefillLimiter(2)
        scheduler = _make_scheduler(
            limiter,
            SimpleNamespace(batch_to_run=None, running_batch=running_batch),
        )
        scheduler.require_mlp_sync = True
        scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)
        idle_batch = _Batch(empty=True)
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, need_sync=None: (
                idle_batch if batch is None else batch
            )
        )

        for _ in range(2):
            Scheduler.get_next_batch_to_run(scheduler, running_batch, None)

        self.assertFalse(limiter.should_force_decode(has_runnable_decode=True))
        self.assertEqual(limiter._consecutive_prefill_batches, 0)


if __name__ == "__main__":
    unittest.main()
