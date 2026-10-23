"""Tests for gating the prefill/decode interval on running batch size."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_device, publish, reset_context
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(*, interval: int, min_bs: int) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.prefill_decode_interval_min_bs = min_bs
    scheduler.require_mlp_sync = False
    return scheduler


def _arm(scheduler: Scheduler) -> None:
    scheduler._arm_prefill_decode_interval(
        SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=False,
        )
    )


def _make_running_batch(batch_size: int):
    return SimpleNamespace(batch_size=lambda: batch_size)


class TestPrefillDecodeIntervalMinBs(unittest.TestCase):
    def test_small_running_batch_admits_prefill_and_clears_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(7)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)
        # The cleared interval does not resume once the batch grows.
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(8)))

    def test_large_running_batch_defers_prefill(self):
        scheduler = _make_scheduler(interval=2, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(9)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(9)))

    def test_running_batch_shrinking_below_threshold_ends_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(5)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)


class TestPrefillOverlapBudgets(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="scheduler")

    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.require_mlp_sync = False
        scheduler.result_queue = []
        scheduler.device = "cuda"
        scheduler.tp_cpu_group = object()
        return scheduler

    def _batch(self, tokens, *, prefill=True):
        return SimpleNamespace(
            extend_num_tokens=tokens,
            forward_mode=SimpleNamespace(
                is_extend=lambda: prefill, is_decode=lambda: not prefill
            ),
            is_extend_in_batch=prefill,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )

    def test_combined_prefill_budget_and_disabled_default(self):
        scheduler = self._scheduler()
        with (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False),
            envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(0),
        ):
            for budget, tokens, expected in [
                (0, 80, False),
                (128, 64, False),
                (128, 65, True),
            ]:
                with (
                    self.subTest(budget=budget, tokens=tokens),
                    envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(budget),
                ):
                    self.assertEqual(
                        scheduler.is_disable_overlap_for_batch(
                            self._batch(tokens), self._batch(tokens)
                        ),
                        expected,
                    )
            with envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(1):
                self.assertFalse(
                    scheduler.is_disable_overlap_for_batch(
                        self._batch(80, prefill=False), self._batch(80)
                    )
                )
                self.assertFalse(
                    scheduler.is_disable_overlap_for_batch(self._batch(80), None)
                )

    def test_free_memory_uses_tp_minimum_without_emptying_allocator(self):
        scheduler = self._scheduler()
        with (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False),
            envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(0),
            envs.SGLANG_PREFILL_OVERLAP_MIN_FREE_GB.override(2),
            get_device().override(gpu_id=0),
        ):
            for free_gb, expected in [(1.5, True), (2, False), (3, False)]:

                def probe(device, gpu_id, *, distributed, empty_cache, cpu_group):
                    self.assertTrue(distributed)
                    self.assertFalse(empty_cache)
                    self.assertIs(cpu_group, scheduler.tp_cpu_group)
                    return free_gb

                with patch(
                    "sglang.srt.managers.scheduler.get_available_gpu_memory",
                    side_effect=probe,
                ):
                    self.assertEqual(
                        scheduler.is_disable_overlap_for_batch(
                            self._batch(8), self._batch(8)
                        ),
                        expected,
                    )

    def test_attention_dp_is_rejected_before_resource_collectives(self):
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                tp_size=2,
                dp_size=2,
                enable_dp_attention=True,
            ),
            role="scheduler",
        )
        with envs.SGLANG_OVERLAP_PREFILL_TOKEN_BUDGET.override(128):
            with self.assertRaisesRegex(ValueError, "attention DP size 1"):
                Scheduler.init_overlap(Scheduler.__new__(Scheduler))


if __name__ == "__main__":
    unittest.main()
