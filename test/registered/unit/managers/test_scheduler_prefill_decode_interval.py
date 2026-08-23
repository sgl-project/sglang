"""Tests for scheduler prefill/decode interleaving."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_scheduler(*, interval: int, require_mlp_sync: bool) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.require_mlp_sync = require_mlp_sync
    return scheduler


def _make_batch(*, local_extend: bool, global_extend: bool):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: local_extend),
        is_extend_in_batch=global_extend,
    )


class TestPrefillDecodeInterval(unittest.TestCase):
    def test_disabled_interval_does_not_arm(self):
        scheduler = _make_scheduler(interval=0, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertFalse(scheduler._should_defer_prefill())

    def test_non_dp_interval_uses_local_forward_mode(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertTrue(scheduler._should_defer_prefill())
        self.assertTrue(scheduler._should_defer_prefill())
        self.assertFalse(scheduler._should_defer_prefill())

    def test_dp_interval_uses_globally_synchronized_extend_flag(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)

        # This rank is locally decoding, but another DP rank is prefilling.
        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=True)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 2)
        self.assertTrue(scheduler._should_defer_prefill())

    def test_decode_batch_does_not_rearm_interval(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._prefill_decode_interval_remaining = 1

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=False)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)


class TestDSV4ContinuationPlannerEligibility(unittest.TestCase):
    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        pool = MagicMock()
        pool.target_pool.unified_swa_window = 128
        pool.logical_page_size = 2048
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(dsv4_continuation_pool=pool)
        )
        scheduler.tree_cache = MagicMock()
        return scheduler, pool

    @staticmethod
    def _batch(mode, req, *, global_extend):
        return SimpleNamespace(
            forward_mode=mode,
            is_extend_in_batch=global_extend,
            reqs=[req],
            decoding_reqs=None,
            seq_lens_cpu=torch.tensor([2048], dtype=torch.int64),
            dsv4_continuation_capture_slots=None,
            dsv4_continuation_capture_batch_indices=None,
            dsv4_continuation_capture_endpoints=None,
            dsv4_continuation_restore_slots=None,
            dsv4_continuation_restore_batch_indices=None,
            dsv4_continuation_restore_endpoints=None,
        )

    @staticmethod
    def _req(*, streaming=False):
        return SimpleNamespace(
            rid="req",
            req_pool_idx=1,
            is_retracted=False,
            skip_radix_cache_insert=False,
            session=(SimpleNamespace(streaming=True) if streaming else None),
            dsv4_continuation_value=None,
            dsv4_continuation_node=7,
            dsv4_continuation_host_hit=False,
            prefix_indices=torch.arange(2048),
        )

    def test_global_extend_does_not_plan_local_decode(self):
        scheduler, pool = self._scheduler()
        batch = self._batch(ForwardMode.DECODE, self._req(), global_extend=True)

        scheduler._prepare_dsv4_continuation_capture(batch)
        scheduler._prepare_dsv4_continuation_restore(batch)

        self.assertIsNone(batch.dsv4_continuation_capture_slots)
        self.assertIsNone(batch.dsv4_continuation_restore_slots)
        pool.alloc.assert_not_called()
        scheduler.tree_cache.get_dsv4_continuation_value.assert_not_called()

    def test_streaming_extend_does_not_capture(self):
        scheduler, pool = self._scheduler()
        batch = self._batch(
            ForwardMode.EXTEND, self._req(streaming=True), global_extend=True
        )

        scheduler._prepare_dsv4_continuation_capture(batch)

        self.assertIsNone(batch.dsv4_continuation_capture_slots)
        pool.available_size.assert_not_called()


if __name__ == "__main__":
    unittest.main()
