"""Tests for scheduler prefill/decode interleaving."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context

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


class TestPrefillDecodePhaseSeparation(CustomTestCase):
    def test_peer_prefill_preserves_local_decode(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._pending_chunked_abort_req = None
        scheduler._abort_on_waiting_timeout = MagicMock()
        scheduler._abort_on_running_timeout = MagicMock()
        scheduler.dllm_config = None
        scheduler.enable_fpm = False
        scheduler.enable_hisparse = False
        scheduler.chunked_req = None
        scheduler.require_mlp_sync = True
        scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        scheduler.prefill_decode_interval = 0
        scheduler._prefill_decode_interval_remaining = 0
        scheduler.ngram_embedding_manager = MagicMock()
        scheduler.ngram_embedding_manager.prepare_for_forward.side_effect = (
            lambda batch, **_: batch
        )

        running_batch = MagicMock(name="running_decode_batch")
        running_batch.is_empty.return_value = False
        running_batch.is_prefill_only = False
        scheduler.get_new_batch_prefill = MagicMock(
            return_value=NextBatchPlan(
                batch_to_run=None, running_batch=running_batch
            )
        )
        scheduler.update_running_batch = MagicMock(
            side_effect=AssertionError("running decode must be preserved")
        )

        idle_batch = MagicMock(name="peer_prefill_idle_batch")
        scheduler.dp_attn_adapter = MagicMock()
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = [
            idle_batch,
            idle_batch,
        ]
        scheduler.dp_attn_adapter.maybe_convert_decode_to_extend.return_value = (
            idle_batch
        )

        with (
            get_context().override_server_args(
                enable_prefill_decode_phase_separation=True
            ),
            patch("sglang.srt.managers.scheduler.set_schedule_time_batch"),
        ):
            plan = scheduler.get_next_batch_to_run(running_batch, last_batch=None)

        self.assertIs(plan.batch_to_run, idle_batch)
        self.assertIs(plan.running_batch, running_batch)
        scheduler.update_running_batch.assert_not_called()
        self.assertEqual(
            scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.call_args_list,
            [call(None), call(idle_batch, need_sync=False)],
        )


if __name__ == "__main__":
    unittest.main()
