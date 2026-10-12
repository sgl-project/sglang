"""Tests for scheduler prefill/decode interleaving."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(
    *, interval: int, require_mlp_sync: bool, chunked_only: bool = False
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler.pdi_chunked_only = chunked_only
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.require_mlp_sync = require_mlp_sync
    return scheduler


def _make_batch(*, local_extend: bool, global_extend: bool, chunked_req=None):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: local_extend),
        is_extend_in_batch=global_extend,
        chunked_req=chunked_req,
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

    def test_completed_prefill_does_not_delay_next_admission(self):
        scheduler = _make_scheduler(
            interval=64, require_mlp_sync=False, chunked_only=True
        )
        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )
        self.assertFalse(scheduler._should_defer_prefill())

    def test_unfinished_prompt_keeps_decode_protection(self):
        scheduler = _make_scheduler(
            interval=2, require_mlp_sync=False, chunked_only=True
        )
        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False, chunked_req=object())
        )
        self.assertTrue(scheduler._should_defer_prefill())
        self.assertTrue(scheduler._should_defer_prefill())
        self.assertFalse(scheduler._should_defer_prefill())

    def test_finishing_chunk_clears_remaining_hold(self):
        scheduler = _make_scheduler(
            interval=64, require_mlp_sync=False, chunked_only=True
        )
        scheduler._prefill_decode_interval_remaining = 17
        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)

    def test_dp_ranks_keep_matching_cadence(self):
        for local_extend, chunked_req in (
            (True, object()),
            (True, None),
            (False, None),
        ):
            with self.subTest(
                local_extend=local_extend, unfinished=chunked_req is not None
            ):
                scheduler = _make_scheduler(
                    interval=2, require_mlp_sync=True, chunked_only=True
                )
                scheduler._arm_prefill_decode_interval(
                    _make_batch(
                        local_extend=local_extend,
                        global_extend=True,
                        chunked_req=chunked_req,
                    )
                )
                self.assertEqual(scheduler._prefill_decode_interval_remaining, 2)

    def test_decode_does_not_clear_chunk_protection(self):
        scheduler = _make_scheduler(
            interval=2, require_mlp_sync=False, chunked_only=True
        )
        scheduler._prefill_decode_interval_remaining = 1
        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=False)
        )
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)

    def test_none_batch_does_not_change_protection(self):
        scheduler = _make_scheduler(
            interval=2, require_mlp_sync=False, chunked_only=True
        )
        scheduler._prefill_decode_interval_remaining = 1
        scheduler._arm_prefill_decode_interval(None)
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)


if __name__ == "__main__":
    unittest.main()
