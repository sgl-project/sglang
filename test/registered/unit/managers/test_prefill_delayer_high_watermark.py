import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
    RecentPrefillBatchSizeTracker,
    _NegotiateOutput,
    _State,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestPrefillDelayerHighWatermark(CustomTestCase):
    def test_peak_expires_after_recent_attempt_window(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        self.assertEqual(tracker.observe_attempt(100), 100)
        for attempted_prefill_bs in [2, 1, 2]:
            self.assertEqual(tracker.observe_attempt(attempted_prefill_bs), 100)

        self.assertEqual(tracker.observe_attempt(2), 2)

    def test_rejected_non_empty_attempt_advances_window(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)
        self.assertEqual(tracker.observe_attempt(10), 10)

        delayer = MagicMock()
        delayer.attn_dp_enabled = True
        delayer.num_dp_ranks = 1
        delayer._metrics_collector = None
        delayer._debug_log_enabled = False
        delayer._negotiate_should_allow_prefill.return_value = _NegotiateOutput(
            next_state=None,
            input_estimation="all",
            output_allow=False,
            output_reason="delay",
            num_prefillable=1,
            num_token_watermark_force_allow=0,
        )

        for _ in range(4):
            executor = PrefillDelayerSinglePassExecutor(delayer, token_usage=0.9)
            self.assertFalse(
                executor.negotiate_should_allow_prefill(
                    local_prefillable=True,
                    running_batch=15,
                    max_prefill_bs=tracker.max_prefill_bs,
                    max_running_requests=20,
                    waiting_queue_len=2,
                )
            )
            attempted_prefill_bs = executor.finalize(actual_prefill_bs=0)
            self.assertEqual(attempted_prefill_bs, 2)
            tracker.observe_attempt(attempted_prefill_bs)

        self.assertEqual(tracker.max_prefill_bs, 2)

    def test_rejects_empty_attempts(self):
        with self.assertRaisesRegex(ValueError, "window_size must be positive"):
            RecentPrefillBatchSizeTracker(window_size=0)

        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        with self.assertRaisesRegex(ValueError, "non-empty attempt"):
            tracker.observe_attempt(0)

        self.assertEqual(tracker.max_prefill_bs, 0)


class TestPrefillDelayerClockSkew(CustomTestCase):
    def test_queue_timeout_follows_gathered_flag(self):
        delayer = PrefillDelayer.__new__(PrefillDelayer)
        delayer.__dict__.update(
            _max_delay_passes=100,
            _token_usage_low_watermark=None,
            _queue_min_ratio=0.5,
            _max_delay_ms=500,
            _queue_trigger_enabled=True,
            _prefill_max_requests=4,
            attn_dp_enabled=False,
            num_dp_ranks=1,
            skip_first_delayer=False,
        )
        for local_expired in (False, True):
            for gathered_expired in (False, True):
                with self.subTest(
                    local_expired=local_expired, gathered_expired=gathered_expired
                ):
                    delayer._gather_info = MagicMock(
                        return_value=torch.tensor(
                            [[1, 0, 8, 4, 1, int(gathered_expired)]]
                        )
                    )
                    start_time = float("-inf") if local_expired else time.perf_counter()
                    out = delayer._negotiate_should_allow_prefill_pure(
                        prev_state=_State(delayed_count=1, start_time=start_time),
                        local_prefillable=True,
                        token_usage=0.8,
                        running_batch=8,
                        max_prefill_bs=4,
                        max_running_requests=128,
                        waiting_queue_len=1,
                    )
                    self.assertEqual(out.output_allow, gathered_expired)
                    kwargs = delayer._gather_info.call_args.kwargs
                    self.assertEqual(kwargs["queue_timeout_expired"], local_expired)


def _gather_on_one_rank(output, local, group=None):
    output.copy_(local)


# 50 running at ratio 0.3 wait for 15 queued; 974 free slots keep the slot
# trigger off.
_QUEUE_ONLY_PASS = dict(
    local_prefillable=True,
    token_usage=0.5,
    running_batch=50,
    max_prefill_bs=16,
    max_running_requests=1024,
    waiting_queue_len=2,
)


class TestPrefillDelayerQueueTrigger(CustomTestCase):
    """A queue-triggered delay holds until the waiting queue reaches
    running_req * ratio or max_delay_ms elapses, whichever comes first."""

    def make_delayer(
        self, *, queue_min_ratio=0.3, max_delay_passes=30, prefill_max_requests=None
    ):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            published_topology(
                enable_prefill_delayer=True,
                prefill_delayer_queue_min_ratio=queue_min_ratio,
                prefill_max_requests=prefill_max_requests,
            )
        )
        stack.enter_context(
            get_parallel().override(
                tp_group=SimpleNamespace(cpu_group=None, device_group=None)
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.managers.prefill_delayer.all_gather_single",
                side_effect=_gather_on_one_rank,
            )
        )
        delayer = PrefillDelayer(
            max_delay_passes=max_delay_passes, token_usage_low_watermark=None
        )
        delayer.skip_first_delayer = False
        return delayer

    def negotiate(self, delayer, **kwargs):
        out = delayer._negotiate_should_allow_prefill(**(_QUEUE_ONLY_PASS | kwargs))
        return out.output_allow, out.output_reason

    def test_threshold_ignores_collapsing_max_prefill_bs(self):
        # Delayed passes report the waiting queue as their attempted batch
        # size, so the max_prefill_bs high-watermark falls to the queue length
        # while the delay runs.
        delayer = self.make_delayer()
        self.assertEqual(self.negotiate(delayer), (False, "delay"))
        self.assertEqual(
            self.negotiate(delayer, max_prefill_bs=11, waiting_queue_len=11),
            (False, "delay"),
        )
        self.assertEqual(
            self.negotiate(delayer, max_prefill_bs=11, waiting_queue_len=15),
            (True, "wait_success"),
        )

    def test_prefill_max_requests_caps_the_threshold(self):
        delayer = self.make_delayer(prefill_max_requests=8)
        self.assertEqual(
            self.negotiate(delayer, waiting_queue_len=8), (True, "no_wait")
        )

    def test_queue_trigger_leaves_skip_first_delayer_to_slot_trigger(self):
        delayer = self.make_delayer()
        delayer.skip_first_delayer = True
        self.assertEqual(self.negotiate(delayer), (False, "delay"))
        self.assertTrue(delayer.skip_first_delayer)

        delayer._curr_state = None
        slot_only = dict(max_running_requests=60, waiting_queue_len=99)
        self.assertEqual(self.negotiate(delayer, **slot_only), (True, "no_wait"))
        self.assertFalse(delayer.skip_first_delayer)

    def test_queue_delay_outlasts_max_delay_passes(self):
        delayer = self.make_delayer(max_delay_passes=3)
        for _ in range(8):
            self.assertEqual(self.negotiate(delayer), (False, "delay"))

    def test_slot_delay_still_capped_by_max_delay_passes(self):
        delayer = self.make_delayer(queue_min_ratio=None, max_delay_passes=3)
        slot_only = dict(max_running_requests=60, waiting_queue_len=99)
        reasons = [self.negotiate(delayer, **slot_only)[1] for _ in range(3)]
        self.assertEqual(reasons, ["delay", "delay", "wait_timeout"])


if __name__ == "__main__":
    unittest.main()
