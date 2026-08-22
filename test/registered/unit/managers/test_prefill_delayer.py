"""Unit tests for sglang.srt.managers.prefill_delayer."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import time
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
    _NegotiateOutput,
    _State,
)
from sglang.test.test_utils import CustomTestCase


def _make_delayer_mock(
    *,
    max_delay_passes: int = 3,
    token_usage_low_watermark=None,
    queue_trigger_enabled: bool = False,
    queue_min_ratio=None,
    max_delay_ms: float = 5000.0,
    enable_dp_attention: bool = True,
    dp_size: int = 1,
    skip_first_delayer: bool = False,
) -> MagicMock:
    """Return a minimal MagicMock usable with _negotiate_should_allow_prefill_pure."""
    mock = MagicMock(spec=PrefillDelayer)
    mock._max_delay_passes = max_delay_passes
    mock._token_usage_low_watermark = token_usage_low_watermark
    mock._queue_trigger_enabled = queue_trigger_enabled
    mock._queue_min_ratio = queue_min_ratio
    mock._max_delay_ms = max_delay_ms
    mock.enable_dp_attention = enable_dp_attention
    mock.dp_size = dp_size
    mock.skip_first_delayer = skip_first_delayer
    return mock


def _gather_side_effect(rows):
    """Return a callable that makes _gather_info return an int64 tensor for the given rows."""
    t = torch.tensor(rows, dtype=torch.int64)

    def _fn(*args, **kwargs):
        return t

    return _fn


class TestState(CustomTestCase):
    def test_initial_delayed_count_is_zero(self):
        """Default _State has delayed_count of zero."""
        state = _State()
        self.assertEqual(state.delayed_count, 0)

    def test_bump_delayed_count_increments_by_one(self):
        """bump_delayed_count returns a new _State with delayed_count incremented by one."""
        state = _State(delayed_count=2)
        self.assertEqual(state.bump_delayed_count().delayed_count, 3)

    def test_bump_delayed_count_returns_new_instance(self):
        """bump_delayed_count produces a new instance without modifying the original."""
        state = _State(delayed_count=0)
        bumped = state.bump_delayed_count()
        self.assertEqual(state.delayed_count, 0)
        self.assertIsNot(state, bumped)

    def test_multiple_bumps_accumulate(self):
        """Successive bump_delayed_count calls accumulate correctly."""
        state = _State()
        for _ in range(5):
            state = state.bump_delayed_count()
        self.assertEqual(state.delayed_count, 5)


class TestNegotiateOutput(CustomTestCase):
    def test_default_wait_fields_are_zero(self):
        """wait_forward_passes and wait_seconds default to 0 and 0.0."""
        out = _NegotiateOutput(
            next_state=None,
            input_estimation="all",
            output_allow=True,
            output_reason="no_wait",
            num_prefillable=1,
            num_token_watermark_force_allow=0,
        )
        self.assertEqual(out.wait_forward_passes, 0)
        self.assertAlmostEqual(out.wait_seconds, 0.0)

    def test_explicit_wait_fields_stored_correctly(self):
        """Explicit wait_forward_passes and wait_seconds are stored correctly."""
        out = _NegotiateOutput(
            next_state=None,
            input_estimation="mixed",
            output_allow=False,
            output_reason="delay",
            num_prefillable=1,
            num_token_watermark_force_allow=0,
            wait_forward_passes=4,
            wait_seconds=2.5,
        )
        self.assertEqual(out.wait_forward_passes, 4)
        self.assertAlmostEqual(out.wait_seconds, 2.5)


class TestNegotiateShouldAllowPrefillPure(CustomTestCase):
    """Tests for PrefillDelayer._negotiate_should_allow_prefill_pure via a mocked self."""

    def _call(
        self,
        mock,
        prev_state=None,
        local_prefillable=True,
        token_usage=0.5,
        running_batch=0,
        max_prefill_bs=0,
        max_running_requests=0,
        waiting_queue_len=0,
    ):
        return PrefillDelayer._negotiate_should_allow_prefill_pure(
            mock,
            prev_state=prev_state,
            local_prefillable=local_prefillable,
            token_usage=token_usage,
            running_batch=running_batch,
            max_prefill_bs=max_prefill_bs,
            max_running_requests=max_running_requests,
            waiting_queue_len=waiting_queue_len,
        )

    def test_none_prefillable_status_always_allows(self):
        """prefillable_status=none (all ranks non-prefillable) always returns output_allow=True."""
        mock = _make_delayer_mock()
        mock._gather_info.side_effect = _gather_side_effect([[0, 0, 0, 0, 0]])
        out = self._call(mock)
        self.assertTrue(out.output_allow)
        self.assertEqual(out.input_estimation, "none")

    def test_all_global_token_watermark_force_allows(self):
        """prefillable_status=all with global watermark force flag returns token_watermark reason."""
        mock = _make_delayer_mock()
        # col 0: prefillable=1 -> "all"; col 1: token_watermark_force_allow=1 -> force allow
        mock._gather_info.side_effect = _gather_side_effect([[1, 1, 0, 0, 0]])
        out = self._call(mock)
        self.assertTrue(out.output_allow)
        self.assertEqual(out.output_reason, "token_watermark")

    def test_all_no_condition_no_prev_state_returns_no_wait(self):
        """prefillable_status=all with no slot/queue condition and no prev_state returns no_wait."""
        mock = _make_delayer_mock()
        # running_batch=1, max_prefill_bs=1; max_running_requests=100 -> slot_condition False
        mock._gather_info.side_effect = _gather_side_effect([[1, 0, 1, 1, 0]])
        out = self._call(mock, max_running_requests=100, max_prefill_bs=1)
        self.assertTrue(out.output_allow)
        self.assertEqual(out.output_reason, "no_wait")

    def test_all_prev_state_no_condition_returns_wait_success(self):
        """prefillable_status=all with prev_state and no condition returns wait_success."""
        mock = _make_delayer_mock()
        mock._gather_info.side_effect = _gather_side_effect([[1, 0, 1, 1, 0]])
        prev = _State(delayed_count=2)
        out = self._call(
            mock, prev_state=prev, max_running_requests=100, max_prefill_bs=1
        )
        self.assertTrue(out.output_allow)
        self.assertEqual(out.output_reason, "wait_success")
        self.assertEqual(out.wait_forward_passes, 2)

    def test_all_skip_first_delayer_allows_on_slot_condition(self):
        """skip_first_delayer=True causes the first slot-condition hit to be skipped."""
        mock = _make_delayer_mock(skip_first_delayer=True)
        # slot_condition: max_running_requests=10, running_batch=9, max_prefill_bs=5 -> 10-9=1 < 5
        mock._gather_info.side_effect = _gather_side_effect([[1, 0, 9, 5, 0]])
        out = self._call(mock, max_running_requests=10, max_prefill_bs=5)
        self.assertTrue(out.output_allow)
        self.assertFalse(mock.skip_first_delayer)

    def test_all_slot_condition_delays_prefill(self):
        """prefillable_status=all with slot condition and skip_first_delayer=False delays."""
        mock = _make_delayer_mock(skip_first_delayer=False)
        # slot_condition: 10 - 9 = 1 < 5
        mock._gather_info.side_effect = _gather_side_effect([[1, 0, 9, 5, 0]])
        out = self._call(mock, max_running_requests=10, max_prefill_bs=5)
        self.assertFalse(out.output_allow)
        self.assertEqual(out.output_reason, "delay")

    def test_mixed_global_token_watermark_force_allows(self):
        """prefillable_status=mixed with global watermark force flag returns token_watermark reason."""
        mock = _make_delayer_mock()
        # two ranks: one prefillable, one not -> "mixed"; first rank has token_watermark=1
        mock._gather_info.side_effect = _gather_side_effect(
            [[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
        )
        out = self._call(mock)
        self.assertTrue(out.output_allow)
        self.assertEqual(out.output_reason, "token_watermark")

    def test_mixed_delays_within_max_passes(self):
        """prefillable_status=mixed delays when delayed_count < max_delay_passes - 1."""
        mock = _make_delayer_mock(max_delay_passes=3)
        mock._gather_info.side_effect = _gather_side_effect(
            [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        )
        prev = _State(delayed_count=0)
        out = self._call(mock, prev_state=prev)
        self.assertFalse(out.output_allow)
        self.assertEqual(out.output_reason, "delay")
        self.assertEqual(out.next_state.delayed_count, 1)

    def test_mixed_timeout_allows_after_max_passes(self):
        """prefillable_status=mixed allows with wait_timeout when delayed_count >= max_delay_passes - 1."""
        mock = _make_delayer_mock(max_delay_passes=3)
        mock._gather_info.side_effect = _gather_side_effect(
            [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        )
        prev = _State(delayed_count=2)
        out = self._call(mock, prev_state=prev)
        self.assertTrue(out.output_allow)
        self.assertEqual(out.output_reason, "wait_timeout")

    def test_all_queue_condition_delays_prefill(self):
        """prefillable_status=all with queue condition delays prefill."""
        mock = _make_delayer_mock(
            queue_trigger_enabled=True,
            queue_min_ratio=0.5,
            skip_first_delayer=False,
        )
        # col 2: running_batch=10, col 3: max_prefill_bs=4, col 4: waiting_queue_len=2
        # queue_min_effective = min(int(10 * 0.5), 4) = 4; waiting_queue_len=2 < 4 -> delay.
        # slot_condition: max_running_requests=100 -> 100 - 10 = 90 < 4 is False, so only
        # the queue trigger fires.
        mock._gather_info.side_effect = _gather_side_effect([[1, 0, 10, 4, 2]])
        out = self._call(mock, max_running_requests=100, max_prefill_bs=4)
        self.assertFalse(out.output_allow)
        self.assertEqual(out.output_reason, "delay")

    def test_all_queue_condition_within_timeout_still_delays(self):
        """prefillable_status=all with queue condition and elapsed < max_delay_ms still delays."""
        mock = _make_delayer_mock(
            queue_trigger_enabled=True,
            queue_min_ratio=0.5,
            max_delay_ms=5000.0,
            skip_first_delayer=False,
        )
        mock._gather_info.side_effect = _gather_side_effect([[1, 0, 10, 4, 2]])
        # prev_state started "now", so elapsed is ~0ms << 5000ms: timeout does not fire,
        # queue_condition stays True and the prefill is delayed.
        prev = _State()
        out = self._call(
            mock, prev_state=prev, max_running_requests=100, max_prefill_bs=4
        )
        self.assertFalse(out.output_allow)
        self.assertEqual(out.output_reason, "delay")

    def test_all_queue_condition_timeout_allows_prefill(self):
        """prefillable_status=all with queue condition but elapsed >= max_delay_ms allows prefill."""
        mock = _make_delayer_mock(
            queue_trigger_enabled=True,
            queue_min_ratio=0.5,
            max_delay_ms=100.0,
            skip_first_delayer=False,
        )
        mock._gather_info.side_effect = _gather_side_effect([[1, 0, 10, 4, 2]])
        # start_time 200ms in the past > 100ms max_delay_ms: the wall-clock timeout
        # resets queue_condition to False, so the prefill is released. Because a
        # prev_state exists, the reason is "wait_success" (not "wait_timeout", which
        # only the "mixed" branch produces).
        prev = _State(start_time=time.perf_counter() - 0.2)
        out = self._call(
            mock, prev_state=prev, max_running_requests=100, max_prefill_bs=4
        )
        self.assertTrue(out.output_allow)
        self.assertEqual(out.output_reason, "wait_success")


class TestPrefillDelayerSinglePassExecutor(CustomTestCase):
    def _make_executor(self, output_allow: bool = True):
        delayer = MagicMock(spec=PrefillDelayer)
        delayer._metrics_collector = None
        delayer._negotiate_should_allow_prefill.return_value = _NegotiateOutput(
            next_state=None,
            input_estimation="all",
            output_allow=output_allow,
            output_reason="no_wait",
            num_prefillable=1,
            num_token_watermark_force_allow=0,
        )
        return PrefillDelayerSinglePassExecutor(delayer, token_usage=0.5), delayer

    def test_negotiate_is_idempotent(self):
        """negotiate_should_allow_prefill called multiple times only invokes delayer once."""
        executor, delayer = self._make_executor(output_allow=True)
        result1 = executor.negotiate_should_allow_prefill(local_prefillable=True)
        result2 = executor.negotiate_should_allow_prefill(local_prefillable=False)
        self.assertEqual(result1, result2)
        delayer._negotiate_should_allow_prefill.assert_called_once()

    def test_finalize_without_prior_negotiate_calls_negotiate_with_false(self):
        """finalize with no prior negotiate call invokes negotiate with local_prefillable=False."""
        executor, delayer = self._make_executor()
        delayer._negotiate_should_allow_prefill.assert_not_called()
        executor.finalize(actual_prefill=False)
        delayer._negotiate_should_allow_prefill.assert_called_once_with(
            local_prefillable=False,
            token_usage=0.5,
            running_batch=0,
            max_prefill_bs=0,
            max_running_requests=0,
            waiting_queue_len=0,
        )

    def test_finalize_after_negotiate_does_not_call_negotiate_again(self):
        """finalize after negotiate_should_allow_prefill does not trigger a second negotiate call."""
        executor, delayer = self._make_executor()
        executor.negotiate_should_allow_prefill(local_prefillable=True)
        delayer._negotiate_should_allow_prefill.assert_called_once()
        executor.finalize(actual_prefill=True)
        delayer._negotiate_should_allow_prefill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
