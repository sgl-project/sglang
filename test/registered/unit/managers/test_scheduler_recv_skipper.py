import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_recv_skipper import (  # noqa: E402
    SchedulerRecvSkipper,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _server_args(interval, enable_dp_attention=False):
    return SimpleNamespace(
        scheduler_recv_interval=interval,
        enable_dp_attention=enable_dp_attention,
    )


def _batch(forward_mode, recv_skipper_forward_mode=None):
    return SimpleNamespace(
        forward_mode=forward_mode,
        recv_skipper_forward_mode=recv_skipper_forward_mode,
    )


class TestSchedulerRecvSkipper(CustomTestCase):
    def test_disabled_at_default_interval(self):
        # interval <= 1 disables the skipper entirely.
        self.assertIsNone(SchedulerRecvSkipper.maybe_create(_server_args(1)))

    def test_enabled_under_dp_attention(self):
        # Regression: the constructor used to assert `not enable_dp_attention`.
        skipper = SchedulerRecvSkipper.maybe_create(
            _server_args(50, enable_dp_attention=True)
        )
        self.assertIsNotNone(skipper)

    def test_no_last_batch_accumulates_slowly(self):
        skipper = SchedulerRecvSkipper.maybe_create(_server_args(50))
        self.assertFalse(skipper.handle(None))

    def test_decode_accumulates_until_threshold(self):
        # DECODE weight = 1: recv only every `interval` decode steps.
        skipper = SchedulerRecvSkipper.maybe_create(_server_args(3))
        decode = _batch(ForwardMode.DECODE)
        self.assertFalse(skipper.handle(decode))  # counter 1
        self.assertFalse(skipper.handle(decode))  # counter 2
        self.assertTrue(skipper.handle(decode))  # counter 3 -> recv, reset
        self.assertFalse(skipper.handle(decode))  # counter 1 again

    def test_prefill_triggers_recv_immediately(self):
        # Non-decode passes use the large default weight: recv right away.
        skipper = SchedulerRecvSkipper.maybe_create(_server_args(50))
        self.assertTrue(skipper.handle(_batch(ForwardMode.EXTEND)))

    def test_dp_uses_synced_mode_not_local(self):
        # Local EXTEND (weight 1000) must be ignored in favor of the synced
        # DECODE (weight 1); a recv here would mean the local mode leaked in.
        skipper = SchedulerRecvSkipper.maybe_create(
            _server_args(50, enable_dp_attention=True)
        )
        self.assertFalse(skipper.handle(_batch(ForwardMode.EXTEND, ForwardMode.DECODE)))

    def test_dp_synced_extend_triggers_recv(self):
        skipper = SchedulerRecvSkipper.maybe_create(
            _server_args(50, enable_dp_attention=True)
        )
        self.assertTrue(skipper.handle(_batch(ForwardMode.IDLE, ForwardMode.EXTEND)))

    def test_derive_forward_mode(self):
        derive = SchedulerRecvSkipper.derive_forward_mode
        decode = ForwardMode.DECODE.value
        extend = ForwardMode.EXTEND.value
        mixed = ForwardMode.MIXED.value
        idle = ForwardMode.IDLE.value
        prebuilt = ForwardMode.PREBUILT.value
        verify = ForwardMode.TARGET_VERIFY.value

        # All ranks idle/prebuilt: same bucket as "no last batch".
        self.assertIsNone(derive([idle, idle]))
        self.assertIsNone(derive([prebuilt, idle]))

        # Any extend-like rank forces the immediate-recv bucket.
        self.assertEqual(derive([decode, extend, idle]), ForwardMode.EXTEND)
        self.assertEqual(derive([mixed, decode]), ForwardMode.EXTEND)

        # Pure decode-like steps keep the slow-recv weights.
        self.assertEqual(derive([decode, idle, decode]), ForwardMode.DECODE)
        self.assertEqual(derive([verify, verify]), ForwardMode.TARGET_VERIFY)
        self.assertEqual(derive([verify, decode]), ForwardMode.DECODE)


if __name__ == "__main__":
    unittest.main()
