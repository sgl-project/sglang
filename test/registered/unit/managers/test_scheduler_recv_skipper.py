import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_recv_skipper import (  # noqa: E402
    SchedulerRecvSkipper,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server_args(interval, enable_dp_attention=False):
    return SimpleNamespace(
        scheduler_recv_interval=interval,
        enable_dp_attention=enable_dp_attention,
    )


def _last_forward_mode(enable_dp_attention, last_batch):
    fake_scheduler = SimpleNamespace(
        server_args=SimpleNamespace(enable_dp_attention=enable_dp_attention),
        last_batch=last_batch,
    )
    return Scheduler._recv_skipper_last_forward_mode(fake_scheduler)


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

    def test_decode_accumulates_until_threshold(self):
        # DECODE weight = 1: recv only every `interval` decode steps.
        skipper = SchedulerRecvSkipper.maybe_create(_server_args(3))
        self.assertFalse(skipper.handle(ForwardMode.DECODE))  # counter 1
        self.assertFalse(skipper.handle(ForwardMode.DECODE))  # counter 2
        self.assertTrue(skipper.handle(ForwardMode.DECODE))  # counter 3 -> recv, reset
        self.assertFalse(skipper.handle(ForwardMode.DECODE))  # counter 1 again

    def test_prefill_triggers_recv_immediately(self):
        # Non-decode passes use the large default weight: recv right away.
        skipper = SchedulerRecvSkipper.maybe_create(_server_args(50))
        self.assertTrue(skipper.handle(ForwardMode.EXTEND))

    def test_derive_global_forward_mode(self):
        derive = SchedulerRecvSkipper.derive_global_forward_mode
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

    def test_derived_mode_is_rank_invariant(self):
        # Permutations of the gathered list must not change the result.
        derive = SchedulerRecvSkipper.derive_global_forward_mode
        modes = [
            ForwardMode.DECODE.value,
            ForwardMode.IDLE.value,
            ForwardMode.EXTEND.value,
            ForwardMode.TARGET_VERIFY.value,
        ]
        results = {derive(modes[i:] + modes[:i]) for i in range(len(modes))}
        self.assertEqual(len(results), 1)


class TestRecvSkipperLastForwardMode(CustomTestCase):
    """The scheduler-side helper that picks what the skipper is fed."""

    def test_no_last_batch(self):
        self.assertIsNone(_last_forward_mode(True, None))
        self.assertIsNone(_last_forward_mode(False, None))

    def test_non_dp_uses_local_mode(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, global_forward_modes=None
        )
        self.assertEqual(_last_forward_mode(False, batch), ForwardMode.DECODE)

    def test_dp_derives_from_gathered_modes_not_local(self):
        # The rank-local mode (IDLE here) differs across ranks and must be
        # ignored in favor of the gathered per-rank modes.
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            global_forward_modes=[
                ForwardMode.EXTEND.value,
                ForwardMode.IDLE.value,
            ],
        )
        self.assertEqual(_last_forward_mode(True, batch), ForwardMode.EXTEND)

    def test_dp_without_gathered_modes_returns_none(self):
        # SGLANG_SCHEDULER_SKIP_ALL_GATHER is global, so None stays consistent.
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, global_forward_modes=None
        )
        self.assertIsNone(_last_forward_mode(True, batch))


if __name__ == "__main__":
    unittest.main()
