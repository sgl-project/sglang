import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

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


class TestSchedulerRecvSkipper(unittest.TestCase):
    def test_disabled_at_default_interval(self):
        # interval <= 1 disables the skipper entirely.
        self.assertIsNone(SchedulerRecvSkipper.maybe_create(_server_args(1)))

    def test_enabled_under_dp_attention(self):
        # Regression: previously this asserted `not enable_dp_attention` and
        # crashed at startup. It must now construct successfully -- the
        # scheduler feeds it a mode derived from the gathered per-rank modes.
        skipper = SchedulerRecvSkipper.maybe_create(
            _server_args(50, enable_dp_attention=True)
        )
        self.assertIsNotNone(skipper)

    def test_derive_global_forward_mode(self):
        derive = SchedulerRecvSkipper.derive_global_forward_mode
        decode = ForwardMode.DECODE.value
        extend = ForwardMode.EXTEND.value
        mixed = ForwardMode.MIXED.value
        idle = ForwardMode.IDLE.value
        prebuilt = ForwardMode.PREBUILT.value
        verify = ForwardMode.TARGET_VERIFY.value

        # All ranks idle/prebuilt: same bucket as "no last batch" so every
        # rank (including those whose last_batch is None) agrees.
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
        # The derived mode only depends on the gathered list, which is
        # identical on every rank -- permutations must not change the result.
        derive = SchedulerRecvSkipper.derive_global_forward_mode
        modes = [
            ForwardMode.DECODE.value,
            ForwardMode.IDLE.value,
            ForwardMode.EXTEND.value,
            ForwardMode.TARGET_VERIFY.value,
        ]
        results = {derive(modes[i:] + modes[:i]) for i in range(len(modes))}
        self.assertEqual(len(results), 1)

    def test_decode_accumulates_until_threshold(self):
        # DECODE weight = 1: recv only every `interval` decode steps.
        skipper = SchedulerRecvSkipper.maybe_create(_server_args(3))
        self.assertFalse(skipper.handle(ForwardMode.DECODE))  # counter 1
        self.assertFalse(skipper.handle(ForwardMode.DECODE))  # counter 2
        self.assertTrue(skipper.handle(ForwardMode.DECODE))  # counter 3 -> recv, reset
        self.assertFalse(skipper.handle(ForwardMode.DECODE))  # counter 1 again

    def test_prefill_triggers_recv_immediately(self):
        # A non-decode pass (e.g. EXTEND/prefill) uses the large default weight,
        # so the scheduler polls for new requests right away.
        skipper = SchedulerRecvSkipper.maybe_create(_server_args(50))
        self.assertTrue(skipper.handle(ForwardMode.EXTEND))


if __name__ == "__main__":
    unittest.main()
