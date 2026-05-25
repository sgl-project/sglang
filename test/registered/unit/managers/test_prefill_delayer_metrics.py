"""Regression tests for PrefillDelayer wait-length metric observation (#25949)."""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.prefill_delayer import (
    _NegotiateOutput,
    _record_single_pass_result,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_output(
    *,
    next_state=None,
    output_allow=True,
    output_reason="no_wait",
    wait_forward_passes=0,
    wait_seconds=0.0,
):
    return _NegotiateOutput(
        next_state=next_state,
        input_estimation="all",
        output_allow=output_allow,
        output_reason=output_reason,
        num_prefillable=4,
        num_token_watermark_force_allow=0,
        wait_forward_passes=wait_forward_passes,
        wait_seconds=wait_seconds,
    )


class TestRecordSinglePassResult(CustomTestCase):
    def test_release_with_prior_wait_observes_nonzero(self):
        mc = MagicMock()
        out = _make_output(
            output_reason="wait_success",
            wait_forward_passes=7,
            wait_seconds=2.5,
        )

        _record_single_pass_result(
            actual_execution=True, output=out, metrics_collector=mc
        )

        kwargs = mc.observe_prefill_delayer_outcome.call_args.kwargs
        self.assertEqual(kwargs["forward_passes"], 7)
        self.assertEqual(kwargs["wait_seconds"], 2.5)
        self.assertEqual(kwargs["output_reason"], "wait_success")
        self.assertTrue(kwargs["actual_execution"])

    def test_no_wait_observes_zero(self):
        mc = MagicMock()
        out = _make_output(output_reason="no_wait")

        _record_single_pass_result(
            actual_execution=True, output=out, metrics_collector=mc
        )

        kwargs = mc.observe_prefill_delayer_outcome.call_args.kwargs
        self.assertEqual(kwargs["forward_passes"], 0)
        self.assertEqual(kwargs["wait_seconds"], 0.0)

    def test_other_fields_pass_through(self):
        mc = MagicMock()
        out = _make_output(
            output_reason="wait_timeout", wait_forward_passes=2, wait_seconds=0.3
        )._replace(input_estimation="mixed")

        _record_single_pass_result(
            actual_execution=False, output=out, metrics_collector=mc
        )

        kwargs = mc.observe_prefill_delayer_outcome.call_args.kwargs
        self.assertEqual(kwargs["input_estimation"], "mixed")
        self.assertFalse(kwargs["actual_execution"])
        self.assertEqual(kwargs["output_reason"], "wait_timeout")
        self.assertEqual(kwargs["forward_passes"], 2)

    def test_metrics_collector_none_is_noop(self):
        out = _make_output(wait_forward_passes=5, wait_seconds=1.0)
        _record_single_pass_result(
            actual_execution=True, output=out, metrics_collector=None
        )


class TestNegotiateOutputDefaults(CustomTestCase):
    def test_wait_fields_default_to_zero(self):
        out = _NegotiateOutput(
            next_state=None,
            input_estimation="all",
            output_allow=True,
            output_reason="no_wait",
            num_prefillable=4,
            num_token_watermark_force_allow=0,
        )
        self.assertEqual(out.wait_forward_passes, 0)
        self.assertEqual(out.wait_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
