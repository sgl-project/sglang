"""Unit tests for the per-run speculative accept length bench_serving reports."""

import unittest

from sglang.benchmark.serving import RequestFuncOutput, calculate_run_accept_length
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _out(output_len, accept_len=0.0, verify_ct=0, success=True):
    return RequestFuncOutput(
        success=success,
        output_len=output_len,
        spec_accept_length=accept_len,
        spec_verify_ct=verify_ct,
    )


class TestCalculateRunAcceptLength(CustomTestCase):
    def test_weights_by_verify_steps(self):
        # 420 tokens in 110 verify steps, not the 3.0 mean of the two requests.
        outputs = [_out(400, 4.0, 100), _out(20, 2.0, 10)]
        self.assertAlmostEqual(calculate_run_accept_length(outputs), 420 / 110)

    def test_recovers_verify_steps_without_verify_ct(self):
        outputs = [_out(400, 4.0), _out(20, 2.0)]
        self.assertAlmostEqual(calculate_run_accept_length(outputs), 420 / 110)

    def test_skips_failed_and_non_speculative(self):
        outputs = [_out(30, 3.0, 10), _out(0, 9.0, 1, success=False), _out(50)]
        self.assertAlmostEqual(calculate_run_accept_length(outputs), 3.0)
        self.assertIsNone(calculate_run_accept_length([_out(8)]))


if __name__ == "__main__":
    unittest.main()
