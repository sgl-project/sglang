"""Focused tests for decode-context-parallel argument validation."""

import unittest

from sglang.srt.arg_groups.parallel_hook import handle_dcp_validation
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDCPTopologyValidation(CustomTestCase):
    def test_dcp_size_must_divide_tp_size(self):
        args = ServerArgs(model_path="dummy", tp_size=8, dcp_size=3)

        with self.assertRaisesRegex(ValueError, "must be divisible"):
            handle_dcp_validation(args)

    def test_dcp_size_dividing_tp_size_passes(self):
        args = ServerArgs(model_path="dummy", tp_size=8, dcp_size=4)

        handle_dcp_validation(args)


if __name__ == "__main__":
    unittest.main()
