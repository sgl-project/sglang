"""
Usage:
python3 test/registered/attention/test_deterministic_tp4.py
"""

import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_deterministic import BenchArgs, test_deterministic
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)

register_amd_ci(est_time=300, suite="stage-c-test-large-8-gpu-amd-mi35x")


class TestTritonDeterministicTP4(TestDeterministicBase):
    """With TP4, the log-probabilities of a prefixed request must not change with the
    batch, including batches whose all-reduce inputs exceed the custom all-reduce buffer."""

    test_single = None

    @classmethod
    def get_server_args(cls):
        return list(COMMON_SERVER_ARGS) + ["--attention-backend", "triton", "--tp", "4"]

    def test_prefix_with_logprobs(self):
        args = BenchArgs()
        args.host, args.port = self._extract_host_and_port(self.base_url)
        args.test_mode = "prefix"
        args.n_start = 10
        # Measured on MI350X without the fix for oversized all-reduce inputs: 50
        # trials fail in 6 of 8 runs (10 trials pass at TP2).
        args.n_trials = 50
        args.temperature = 0.5
        args.return_logprob = True
        for result in test_deterministic(args):
            assert result == 1


if __name__ == "__main__":
    unittest.main()
