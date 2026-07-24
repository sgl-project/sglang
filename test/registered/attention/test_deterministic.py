"""
Usage:
cd test/srt
python3 -m unittest test_deterministic.TestDeterministic.TESTCASE

Note that there is also `python/sglang/test/test_deterministic.py` as an interactive test. We are converting that
test into unit tests so that's easily reproducible in CI.
"""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)
from sglang.test.test_utils import is_in_amd_ci

register_cuda_ci(est_time=207, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=278, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipIf(is_in_amd_ci(), "Skip for AMD CI.")
class TestFlashinferDeterministic(TestDeterministicBase):
    # Test with flashinfer attention backend
    @classmethod
    def get_server_args(cls):
        return list(COMMON_SERVER_ARGS) + [
            "--attention-backend",
            "flashinfer",
        ]


@unittest.skipIf(is_in_amd_ci(), "Skip for AMD CI.")
class TestFa3Deterministic(TestDeterministicBase):
    # Test with fa3 attention backend
    @classmethod
    def get_server_args(cls):
        return list(COMMON_SERVER_ARGS) + [
            "--attention-backend",
            "fa3",
        ]


class TestTritonDeterministic(TestDeterministicBase):
    # Test with triton attention backend
    @classmethod
    def get_server_args(cls):
        return list(COMMON_SERVER_ARGS) + [
            "--attention-backend",
            "triton",
        ]


if __name__ == "__main__":
    unittest.main()
