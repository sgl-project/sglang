"""
Usage:
cd test/srt
python3 -m unittest test_deterministic.TestDeterministic.TESTCASE

Note that there is also `python/sglang/test/test_deterministic.py` as an interactive test. We are converting that
test into unit tests so that's easily reproducible in CI.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)

register_cuda_ci(est_time=228, suite="stage-b-test-small-1-gpu")


class TestFlashinferDeterministic(TestDeterministicBase):
    # Test with flashinfer attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(
            [
                "--attention-backend",
                "flashinfer",
            ]
        )
        return args


class TestFa3Deterministic(TestDeterministicBase):
    # Test with fa3 attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(
            [
                "--attention-backend",
                "fa3",
            ]
        )
        return args


class TestTritonDeterministic(TestDeterministicBase):
    # Test with triton attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(
            [
                "--attention-backend",
                "triton",
            ]
        )
        return args


if __name__ == "__main__":
    unittest.main()
