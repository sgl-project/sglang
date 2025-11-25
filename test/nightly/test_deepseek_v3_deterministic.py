"""
Usage:
cd test/srt
python3 -m unittest test_deepseek_v3_deterministic.TestFa3Deterministic
"""

import unittest

from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)

DEEPSEEK_MODEL = "lmsys/sglang-ci-dsv3-test"


class TestFa3Deterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return DEEPSEEK_MODEL

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
    @classmethod
    def get_model(cls):
        return DEEPSEEK_MODEL

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
