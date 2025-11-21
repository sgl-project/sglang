"""
Usage:
cd test/srt
python3 -m unittest test_qwen3_next_deterministic.TestFlashInferDeterministic
"""

import unittest

from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)

QWEN3_NEXT = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestFlashInferDeterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return QWEN3_NEXT

    # Test with flashinfer attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(["--attention-backend", "flashinfer", "--tp", "4"])
        return args


class TestTritonDeterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return QWEN3_NEXT

    # Test with triton attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(["--attention-backend", "triton", "--tp", "4"])
        return args


if __name__ == "__main__":
    unittest.main()
