"""
Usage:
cd test/srt
python3 -m unittest test_qwen3_next_deterministic.TestFlashInferDeterministic
"""

import unittest

from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    TestDeterministicBase,
    popen_launch_server,
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

    @classmethod
    def setUpClass(cls):
        import os

        cls.model = cls.get_model()
        cls.base_url = DEFAULT_URL_FOR_TEST
        if "--attention-backend" not in cls.get_server_args():
            raise unittest.SkipTest("Skip the base test class")

        custom_env = os.environ.copy()
        custom_env["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"] = "false"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
            env=custom_env,
        )


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

    @classmethod
    def setUpClass(cls):
        import os

        cls.model = cls.get_model()
        cls.base_url = DEFAULT_URL_FOR_TEST
        if "--attention-backend" not in cls.get_server_args():
            raise unittest.SkipTest("Skip the base test class")

        custom_env = os.environ.copy()
        custom_env["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"] = "false"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
            env=custom_env,
        )


if __name__ == "__main__":
    unittest.main()
