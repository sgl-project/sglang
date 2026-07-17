"""
Usage:
cd test/srt
python3 -m unittest test_deterministic.TestDeterministic.TESTCASE

Note that there is also `python/sglang/test/test_deterministic.py` as an interactive test. We are converting that
test into unit tests so that's easily reproducible in CI.
"""

import unittest

from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    is_in_amd_ci,
)
from sglang.srt.utils import is_xpu

register_cuda_ci(est_time=207, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=278, suite="stage-b-test-1-gpu-small-amd")
register_xpu_ci(est_time=207, suite="stage-b-test-1-gpu-xpu")


@unittest.skipIf(is_in_amd_ci(), "Skip for AMD CI.")
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


@unittest.skipIf(is_in_amd_ci(), "Skip for AMD CI.")
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


@unittest.skipUnless(is_xpu(), "Skip unless XPU is available.")
class TestIntelXPUDeterministic(TestDeterministicBase):
    # Test with intel_xpu attention backend using smaller model to avoid OOM
    @classmethod
    def get_model(cls):
        # Use smaller model for XPU to avoid OOM
        return DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(
            [
                "--attention-backend",
                "intel_xpu",
                "--device",
                "xpu",
                "--mem-fraction-static",
                "0.80",
            ]
        )
        return args


if __name__ == "__main__":
    unittest.main()
