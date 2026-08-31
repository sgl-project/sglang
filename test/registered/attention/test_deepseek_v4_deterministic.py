"""
Usage:
cd test/registered/attention
python3 -m unittest test_deepseek_v4_deterministic.TestDeepseekV4Deterministic
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)

register_cuda_ci(est_time=1200, stage="weekly", runner_config="4-gpu-h100")

DEEPSEEK_V4_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
DEEPSEEK_V4_LAUNCH_TIMEOUT = 3600


class TestDeepseekV4Deterministic(TestDeterministicBase):
    """DeepSeek-V4 on its own dsv4 attention backend.

    The backend is forced by the model's config-time overrides, so this covers
    the one attention backend DeepSeek-V4 can run on rather than a choice
    between several.
    """

    @classmethod
    def get_model(cls):
        return DEEPSEEK_V4_MODEL

    @classmethod
    def get_launch_timeout(cls):
        return DEEPSEEK_V4_LAUNCH_TIMEOUT

    @classmethod
    def get_server_args(cls):
        return COMMON_SERVER_ARGS + ["--tp", "4"]


if __name__ == "__main__":
    unittest.main()
