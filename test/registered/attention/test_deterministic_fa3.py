"""FA3 class of test_deterministic.py, kept on H100 because FA3 is SM80/SM90 only."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)

register_cuda_ci(est_time=95, stage="base-b", runner_config="1-gpu-large")


class TestFa3Deterministic(TestDeterministicBase):
    @classmethod
    def get_server_args(cls):
        return COMMON_SERVER_ARGS + ["--attention-backend", "fa3"]


if __name__ == "__main__":
    unittest.main()
