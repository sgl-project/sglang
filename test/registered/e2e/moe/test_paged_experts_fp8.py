"""Paged experts with block-FP8 experts (128x128 scales, triton runner) reproduces the unpaged
model bit for bit."""

import unittest

from sglang.test import paged_experts_utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE

register_cuda_ci(est_time=450, stage="base-b", runner_config="1-gpu-large")


class TestPagedExpertsFp8Block(paged_experts_utils.PagedMatchesUnpagedBase):
    model = DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE


if __name__ == "__main__":
    unittest.main(verbosity=3)
