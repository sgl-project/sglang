"""Paged experts with unquantized (bf16) experts reproduces the unpaged model bit for bit."""

import unittest

from sglang.test import paged_experts_utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE

register_cuda_ci(est_time=450, stage="base-b", runner_config="1-gpu-large")


class TestPagedExpertsUnquantized(paged_experts_utils.PagedMatchesUnpagedBase):
    model = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE


if __name__ == "__main__":
    unittest.main(verbosity=3)
