"""Paged experts with GPTQ int4 experts (Marlin runner) stays within Marlin's own noise of the
unpaged model."""

import unittest

from sglang.test import paged_experts_utils
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=450, stage="base-b", runner_config="1-gpu-large")


class TestPagedExpertsGptq(paged_experts_utils.PagedMatchesUnpagedBase):
    model = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    exact = False
    max_abs_tolerance = 0.1
    mean_abs_tolerance = 0.01


if __name__ == "__main__":
    unittest.main(verbosity=3)
