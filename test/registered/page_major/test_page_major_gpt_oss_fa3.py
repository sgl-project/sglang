"""fa3 cell of test_page_major_gpt_oss.py, kept on H100 because FA3 is SM80/SM90 only."""

import unittest

import test_page_major_gpt_oss as base

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=140, stage="extra-a", runner_config="1-gpu-large")


class TestUnifiedGptOssFa3(base.TestUnifiedGptOssTriton):
    """fa3 pinned: the per-layer views read through the translator's read
    tables (eager direct-bind + captured fused copy)."""

    other_args = base._UNIFIED_COMMON_ARGS + ["--attention-backend", "fa3"]


if __name__ == "__main__":
    unittest.main()
