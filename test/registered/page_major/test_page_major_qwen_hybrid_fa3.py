"""fa3 cell of test_page_major_qwen_hybrid.py, kept on H100 because FA3 is SM80/SM90 only."""

import unittest

import test_page_major_qwen_hybrid as base

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="extra-a", runner_config="1-gpu-large")


class TestUnifiedQwenHybridFa3(base.TestUnifiedQwenHybridTriton):
    """fa3 pinned: read tables, eager direct-bind + captured fused copy."""

    other_args = base._UNIFIED_COMMON_ARGS + ["--attention-backend", "fa3"]


if __name__ == "__main__":
    unittest.main()
