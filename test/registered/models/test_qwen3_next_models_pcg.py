"""
Qwen3 Next piecewise CUDA graph tests.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(
    est_time=400,
    suite="stage-c-test-4-gpu-h100",
)

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestQwen3NextPiecewiseCudaGraph(GSM8KMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    other_args = [
        "--tp",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
