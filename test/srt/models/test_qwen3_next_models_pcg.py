"""
Qwen3 Next piecewise CUDA graph tests.

DISABLED: See https://github.com/sgl-project/sglang/issues/17039
PCG tests for Qwen3 Next have intermittent failures (5-10% probability).
Investigation ongoing by @YuweiAn.
"""

import unittest

from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


@unittest.skip("Disabled: intermittent failures, see #17039")
class TestQwen3NextPiecewiseCudaGraph(GSM8KMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    other_args = [
        "--tp",
        "4",
        "--enable-piecewise-cuda-graph",
        "--piecewise-cuda-graph-compiler",
        "eager",
    ]


if __name__ == "__main__":
    unittest.main()
