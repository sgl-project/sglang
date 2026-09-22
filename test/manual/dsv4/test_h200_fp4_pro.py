"""H200 (FP4 / Marlin) x DeepSeek-V4-Pro.

The cookbook disables Context-Parallel for the H200 FP4 (Marlin)
hardware, so this file only covers Low-Latency, Balanced, and
Max-Throughput.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DSV4ProAime25TestBase

MODEL = "deepseek-ai/DeepSeek-V4-Pro"


class TestH200Fp4ProLowLatency(DSV4ProAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--moe-runner-backend",
        "marlin",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.88",
    ]
    EXTRA_ENV = {}


class TestH200Fp4ProBalanced(DSV4ProAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--moe-runner-backend",
        "marlin",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
        "--mem-fraction-static",
        "0.88",
    ]
    EXTRA_ENV = {}


class TestH200Fp4ProMaxThroughput(DSV4ProAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--moe-runner-backend",
        "marlin",
        "--mem-fraction-static",
        "0.88",
    ]
    EXTRA_ENV = {}


if __name__ == "__main__":
    unittest.main()
