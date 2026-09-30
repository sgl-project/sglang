"""H200 (FP4 / Marlin) x DeepSeek-V4-Flash.

The cookbook disables Context-Parallel for the H200 FP4 (Marlin)
hardware, so this file only covers Low-Latency, Balanced, and
Max-Throughput.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DSV4FlashAime25TestBase

MODEL = "deepseek-ai/DeepSeek-V4-Flash"


class TestH200Fp4FlashLowLatency(DSV4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
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
    ]
    EXTRA_ENV = {}


class TestH200Fp4FlashBalanced(DSV4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
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
    ]
    EXTRA_ENV = {}


class TestH200Fp4FlashMaxThroughput(DSV4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
        "--moe-runner-backend",
        "marlin",
    ]
    EXTRA_ENV = {}


if __name__ == "__main__":
    unittest.main()
