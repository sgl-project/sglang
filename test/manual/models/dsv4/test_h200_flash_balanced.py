"""H200 (FP4) x DeepSeek-V4-Flash x Balanced.

Mirrors the cookbook command:
  sglang serve --trust-remote-code --model-path deepseek-ai/DeepSeek-V4-Flash \
    --tp 4 --moe-runner-backend marlin \
    --speculative-algo EAGLE --speculative-num-steps 1 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Dsv4Aime25TestBase


class TestH200FlashBalanced(Dsv4Aime25TestBase):
    MODEL = "deepseek-ai/DeepSeek-V4-Flash"
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


if __name__ == "__main__":
    unittest.main()
