"""H200 (FP4) x DeepSeek-V4-Flash x Max-Throughput.

Mirrors the cookbook command:
  sglang serve --trust-remote-code --model-path deepseek-ai/DeepSeek-V4-Flash \
    --tp 4 --moe-runner-backend marlin
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Dsv4Aime25TestBase


class TestH200FlashMaxThroughput(Dsv4Aime25TestBase):
    MODEL = "deepseek-ai/DeepSeek-V4-Flash"
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
