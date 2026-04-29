"""B200 (FP4) x DeepSeek-V4-Pro x Max-Throughput.

Mirrors the cookbook command:
  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
  sglang serve --trust-remote-code --model-path deepseek-ai/DeepSeek-V4-Pro \
    --tp 8 --dp 8 --enable-dp-attention --moe-a2a-backend deepep \
    --mem-fraction-static 0.82 --cuda-graph-max-bs 64 --max-running-requests 256 \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Dsv4Aime25TestBase


class TestB200ProMaxThroughput(Dsv4Aime25TestBase):
    MODEL = "deepseek-ai/DeepSeek-V4-Pro"
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--dp",
        "8",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--mem-fraction-static",
        "0.82",
        "--cuda-graph-max-bs",
        "64",
        "--max-running-requests",
        "256",
        "--deepep-config",
        '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}',
    ]
    EXTRA_ENV = {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256"}


if __name__ == "__main__":
    unittest.main()
