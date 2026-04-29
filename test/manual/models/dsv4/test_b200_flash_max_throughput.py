"""B200 (FP4) x DeepSeek-V4-Flash x Max-Throughput.

Mirrors the cookbook command:
  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
  sglang serve --trust-remote-code --model-path deepseek-ai/DeepSeek-V4-Flash \
    --tp 4 --dp 4 --enable-dp-attention --moe-a2a-backend deepep \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Dsv4Aime25TestBase


class TestB200FlashMaxThroughput(Dsv4Aime25TestBase):
    MODEL = "deepseek-ai/DeepSeek-V4-Flash"
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
        "--dp",
        "4",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-config",
        '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}',
    ]
    EXTRA_ENV = {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024"}


if __name__ == "__main__":
    unittest.main()
