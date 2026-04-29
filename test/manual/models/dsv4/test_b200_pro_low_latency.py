"""B200 (FP4) x DeepSeek-V4-Pro x Low-Latency.

Mirrors the cookbook command:
  sglang serve --trust-remote-code --model-path deepseek-ai/DeepSeek-V4-Pro \
    --tp 8 --moe-runner-backend flashinfer_mxfp4 \
    --speculative-algo EAGLE --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 4096 --disable-flashinfer-autotune \
    --mem-fraction-static 0.88
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Dsv4Aime25TestBase


class TestB200ProLowLatency(Dsv4Aime25TestBase):
    MODEL = "deepseek-ai/DeepSeek-V4-Pro"
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--chunked-prefill-size",
        "4096",
        "--disable-flashinfer-autotune",
        "--mem-fraction-static",
        "0.88",
    ]
    EXTRA_ENV = {}


if __name__ == "__main__":
    unittest.main()
