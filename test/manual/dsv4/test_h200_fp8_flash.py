"""H200 (FP8) x DeepSeek-V4-Flash.

Uses the FP8-repackaged repo (sgl-project/DeepSeek-V4-Flash-FP8); the
routed-expert layout is auto-detected from the safetensors header. Covers
Low-Latency, Balanced, Max-Throughput, CP.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEEPEP_LARGE_SMS_CONFIG, Dsv4FlashAime25TestBase

MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"


class TestH200Fp8FlashLowLatency(Dsv4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]


class TestH200Fp8FlashBalanced(Dsv4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
        "--dp",
        "4",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
        "--cuda-graph-max-bs",
        "128",
        "--max-running-requests",
        "128",
        "--deepep-config",
        DEEPEP_LARGE_SMS_CONFIG,
    ]
    EXTRA_ENV = {
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    }


class TestH200Fp8FlashMaxThroughput(Dsv4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
        "--dp",
        "4",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--cuda-graph-max-bs",
        "128",
        "--max-running-requests",
        "256",
        "--deepep-config",
        DEEPEP_LARGE_SMS_CONFIG,
    ]
    EXTRA_ENV = {
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    }


class TestH200Fp8FlashCP(Dsv4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
        "--moe-a2a-backend",
        "deepep",
        "--enable-nsa-prefill-context-parallel",
        "--nsa-prefill-cp-mode",
        "round-robin-split",
        "--chunked-prefill-size",
        "16384",
        "--mem-fraction-static",
        "0.78",
        "--max-running-requests",
        "1024",
        "--deepep-config",
        DEEPEP_LARGE_SMS_CONFIG,
    ]
    EXTRA_ENV = {
        "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
    }


if __name__ == "__main__":
    unittest.main()
