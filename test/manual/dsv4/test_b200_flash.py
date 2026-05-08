"""B200 (FP4) x DeepSeek-V4-Flash.

Covers the four cookbook recipes for this hardware x model_size cell:
Low-Latency, Balanced, Max-Throughput, Context-Parallel (CP).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEEPEP_LARGE_SMS_CONFIG, DSV4FlashAime25TestBase

MODEL = "deepseek-ai/DeepSeek-V4-Flash"


class TestB200FlashLowLatency(DSV4FlashAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "4",
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
    ]
    EXTRA_ENV = {}


class TestB200FlashBalanced(DSV4FlashAime25TestBase):
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
        "--deepep-config",
        DEEPEP_LARGE_SMS_CONFIG,
    ]
    EXTRA_ENV = {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024"}


class TestB200FlashMaxThroughput(DSV4FlashAime25TestBase):
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
        "--deepep-config",
        DEEPEP_LARGE_SMS_CONFIG,
    ]
    EXTRA_ENV = {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024"}


class TestB200FlashCP(DSV4FlashAime25TestBase):
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
