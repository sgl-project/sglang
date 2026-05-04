"""B200 (FP4) x DeepSeek-V4-Pro.

Covers the four cookbook recipes for this hardware x model_size cell:
Low-Latency, Balanced, Max-Throughput, Context-Parallel (CP).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEEPEP_LARGE_SMS_CONFIG, DSV4ProAime25TestBase

MODEL = "deepseek-ai/DeepSeek-V4-Pro"


class TestB200ProLowLatency(DSV4ProAime25TestBase):
    MODEL = MODEL
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


class TestB200ProBalanced(DSV4ProAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--dp",
        "8",
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
        "--mem-fraction-static",
        "0.82",
        "--cuda-graph-max-bs",
        "64",
        "--max-running-requests",
        "128",
        "--deepep-config",
        DEEPEP_LARGE_SMS_CONFIG,
    ]
    EXTRA_ENV = {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256"}


class TestB200ProMaxThroughput(DSV4ProAime25TestBase):
    MODEL = MODEL
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
        DEEPEP_LARGE_SMS_CONFIG,
    ]
    EXTRA_ENV = {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256"}


class TestB200ProCP(DSV4ProAime25TestBase):
    MODEL = MODEL
    OTHER_ARGS = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--moe-a2a-backend",
        "deepep",
        "--enable-nsa-prefill-context-parallel",
        "--nsa-prefill-cp-mode",
        "round-robin-split",
        "--chunked-prefill-size",
        "16384",
        "--mem-fraction-static",
        "0.78",
        "--cuda-graph-max-bs",
        "256",
        "--max-running-requests",
        "256",
        "--deepep-config",
        DEEPEP_LARGE_SMS_CONFIG,
    ]
    EXTRA_ENV = {
        "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    }


if __name__ == "__main__":
    unittest.main()
