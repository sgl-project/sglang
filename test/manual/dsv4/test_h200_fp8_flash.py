"""H200 (FP8) x DeepSeek-V4-Flash.

Uses the FP8-repackaged repo (sgl-project/DeepSeek-V4-Flash-FP8) and
the SGLANG_DSV4_FP4_EXPERTS=0 env that the cookbook generator emits
for H200 FP8 cells. Covers Low-Latency, Balanced, Max-Throughput, CP.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEEPEP_LARGE_SMS_CONFIG, DSV4FlashAime25TestBase

MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
H200_FP8_ENV = {"SGLANG_DSV4_FP4_EXPERTS": "0"}


class TestH200Fp8FlashLowLatency(DSV4FlashAime25TestBase):
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
    EXTRA_ENV = dict(H200_FP8_ENV)


class TestH200Fp8FlashBalanced(DSV4FlashAime25TestBase):
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
        **H200_FP8_ENV,
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    }


class TestH200Fp8FlashMaxThroughput(DSV4FlashAime25TestBase):
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
        **H200_FP8_ENV,
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    }


class TestH200Fp8FlashCP(DSV4FlashAime25TestBase):
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
        **H200_FP8_ENV,
        "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
    }


if __name__ == "__main__":
    unittest.main()
