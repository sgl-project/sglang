"""H200 (FP8) x DeepSeek-V4-Pro.

The cookbook ships this cell as a multi-node (2 nodes, TP=16) launch
using the FP8-repackaged repo (sgl-project/DeepSeek-V4-Pro-FP8).
Each test class skips itself unless DSV4_NODE_RANK and
DSV4_DIST_INIT_ADDR are exported. Runtime expectation:

    On every node:
        DSV4_NODE_RANK=<0 or 1> \\
        DSV4_DIST_INIT_ADDR=<head-node-ip>:20000 \\
        python test/manual/models/dsv4/test_h200_fp8_pro.py

Context-Parallel is marked TBD in the cookbook for this cell, so it
is intentionally omitted.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DSV4ProAime25TestBase, multinode_args

MODEL = "sgl-project/DeepSeek-V4-Pro-FP8"
H200_FP8_PRO_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
}


class TestH200Fp8ProLowLatency(DSV4ProAime25TestBase):
    MODEL = MODEL
    EXTRA_ENV = dict(H200_FP8_PRO_ENV)

    @classmethod
    def setUpClass(cls):
        cls.OTHER_ARGS = [
            "--trust-remote-code",
            "--tp",
            "16",
            "--dp",
            "16",
            "--enable-dp-attention",
            *multinode_args(2),
            "--moe-a2a-backend",
            "deepep",
            "--cuda-graph-max-bs",
            "8",
            "--max-running-requests",
            "32",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-fraction-static",
            "0.88",
        ]
        super().setUpClass()


class TestH200Fp8ProBalanced(DSV4ProAime25TestBase):
    MODEL = MODEL
    EXTRA_ENV = dict(H200_FP8_PRO_ENV)

    @classmethod
    def setUpClass(cls):
        cls.OTHER_ARGS = [
            "--trust-remote-code",
            "--tp",
            "16",
            "--dp",
            "16",
            "--enable-dp-attention",
            *multinode_args(2),
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
            "0.88",
            "--cuda-graph-max-bs",
            "8",
            "--max-running-requests",
            "32",
        ]
        super().setUpClass()


class TestH200Fp8ProMaxThroughput(DSV4ProAime25TestBase):
    MODEL = MODEL
    EXTRA_ENV = dict(H200_FP8_PRO_ENV)

    @classmethod
    def setUpClass(cls):
        cls.OTHER_ARGS = [
            "--trust-remote-code",
            "--tp",
            "16",
            "--dp",
            "16",
            "--enable-dp-attention",
            *multinode_args(2),
            "--moe-a2a-backend",
            "deepep",
            "--mem-fraction-static",
            "0.88",
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "256",
        ]
        super().setUpClass()


if __name__ == "__main__":
    unittest.main()
