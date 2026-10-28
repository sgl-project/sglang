"""MiniMax-M3.1-Flash Quark MXFP4 on the MI355X Low-Latency recipe, with DSpark.

Runs the cookbook's MI355X / Flash / MXFP4 / Low-Latency / single-node / DSpark
cell as published: AMD's Quark MXFP4 export at TP4 on four of the runner's
eight GPUs, with the DSpark draft the checkpoint ships in its ``dspark/``
subfolder. It covers the gfx950 M3.1 path end to end: the MXFP4 routed experts
on AITER, the FP8 attention and dense GEMMs on AITER, the sparse attention
kernels, and DSpark verify.

GSM8K settings and the 0.93 threshold match the H200 and B300 M3.1 nightlies,
so the three gates read the same way.

Registry: nightly-amd-4-gpu-mi35x-minimax-m31-flash-mxfp4 suite
"""

import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-4-gpu-mi35x-minimax-m31-flash-mxfp4",
    nightly=True,
)


class TestMiniMaxM31FlashMXFP4MI35x(GSM8KMixin, DefaultServerBase):
    model = "amd/MiniMax-M3.1-Flash-Quark-MXFP4"
    # Covers a cold-cache download of the checkpoint plus weight load.
    timeout = 3600
    server_env = {"SGLANG_USE_AITER": "1"}
    other_args = [
        "--trust-remote-code",
        "--reasoning-parser",
        "minimax-m3",
        "--tool-call-parser",
        "minimax-m3",
        "--tp",
        "4",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-use-block-verification",
        "--cuda-graph-backend-prefill",
        "breakable",
        "--fp8-gemm-backend",
        "aiter",
        "--chunked-prefill-size",
        "32768",
        "--max-running-requests",
        "128",
        "--cuda-graph-max-bs-decode",
        "128",
        "--mem-fraction-static",
        "0.8",
        "--tokenizer-worker-num",
        "8",
    ]

    gsm8k_backend = "sgl_eval"
    gsm8k_num_examples = 1319
    gsm8k_num_threads = 64
    gsm8k_max_tokens = 8192
    gsm8k_temperature = 1.0
    gsm8k_top_p = 0.95
    gsm8k_score_threshold = 0.93
    # DSpark accept length is printed but not gated until it is measured on
    # this checkpoint.


if __name__ == "__main__":
    unittest.main()
