"""MiniMax-M3.1-Flash on the H200 balanced recipe with DSpark.

Hopper serves the NVFP4 experts through Marlin W4A16, runs the sparse attention
without the Blackwell MSA kernels, and verifies DSpark drafts as per-row decode
queries, so this recipe needs no extra install on the CI runner.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(
    est_time=960,
    stage="nightly",
    runner_config="8-gpu-h200",
    disabled="times out on every nightly run; disabled while under investigation",
)


class TestMiniMaxM31FlashH200(GSM8KMixin, DefaultServerBase):
    model = "MiniMaxAI/MiniMax-M3.1-Flash"
    other_args = [
        "--trust-remote-code",
        "--reasoning-parser",
        "minimax-m3",
        "--tool-call-parser",
        "minimax-m3",
        "--tp",
        "8",
        "--context-length",
        "140000",
        "--cuda-graph-backend-prefill",
        "breakable",
        "--mem-fraction-static",
        "0.85",
        "--cuda-graph-max-bs-decode",
        "64",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-use-block-verification",
    ]

    gsm8k_backend = "sgl_eval"
    gsm8k_num_examples = 1319
    gsm8k_num_threads = 64
    gsm8k_max_tokens = 8192
    gsm8k_temperature = 1.0
    gsm8k_top_p = 0.95
    # Measured 0.9575 and accept length 3.87 on 8x H200.
    gsm8k_score_threshold = 0.93
    gsm8k_accept_length_thres = 3.4


if __name__ == "__main__":
    unittest.main()
