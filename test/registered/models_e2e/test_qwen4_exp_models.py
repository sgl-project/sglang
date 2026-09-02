"""B200 E2E coverage for Qwen3.8-Flash-Next (Qwen4-Exp).

Two cases: plain serving (exercises the QSA sparse-decode path that MTP's
verify widths otherwise mask, plus GDN, PLE and HC), and NEXTN MTP (the
draft layer ships inside the target checkpoint; covers the shared QSA
indexer across draft/verify and the PLE/mamba state rollback after verify).
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=1500, stage="base-c", runner_config="4-gpu-b200")

MODEL = "Qwen/Qwen3.8-Flash-Next"

SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_SCORE_THRESHOLD = 0.94

BASE_ARGS = [
    "--tp-size",
    "4",
    "--mem-fraction-static",
    "0.85",
    "--chunked-prefill-size",
    "8192",
    "--linear-attn-prefill-backend",
    "flashinfer",
    "--linear-attn-decode-backend",
    "flashinfer",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3-thinking",
]


class _Qwen4ExpServer:
    speculative_args: list[str] = []
    model = try_cached_model(MODEL)
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_max_tokens = 16384
    gsm8k_score_threshold = GSM8K_SCORE_THRESHOLD

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=BASE_ARGS + cls.speculative_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestQwen4ExpBase(_Qwen4ExpServer, GSM8KMixin, CustomTestCase):
    """Normal autoregressive serving."""


class TestQwen4ExpMTP(_Qwen4ExpServer, GSM8KMixin, CustomTestCase):
    """NEXTN MTP serving (3 steps, topk 1, 4 draft tokens)."""

    # Accept length measured at 2.46-2.49 on thinking workloads; 2.1 leaves
    # noise margin while still failing on a real drop.
    gsm8k_accept_length_thres = 2.1
    speculative_args = [
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
