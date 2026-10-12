"""DeepSeek-V4.1-Flash accuracy with DSPARK on four GB300 GPUs.

Run GSM8K and MMLU in TP4+EP4 and DP-attention4 + EP4 configurations.
The DP+EP recipe is intentionally exercised rather than skipped: startup or
accuracy failures must be visible in CI while this path is being validated.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin, _run_accuracy_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=1800, stage="base-c", runner_config="4-gpu-gb300")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


class DSV41FlashAccuracyMixin(GSM8KMixin):
    # Use the zero-shot chat evaluator, not the legacy five-shot completion path.
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    gsm8k_num_threads = 64
    gsm8k_max_tokens = 4096
    mmlu_score_threshold = 0.90
    mmlu_num_examples = 500
    mmlu_num_threads = 64
    mmlu_max_tokens = 4096

    parallel_args = []
    server_env = {}

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-dspark-block-size",
                "5",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-max-bs-decode",
                "64",
                "--max-running-requests",
                "64",
                "--reasoning-parser",
                "auto",
                "--tool-call-parser",
                "auto",
                *cls.parallel_args,
            ],
            env=cls.server_env,
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

    def assert_dspark_accepts_draft_tokens(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        states = response.json()["internal_states"]
        self.assertTrue(states, "Missing scheduler statistics")
        accept_lengths = [state["avg_spec_accept_length"] for state in states]
        self.assertGreater(
            max(accept_lengths),
            1.0,
            f"DSPARK accepted no draft tokens: {accept_lengths}",
        )

    def test_gsm8k(self):
        super().test_gsm8k()
        self.assert_dspark_accepts_draft_tokens()

    def test_mmlu(self):
        _run_accuracy_eval(
            self,
            eval_name="mmlu",
            score_threshold=self.mmlu_score_threshold,
            num_examples=self.mmlu_num_examples,
            num_threads=self.mmlu_num_threads,
            max_tokens=self.mmlu_max_tokens,
        )
        self.assert_dspark_accepts_draft_tokens()


class TestDSV41FlashTP4EP4DSpark(DSV41FlashAccuracyMixin, CustomTestCase):
    """Tensor-parallel attention and expert-parallel MoE, without DP attention."""

    parallel_args = ["--attn-dp-size", "1", "--ep-size", "4"]


class TestDSV41FlashDP4EP4DSpark(DSV41FlashAccuracyMixin, CustomTestCase):
    """DP attention and EP MoE using the built-in gather/reduce path.

    DSPARK with DP attention does not support the DeepEP backend.
    """

    parallel_args = [
        "--attn-dp-size",
        "4",
        "--ep-size",
        "4",
        "--enable-dp-lm-head",
        "--moe-a2a-backend",
        "none",
    ]


if __name__ == "__main__":
    unittest.main()
