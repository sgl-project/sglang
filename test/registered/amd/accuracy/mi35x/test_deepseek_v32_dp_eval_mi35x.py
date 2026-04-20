"""MI35x DeepSeek-V3.2 DP GSM8K Accuracy Evaluation Test (8-GPU)

Tests DeepSeek-V3.2 with DP=8 + TP=8 + dp-attention using few-shot
completion benchmark on MI35x.

Registry: nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32-dp suite
"""

import os

# Set HF cache for MI35x
os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Register for AMD CI - MI35x DeepSeek-V3.2 DP accuracy test
register_amd_ci(
    est_time=3600,
    suite="nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32-dp",
    nightly=True,
)

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"

# Accuracy threshold
GSM8K_ACCURACY_THRESHOLD = 0.935


class TestDeepseekV32DP(CustomTestCase):
    """Test DeepSeek V3.2 with DP=8 + TP=8 + dp-attention.

    This test runs GSM8K evaluation and measures accuracy on MI35x.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--nsa-prefill-backend",
            "tilelang",
            "--nsa-decode-backend",
            "tilelang",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        """GSM8K evaluation for DP configuration.

        Named with 'a' prefix to run first (alphabetically) to warm up the server.
        """
        args = SimpleNamespace(
            num_shots=20,
            data_path=None,
            num_questions=1400,
            parallel=1400,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v32 DP MI35x)\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)

    def test_bs_1_speed(self):
        """Single batch speed test for DP configuration."""
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (deepseek-v32 DP MI35x)\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, 10)


if __name__ == "__main__":
    unittest.main()
