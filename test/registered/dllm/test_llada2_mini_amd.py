"""
Test LLaDA2 (Diffusion Language Model) on AMD GPUs.

This test verifies that DLLM works on AMD with triton attention backend.
"""

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

register_amd_ci(est_time=520, suite="stage-b-test-small-1-gpu-amd")


class TestLLaDA2MiniAMD(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "inclusionAI/LLaDA2.0-mini"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "1",
            "--attention-backend",
            "triton",  # Use triton for AMD instead of flashinfer
            "--dllm-algorithm",
            "LowConfidence",
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

    def test_gsm8k(self):
        """Test GSM8K accuracy with DLLM on AMD."""
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        # Relaxed thresholds for AMD - may need adjustment
        self.assertGreater(metrics["accuracy"], 0.80)
        self.assertGreater(metrics["output_throughput"], 50)

    def test_bs_1_speed(self):
        """Test single batch inference speed."""
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (llada2-mini AMD) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            # Relaxed threshold for AMD
            self.assertGreater(speed, 10)


if __name__ == "__main__":
    unittest.main()
