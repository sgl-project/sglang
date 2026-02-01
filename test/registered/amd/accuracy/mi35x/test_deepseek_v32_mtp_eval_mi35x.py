"""MI35x DeepSeek-V3.2 TP+MTP GSM8K Accuracy Evaluation Test (8-GPU)

Tests DeepSeek-V3.2 with TP=8 + MTP (EAGLE speculative decoding) using few-shot
completion benchmark on MI35x.

Registry: nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32-mtp suite
"""

import os

# Set HF cache for MI35x
os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

import unittest
from types import SimpleNamespace

import requests

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

# Register for AMD CI - MI35x DeepSeek-V3.2 TP+MTP accuracy test
register_amd_ci(
    est_time=3600,
    suite="nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32-mtp",
    nightly=True,
)

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"

# Accuracy and performance thresholds
GSM8K_ACCURACY_THRESHOLD = 0.94
AVG_SPEC_ACCEPT_LENGTH_THRESHOLD = 2.7


class TestDeepseekV32TPMTP(CustomTestCase):
    """Test DeepSeek V3.2 with TP=8 + MTP (EAGLE speculative decoding).

    This test runs GSM8K evaluation and measures both accuracy and
    speculative decoding acceptance length on MI35x.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-frac",
            "0.7",
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
        """GSM8K evaluation for TP+MTP configuration.

        Named with 'a' prefix to run first (alphabetically) to warm up the server.
        """
        requests.get(self.base_url + "/flush_cache")

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

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v32 TP+MTP MI35x)\n"
                f'{metrics["accuracy"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)
            self.assertGreater(avg_spec_accept_length, AVG_SPEC_ACCEPT_LENGTH_THRESHOLD)

    def test_bs_1_speed(self):
        """Single batch speed test for TP+MTP configuration."""
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (deepseek-v32 TP+MTP MI35x)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(acc_length, AVG_SPEC_ACCEPT_LENGTH_THRESHOLD)
            self.assertGreater(speed, 55)  # Lowered from 60 for AMD MI35x


if __name__ == "__main__":
    unittest.main()
