import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
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

MISTRAL_LARGE3_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512"


class TestMistralLarge3Basic(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Set environment variable to disable JIT DeepGemm
        os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

        cls.model = MISTRAL_LARGE3_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--attention-backend",
            "trtllm_mla",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--chat-template",
            "mistral",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        # Clean up environment variable
        if "SGLANG_ENABLE_JIT_DEEPGEMM" in os.environ:
            del os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"]

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            num_shots=8,
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
                f"### test_gsm8k (mistral-large-3)\n" f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.90)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (mistral-large-3)\n" f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, 50)


if __name__ == "__main__":
    unittest.main()
