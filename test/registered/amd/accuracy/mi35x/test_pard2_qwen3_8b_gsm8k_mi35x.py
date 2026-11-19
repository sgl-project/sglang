"""MI35x PARD-2 GSM8K Accuracy Test (1-GPU)

Qwen/Qwen3-8B with the amd/PARD2-Qwen3-8B draft (--speculative-algorithm PARD2;
auto mode resolves to target-dependent, K=15): few-shot GSM8K over all 1319 test
questions, greedy.

Registry: nightly-test-1-gpu-amd-mi35x suite
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=600, suite="nightly-test-1-gpu-amd-mi35x", nightly=True)

TARGET_MODEL_PATH = "Qwen/Qwen3-8B"
DRAFT_MODEL_PATH = "amd/PARD2-Qwen3-8B"

GSM8K_ACCURACY_THRESHOLD = 0.88
AVG_SPEC_ACCEPT_LENGTH_THRESHOLD = 5.0


class TestPard2Qwen3Gsm8k(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "PARD2",
                "--speculative-draft-model-path",
                DRAFT_MODEL_PATH,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
            parallel=128,
            max_new_tokens=512,
            host="127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{metrics=}")
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (Qwen3-8B + PARD-2, MI35x)\n"
                f'{metrics["accuracy"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
        self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)
        self.assertGreater(avg_spec_accept_length, AVG_SPEC_ACCEPT_LENGTH_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
