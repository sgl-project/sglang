import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class TestKimiK2Thinking(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "moonshotai/Kimi-K2-Thinking"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--tool-call-parser",
            "kimi_k2",
            "--reasoning-parser",
            "kimi_k2",
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

    def test_a_gsm8k(
        self,
    ):
        requests.get(self.base_url + "/flush_cache")

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

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (Kimi-K2-Thinking)\n" f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
