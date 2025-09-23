import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_DEEPPEP_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDeepseek(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--enable-dp-attention",
                "--dp",
                "8",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--moe-a2a-backend",
                "deepep",
                "--enable-two-batch-overlap",
                "--ep-num-redundant-experts",
                "32",
                "--ep-dispatch-algorithm",
                "dynamic",
                "--eplb-algorithm",
                "deepseek",
                "--cuda-graph-bs",
                "256",
                "--max-running-requests",
                "2048",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1200,
            parallel=1200,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["accuracy"], 0.92)


class TestDeepseekMTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--enable-dp-attention",
                "--dp",
                "8",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--moe-a2a-backend",
                "deepep",
                "--enable-two-batch-overlap",
                "--ep-num-redundant-experts",
                "32",
                "--ep-dispatch-algorithm",
                "dynamic",
                "--eplb-algorithm",
                "deepseek",
                "--cuda-graph-bs",
                "64",  # TODO: increase it to 128 when TBO is supported in draft_extend
                "--max-running-requests",
                "512",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1200,
            parallel=1200,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["accuracy"], 0.92)

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(
            f"###test_gsm8k:\n"
            f"accuracy={metrics['accuracy']=:.3f}\n"
            f"{avg_spec_accept_length=:.3f}\n"
        )
        self.assertGreater(avg_spec_accept_length, 1.85)


if __name__ == "__main__":
    unittest.main()
