import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def test_gsm8k(base_url: str):
    requests.get(base_url + "/flush_cache")

    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
        host="http://127.0.0.1",
        port=int(base_url.split(":")[-1]),
    )
    metrics = run_eval_few_shot_gsm8k(args)
    server_info = requests.get(base_url + "/get_server_info")
    avg_spec_accept_length = server_info.json()["internal_states"][0][
        "avg_spec_accept_length"
    ]

    print(f"{metrics=}")
    print(f"{avg_spec_accept_length=}")
    return metrics, avg_spec_accept_length


class TestEagleDPAttnServerSmall(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--enable-dp-attention",
            "--speculative-draft-model-path",
            DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
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
        metrics, avg_spec_accept_length = test_gsm8k(self.base_url)
        self.assertGreater(metrics["accuracy"], 0.64)
        self.assertGreater(avg_spec_accept_length, 1.4)


if __name__ == "__main__":
    unittest.main()
