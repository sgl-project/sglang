import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# EAGLE with DP attention on B200 (tp=2, dp=2, requires 4 B200 GPUs)
register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-b200")


def test_gsm8k(base_url: str, model: str):
    requests.get(base_url + "/flush_cache")

    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="gsm8k",
        api="completion",
        max_tokens=512,
        num_examples=200,
        num_threads=128,
    )
    metrics = run_eval(args)
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
        with envs.SGLANG_ENABLE_SPEC_V2.override(
            True
        ), envs.SGLANG_SPEC_NAN_DETECTION.override(
            True
        ), envs.SGLANG_SPEC_OOB_DETECTION.override(
            True
        ):
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
        metrics, avg_spec_accept_length = test_gsm8k(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.62)
        self.assertGreater(avg_spec_accept_length, 2.7)


if __name__ == "__main__":
    unittest.main()
