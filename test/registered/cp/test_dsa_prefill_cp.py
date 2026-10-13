import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    write_github_step_summary,
)

register_cuda_ci(est_time=314, stage="extra-b", runner_config="8-gpu-h200")
GLM52_MODEL_PATH = "zai-org/GLM-5.2-FP8"
SERVER_LAUNCH_TIMEOUT = max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1800)


class TestDSACPInterleave(CustomTestCase):
    attn_cp_size = 2
    attn_dp_size = 2
    attn_tp_size = 2

    @classmethod
    def setUpClass(cls):
        cls.model = GLM52_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--attn-cp-size",
            str(cls.attn_cp_size),
            "--attn-dp-size",
            str(cls.attn_dp_size),
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-frac",
            "0.85",
            "--cuda-graph-max-bs-decode",
            "32",
            "--max-running-requests",
            "32",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        response = requests.get(f"{self.base_url}/server_info", timeout=30)
        response.raise_for_status()
        info = response.json()
        self.assertEqual(info["attn_cp_size"], self.attn_cp_size)
        self.assertEqual(info["attn_dp_size"], self.attn_dp_size)
        self.assertEqual(
            info["tp_size"] // (info["attn_dp_size"] * info["attn_cp_size"]),
            self.attn_tp_size,
        )
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=500,
            num_threads=32,
            num_shots=20,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f'### test_a_gsm8k (dsa-cp-interleave)\n{metrics["score"]=:.3f}\n'
            )
        self.assertGreater(metrics["score"], 0.935)


if __name__ == "__main__":
    unittest.main()
