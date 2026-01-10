import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    kill_process_tree,
    popen_launch_server,
    write_github_step_summary,
)

# EAGLE3 with DP attention (tp=2, dp=2, requires 4 GPUs)
register_cuda_ci(est_time=200, suite="stage-c-test-large-4-gpu")


class TestEAGLE3EngineDPAttention(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-num-steps",
            "6",
            "--speculative-eagle-topk",
            "10",
            "--speculative-num-draft-tokens",
            "32",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE_DP_ATTN,
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--moe-dense-tp-size",
            "1",
            "--attention-backend",
            "fa3",
            "--mem-fraction-static",
            "0.75",
            "--cuda-graph-max-bs",
            "64",
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
        """Test GSM8K evaluation - append 'a' to run first alphabetically"""
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

        server_info = requests.get(self.base_url + "/get_server_info")
        server_data = server_info.json()

        # Try to get avg_spec_accept_length
        avg_spec_accept_length = None
        if "internal_states" in server_data and len(server_data["internal_states"]) > 0:
            internal_state = server_data["internal_states"][0]
            if "avg_spec_accept_length" in internal_state:
                avg_spec_accept_length = internal_state["avg_spec_accept_length"]
            elif "spec_accept_length" in internal_state:
                avg_spec_accept_length = internal_state["spec_accept_length"]

        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (EAGLE3 DP Attention)\n"
                f'{metrics["accuracy"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["accuracy"], 0.91)
            if avg_spec_accept_length is not None:
                self.assertGreater(avg_spec_accept_length, 2.5)

    def test_bs_1_speed(self):
        """Test batch size 1 speed with EAGLE3 DP Attention"""
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (EAGLE3 DP Attention)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(acc_length, 2.0)
            else:
                self.assertGreater(acc_length, 2.3)
            if is_in_amd_ci():
                self.assertGreater(speed, 10)
            else:
                self.assertGreater(speed, 40)


if __name__ == "__main__":
    unittest.main()
