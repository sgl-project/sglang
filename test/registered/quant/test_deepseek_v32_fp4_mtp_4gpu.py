import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=1036, suite="stage-c-test-4-gpu-b200")

FULL_DEEPSEEK_V3_FP4_MODEL_PATH = "nvidia/DeepSeek-V3.2-NVFP4"
SERVER_LAUNCH_TIMEOUT = 1200


class TestDeepseekV32FP4DPSpecV2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "4",
            "--dp",
            "4",
            "--enable-dp-attention",
            "--attention-backend",
            "nsa",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--quantization",
            "modelopt_fp4",
            "--tool-call-parser",
            "deepseekv32",
            "--reasoning-parser",
            "deepseek-v3",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=other_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=500,
            num_threads=500,
            num_shots=20,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v32 mtp)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["score"], 0.93)
            self.assertGreater(avg_spec_accept_length, 2.7)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (deepseek-v32 mtp)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )

            self.assertGreater(acc_length, 2.7)
            self.assertGreater(speed, 90)


class TestDeepseekV32FP4TPSpecV2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "4",
            "--attention-backend",
            "nsa",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--quantization",
            "modelopt_fp4",
            "--tool-call-parser",
            "deepseekv32",
            "--reasoning-parser",
            "deepseek-v3",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=other_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=500,
            num_threads=500,
            num_shots=20,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v32 mtp)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["score"], 0.93)
            self.assertGreater(avg_spec_accept_length, 2.7)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (deepseek-v32 mtp)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )

            self.assertGreater(acc_length, 2.7)
            self.assertGreater(speed, 150)


if __name__ == "__main__":
    unittest.main()
