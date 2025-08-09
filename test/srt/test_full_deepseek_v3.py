import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

FULL_DEEPSEEK_V3_MODEL_PATH = "deepseek-ai/DeepSeek-V3-0324"


class TestDeepseekV3(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code", "--tp", "8"]
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
                f"### test_gsm8k (deepseek-v3)\n" f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.935)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (deepseek-v3)\n" f"{speed=:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(speed, 12)
            else:
                self.assertGreater(speed, 75)


class TestDeepseekV3Small(TestDeepseekV3):
    @classmethod
    def setUpClass(cls):
        cls.model = "/data00/697678ab8e528b85a2a7bddafea1fa4f/DeepSeek-V3-5layer"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "low_latency",
            "--enable-eplb",
            "--ep-num-redundant-experts",
            "32",
            "--enable-dp-attention",
            "--dp-size",
            "8",
            "--moe-dense-tp-size",
            "1",
            "--max-prefill-tokens",
            "128",
            "--chunked-prefill-size",
            "1024",
            "--mem-fraction-static",
            "0.8",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )


class TestDeepseekV3MTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]
        if not is_in_amd_ci():
            other_args += ["--mem-frac", "0.7"]
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
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
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
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3 mtp)\n"
                f'{metrics["accuracy"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["accuracy"], 0.935)
            self.assertGreater(avg_spec_accept_length, 2.9)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (deepseek-v3 mtp)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(acc_length, 2.8)
            else:
                self.assertGreater(acc_length, 2.9)
            if is_in_amd_ci():
                self.assertGreater(speed, 15)
            else:
                self.assertGreater(speed, 130)


if __name__ == "__main__":
    unittest.main()
