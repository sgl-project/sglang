import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DEEPSEEK_W4AFP8_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    try_cached_model,
    write_github_step_summary,
)

register_cuda_ci(est_time=714, suite="stage-c-test-8-gpu-h20")


class TestDeepseekV3W4afp8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(DEFAULT_DEEPSEEK_W4AFP8_MODEL_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--trust-remote-code", "--tp", "8", "--ep-size", "8"]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1200,
            num_threads=1200,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.92)


class TestDeepseekV3W4Afp8Mtp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(DEFAULT_DEEPSEEK_W4AFP8_MODEL_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--ep-size",
            "8",
            "--cuda-graph-bs",
            "256",
            "--disable-radix-cache",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "2",
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

    def test_gsm8k(
        self,
    ):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
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
                f"### test_gsm8k (deepseek-v3 mtp)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["score"], 0.935)
            self.assertGreater(avg_spec_accept_length, 2.9)


class TestDeepseekV3W4Afp8DeepepNormal(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(DEFAULT_DEEPSEEK_W4AFP8_MODEL_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--ep-size",
            "8",
            "--cuda-graph-bs",
            "256",
            "--disable-radix-cache",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "normal",
            "--dp",
            "8",
            "--enable-dp-attention",
            "--moe-runner-backend",
            "cutlass",
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

    def test_gsm8k(
        self,
    ):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.92)


class TestDeepseekV3W4Afp8DeepepAutoMtp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(DEFAULT_DEEPSEEK_W4AFP8_MODEL_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--ep-size",
            "8",
            "--cuda-graph-bs",
            "256",
            "--disable-radix-cache",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--dp",
            "8",
            "--enable-dp-attention",
            "--moe-runner-backend",
            "cutlass",
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
            env={
                **os.environ,
                "SGLANG_DEEPEP_BF16_DISPATCH": "1",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(
        self,
    ):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.92)


if __name__ == "__main__":
    unittest.main()
