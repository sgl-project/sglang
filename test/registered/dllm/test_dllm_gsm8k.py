from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, suite="stage-b-test-large-1-gpu")

import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


def create_dllm_config(threshold: float, cache_delay_iter: int) -> str:
    """Create a temporary YAML config file for dLLM algorithm."""
    import yaml

    config = {
        "threshold": threshold,
        "block_size": 32,
        "cache_delay_iter": cache_delay_iter,
    }
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="dllm_config_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(config, f)
    return path


class TestD3LLMLLaDA(CustomTestCase):
    config_path = None

    @classmethod
    def setUpClass(cls):
        cls.model = "d3LLM/d3LLM_LLaDA"
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Create config with threshold=0.5, cache_delay_iter=2 (matching sglang_gsm8k_cot.sh)
        cls.config_path = create_dllm_config(threshold=0.5, cache_delay_iter=2)

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "2",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "FullAttnMultiBlock",
            "--dllm-algorithm-config",
            cls.config_path,
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
        if cls.config_path and os.path.exists(cls.config_path):
            os.remove(cls.config_path)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=1,
            data_path=None,
            num_questions=100,
            max_new_tokens=256,
            parallel=16,
            host="127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.70)
        self.assertGreater(metrics["output_throughput"], 100)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=256)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (d3LLM_LLaDA) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, 200)


class TestD3LLMDream(CustomTestCase):
    config_path = None

    @classmethod
    def setUpClass(cls):
        cls.model = "d3LLM/d3LLM_Dream"
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Create config with threshold=0.4, cache_delay_iter=1 (matching sglang_gsm8k_cot.sh)
        cls.config_path = create_dllm_config(threshold=0.4, cache_delay_iter=1)

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "2",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "FullAttnMultiBlock",
            "--dllm-algorithm-config",
            cls.config_path,
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
        if cls.config_path and os.path.exists(cls.config_path):
            os.remove(cls.config_path)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=1,
            data_path=None,
            num_questions=100,
            max_new_tokens=256,
            parallel=16,
            host="127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.75)
        self.assertGreater(metrics["output_throughput"], 90)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=256)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (d3LLM_Dream) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, 100)


if __name__ == "__main__":
    unittest.main()
