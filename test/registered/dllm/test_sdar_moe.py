from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=181, suite="stage-b-test-large-1-gpu")
register_amd_ci(est_time=330, suite="stage-b-test-small-1-gpu-amd")

import unittest
from types import SimpleNamespace

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


class TestSDARMiniBase(CustomTestCase):
    """Base class for SDAR tests with different parallelism configs"""

    model = "JetLM/SDAR-30B-A3B-Chat"
    base_url = DEFAULT_URL_FOR_TEST
    tp_size = 1
    ep_size = 1

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "64",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "LowConfidence",
            "--tp",
            str(cls.tp_size),
            "--ep",
            str(cls.ep_size),
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

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=1024,
            parallel=64,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)

        config_name = f"TP{self.tp_size}-EP{self.ep_size}"
        print(f"[{config_name}] {metrics=}")

        self.assertGreater(metrics["accuracy"], 0.92)

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k_accuracy (SDAR-30B-A3B-Chat) with {config_name}\n"
                f"accuracy={metrics['accuracy']:.4f}\n"
                f"output_throughput={metrics['output_throughput']:.2f} token/s\n"
            )

        if is_in_amd_ci():
            self.assertGreater(metrics["output_throughput"], 80)
        else:
            # Adjust throughput threshold based on parallelism
            min_throughput = self._get_min_throughput()
            self.assertGreater(metrics["output_throughput"], min_throughput)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        config_name = f"TP{self.tp_size}-EP{self.ep_size}"
        print(f"[{config_name}] {speed=:.2f} token/s")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (SDAR-30B-A3B-Chat) with {config_name}\n"
                f"{speed=:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(speed, 10)
            else:
                min_speed = self._get_min_speed()
                self.assertGreater(speed, min_speed)

    def _get_min_throughput(self):
        """Get minimum throughput threshold based on parallelism config"""
        # Baseline for TP1-EP1
        if self.tp_size == 1 and self.ep_size == 1:
            return 400
        elif self.tp_size == 2 and self.ep_size == 1:
            return 500
        elif self.tp_size == 1 and self.ep_size == 2:
            return 550
        elif self.tp_size == 2 and self.ep_size == 2:
            return 450
        return 400

    def _get_min_speed(self):
        """Get minimum speed threshold based on parallelism config"""
        # Baseline for TP1-EP1
        if self.tp_size == 1 and self.ep_size == 1:
            return 70
        elif self.tp_size == 2 and self.ep_size == 1:
            return 60
        elif self.tp_size == 1 and self.ep_size == 2:
            return 65
        elif self.tp_size == 2 and self.ep_size == 2:
            return 55
        return 50


class TestSDARMini_TP1_EP1(TestSDARMiniBase):
    """Test SDAR with TP=1, EP=1"""

    tp_size = 1
    ep_size = 1


class TestSDARMini_TP2_EP1(TestSDARMiniBase):
    """Test SDAR with TP=2, EP=1"""

    tp_size = 2
    ep_size = 1


class TestSDARMini_TP2_EP2(TestSDARMiniBase):
    """Test SDAR with TP=2, EP=2"""

    tp_size = 2
    ep_size = 2


if __name__ == "__main__":
    unittest.main()
