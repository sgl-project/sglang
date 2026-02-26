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


class TestLLaDA2Mini(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "inclusionAI/LLaDA2.0-mini"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "4",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "LowConfidence",
            "--cuda-graph-bs",
            "1",
            "2",
            "3",
            "4",
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

    def test_gsm8k(self, mocking_lower_arch: bool = False):
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

        self.assertGreater(metrics["accuracy"], 0.88)
        if is_in_amd_ci():
            self.assertGreater(metrics["output_throughput"], 80)
        elif mocking_lower_arch:
            self.assertGreater(metrics["output_throughput"], 100)
        else:
            self.assertGreater(metrics["output_throughput"], 250)

    def test_bs_1_speed(self, mocking_lower_arch: bool = False):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (llada2-mini) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(speed, 10)
            elif mocking_lower_arch:
                self.assertGreater(speed, 100)
            else:
                self.assertGreater(speed, 250)

    def test_for_lower_arch(self):
        """Simulate sm_86 to verify fallback path works (fused_topk_deepseek unsupported)."""
        if not is_in_amd_ci():
            with unittest.mock.patch(
                "sglang.srt.layers.moe.topk.get_device_capability",
                return_value=(8, 6),
            ):
                self.test_gsm8k(mocking_lower_arch=True)
                self.test_bs_1_speed(mocking_lower_arch=True)


if __name__ == "__main__":
    unittest.main()
