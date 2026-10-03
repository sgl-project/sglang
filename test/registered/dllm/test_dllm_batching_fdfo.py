from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=141, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=345, suite="stage-b-test-1-gpu-small-amd")

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
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


class TestBatchingFDFO(CustomTestCase):
    """End-to-end dLLM coverage on the default First-Done-First-Out scheduler."""

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
            "--dllm-fdfo",
            "--cuda-graph-bs-decode",
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

    def test_gsm8k(self):
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

        self.assertGreater(metrics["score"], 0.88)
        if is_in_amd_ci():
            self.assertGreater(metrics["output_throughput"], 80)
        else:
            self.assertGreater(metrics["output_throughput"], 450)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (llada2-mini FDFO) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(speed, 10)
            else:
                self.assertGreater(speed, 250)

    def _completion(self, prompt: str, **sampling_params) -> str:
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 128,
                **sampling_params,
            },
            timeout=120,
        )
        return response.json()["choices"][0]["text"]

    def test_sampling_params_reach_the_denoise_step(self):
        """Regression: dLLM decoding accepted temperature/top_p and ignored them.

        The denoise step reads its sampling params off the ForwardBatch rather
        than a dedicated field, so a refactor of dLLM batch construction can stop
        populating them and leave every request silently greedy again. Greedy is
        reproduced first, so a difference is attributable to the params and not to
        run-to-run nondeterminism.
        """
        prompt = "Question: Why is the sea salty? Answer:"
        greedy = self._completion(prompt, temperature=0)
        self.assertEqual(
            self._completion(prompt, temperature=0),
            greedy,
            "greedy dLLM decoding must be reproducible for this test to mean "
            "anything; a mismatch here invalidates the comparison below.",
        )
        self.assertNotEqual(
            self._completion(prompt, temperature=1.0, top_p=0.95),
            greedy,
            "temperature and top_p must change dLLM output, not be dropped.",
        )


if __name__ == "__main__":
    unittest.main()
