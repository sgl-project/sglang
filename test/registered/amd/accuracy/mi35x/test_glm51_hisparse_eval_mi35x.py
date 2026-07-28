"""MI35x GLM-5.1 HiSparse GSM8K evaluation test (8-GPU)."""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=5400,
    suite="nightly-amd-8-gpu-mi35x-glm51-hisparse",
    nightly=True,
)


class TestGLM51HiSparseEvalMI35x(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/models/GLM-5.1-FP8/"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=7200,
            other_args=[
                "--tp",
                "8",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
                "--trust-remote-code",
                "--tool-call-parser",
                "glm47",
                "--reasoning-parser",
                "glm45",
                "--mem-fraction-static",
                "0.65",
                "--dsa-prefill-backend",
                "aiter",
                "--dsa-decode-backend",
                "aiter",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--max-running-requests",
                "2",
                "--watchdog-timeout",
                "1200",
                "--skip-server-warmup",
                "--enable-hisparse",
                "--hisparse-config",
                '{"top_k": 2048, "device_buffer_size": 2048, "host_to_device_ratio": 1}',
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=4000,
            num_examples=500,
            num_threads=100,
            num_shots=24,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (glm-5.1 hisparse mi35x)\n"
                f'{metrics["score"]=:.3f}\n'
            )
        self.assertGreater(metrics["score"], 0.93)


if __name__ == "__main__":
    unittest.main()
