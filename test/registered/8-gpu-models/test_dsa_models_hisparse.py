import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=720, suite="stage-c-test-8-gpu-h200", nightly=True)

GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"


class TestGLM5DPHiSparse(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GLM5_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
            "--page-size",
            "64",
            "--max-running-requests",
            "200",
            "--mem-fraction-static",
            "0.85",
            "--disable-radix-cache",
            "--kv-cache-dtype",
            "bfloat16",
            "--nsa-decode-backend",
            "flashmla_sparse",
            "--enable-hisparse",
            "--hisparse-config",
            '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 5}',
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=7200,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
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
                f"### test_gsm8k (glm-5 hisparse)\n" f'{metrics["score"]=:.3f}\n'
            )
            self.assertGreater(metrics["score"], 0.94)


if __name__ == "__main__":
    unittest.main()
