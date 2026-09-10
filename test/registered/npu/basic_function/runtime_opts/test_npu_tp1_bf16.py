import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.npu_eval_accuracy_kit import _is_pr_pipeline, run_npu_pr_smoke
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval as run_gsm8k_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct": {
        "accuracy": 0.84,
        "latency": 150,
        "output_throughput": 30,
    },
}


class TestAscendTp1Bf16(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.common_args = [
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
        ]

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        *self.common_args,
                    ],
                )

                try:
                    if _is_pr_pipeline:
                        run_npu_pr_smoke(self.base_url)
                    else:
                        print(f"##=== Testing accuracy: {model} ===##")

                        args = SimpleNamespace(
                            eval_name="gsm8k",
                            num_examples=1319,
                            max_tokens=512,
                            num_threads=128,
                            host=f"http://{self.url.hostname}",
                            port=int(self.url.port),
                        )

                        metrics = run_gsm8k_eval(args)
                        self.assertGreaterEqual(
                            metrics["accuracy"],
                            TEST_MODEL_MATRIX[model]["accuracy"],
                        )
                finally:
                    kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
