import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

TEST_MODEL_MATRIX = {
    "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-R1-0528-W8A8": {
        "accuracy": 0.95,
        "latency": 1000,
        "output_throughput": 6,
    },
}


class TestAscendDeepSeekMTP(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "w8a8_int8",
            "--mem-fraction-static",
            0.8,
            "--disable-radix-cache",
            "--chunked-prefill-size",
            32768,
            "--tp-size",
            16,
            "--dp-size",
            2,
            "--enable-dp-attention",
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            1,
            "--speculative-eagle-topk",
            1,
            "--speculative-num-draft-tokens",
            2,
        ]

        cls.extra_envs = {
            "SGLANG_NPU_USE_MLAPO": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        }
        os.environ.update(cls.extra_envs)

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing accuracy: {model} ===##")

                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=1500,
                    other_args=[
                        *self.common_args,
                    ],
                )

                try:
                    args = SimpleNamespace(
                        num_shots=5,
                        data_path=None,
                        num_questions=1319,
                        max_new_tokens=512,
                        parallel=128,
                        host=f"http://{self.url.hostname}",
                        port=int(self.url.port),
                    )

                    metrics = run_eval_few_shot_gsm8k(args)
                    self.assertGreaterEqual(
                        metrics["accuracy"],
                        TEST_MODEL_MATRIX[model]["accuracy"],
                    )
                finally:
                    kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
