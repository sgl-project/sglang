import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

TEST_MODEL_MATRIX = {
    "Qwen/Qwen3-Next-80B-A3B-Instruct": {
        "accuracy": 0.90,
        "latency": 300,
        "output_throughput": 30,
    },
}


class TestAscendTp8Bf16(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.common_args = [
            "--tp",
            "8",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--mem-fraction-static",
            "0.85",
            "--max-total-tokens",
            "1126400",
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--max-running-requests",
            "32",
            "--cuda-graph-bs",
            "32",
            "--disable-overlap-schedule",
            "--mamba-ssm-dtype",
            "bfloat16",
            "--context-length",
            "262144",
            "--chunked-prefill-size",
            "71680",
            "--max-prefill-tokens",
            "262144",
            "--skip-server-warmup",
            "--disable-radix-cache",
        ]
        cls.extra_envs = {
            "HCCL_BUFFSIZE": "2048",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
            "SGLANG_DEEPEP_BF16_DISPATCH": "1",
            "ENABLE_ASCENDC_FUSION_GDN": "false",  # only latest CANN supports this api
            "ASCEND_USE_FIA": "true",
        }
        os.environ.update(cls.extra_envs)

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing accuracy: {model} ===##")

                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=1800,
                    other_args=[
                        *self.common_args,
                    ],
                )

                try:
                    args = SimpleNamespace(
                        num_shots=5,
                        data_path=None,
                        num_questions=200,
                        max_new_tokens=512,
                        parallel=32,
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

    def test_b_throughput(self):
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing throughput: {model} ===##")

                output_throughput = run_bench_offline_throughput(
                    model,
                    [
                        *self.common_args,
                    ],
                )

                print(f"##=== {model} throughput: {output_throughput} ===##")

                if is_in_ci():
                    self.assertGreater(
                        output_throughput,
                        TEST_MODEL_MATRIX[model]["output_throughput"],
                    )


if __name__ == "__main__":
    unittest.main()
