import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

FULL_DEEPSEEK_V3_MODEL_PATH = "deepseek-ai/DeepSeek-V3-0324"


class TestDeepseekR1Fp8Flashinfer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(FULL_DEEPSEEK_V3_MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--disable-radix-cache",
            "--max-running-requests",
            "512",
            "--chunked-prefill-size",
            "8192",
            "--mem-fraction-static",
            "0.9",
            "--cuda-graph-max-bs",
            "128",
            "--max-prefill-tokens",
            "8192",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--quantization",
            "fp8",
            "--tensor-parallel-size",
            "8",
            "--data-parallel-size",
            "1",
            "--expert-parallel-size",
            "1",
            "--scheduler-recv-interval",
            "10",
            "--stream-interval",
            "10",
            "--attention-backend",
            "trtllm_mla",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--enable-symm-mem",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                **os.environ,
                "SGLANG_ENABLE_FLASHINFER_FP8_GEMM": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=512,
            parallel=512,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["accuracy"], 0.92)


if __name__ == "__main__":
    unittest.main()
