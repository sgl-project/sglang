"""
Usage:
python3 -m unittest test_flash_mla_attention_backend.TestFlashMLAAttnBackend.test_mmlu
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch,
)

# Use DeepSeek V3 model for testing
DSV3_MODEL_FOR_TEST = "deepseek-ai/DeepSeek-V3"

class TestFlashMLAAttnBackend(unittest.TestCase):
    def test_latency(self):
        output_throughput = run_bench_one_batch(
            DSV3_MODEL_FOR_TEST,
            [
                "--attention-backend",
                "flashinfer",
                "--enable-torch-compile",
                "--cuda-graph-max-bs",
                "16",
            ],
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 153)

    def test_mmlu(self):
        model = DSV3_MODEL_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "flashinfer", "--trust-remote-code"],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], 0.87)  # Higher threshold based on DSV3 MMLU score from PR
        finally:
            kill_process_tree(process.pid)

    def test_gsm8k(self):
        model = DSV3_MODEL_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "flashinfer", "--trust-remote-code"],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="gsm8k",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], 0.97)  # Higher threshold based on DSV3 GSM8K score from PR
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
