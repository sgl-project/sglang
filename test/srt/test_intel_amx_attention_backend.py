"""
Usage:
python3 -m unittest test_intel_amx_attention_backend.TestIntelAMXAttnBackend.test_mmlu
"""

import unittest
from functools import wraps
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8,
    DEFAULT_MODEL_NAME_FOR_TEST_W8A8,
    DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch,
)


def intel_amx_benchmark(extra_args=None, min_throughput=None):
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(self):
            common_args = [
                "--attention-backend",
                "intel_amx",
                "--disable-radix",
                "--trust-remote-code",
            ]
            full_args = common_args + (extra_args or [])

            model = test_func(self)
            prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
                model, full_args
            )

            print(f"{model=}")
            print(f"{prefill_latency=}")
            print(f"{decode_throughput=}")
            print(f"{decode_latency=}")

            if is_in_ci() and min_throughput is not None:
                self.assertGreater(decode_throughput, min_throughput)

        return wrapper

    return decorator


class TestIntelAMXAttnBackend(CustomTestCase):

    @intel_amx_benchmark(extra_args=["--batch-size", "4"], min_throughput=10)
    def test_latency_mla_model(self):
        return DEFAULT_MLA_MODEL_NAME_FOR_TEST

    @intel_amx_benchmark(extra_args=["--batch-size", "4"], min_throughput=40)
    def test_latency_default_model(self):
        return DEFAULT_MODEL_NAME_FOR_TEST

    @intel_amx_benchmark(extra_args=["--batch-size", "4"], min_throughput=150)
    def test_latency_fp8_qwen(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8

    @intel_amx_benchmark(extra_args=["--batch-size", "4"], min_throughput=50)
    def test_latency_fp8_moe_model(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE

    @intel_amx_benchmark(
        extra_args=["--batch-size", "4", "--quantization", "w8a8_int8"],
        min_throughput=100,
    )
    def test_latency_w8a8_default_model(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_W8A8

    @intel_amx_benchmark(
        extra_args=[
            "--batch-size",
            "4",
            "--quantization",
            "w8a8_int8",
            "--mem-fraction-static",
            "0.9",
            "--max-total-tokens",
            "65536",
            "--tp",
            "6",
        ],
        min_throughput=100,
    )
    def test_latency_w8a8_moe_model(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE

    def test_mmlu(self):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.3",
                "--disable-radix",
                "--trust-remote-code",
                "--disable-overlap-schedule",
            ],
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
            if is_in_ci():
                self.assertGreater(metrics["score"], 0.45)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
