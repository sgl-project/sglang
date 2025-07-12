"""
Usage:
python3 -m unittest test_intel_amx_attention_backend.TestIntelAMXAttnBackend.test_mmlu
"""

import os
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


def with_cpu_omp_threads_bind(value="0-15|16-31|32-47|48-63|64-79|80-95"):
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            original = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND")
            os.environ["SGLANG_CPU_OMP_THREADS_BIND"] = value
            try:
                print("[Decorator] Set env:", value)
                return test_func(*args, **kwargs)
            finally:
                if original is not None:
                    os.environ["SGLANG_CPU_OMP_THREADS_BIND"] = original
                else:
                    del os.environ["SGLANG_CPU_OMP_THREADS_BIND"]
                print("[Decorator] Restored env.")

        return wrapper

    return decorator


class TestIntelAMXAttnBackend(CustomTestCase):
    def test_latency_mla_model(self):
        prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
            DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--batch-size",
                "4",
            ],
        )

        print(f"{prefill_latency=}")
        print(f"{decode_throughput=}")
        print(f"{decode_latency=}")

        if is_in_ci():
            self.assertGreater(decode_throughput, 10)

    def test_latency_default_model(self):
        prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--batch-size",
                "4",
            ],
        )

        print(f"{prefill_latency=}")
        print(f"{decode_throughput=}")
        print(f"{decode_latency=}")

        if is_in_ci():
            self.assertGreater(decode_throughput, 40)

    def test_latency_fp8_qwen(self):
        prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8,
            [
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--batch-size",
                "4",
            ],
        )

        print(f"{prefill_latency=}")
        print(f"{decode_throughput=}")
        print(f"{decode_latency=}")

        if is_in_ci():
            self.assertGreater(decode_throughput, 150)

    def test_latency_fp8_moe_model(self):
        prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
            [
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--batch-size",
                "4",
            ],
        )

        print(f"{prefill_latency=}")
        print(f"{decode_throughput=}")
        print(f"{decode_latency=}")

        if is_in_ci():
            self.assertGreater(decode_throughput, 50)

    def test_latency_w8a8_default_model(self):
        prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST_W8A8,
            [
                "--attention-backend",
                "intel_amx",
                "--quantization",
                "w8a8_int8",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--batch-size",
                "4",
            ],
        )

        print(f"{prefill_latency=}")
        print(f"{decode_throughput=}")
        print(f"{decode_latency=}")

        if is_in_ci():
            self.assertGreater(decode_throughput, 100)

    @with_cpu_omp_threads_bind()
    def test_latency_w8a8_moe_model(self):
        prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE,
            [
                "--attention-backend",
                "intel_amx",
                "--quantization",
                "w8a8_int8",
                "--mem-fraction-static",
                "0.9",
                "--disable-radix",
                "--trust-remote-code",
                "--batch-size",
                "4",
            ],
        )

        print(f"{prefill_latency=}")
        print(f"{decode_throughput=}")
        print(f"{decode_latency=}")

        if is_in_ci():
            self.assertGreater(decode_throughput, 100)

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
                "0.05",
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
            self.assertGreater(metrics["score"], 0.5)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
