import os
import unittest

from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, run_bench_serving


class TestBenchServing(unittest.TestCase):

    def test_offline_throughput_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[],
        )

        if os.getenv("SGLANG_IS_IN_CI", "false") == "true":
            assert res["output_throughput"] > 2400

    def test_offline_throughput_without_radix_cache(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=["--disable-radix-cache"],
        )

        if os.getenv("SGLANG_IS_IN_CI", "false") == "true":
            assert res["output_throughput"] > 2800

    def test_offline_throughput_without_chunked_prefill(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=["--chunked-prefill-size", "-1"],
        )

        if os.getenv("SGLANG_IS_IN_CI", "false") == "true":
            assert res["output_throughput"] > 2400

    def test_offline_throughput_with_triton_attention_backend(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
                "--attention-backend",
                "triton",
                "--context-length",
                "8192",
            ],
        )

    def test_online_latency_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[],
        )


if __name__ == "__main__":
    unittest.main()
