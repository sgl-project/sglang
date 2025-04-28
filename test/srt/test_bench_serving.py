import unittest

from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_FP8,
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    run_bench_serving,
    write_github_step_summary,
)


class TestBenchServing(CustomTestCase):

    def test_offline_throughput_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_default\n"
                f'Output throughput: {res["output_throughput"]:.2f} token/s\n'
            )
            self.assertGreater(res["output_throughput"], 3800)

    def test_offline_throughput_non_stream_small_batch_size(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=200,
            request_rate=float("inf"),
            other_server_args=["--max-running-requests", "10"],
            dataset_name="sharegpt",
            random_input_len=None,
            random_output_len=None,
            disable_stream=True,
            need_warmup=True,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_non_stream_small_batch_size\n"
                f'Output throughput: {res["output_throughput"]:.2f} token/s\n'
            )
            self.assertGreater(res["output_throughput"], 1050)

    def test_offline_throughput_without_radix_cache(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=["--disable-radix-cache"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_without_radix_cache\n"
                f'Output throughput: {res["output_throughput"]:.2f} token/s\n'
            )
            self.assertGreater(res["output_throughput"], 3800)

    def test_offline_throughput_without_chunked_prefill(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=["--chunked-prefill-size", "-1"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_without_chunked_prefill\n"
                f'Output throughput: {res["output_throughput"]:.2f} token/s\n'
            )
            self.assertGreater(res["output_throughput"], 2600)

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

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_with_triton_attention_backend\n"
                f'Output throughput: {res["output_throughput"]:.2f} token/s\n'
            )
            self.assertGreater(res["output_throughput"], 3700)

    def test_offline_throughput_default_fp8(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST_FP8,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_default_fp8\n"
                f'Output throughput: {res["output_throughput"]:.2f} token/s\n'
            )
            self.assertGreater(res["output_throughput"], 4300)

    def test_online_latency_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=100,
            request_rate=1,
            other_server_args=[],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_latency_default\n"
                f'median_e2e_latency_ms: {res["median_e2e_latency_ms"]:.2f} ms\n'
            )
            self.assertLess(res["median_e2e_latency_ms"], 11000)
            self.assertLess(res["median_ttft_ms"], 86)
            self.assertLess(res["median_itl_ms"], 10)

    def test_online_latency_eagle(self):
        res = run_bench_serving(
            model=DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            num_prompts=300,
            request_rate=8,
            sharegpt_context_len=3072,
            disable_ignore_eos=True,
            dataset_name="sharegpt",
            other_server_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "4",
                "--speculative-num-draft-tokens",
                "16",
                "--mem-fraction-static",
                "0.7",
            ],
            need_warmup=True,
            seed=42,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_latency_eagle\n"
                f'median_e2e_latency_ms: {res["median_e2e_latency_ms"]:.2f} ms\n'
                f'accept_length: {res["accept_length"]:.2f} \n'
            )
            self.assertLess(res["median_e2e_latency_ms"], 900)
            self.assertGreater(res["accept_length"], 3.0)

    def test_moe_offline_throughput_default(self):
        res = run_bench_serving(
            model=DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            num_prompts=300,
            request_rate=float("inf"),
            other_server_args=["--tp", "2"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_offline_throughput_default\n"
                f'Output throughput: {res["output_throughput"]:.2f} token/s\n'
            )
            self.assertGreater(res["output_throughput"], 2200)

    def test_moe_offline_throughput_without_radix_cache(self):
        res = run_bench_serving(
            model=DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            num_prompts=300,
            request_rate=float("inf"),
            other_server_args=["--tp", "2", "--disable-radix-cache"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_offline_throughput_without_radix_cache\n"
                f'Output throughput: {res["output_throughput"]:.2f} token/s\n'
            )
            self.assertGreater(res["output_throughput"], 2200)


if __name__ == "__main__":
    unittest.main()
