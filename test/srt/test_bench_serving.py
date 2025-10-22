import asyncio
import itertools
import unittest

import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_FP8,
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    run_bench_serving,
    run_score_benchmark,
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
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 3050)
            else:
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
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 1000)
            else:
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
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 3050)
            else:
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
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
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
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 3500)
            else:
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
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 3500)
            else:
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
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 11000)
            if is_in_amd_ci():
                self.assertLess(res["median_ttft_ms"], 115)
            else:
                self.assertLess(res["median_ttft_ms"], 86)
            self.assertLess(res["median_itl_ms"], 10)

    def test_vlm_offline_throughput(self):
        res = run_bench_serving(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            num_prompts=200,
            request_rate=float("inf"),
            other_server_args=[
                "--mem-fraction-static",
                "0.7",
            ],
            dataset_name="mmmu",
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_vlm_offline_throughput\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 2000)
                # TODO: not set yet, need AMD machine
            else:
                self.assertGreater(res["output_throughput"], 2500)

    def test_vlm_online_latency(self):
        res = run_bench_serving(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            num_prompts=250,
            request_rate=1,
            other_server_args=[
                "--mem-fraction-static",
                "0.7",
            ],
            dataset_name="mmmu",
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_vlm_online_latency\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 16500)
            if is_in_amd_ci():
                self.assertLess(res["median_ttft_ms"], 150)
                # TODO: not set yet, need AMD machine
            else:
                self.assertLess(res["median_ttft_ms"], 100)
            self.assertLess(res["median_itl_ms"], 8)

    def test_lora_online_latency(self):
        # TODO (lifuhuang): verify LoRA support in AMD.
        if is_in_amd_ci():
            pass

        res = self._run_lora_latency_test(enable_background_task=False)

        if is_in_ci():
            write_github_step_summary(
                f"### test_lora_online_latency\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 2400)
            self.assertLess(res["median_ttft_ms"], 58)

    def test_lora_online_latency_with_concurrent_adapter_updates(self):
        # TODO (lifuhuang): verify LoRA support in AMD.
        if is_in_amd_ci():
            pass

        res = self._run_lora_latency_test(enable_background_task=True)

        if is_in_ci():
            write_github_step_summary(
                f"### test_lora_online_latency\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 4000)
            self.assertLess(res["median_ttft_ms"], 80)

    def _run_lora_latency_test(self, enable_background_task: bool):
        """
        Run a latency test for LoRA with the specified background task setting.
        """

        async def lora_loader_unloader_task(
            base_url: str,
            start_event: asyncio.Event,
            stop_event: asyncio.Event,
        ):
            """
            A background task that repeatedly loads and unloads a LoRA adapter.
            """
            await start_event.wait()

            path_cycler = itertools.cycle(
                [
                    "pbevan11/llama-3.1-8b-ocr-correction",
                    "faridlazuarda/valadapt-llama-3.1-8B-it-chinese",
                    "philschmid/code-llama-3-1-8b-text-to-sql-lora",
                ]
            )
            load_url = f"{base_url}/load_lora_adapter"
            unload_url = f"{base_url}/unload_lora_adapter"
            num_updates = 0

            while not stop_event.is_set():
                # 1. Load the LoRA adapter
                lora_path = next(path_cycler)
                response = await asyncio.to_thread(
                    requests.post,
                    load_url,
                    json={"lora_name": lora_path, "lora_path": lora_path},
                )
                self.assertTrue(
                    response.ok, f"Failed to load LoRA adapter: {response.text}"
                )
                num_updates += 1

                if stop_event.is_set():
                    break

                # Yield control to allow other tasks to run.
                await asyncio.sleep(1)

                # 2. Unload the LoRA adapter
                response = await asyncio.to_thread(
                    requests.post,
                    unload_url,
                    json={"lora_name": lora_path},
                )
                self.assertTrue(
                    response.ok, f"Failed to unload LoRA adapter: {response.text}"
                )
                num_updates += 1

                # Yield control to allow other tasks to run.
                await asyncio.sleep(1)

        background_task = lora_loader_unloader_task if enable_background_task else None
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=400,
            request_rate=8,
            other_server_args=[
                "--enable-lora",
                "--max-loras-per-batch",
                "1",
                "--disable-radix-cache",
                "--random-seed",
                "42",
                "--mem-fraction-static",
                "0.8",
                "--lora-paths",
                "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                "--max-lora-rank",
                "256",
            ],
            dataset_name="random",
            random_input_len=256,
            random_output_len=256,
            lora_name=["Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"],
            background_task=background_task,
        )

        return res

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
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
                f"accept_length: {res['accept_length']:.2f} \n"
            )
            if is_in_amd_ci():
                self.assertLess(res["median_e2e_latency_ms"], 1800)
            else:
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
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 2100)
            else:
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
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 2100)
            else:
                self.assertGreater(res["output_throughput"], 2200)

    def test_pp_offline_throughput_default_decode(self):
        res = run_bench_serving(
            model=DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            num_prompts=1000,
            request_rate=float("inf"),
            random_input_len=1,
            random_output_len=1024,
            other_server_args=["--pp-size", "2"],
            need_warmup=True,
            seed=42,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_pp_offline_throughput_default_decode\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 6700)

    def test_pp_long_context_prefill(self):
        res = run_bench_serving(
            model="meta-llama/Llama-3.3-70B-Instruct",
            num_prompts=4,
            request_rate=float("inf"),
            random_input_len=128000,
            random_output_len=1,
            dataset_name="random",
            other_server_args=[
                "--quantization",
                "fp8",
                "--pp-size",
                "2",
            ],
            need_warmup=False,
            seed=42,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_pp_long_context_latency_prefill\n"
                f"input_throughput: {res['input_throughput']:.2f} ms\n"
            )
            self.assertGreater(res["input_throughput"], 4000)

    def test_score_api_latency_throughput(self):
        """Test score API latency and throughput performance"""
        res = run_score_benchmark(
            model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
            num_requests=1000,
            batch_size=10,
            other_server_args=[],
            need_warmup=True,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_score_api_throughput\n"
                f"Average latency: {res['avg_latency_ms']:.2f} ms\n"
                f"P95 latency: {res['p95_latency_ms']:.2f} ms\n"
                f"Score API throughput: {res['throughput']:.2f} req/s\n"
                f"Successful requests: {res['successful_requests']}/{res['total_requests']}\n"
            )

        self.assertEqual(res["successful_requests"], res["total_requests"])
        self.assertLess(res["avg_latency_ms"], 48)
        self.assertLess(res["p95_latency_ms"], 50)
        self.assertGreater(res["throughput"], 20)

    def test_score_api_batch_scaling(self):
        """Test score API performance with different batch sizes"""
        batch_sizes = [10, 25, 50]

        for batch_size in batch_sizes:
            res = run_score_benchmark(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
                num_requests=500,
                batch_size=batch_size,
            )

            if is_in_ci():
                write_github_step_summary(
                    f"### test_score_api_batch_scaling_size_{batch_size}\n"
                    f"Batch size: {batch_size}\n"
                    f"Average latency: {res['avg_latency_ms']:.2f} ms\n"
                    f"P95 latency: {res['p95_latency_ms']:.2f} ms\n"
                    f"Throughput: {res['throughput']:.2f} req/s\n"
                    f"Successful requests: {res['successful_requests']}/{res['total_requests']}\n"
                )

            self.assertEqual(res["successful_requests"], res["total_requests"])
            if batch_size == 10:
                avg_latency_bound = 45
            elif batch_size == 25:
                avg_latency_bound = 50
            elif batch_size == 50:
                avg_latency_bound = 60
            else:
                avg_latency_bound = 60
            self.assertLess(res["avg_latency_ms"], avg_latency_bound)
            if batch_size == 10:
                p95_latency_bound = 50
            elif batch_size == 25:
                p95_latency_bound = 60
            elif batch_size == 50:
                p95_latency_bound = 65
            else:
                p95_latency_bound = 65
            self.assertLess(res["p95_latency_ms"], p95_latency_bound)


if __name__ == "__main__":
    unittest.main()
