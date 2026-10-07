"""Latency of the LoRA serving path, with and without adapter churn."""

import asyncio
import itertools
import unittest

import requests

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.perf_bench_kit import at_most, check_perf
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_serving,
)

register_cuda_ci(est_time=490, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=430, suite="stage-b-test-1-gpu-large-amd")


class TestLoRALatency(CustomTestCase):
    def test_online_lora_latency(self):
        res = self._run_lora_latency_test(enable_background_task=False)

        check_perf(
            self,
            at_most(
                "median_e2e_latency_ms",
                res["median_e2e_latency_ms"],
                2270,
                amd=3320,
                unit="ms",
            ),
            # mi300x is about twice as slow as mi325 on LoRA TTFT.
            at_most("median_ttft_ms", res["median_ttft_ms"], 52, amd=100, unit="ms"),
        )

    def test_online_lora_latency_with_concurrent_adapter_updates(self):
        res = self._run_lora_latency_test(enable_background_task=True)

        check_perf(
            self,
            at_most(
                "median_e2e_latency_ms",
                res["median_e2e_latency_ms"],
                3420,
                amd=6000,
                unit="ms",
            ),
            at_most("median_ttft_ms", res["median_ttft_ms"], 55, amd=130, unit="ms"),
        )

    def _run_lora_latency_test(self, enable_background_task: bool):
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

                await asyncio.sleep(1)

                response = await asyncio.to_thread(
                    requests.post,
                    unload_url,
                    json={"lora_name": lora_path},
                )
                self.assertTrue(
                    response.ok, f"Failed to unload LoRA adapter: {response.text}"
                )
                num_updates += 1

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
                "nvidia/llama-3.1-nemoguard-8b-topic-control",
                "--max-lora-rank",
                "256",
            ],
            dataset_name="random",
            random_input_len=256,
            random_output_len=256,
            lora_name=["nvidia/llama-3.1-nemoguard-8b-topic-control"],
            background_task=background_task,
        )

        return res


if __name__ == "__main__":
    unittest.main()
