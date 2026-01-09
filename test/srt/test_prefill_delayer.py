import asyncio
import os
import re
import time
import unittest
from types import SimpleNamespace

import openai
import requests

from sglang.bench_serving import run_benchmark
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)

# ============================ Unit Tests ============================


TODO


# ============================ E2E Tests ============================


class TestPrefillDelayerThroughputOnlineServing(CustomTestCase):
    def test_1_has_prefill_delayer(self):
        self._run(prefill_delayer=True)

    def test_2_no_prefill_delayer(self):
        self._run(prefill_delayer=False)

    def _run(self, prefill_delayer: bool):
        _run_throughput_test(
            debug_name=f"online_serving ({prefill_delayer=})",
            prefill_delayer=prefill_delayer,
            other_launch_args=[
                # Not really needed, only to test support non-FCFS algorithms
                "--schedule-policy",
                "lpm",
            ],
            other_benchmark_args=dict(
                num_prompts=500,
                random_input_len=30000,
                random_output_len=256,
                request_rate=32,
            ),
        )


class TestPrefillDelayerThroughputOfflineGen(CustomTestCase):
    def test_1_has_prefill_delayer(self):
        self._run(prefill_delayer=True)

    def test_2_no_prefill_delayer(self):
        self._run(prefill_delayer=False)

    def _run(self, prefill_delayer: bool):
        _run_throughput_test(
            debug_name=f"offline_gen ({prefill_delayer=})",
            prefill_delayer=prefill_delayer,
            other_benchmark_args=dict(
                num_prompts=800,
                random_input_len=30000,
                random_output_len=500,
            ),
            other_launch_args=[
                "--max-total-tokens",
                "200000",
            ],
            token_usage_low_watermark=0.8,
        )


def _run_throughput_test(
    debug_name: str,
    prefill_delayer: bool,
    other_launch_args,
    other_benchmark_args,
    token_usage_low_watermark: float = None,
):
    model = "Qwen/Qwen3-0.6B"
    base_url = DEFAULT_URL_FOR_TEST

    process = _launch_server(
        prefill_delayer=prefill_delayer,
        model=model,
        base_url=base_url,
        other_args=other_launch_args,
        token_usage_low_watermark=token_usage_low_watermark,
    )

    try:
        args = get_benchmark_args(
            base_url=base_url,
            dataset_name="random",
            tokenizer=model,
            **other_benchmark_args,
        )
        res = run_benchmark(args)
        _print_prefill_delayer_metrics(base_url, expect_metrics=prefill_delayer)
    finally:
        kill_process_tree(process.pid)

    print(f"=== {debug_name} ===")
    print(f"Input throughput: {res['input_throughput']:.2f} token/s")
    print(f"Output throughput: {res['output_throughput']:.2f} token/s")


class TestPrefillDelayerTokenUsageLowWatermark(CustomTestCase):
    def test_1_with_low_watermark(self):
        self._run(token_usage_low_watermark=0.8)

    def test_2_without_low_watermark(self):
        self._run(token_usage_low_watermark=None)

    def _run(self, token_usage_low_watermark):
        model = "Qwen/Qwen3-0.6B"
        base_url = DEFAULT_URL_FOR_TEST
        world_size = int(os.environ.get("SGLANG_TEST_WORLD_SIZE", "8"))

        process = _launch_server(
            model=model,
            base_url=base_url,
            prefill_delayer=True,
            other_args=["--max-total-tokens", "50000"],
            # e.g. gen throughput is 370 tok/s
            max_delay_passes=3000,
            token_usage_low_watermark=token_usage_low_watermark,
        )

        async def run_test():
            client = openai.AsyncClient(base_url=f"{base_url}/v1", api_key="EMPTY")
            long_prompt = "Hello " * 5000

            async def send_blocking_request():
                return await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": long_prompt}],
                    max_tokens=10000,
                    extra_body={"data_parallel_rank": 0},
                )

            async def send_normal_request(dp_rank, req_idx):
                start = time.time()
                await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say hi"}],
                    max_tokens=10,
                    extra_body={"data_parallel_rank": dp_rank},
                )
                elapsed = time.time() - start
                return dp_rank, req_idx, elapsed

            asyncio.create_task(send_blocking_request())
            await asyncio.sleep(3)

            num_reqs_per_rank = 10
            results = await asyncio.gather(
                *[
                    send_normal_request(dp_rank, req_idx)
                    for dp_rank in range(1, world_size)
                    for req_idx in range(num_reqs_per_rank)
                ]
            )

            enabled = token_usage_low_watermark is not None
            thresh = 5
            for dp_rank, req_idx, elapsed in results:
                print(f"DP rank {dp_rank} req {req_idx} completed in {elapsed:.2f}s")
                self.assertTrue(
                    (elapsed < thresh) if enabled else (elapsed > thresh),
                    f"DP rank {dp_rank} req {req_idx}: elapsed={elapsed:.2f}s, thresh={thresh}, enabled={enabled}",
                )

        try:
            asyncio.run(run_test())

            metrics_text = _print_prefill_delayer_metrics(base_url, expect_metrics=True)
            if token_usage_low_watermark is not None:
                matches = re.findall(
                    r'outcome="token_watermark_force_allow".*?\} (\d+)', metrics_text
                )
                total_force_allow = sum(int(m) for m in matches)
                self.assertGreater(
                    total_force_allow,
                    0,
                    "Expected token_watermark_force_allow > 0 when low watermark is enabled",
                )
                print(f"total token_watermark_force_allow: {total_force_allow}")
        finally:
            kill_process_tree(process.pid)


class TestPrefillDelayerAccuracy(CustomTestCase):
    def test_1_mgsm_en_has_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=True)

    def test_2_mgsm_en_no_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=False)

    def _run_accuracy_test(self, prefill_delayer: bool):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = _launch_server(
            prefill_delayer=prefill_delayer,
            model=model,
            base_url=base_url,
            other_args=[
                # Not really needed, only to test support non-FCFS algorithms
                "--schedule-policy",
                "lpm",
                # Use this to ensure prefill delayer will be run
                "--max-total-tokens",
                "4096",
            ],
        )
        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mgsm_en",
                num_examples=None,
                num_threads=1024,
            )
            metrics = run_eval(args)
            print(f"=== mgsm_en ({prefill_delayer=}) ===")
            print(f"{metrics=}")
            self.assertGreater(metrics["score"], 0.87)
        finally:
            kill_process_tree(process.pid)


def _launch_server(
    *,
    model,
    base_url,
    prefill_delayer: bool,
    other_args,
    max_delay_passes: int = 100,
    token_usage_low_watermark: float = None,
):
    os.environ["SGLANG_PREFILL_DELAYER_DEBUG_LOG"] = "1"
    world_size = os.environ.get("SGLANG_TEST_WORLD_SIZE", "8")

    with envs.SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE.override(
        prefill_delayer
    ), envs.SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES.override(
        max_delay_passes
    ), envs.SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK.override(
        token_usage_low_watermark
    ):
        return popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                world_size,
                "--enable-dp-attention",
                "--dp",
                world_size,
                "--chunked-prefill-size",
                "131072",
                "--mem-fraction-static",
                "0.6",
                "--enable-metrics",
                *(other_args or []),
            ],
        )


def _print_prefill_delayer_metrics(base_url: str, expect_metrics: bool) -> str:
    metrics_response = requests.get(f"{base_url}/metrics")
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text
    prefill_delayer_metrics = [
        line for line in metrics_text.split("\n") if "prefill_delayer" in line
    ]
    print("=== PrefillDelayer Metrics ===")
    for line in prefill_delayer_metrics:
        print(line)
    if expect_metrics:
        assert "sglang:prefill_delayer_wait_forward_passes" in metrics_text
        assert "sglang:prefill_delayer_wait_seconds" in metrics_text
        assert "sglang:prefill_delayer_outcomes_total" in metrics_text
    return metrics_text


if __name__ == "__main__":
    unittest.main()
