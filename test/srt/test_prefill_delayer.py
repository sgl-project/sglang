import asyncio
import os
import re
import time
import unittest
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import openai
import requests
import torch
import torch.multiprocessing as mp

from sglang.bench_serving import run_benchmark
from sglang.srt.managers.prefill_delayer import PrefillDelayer
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

WORLD_SIZE = os.environ.get("SGLANG_TEST_WORLD_SIZE", "8")

# ============================ Unit Tests ============================


@dataclass
class NegotiateCall:
    prefillable: List[bool]
    token_usage: List[float]


@dataclass
class NegotiateTestCase:
    name: str
    max_delay_passes: int
    token_usage_low_watermark: Optional[float]
    calls: List[NegotiateCall]
    expected_allow: bool
    expected_reason: str


def _run_negotiate_test(rank, world_size, test_cases, results_queue, port):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    cpu_group = torch.distributed.new_group(backend="gloo")

    for case in test_cases:
        delayer = PrefillDelayer(
            dp_size=world_size,
            attn_tp_size=1,
            cpu_group=cpu_group,
            server_args=SimpleNamespace(
                enable_dp_attention=True,
                disaggregation_mode="null",
                disable_overlap_schedule=False,
            ),
            max_delay_passes=case.max_delay_passes,
            token_usage_low_watermark=case.token_usage_low_watermark,
        )

        for call in case.calls:
            result = delayer._negotiate_should_allow_prefill(
                local_prefillable=call.prefillable[rank],
                token_usage=call.token_usage[rank],
            )

        results_queue.put((rank, case.name, result.output_allow, result.output_reason))

    torch.distributed.destroy_process_group()


_NEGOTIATE_TEST_CASES = [
    NegotiateTestCase(
        name="all_prefillable",
        max_delay_passes=100,
        token_usage_low_watermark=0.8,
        calls=[
            NegotiateCall(
                prefillable=[True, True, True, True],
                token_usage=[0.9, 0.9, 0.9, 0.9],
            )
        ],
        expected_allow=True,
        expected_reason="no_wait",
    ),
    NegotiateTestCase(
        name="all_prefillable_with_previous_wait",
        max_delay_passes=100,
        token_usage_low_watermark=0.8,
        calls=[
            NegotiateCall(
                prefillable=[True, False, True, False],
                token_usage=[0.9, 0.9, 0.9, 0.9],
            ),
            NegotiateCall(
                prefillable=[True, True, True, True],
                token_usage=[0.9, 0.9, 0.9, 0.9],
            ),
        ],
        expected_allow=True,
        expected_reason="wait_success",
    ),
    NegotiateTestCase(
        name="none_prefillable",
        max_delay_passes=100,
        token_usage_low_watermark=0.8,
        calls=[
            NegotiateCall(
                prefillable=[False, False, False, False],
                token_usage=[0.9, 0.9, 0.9, 0.9],
            )
        ],
        expected_allow=True,
        expected_reason="",
    ),
    NegotiateTestCase(
        name="mixed_delay",
        max_delay_passes=100,
        token_usage_low_watermark=0.8,
        calls=[
            NegotiateCall(
                prefillable=[True, False, True, False],
                token_usage=[0.9, 0.9, 0.9, 0.9],
            )
        ],
        expected_allow=False,
        expected_reason="delay",
    ),
    NegotiateTestCase(
        name="mixed_watermark_force_allow",
        max_delay_passes=100,
        token_usage_low_watermark=0.8,
        calls=[
            NegotiateCall(
                prefillable=[True, False, True, False],
                token_usage=[0.5, 0.9, 0.9, 0.9],
            )
        ],
        expected_allow=True,
        expected_reason="token_watermark",
    ),
    NegotiateTestCase(
        name="mixed_watermark_disabled",
        max_delay_passes=100,
        token_usage_low_watermark=None,
        calls=[
            NegotiateCall(
                prefillable=[True, False, True, False],
                token_usage=[0.5, 0.9, 0.9, 0.9],
            )
        ],
        expected_allow=False,
        expected_reason="delay",
    ),
    NegotiateTestCase(
        name="mixed_watermark_not_prefillable",
        max_delay_passes=100,
        token_usage_low_watermark=0.8,
        calls=[
            NegotiateCall(
                prefillable=[False, False, True, False],
                token_usage=[0.5, 0.9, 0.9, 0.9],
            )
        ],
        expected_allow=False,
        expected_reason="delay",
    ),
    NegotiateTestCase(
        name="mixed_timeout",
        max_delay_passes=3,
        token_usage_low_watermark=0.8,
        calls=[
            NegotiateCall(
                prefillable=[True, False, True, False],
                token_usage=[0.9, 0.9, 0.9, 0.9],
            ),
            NegotiateCall(
                prefillable=[True, False, True, False],
                token_usage=[0.9, 0.9, 0.9, 0.9],
            ),
            NegotiateCall(
                prefillable=[True, False, True, False],
                token_usage=[0.9, 0.9, 0.9, 0.9],
            ),
        ],
        expected_allow=True,
        expected_reason="wait_timeout",
    ),
]


class TestPrefillDelayerNegotiate(unittest.TestCase):
    def test_negotiate(self):
        world_size = 4
        test_cases = _NEGOTIATE_TEST_CASES

        ctx = mp.get_context("spawn")
        results_queue = ctx.Queue()
        port = 29500 + os.getpid() % 1000

        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=_run_negotiate_test,
                args=(rank, world_size, test_cases, results_queue, port),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results = defaultdict(dict)
        for _ in range(world_size * len(test_cases)):
            rank, case_name, output_allow, output_reason = results_queue.get()
            results[case_name][rank] = (output_allow, output_reason)

        for case in test_cases:
            for rank in range(world_size):
                output_allow, output_reason = results[case.name][rank]
                self.assertEqual(
                    (output_allow, output_reason),
                    (case.expected_allow, case.expected_reason),
                    f"Case {case.name} rank {rank}",
                )


# ============================ E2E Tests ============================


class TestPrefillDelayerThroughputOnlineServing(CustomTestCase):
    def test_throughput_comparison(self):
        _run_throughput_comparison(
            self,
            test_name="online_serving",
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
            min_improvement_pct=5,
        )


class TestPrefillDelayerThroughputOfflineGen(CustomTestCase):
    def test_throughput_comparison(self):
        _run_throughput_comparison(
            self,
            test_name="offline_gen",
            other_launch_args=["--max-total-tokens", "200000"],
            other_benchmark_args=dict(
                num_prompts=800,
                random_input_len=30000,
                random_output_len=500,
            ),
            token_usage_low_watermark=0.8,
            min_improvement_pct=20,
        )


def _run_throughput_comparison(
    test_case,
    test_name: str,
    other_launch_args,
    other_benchmark_args,
    min_improvement_pct: float,
    token_usage_low_watermark: float = None,
):
    common_kwargs = dict(
        debug_name=test_name,
        other_launch_args=other_launch_args,
        other_benchmark_args=other_benchmark_args,
        token_usage_low_watermark=token_usage_low_watermark,
    )
    res_enabled = _run_throughput_test(prefill_delayer=True, **common_kwargs)
    res_disabled = _run_throughput_test(prefill_delayer=False, **common_kwargs)

    _assert_throughput_improvement(
        test_case,
        test_name=test_name,
        res_enabled=res_enabled,
        res_disabled=res_disabled,
        min_improvement_pct=min_improvement_pct,
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

    print(f"=== {debug_name} ({prefill_delayer=}) ===")
    res["total_throughput"] = res["input_throughput"] + res["output_throughput"]
    print(f"Input throughput: {res['input_throughput']:.2f} token/s")
    print(f"Output throughput: {res['output_throughput']:.2f} token/s")
    print(f"Total throughput: {res['total_throughput']:.2f} token/s")

    return res


def _assert_throughput_improvement(
    test_case,
    test_name: str,
    res_enabled: dict,
    res_disabled: dict,
    min_improvement_pct: float,
):
    test_case.assertEqual(
        WORLD_SIZE,
        "8",
        f"This test requires 8 GPUs to properly measure throughput improvement, got {WORLD_SIZE}",
    )

    enabled = res_enabled["total_throughput"]
    disabled = res_disabled["total_throughput"]
    improvement_pct = (enabled - disabled) / disabled * 100

    print(f"\n=== {test_name} Throughput Comparison ===")
    print(
        f"Total: enabled={enabled:.2f}, disabled={disabled:.2f}, improvement={improvement_pct:.2f}%"
    )

    test_case.assertGreaterEqual(
        improvement_pct,
        min_improvement_pct,
        f"{test_name}: Throughput improvement ({improvement_pct:.2f}%) < {min_improvement_pct}%",
    )


class TestPrefillDelayerTokenUsageLowWatermark(CustomTestCase):
    def test_1_with_low_watermark(self):
        # The kv cache size here is deliberately small, thus we use smaller token usage
        self._run(token_usage_low_watermark=0.5)

    def test_2_without_low_watermark(self):
        self._run(token_usage_low_watermark=None)

    def _run(self, token_usage_low_watermark):
        model = "Qwen/Qwen3-0.6B"
        base_url = DEFAULT_URL_FOR_TEST
        world_size = int(WORLD_SIZE)

        process = _launch_server(
            model=model,
            base_url=base_url,
            prefill_delayer=True,
            other_args=["--max-total-tokens", "50000"],
            # e.g. gen throughput is 370 tok/s on H200.
            # Will need a different threshold on B200
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
                    f"DP rank {dp_rank} req {req_idx}: elapsed={elapsed:.2f}s, thresh={thresh}, enabled={enabled}. "
                    f"Maybe you need a different `max_delay_passes` when using hardware other than H200.",
                )

        try:
            asyncio.run(run_test())

            metrics_text = _print_prefill_delayer_metrics(base_url, expect_metrics=True)
            if token_usage_low_watermark is not None:
                total = _sum_prometheus_metric_values(metrics_text, "token_watermark")
                self.assertGreater(total, 0, "Expected token_watermark > 0")
                print(f"total token_watermark: {total}")
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

    return popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=[
            "--trust-remote-code",
            "--tp",
            WORLD_SIZE,
            "--enable-dp-attention",
            "--dp",
            WORLD_SIZE,
            "--chunked-prefill-size",
            "131072",
            "--mem-fraction-static",
            "0.6",
            "--enable-metrics",
            *(["--enable-prefill-delayer"] if prefill_delayer else []),
            "--prefill-delayer-max-delay-passes",
            str(max_delay_passes),
            *(
                [
                    "--prefill-delayer-token-usage-low-watermark",
                    str(token_usage_low_watermark),
                ]
                if token_usage_low_watermark is not None
                else []
            ),
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


def _sum_prometheus_metric_values(metrics_text: str, label_value: str) -> int:
    matches = re.findall(rf'{label_value}".*?\}} (\d+)', metrics_text)
    return sum(int(m) for m in matches)


if __name__ == "__main__":
    unittest.main()
