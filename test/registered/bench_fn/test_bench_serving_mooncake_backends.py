"""Replay the same timestamped workload through native and OpenAI clients."""

import asyncio
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.benchmark import serving
from sglang.benchmark.datasets import mooncake
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class MetricsReached(Exception):
    """Stop after the real benchmark scheduler completes all client calls."""


class TestMooncakeBackends(unittest.TestCase):
    def test_native_and_openai_replay_raw_trace_rows_at_trace_rate(self):
        for backend in ["sglang", "vllm", "sglang-oai", "sglang-oai-chat"]:
            with self.subTest(backend=backend):
                self._replay(backend)

    def test_burst_records_client_wait_and_failed_request_terminal_times(self):
        outputs = self._replay("vllm", concurrency=1, burst=True)
        timings = [output.client_timing for output in outputs]
        self.assertEqual([t["scheduled_arrival_s"] for t in timings], [0, 0, 0])
        self.assertEqual([t["ready_s"] for t in timings], [0, 0, 0])
        self.assertEqual([t["dispatch_s"] for t in timings], [0, 1, 2])
        self.assertEqual([t["backend_start_s"] for t in timings], [0, 1, 2])
        self.assertEqual([t["terminal_s"] for t in timings], [1, 2, 3])
        self.assertEqual([t["first_token_s"] for t in timings], [0.25, None, 2.25])
        self.assertEqual([t["success"] for t in timings], [True, False, False])
        self.assertEqual([o.latency for o in outputs], [1, 0, 0])
        self.assertEqual(outputs[1].error, "queue full")
        self.assertEqual(outputs[2].error, "failure after partial output")

    def test_prompt_preparation_does_not_shift_planned_arrival(self):
        outputs = self._replay("vllm", preparation_delay=0.25)
        self.assertEqual(
            [o.client_timing["ready_s"] for o in outputs], [0.25, 2.25, 4.25]
        )

    def _replay(self, backend, *, concurrency=8, burst=False, preparation_delay=0):
        serving.set_global_args(
            Namespace(
                dataset_name="mooncake",
                warmup_requests=1,
                mooncake_num_rounds=1,
                plot_throughput=False,
            )
        )
        clock = [100.0]
        sleeps = []
        calls = []
        original_sleep = asyncio.sleep

        async def sleep(delay):
            sleeps.append(delay)
            clock[0] += delay
            await original_sleep(0)

        async def request(request_func_input, pbar=None):
            calls.append(request_func_input)
            if burst and len(calls) > 1:
                index = len(calls) - 2
                started = clock[0]
                await original_sleep(0)
                clock[0] += 1
                return serving.RequestFuncOutput(
                    success=index == 0,
                    start_time=started,
                    ttft=0.25 if index != 1 else 0,
                    latency=1 if index == 0 else 0,
                    output_len=4 if index != 1 else 0,
                    error=["", "queue full", "failure after partial output"][index],
                )
            return serving.RequestFuncOutput(success=True, output_len=4)

        tokenizer = Mock()

        def encode(text):
            clock[0] += preparation_delay
            return list(range(len(text.split())))

        tokenizer.encode.side_effect = encode
        tokenizer.apply_chat_template.side_effect = lambda messages, **_: messages[0][
            "content"
        ]
        trace = [
            {"timestamp": 3000, "hash_ids": [303], "output_length": 4},
            {"timestamp": 1000, "hash_ids": [101], "output_length": 4},
            {"timestamp": 2000, "hash_ids": [202], "output_length": 4},
        ]
        if burst:
            for row in trace:
                row["timestamp"] = 1000
        recorded = {}

        def metrics(**kwargs):
            recorded.update(kwargs)
            raise MetricsReached

        def blocking_sleep(delay):
            sleeps.append(delay)
            clock[0] += delay

        fake_time = SimpleNamespace(perf_counter=lambda: clock[0], sleep=blocking_sleep)
        with (
            patch.dict(serving.ASYNC_REQUEST_FUNCS, {backend: request}),
            patch.object(serving, "time", fake_time),
            patch.object(mooncake, "time", fake_time),
            patch.object(asyncio, "sleep", sleep),
            patch.object(serving, "_get_bool_env_var", return_value=False),
            patch.object(serving, "get_auth_headers", return_value={}),
            patch.object(serving.requests, "get", return_value=Mock(status_code=404)),
            patch.object(serving, "calculate_metrics", side_effect=metrics),
        ):
            with self.assertRaises(MetricsReached):
                asyncio.run(
                    serving.benchmark(
                        backend=backend,
                        api_url="http://fixture/v1/completions",
                        base_url="http://fixture",
                        model_id="fixture",
                        tokenizer=tokenizer,
                        input_requests=trace,
                        request_rate=float("inf"),
                        max_concurrency=concurrency,
                        disable_tqdm=True,
                        lora_names=None,
                        lora_request_distribution=None,
                        lora_zipf_alpha=None,
                        extra_request_body={},
                        profile=False,
                        warmup_requests=1,
                        mooncake_slowdown_factor=2,
                        mooncake_num_rounds=1,
                    )
                )
        self.assertEqual(
            sleeps,
            [1.0] if burst else [1.0, 2.0 - preparation_delay, 2.0 - preparation_delay],
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0].prompt, calls[1].prompt)
        self.assertEqual(calls[0].prompt_len, calls[1].prompt_len)
        markers = [303, 101, 202] if burst else [101, 202, 303]
        for call, marker in zip(calls[1:], markers):
            self.assertIn(str(marker), call.prompt)
            self.assertEqual(call.output_len, 4)
        self.assertEqual(len(recorded["outputs"]), 3)
        self.assertEqual(
            [o.client_timing["scheduled_arrival_s"] for o in recorded["outputs"]],
            [0, 0, 0] if burst else [0, 2, 4],
        )
        self.assertEqual(
            [r.prompt for r in recorded["input_requests"]],
            [r.prompt for r in calls[1:]],
        )
        return recorded["outputs"]


if __name__ == "__main__":
    unittest.main()
