"""Regressions for continuous benchmark scheduling and durable phase results."""

import asyncio
import contextlib
import io
import json
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from sglang.benchmark import serving
from sglang.benchmark.datasets import DatasetRow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _row(prompt, phase, limit=1, timestamp=0):
    return DatasetRow(
        prompt,
        1,
        1,
        timestamp=timestamp,
        phase=phase,
        phase_duration=1,
        phase_request_rate=1,
        phase_max_concurrency=limit,
    )


def _output():
    return serving.RequestFuncOutput(
        success=True,
        generated_text="hi",
        output_len=1,
        prompt_len=1,
        latency=0.1,
        ttft=0.01,
        stream_complete=True,
        usage_present=True,
    )


class TestDynamicServing(CustomTestCase):
    def _benchmark(self, directory, rows, request_func, concurrency=2):
        options = Namespace(
            dataset_name="dynamic",
            backend="vllm-chat",
            num_prompts=len(rows),
            warmup_requests=0,
            plot_throughput=False,
            goodput=[],
            output_details=True,
            sharegpt_output_len=None,
            random_input_len=1,
            random_output_len=1,
            random_range_ratio=1.0,
            cache_report=False,
            output_file=str(Path(directory) / "results.jsonl"),
        )

        async def run():
            with (
                patch.object(serving, "args", options, create=True),
                patch.dict(serving.ASYNC_REQUEST_FUNCS, {"vllm-chat": request_func}),
                patch.object(
                    serving.requests, "get", return_value=Namespace(status_code=404)
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                return await serving.benchmark(
                    backend="vllm-chat",
                    api_url="http://stub/v1/chat/completions",
                    base_url="http://stub",
                    model_id="test",
                    tokenizer=Namespace(
                        encode=lambda text, **kwargs: [1] if text else []
                    ),
                    input_requests=rows,
                    request_rate=float("inf"),
                    max_concurrency=concurrency,
                    disable_tqdm=True,
                    lora_names=[],
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                    warmup_requests=0,
                    max_pending_requests=4096,
                    phase_output_file=str(Path(directory) / "phases.jsonl"),
                )

        return run()

    def test_completed_phase_is_written_while_later_request_is_pending(self):
        """Below the pending limit, a finished phase still needs durable evidence."""

        async def run(directory):
            slow_started, release = asyncio.Event(), asyncio.Event()

            async def request(request_func_input, pbar):
                if request_func_input.prompt == "slow":
                    slow_started.set()
                    await release.wait()
                return _output()

            task = asyncio.create_task(
                self._benchmark(
                    directory,
                    [_row("fast", "first"), _row("slow", "second")],
                    request,
                )
            )
            try:
                await asyncio.wait_for(slow_started.wait(), 5)
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                path = Path(directory) / "phases.jsonl"
                self.assertTrue(
                    path.exists(), "finished phase was buffered until dispatch drain"
                )
                phases = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([phase["name"] for phase in phases], ["first"])
                self.assertEqual(phases[0]["completed"], 1)
            finally:
                release.set()
                await task
            phases = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([phase["name"] for phase in phases], ["first", "second"])

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory))

    def test_phase_waiter_does_not_hold_a_global_slot(self):
        """A queued request in phase A must leave a global slot for phase B."""

        async def run(directory):
            first_started, other_started = asyncio.Event(), asyncio.Event()
            release = asyncio.Event()

            async def request(request_func_input, pbar):
                if request_func_input.prompt == "a1":
                    first_started.set()
                    await release.wait()
                elif request_func_input.prompt == "b1":
                    other_started.set()
                return _output()

            task = asyncio.create_task(
                self._benchmark(
                    directory,
                    [_row("a1", "a"), _row("a2", "a"), _row("b1", "b")],
                    request,
                )
            )
            try:
                await asyncio.wait_for(first_started.wait(), 5)
                # a2 and b1 are already queued; yield once to finish that ready batch.
                await asyncio.sleep(0)
                self.assertTrue(
                    other_started.is_set(), "a2 occupied the spare global slot"
                )
            finally:
                release.set()
                await task

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory))

    def test_slow_metrics_does_not_delay_phase_evidence_or_event_loop(self):
        """A blocked /metrics fetch must let another phase finish and flush."""
        entered, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def snapshot(base_url, metrics):
            entered.set()
            try:
                # Timeout only prevents a deadlock on the old synchronous code.
                release.wait(timeout=2)
                return {"requests_total": 1}
            finally:
                finished.set()

        async def run(directory):
            rows = [_row("one", "first"), _row("two", "second")]
            path = Path(directory) / "phases.jsonl"
            writer = serving.PhaseResultWriter(
                rows, str(path), "http://stub", ["requests_total"]
            )
            with patch.object(serving, "_prometheus_snapshot", side_effect=snapshot):
                try:
                    writer.record(rows[0], _output())
                    self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                    self.assertFalse(
                        finished.is_set(), "snapshot blocked the event loop"
                    )
                    writer.record(rows[1], _output())
                    phases = [
                        json.loads(line) for line in path.read_text().splitlines()
                    ]
                    self.assertEqual(
                        [phase["name"] for phase in phases], ["first", "second"]
                    )
                finally:
                    release.set()
                    await writer.finish()
            snapshots = [
                json.loads(line)
                for line in path.with_suffix(".metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [entry["name"] for entry in snapshots], ["first", "second"]
            )
            self.assertTrue(all("prometheus_snapshot" in entry for entry in snapshots))

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory))

    def test_failed_metrics_are_reported_without_losing_completed_phases(self):
        """A missing metrics endpoint must be visible instead of silently disabled."""

        async def run(directory):
            rows = [_row("one", "first"), _row("two", "second")]
            path = Path(directory) / "phases.jsonl"
            writer = serving.PhaseResultWriter(
                rows, str(path), "http://stub", ["requests_total"]
            )
            with patch.object(
                serving,
                "_prometheus_snapshot",
                side_effect=serving.requests.HTTPError("404 Not Found"),
            ):
                with self.assertWarnsRegex(
                    UserWarning, "Prometheus snapshots disabled"
                ):
                    for row in rows:
                        writer.record(row, _output())
                    await writer.finish()
            self.assertEqual(len(path.read_text().splitlines()), 2)
            errors = [
                json.loads(line)
                for line in path.with_suffix(".metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([item["name"] for item in errors], ["first", "second"])
            self.assertTrue(
                all("404 Not Found" in item["prometheus_error"] for item in errors)
            )

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory))

    def test_trace_timestamp_validation_precedes_sorting(self):
        """Missing or nonfinite timestamps report the row instead of a sort TypeError."""

        async def consume(rows):
            async def collect():
                return [
                    row
                    async for row in serving.get_request(
                        rows, float("inf"), use_trace_timestamps=True
                    )
                ]

            return await asyncio.wait_for(collect(), 1)

        for value in (None, float("nan"), float("inf"), -1):
            with self.subTest(timestamp=value):
                rows = [
                    _row("one", "first", timestamp=0),
                    _row("two", "second", timestamp=value),
                ]
                with self.assertRaisesRegex(
                    ValueError, "--use-trace-timestamps.*row 1"
                ):
                    asyncio.run(consume(rows))
        rows = [
            _row("late", "second", timestamp=1),
            _row("early", "first", timestamp=0),
        ]
        self.assertEqual(
            [row.prompt for row in asyncio.run(consume(rows))], ["early", "late"]
        )


if __name__ == "__main__":
    unittest.main()
