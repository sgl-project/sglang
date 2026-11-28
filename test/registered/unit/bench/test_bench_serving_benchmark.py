"""Regression tests for bench_serving's benchmark() run, against a stub server."""

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import benchmark, get_request, set_global_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=18, suite="base-a-test-cpu")

# Gap between the two trace rows; far above the stub server's response time.
TRACE_GAP_MS = 400.0


class _ServerHandler(BaseHTTPRequestHandler):
    """Records the prompt and arrival time of every chat request, and every
    cache flush."""

    def do_POST(self):  # noqa: N802
        arrival = time.monotonic()
        if self.path.startswith("/flush_cache"):
            self.server.flushes.append(self.path)
            self.send_response(200)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.arrivals.append((body["messages"][0]["content"], arrival))
        message = {"role": "assistant", "content": "hi"}
        choice = {"index": 0, "message": message, "finish_reason": "length"}
        usage = {"completion_tokens": 1, "prompt_tokens": 1}
        payload = json.dumps({"choices": [choice], "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class _BenchmarkRunTestCase(CustomTestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ServerHandler)
        self.server.arrivals = []
        self.server.flushes = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.output_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.output_dir.cleanup)

    def _run(self, rows, backend, use_trace_timestamps=False):
        set_global_args(
            Namespace(
                backend=backend,
                dataset_name="custom",
                num_prompts=len(rows),
                warmup_requests=0,
                disable_stream=True,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
                cache_report=False,
                plot_throughput=False,
                goodput=None,
                sharegpt_output_len=None,
                random_input_len=None,
                random_output_len=None,
                random_range_ratio=None,
                output_file=os.path.join(self.output_dir.name, "result.jsonl"),
                output_details=False,
            )
        )
        base_url = f"http://127.0.0.1:{self.server.server_port}"
        return asyncio.run(
            benchmark(
                backend=backend,
                api_url=base_url + "/v1/chat/completions",
                base_url=base_url,
                model_id="dummy-model",
                tokenizer=_Tokenizer(),
                input_requests=rows,
                request_rate=float("inf"),
                max_concurrency=None,
                disable_tqdm=True,
                lora_names=None,
                lora_request_distribution=None,
                lora_zipf_alpha=None,
                extra_request_body={},
                profile=False,
                warmup_requests=0,
                use_trace_timestamps=use_trace_timestamps,
            )
        )


class TestBenchServingTraceTimestamps(_BenchmarkRunTestCase):
    def test_requests_are_sent_at_their_trace_timestamps(self):
        """With use_trace_timestamps, benchmark() must send each request at its trace
        offset, not all at once at the request rate while reporting a "trace" run."""
        rows = [
            DatasetRow(
                prompt="late", prompt_len=1, output_len=1, timestamp=TRACE_GAP_MS
            ),
            DatasetRow(prompt="early", prompt_len=1, output_len=1, timestamp=0.0),
        ]

        result = self._run(rows, backend="vllm-chat", use_trace_timestamps=True)

        self.assertEqual(result["request_rate"], "trace")
        self.assertEqual(result["completed"], 2)
        (first, first_arrival), (second, second_arrival) = self.server.arrivals
        self.assertEqual([first, second], ["early", "late"])
        self.assertGreaterEqual(
            second_arrival - first_arrival, TRACE_GAP_MS / 1000.0 * 0.75
        )

    def test_rows_without_timestamps_are_rejected(self):
        """A dataset that carries no timestamps cannot be replayed as a trace."""
        rows = [DatasetRow(prompt="p", prompt_len=1, output_len=1) for _ in range(2)]

        async def drain():
            return [
                row
                async for row in get_request(
                    rows, float("inf"), use_trace_timestamps=True
                )
            ]

        with self.assertRaisesRegex(ValueError, "timestamp"):
            asyncio.run(drain())


class TestBenchServingCiCacheFlush(_BenchmarkRunTestCase):
    def test_ci_flag_is_read_as_the_declared_boolean(self):
        """In CI an sglang backend's cache is flushed before the measured run;
        SGLANG_IS_IN_CI must count for every spelling the declared boolean takes."""
        row = DatasetRow(prompt="p", prompt_len=1, output_len=1)
        for value, flushes in [("yes", 1), ("0", 0)]:
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"SGLANG_IS_IN_CI": value}),
            ):
                self.server.flushes.clear()
                self._run([row], backend="sglang-oai-chat")
                self.assertEqual(len(self.server.flushes), flushes)


class _GenerateSSEHandler(BaseHTTPRequestHandler):
    """Stub /generate endpoint streaming configured SSE frames."""

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/flush_cache"):
            self.send_response(200)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        frames = self.server.response_frames
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for frame in frames:
            self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def do_GET(self):  # noqa: N802
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


class TestBenchServingShortOutputWarning(CustomTestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _GenerateSSEHandler)
        self.server.requests = []
        self.server.response_frames = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.output_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.output_dir.cleanup)

    def _run_generate(
        self,
        rows,
        frames,
        *,
        disable_ignore_eos=False,
        extra_request_body=None,
    ):
        self.server.requests.clear()
        self.server.response_frames = frames
        set_global_args(
            Namespace(
                backend="sglang",
                dataset_name="custom",
                num_prompts=len(rows),
                warmup_requests=0,
                disable_stream=False,
                disable_ignore_eos=disable_ignore_eos,
                temperature=0.0,
                top_p=1.0,
                return_logprob=False,
                return_routed_experts=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
                print_requests=False,
                tokenizer="",
                header=None,
                cache_report=False,
                plot_throughput=False,
                goodput=None,
                sharegpt_output_len=None,
                random_input_len=None,
                random_output_len=None,
                random_range_ratio=None,
                output_file=os.path.join(self.output_dir.name, "result.jsonl"),
                output_details=False,
            )
        )
        base_url = f"http://127.0.0.1:{self.server.server_port}"
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = asyncio.run(
                benchmark(
                    backend="sglang",
                    api_url=base_url + "/generate",
                    base_url=base_url,
                    model_id="dummy-model",
                    tokenizer=_Tokenizer(),
                    input_requests=rows,
                    request_rate=float("inf"),
                    max_concurrency=None,
                    disable_tqdm=True,
                    lora_names=None,
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body=extra_request_body or {},
                    profile=False,
                    warmup_requests=0,
                    use_trace_timestamps=False,
                )
            )
        return result, [
            str(w.message)
            for w in caught
            if "fixed-length benchmark request(s) finished early" in str(w.message)
        ]

    def test_warns_once_when_fixed_length_requests_are_clamped(self):
        rows = [
            DatasetRow(prompt=f"p{i}", prompt_len=10, output_len=1000) for i in range(8)
        ]
        frames = [
            {
                "text": "x",
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "finish_reason": {"type": "length", "length": 1},
                },
            }
        ]
        result, warns = self._run_generate(rows, frames)
        self.assertEqual(result["completed"], 8)
        self.assertEqual(result["total_output_tokens"], 8)
        self.assertEqual(len(warns), 1)
        self.assertIn("8 fixed-length benchmark request(s)", warns[0])
        self.assertIn("8/8000", warns[0])

    def test_captures_textless_terminal_sse_frame(self):
        rows = [DatasetRow(prompt="p0", prompt_len=10, output_len=1000)]
        frames = [
            {"text": "x", "meta_info": {"prompt_tokens": 10, "completion_tokens": 1}},
            {
                "text": "",
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "finish_reason": {"type": "length", "length": 1},
                },
            },
        ]
        _result, warns = self._run_generate(rows, frames)
        self.assertEqual(len(warns), 1)
        self.assertIn("1/1000", warns[0])

    def test_no_warning_for_full_output_natural_stop_or_disabled_checks(self):
        rows = [DatasetRow(prompt="p0", prompt_len=10, output_len=100)]
        full_frames = [
            {
                "text": "x",
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 100,
                    "finish_reason": {"type": "length", "length": 100},
                },
            }
        ]
        self.assertEqual(self._run_generate(rows, full_frames)[1], [])

        eos_frames = [
            {
                "text": "x",
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "finish_reason": {"type": "stop"},
                },
            }
        ]
        self.assertEqual(self._run_generate(rows, eos_frames)[1], [])

        short_frames = [
            {
                "text": "x",
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "finish_reason": {"type": "length", "length": 1},
                },
            }
        ]
        self.assertEqual(
            self._run_generate(rows, short_frames, disable_ignore_eos=True)[1], []
        )

        missing_count_frames = [
            {"text": "", "meta_info": {"finish_reason": {"type": "length"}}}
        ]
        self.assertEqual(self._run_generate(rows, missing_count_frames)[1], [])

        self.assertEqual(
            self._run_generate(
                rows,
                short_frames,
                extra_request_body={
                    "sampling_params": {
                        "max_new_tokens": 1,
                        "ignore_eos": True,
                    }
                },
            )[1],
            [],
        )
        self.assertEqual(
            self._run_generate(
                rows,
                short_frames,
                extra_request_body={
                    "sampling_params": {
                        "max_new_tokens": 100,
                        "ignore_eos": True,
                        "stop": ["END"],
                    }
                },
            )[1],
            [],
        )


if __name__ == "__main__":
    unittest.main()
