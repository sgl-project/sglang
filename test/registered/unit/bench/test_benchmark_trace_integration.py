"""Exercise real benchmark timing/export against a local streaming HTTP server."""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sglang.benchmark import serving
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/convert_benchmark_to_trace.py"


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class _StreamingHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.startswith("/flush_cache"):
            self.send_response(200)
            self.end_headers()
            return
        payload = json.loads(body)
        prompt = payload.get("prompt", payload.get("text"))
        record = {"received": time.perf_counter(), "sent": []}
        with self.server.record_lock:
            self.server.records[prompt] = record
        if prompt == "fail":
            self.send_error(503, "intentional test failure")
            return
        if self.server.barrier is not None:
            try:
                self.server.barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                self.send_error(500, "concurrent request did not arrive")
                return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for index, text in enumerate(("a", "abc", "abcde")):
            time.sleep(0.04)
            if self.path == "/generate":
                data = {"text": text, "meta_info": {"completion_tokens": len(text)}}
            else:
                data = {
                    "choices": [{"text": ("a", "bc", "de")[index]}],
                    "usage": {"completion_tokens": len(text)},
                }
            record["sent"].append(time.perf_counter())
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class TestBenchmarkTraceIntegration(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "benchmark.jsonl"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingHandler)
        self.server.daemon_threads = True
        self.server.records = {}
        self.server.record_lock = threading.Lock()
        self.server.barrier = None
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.previous_args = getattr(serving, "args", None)
        self.addCleanup(serving.set_global_args, self.previous_args)

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _benchmark(self, backend, prompts, concurrency, details=True):
        serving.set_global_args(
            argparse.Namespace(
                backend=backend,
                dataset_name="random",
                disable_stream=False,
                disable_ignore_eos=False,
                return_logprob=False,
                return_routed_experts=False,
                top_logprobs_num=0,
                token_ids_logprob=None,
                logprob_start_len=-1,
                temperature=0.0,
                top_p=1.0,
                warmup_requests=0,
                plot_throughput=False,
                cache_report=False,
                output_file=str(self.output),
                output_details=details,
                sharegpt_output_len=None,
                random_input_len=1,
                random_output_len=5,
                random_range_ratio=1.0,
            )
        )
        path = "/generate" if backend == "sglang" else "/v1/completions"
        return asyncio.run(
            asyncio.wait_for(
                serving.benchmark(
                    backend=backend,
                    api_url=self.base_url + path,
                    base_url=self.base_url,
                    model_id="local-test-model",
                    tokenizer=_Tokenizer(),
                    input_requests=[
                        serving.DatasetRow(prompt=prompt, prompt_len=1, output_len=5)
                        for prompt in prompts
                    ],
                    request_rate=float("inf"),
                    max_concurrency=concurrency,
                    disable_tqdm=True,
                    lora_names=None,
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                    warmup_requests=0,
                ),
                timeout=15,
            )
        )

    def test_serialized_openai_arrivals_exclude_concurrency_queue(self):
        result = self._benchmark("sglang-oai", ["first", "second"], 1)
        self.assertEqual(result["successes"], [True, True])
        self.assertEqual(result["output_lens"], [5, 5])
        self.assertEqual([len(itls) for itls in result["itls"]], [2, 2])
        first_end = (
            result["arrival_times"][0] + result["ttfts"][0] + sum(result["itls"][0])
        )
        self.assertGreaterEqual(result["arrival_times"][1], first_end - 0.001)
        self.assertGreaterEqual(
            self.server.records["second"]["received"],
            self.server.records["first"]["sent"][-1],
        )
        self.assertTrue(all(ttft > 0 for ttft in result["ttfts"]))
        saved = json.loads(self.output.read_text())
        self.assertEqual(saved["arrival_times"], result["arrival_times"])
        self.assertEqual(saved["model_id"], "local-test-model")

    def test_native_concurrent_chunks_preserve_interpolated_itls(self):
        self.server.barrier = threading.Barrier(2)
        result = self._benchmark("sglang", ["first", "second"], 2)
        self.assertEqual(result["successes"], [True, True])
        self.assertEqual(result["output_lens"], [5, 5])
        self.assertEqual([len(itls) for itls in result["itls"]], [4, 4])
        for index, prompt in enumerate(("first", "second")):
            intervals = result["itls"][index]
            self.assertEqual(intervals[0], intervals[1])
            self.assertEqual(intervals[2], intervals[3])
            self.assertTrue(all(interval > 0 for interval in intervals))
            record = self.server.records[prompt]
            self.assertEqual(len(record["sent"]), 3)
            self.assertGreater(record["sent"][0], record["received"])
        latest_start = max(result["arrival_times"])
        earliest_output = min(
            start + ttft
            for start, ttft in zip(result["arrival_times"], result["ttfts"])
        )
        self.assertLess(latest_start, earliest_output)

    def test_failed_request_jsonl_roundtrip_and_summary_only_output(self):
        result = self._benchmark("sglang-oai", ["first", "fail"], 2)
        self.assertEqual(result["successes"], [True, False])
        self.assertIn("intentional test failure", result["errors"][1])
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["output_lens"][1], 0)
        self._benchmark("sglang-oai", ["summary"], 1, details=False)
        records = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(len(records), 2)
        for key in ("arrival_times", "successes", "ttfts", "itls", "model_id"):
            self.assertNotIn(key, records[1])
        process = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(SCRIPT),
                "--input",
                str(self.output),
                "--run-index",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        trace = json.loads(self.output.with_name("benchmark_trace.json").read_text())
        self.assertEqual(trace["metadata"]["requests_included"], 1)
        self.assertEqual(trace["metadata"]["requests_skipped"]["failed"], 1)
        events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
        self.assertEqual(len(events), 3)
        timestamp = result["arrival_times"][0]
        for event, duration in zip(events, [result["ttfts"][0], *result["itls"][0]]):
            self.assertEqual(event["tid"], 1)
            start_us = round(timestamp * 1e6)
            timestamp += duration
            self.assertIsInstance(event["ts"], int)
            self.assertIsInstance(event["dur"], int)
            self.assertEqual(event["ts"], start_us)
            self.assertEqual(event["dur"], round(timestamp * 1e6) - start_us)
        for previous, current in zip(events, events[1:]):
            self.assertEqual(previous["ts"] + previous["dur"], current["ts"])


if __name__ == "__main__":
    unittest.main()
