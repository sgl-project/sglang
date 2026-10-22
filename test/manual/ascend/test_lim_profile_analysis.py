"""Derived properties for diagnostic CANN timing and complete replay selection.

Manual because the measured runtime is an Ascend-specific benchmark helper,
rather than an SRT module. No server or accelerator is required by these cases.
"""

import importlib.util
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.test.test_utils import CustomTestCase

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "lim_profile_analysis", ROOT / "benchmark/hisparse/analyze_lim_profile.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
client_spec = importlib.util.spec_from_file_location(
    "lim_profile_client", ROOT / "benchmark/hisparse/profile_lim_model.py"
)
client = importlib.util.module_from_spec(client_spec)
client_spec.loader.exec_module(client)
precision_spec = importlib.util.spec_from_file_location(
    "lim_precision_client", ROOT / "benchmark/hisparse/check_lim_decode_precision.py"
)
precision = importlib.util.module_from_spec(precision_spec)
precision_spec.loader.exec_module(precision)


class TestLIMProfileAnalysis(CustomTestCase):
    def test_discarded_chunk_samples_do_not_advance_teacher_history(self):
        # Intermediate prefill chunks invoke the processor but emit no token.
        # Forcing/capturing those calls shifts the history before decode starts.
        encoded = precision.processor_string()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            torch.distributed, "get_rank", return_value=0
        ):
            params = {
                "capture_root": directory,
                "tag": "test",
                "case": "one",
                "force_ids": [0, 1],
                "prefill_sampling_calls_to_skip": 2,
            }
            original = torch.tensor([[1.0, 2.0, 3.0]])
            processor = CustomLogitProcessor.from_str(encoded)
            for _ in range(2):
                actual = processor(original.clone(), [params])
                torch.testing.assert_close(actual, original)
            self.assertFalse((Path(directory) / "test").exists())
            processor(original.clone(), [params])
            # Rebuilding at a batch/prefill boundary must preserve the request's
            # raw-call counter, rather than restarting instance-local state.
            processor = type(processor)()
            processor(original.clone(), [params])
            target = Path(directory) / "test/one"
            steps = [
                json.loads(r) for r in (target / "steps.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["step"] for r in steps], [0, 1])
            self.assertEqual([r["raw_sampling_call"] for r in steps], [2, 3])
            self.assertEqual([r["forced_token"] for r in steps], [0, 1])
            for step in range(2):
                torch.testing.assert_close(
                    torch.load(target / f"{step:04d}.pt", weights_only=True),
                    original[0],
                )
            self.assertEqual(precision.partial_prefill_sample_count(8193, 4096), 2)
            self.assertEqual(precision.partial_prefill_sample_count(8192, 4096), 1)

    def test_decode_window_excludes_prefill_and_batch_drain(self):
        # Counts jump at prefill, final SSE metadata can repeat a count, and
        # one request keeps decoding after the first request ends. None of
        # those should inflate the full-batch throughput numerator.
        states = [
            {"events": [(1, 10), (2, 11), (3, 12), (4, 13), (4.2, 13)]},
            {"events": [(1.2, 10), (2.2, 11), (3.2, 12), (4.5, 13)]},
        ]
        result = client.measure_full_batch(states, 1.25)
        self.assertEqual(result["output_tokens_in_window"], 5)
        self.assertAlmostEqual(result["output_tokens_per_second"], 5 / 2.95)
        self.assertEqual(result["median_stream_token_interval_ms"], 1000)

    def test_profile_control_plain_text_and_server_info_json(self):
        # The real profiler endpoint returns plain text. Parsing every success
        # as JSON aborted the workload immediately after arming the profiler.
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                json_reply = self.path == "/get_server_info"
                self.send_response(200)
                self.send_header(
                    "Content-Type", "application/json" if json_reply else "text/plain"
                )
                self.end_headers()
                self.wfile.write(
                    b'{"tp_size":16}' if json_reply else b"Profiling started"
                )

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        args = SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}")
        try:
            self.assertEqual(client.http(args, "/start_profile"), "Profiling started")
            self.assertEqual(client.http(args, "/get_server_info"), {"tp_size": 16})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_nested_overlap_is_not_cumulative_time(self):
        # The same 10 us span appears twice, and a second stream is nested.
        # Adding durations would report 25 us of work in a 10 us window.
        left = [(0, 10), (0, 10), (2, 7)]
        right = [(5, 15), (15, 20)]
        self.assertEqual(analysis.interval_union(left), 10)
        self.assertEqual(analysis.overlap_length(left, right), 5)
        self.assertEqual(analysis.interval_union(left + right), 20)

    def test_partial_graph_and_copy_direction_are_rejected(self):
        # A complete two-layer replay followed by a truncated replay. Epoch
        # timestamps are string decimals; 0.125 us must survive normalization.
        origin = 1792630000000000
        events = []

        def task(name, start, duration, task_id):
            events.append(
                {
                    "ph": "X",
                    "name": name,
                    "ts": f"{origin + start}.125",
                    "dur": duration,
                    "args": {
                        "Task Type": "AI_VECTOR_CORE",
                        "Model Id": 7,
                        "Physic Stream Id": 4,
                        "Task Id": task_id,
                    },
                }
            )

        task("anchor", 0, 1, 0)
        for index in range(5):
            task("unidex_copy_21", 2 + 2 * index, 1, index + 1)
        task("SparseFlashAttention", 12, 2, 6)
        # Four copies cannot be attributed to five directions, even though an
        # SFA follows them. Count the SFA but reject the direction assignment.
        for index in range(4):
            task("unidex_copy_21", 16 + 2 * index, 1, 7 + index)
        task("SparseFlashAttention", 24, 2, 11)
        task("anchor", 100, 1, 0)
        task("SparseFlashAttention", 110, 2, 6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(events))
            report = analysis.analyze(path, expected_layers=2)
        self.assertEqual(report["accepted_graph_replays"], 1)
        self.assertEqual(report["components"]["SFA"]["calls"], 2)
        self.assertEqual(report["raw_components"]["SFA"]["calls"], 3)
        self.assertEqual(report["median_device_graph_span_ms"], 0.026)
        self.assertEqual(report["complete_serial_five_copy_groups"], 1)
        self.assertEqual(report["ambiguous_copy_group_sizes"], {4: 1})
        self.assertEqual(report["timestamp_origin_us"], f"{origin}.125")


if __name__ == "__main__":
    unittest.main()
