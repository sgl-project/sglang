"""CPU-only protocol, isolation, metric, and comparison tests (no model needed)."""

import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bench_code_edits as bench
from code_edit_report import build_summary, compare_outputs, summarize_trial


def event(ids, finished=False):
    return {
        "output_ids": ids,
        "text": "",
        "prompt_token_ids": [128000, 9],
        "meta_info": {
            "prompt_tokens": 2,
            "completion_tokens": len(ids),
            "finish_reason": {"type": "length"} if finished else None,
            "spec_verify_ct": 64,
            "spec_num_correct_drafts": 192,
            "spec_num_proposed_drafts": 256,
        },
    }


def stream_bytes(events, done=True):
    data = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)
    return data + (b"data: [DONE]\n\n" if done else b"")


class TestStreaming(unittest.TestCase):
    def test_empty_text_and_multitoken_events_count_as_tokens(self):
        with patch.object(bench.time, "perf_counter", side_effect=[10.0, 10.1, 10.3]):
            result = bench.collect_stream(
                io.BytesIO(
                    stream_bytes(
                        [
                            event([]),
                            event([3, 4]),
                            event([3, 4, 5, 6], True),
                        ]
                    )
                ),
                10.0,
            )
        self.assertAlmostEqual(result["ttft_s"], 0.1)
        self.assertAlmostEqual(result["tpot_s"], 0.2 / 3)
        self.assertEqual(result["output_ids"], [3, 4, 5, 6])
        self.assertEqual(len(result["stream_events"]), 2)

    def test_truncated_or_incremental_stream_is_rejected(self):
        for payload in (
            stream_bytes([event([1], True)], done=False),
            stream_bytes([event([1])]),
            stream_bytes([event([1]), event([2], True)]),
            stream_bytes([{"error": "aborted"}]),
        ):
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                bench.collect_stream(io.BytesIO(payload), time.perf_counter())


class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send(self, body, status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        self.server.calls.append(self.path)
        self.send({"success": True, "corpus_token_counts": self.server.corpora})

    def do_POST(self):
        body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null"
        )
        self.server.calls.append(self.path)
        if self.path.startswith("/flush_cache"):
            self.send(b"flushed", 400 if self.server.fail_flush else 200)
        elif self.path == "/add_external_corpus":
            count = sum(len(c) for c in body["token_chunks"])
            self.server.corpora[body["corpus_id"]] = count
            self.send({"success": True, "loaded_token_count": count})
        elif self.path == "/remove_external_corpus":
            self.server.corpora.pop(body["corpus_id"], None)
            self.send({"success": True})
        elif self.path == "/generate":
            self.server.bodies.append(body)
            if self.server.fail_generate:
                self.send(b"generation error", 500)
            else:
                self.send(stream_bytes([event([3]), event([3] * 256, True)]))


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.server.corpora = {}
        self.server.calls = []
        self.server.bodies = []
        self.server.fail_flush = False
        self.server.fail_generate = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = bench.Client(self.server.server_port)
        self.row = {"task_id": 0, "input_ids": [128000, 9], "output_tokens": 256}

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_two_requests_are_isolated_and_send_exact_ids(self):
        for _ in range(2):
            record = bench.measure_request(self.client, self.row, "path_probability")
            self.assertEqual(record["validation_errors"], [])
            self.assertGreater(record["workflow_s"], record["generation_s"])
        expected = [
            "/list_external_corpora",
            "/flush_cache?timeout=10",
            "/add_external_corpus",
            "/list_external_corpora",
            "/generate",
            "/remove_external_corpus",
            "/list_external_corpora",
        ]
        self.assertEqual(self.server.calls, expected * 2)
        self.assertEqual(self.server.corpora, {})
        self.assertEqual(self.server.bodies[0]["input_ids"], self.row["input_ids"])
        self.assertTrue(self.server.bodies[0]["sampling_params"]["ignore_eos"])

    def test_baseline_has_no_corpus_api_calls(self):
        bench.measure_request(self.client, self.row, "baseline")
        self.assertEqual(self.server.calls, ["/flush_cache?timeout=10", "/generate"])

    def test_failed_generation_still_removes_corpus(self):
        self.server.fail_generate = True
        with self.assertRaisesRegex(RuntimeError, "generation error"):
            bench.measure_request(self.client, self.row, "legacy_prob")
        self.assertEqual(self.server.corpora, {})
        self.assertIn("/remove_external_corpus", self.server.calls)

    def test_failed_flush_prevents_loading_and_generation(self):
        self.server.fail_flush = True
        with self.assertRaises(RuntimeError):
            bench.measure_request(self.client, self.row, "legacy_bfs")
        self.assertNotIn("/add_external_corpus", self.server.calls)
        self.assertEqual(self.server.bodies, [])

    def test_unexpected_corpus_prevents_generation(self):
        self.server.corpora = {"leaked-answer": 100}
        with self.assertRaisesRegex(RuntimeError, "isolation"):
            bench.measure_request(self.client, self.row, "path_probability")
        self.assertEqual(self.server.bodies, [])

    def test_repeated_matrix_excludes_warmup_rotates_modes_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokens = root / "tokens.json"
            tokens.write_text("{}")
            args = SimpleNamespace(
                model_path=root / "model",
                inputs_dir=root,
                tokens=tokens,
                requests=2,
                repeats=2,
                warmup=1,
                modes=["baseline", "legacy_prob"],
                max_trie_depth=18,
                ngram_capacity=1000,
                directory=root / "results",
                attention_backend="flashinfer",
                port=self.server.server_port,
                seed=0,
                mem_fraction_static=0.8,
                disable_overlap_schedule=False,
                disable_cuda_graph=False,
                request_timeout=10,
                trt_results=None,
                fail_on_mismatch=True,
            )
            artifact = {"requests": [self.row, {**self.row, "task_id": 1}]}
            launch_order = []

            @contextmanager
            def fake_launch(args, command, path):
                launch_order.append(path.name.split(".")[0])
                yield 0.5

            with (
                patch.object(bench, "verify_inputs", return_value=(artifact, 0.1)),
                patch.object(bench, "launch_server", side_effect=fake_launch),
                patch.object(bench, "validate_server"),
                patch.object(bench, "capture", return_value="test"),
                patch("builtins.print"),
            ):
                bench.run(args)
            self.assertEqual(
                launch_order,
                ["baseline-1", "legacy_prob-1", "legacy_prob-2", "baseline-2"],
            )
            self.assertEqual(
                len(self.server.bodies), 12
            )  # 4 trials * (1 warmup + 2 measured).
            for path in args.directory.glob("*.requests.jsonl"):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([row["task_id"] for row in rows], [0, 1])
            summary = json.loads((args.directory / "summary.json").read_text())
            self.assertEqual(summary["modes"]["legacy_prob"]["mismatch_count"], 0)
            self.assertEqual(len(summary["modes"]["baseline"]["trials"]), 2)
            self.assertFalse((args.directory / "failure.json").exists())


def record(task_id, seconds=1.0, verify=64):
    return {
        **event([3] * 256, True),
        "task_id": task_id,
        "generation_s": seconds,
        "workflow_s": seconds + 0.5,
        "ttft_s": 0.01,
        "tpot_s": 0.003,
        "setup_s": 0.4,
        "corpus_load_s": 0.2,
        "meta_info": {
            "spec_verify_ct": verify,
            "spec_num_correct_drafts": 256 - verify,
            "spec_num_proposed_drafts": verify * 4,
        },
    }


class TestReports(unittest.TestCase):
    def test_ratios_use_raw_counts_and_workflow_cost(self):
        report = summarize_trial(
            "legacy_bfs", 1, [record(0, verify=64), record(1, verify=128)], 4
        )
        self.assertAlmostEqual(report["acceptance_length"], 512 / 192)
        self.assertAlmostEqual(report["draft_acceptance_rate"], 320 / 768)
        self.assertEqual(report["generation_tok_s"], 256)
        self.assertAlmostEqual(report["workflow_tok_s"], 512 / 3)
        self.assertEqual(report["window_tok_s"], 128)

    def test_pairing_repeated_baselines_and_first_divergence(self):
        trials = []
        for rep in (1, 2):
            for mode in ("baseline", "legacy_bfs"):
                rows = [record(0, seconds=2 if mode == "baseline" else 1)]
                if rep == 2 or mode == "legacy_bfs":
                    rows[0]["output_ids"][17] = 8
                trials.append({**summarize_trial(mode, rep, rows, 4), "requests": rows})
        summary = build_summary(list(reversed(trials)))
        self.assertEqual(summary["modes"]["baseline"]["mismatch_count"], 1)
        mode = summary["modes"]["legacy_bfs"]
        self.assertEqual(mode["mismatch_count"], 1)
        self.assertEqual(mode["generation_speedup"]["median"], 2)
        divergence = next(t for t in mode["trials"] if t["repetition"] == 1)[
            "mismatches"
        ][0]
        self.assertEqual(divergence["token_index"], 17)
        self.assertEqual(divergence["baseline_token_id"], 3)

    def test_pairing_rejects_missing_or_reordered_tasks(self):
        with self.assertRaises(ValueError):
            compare_outputs([record(0)], [record(1)])

    def test_input_manifest_detects_dataset_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dataset = directory / "dataset.jsonl"
            dataset.write_text(
                json.dumps({"task_id": 0, "prompt": "code", "output_tokens": 256})
                + "\n"
            )
            bench.write_json(
                directory / "manifest.json",
                {
                    "model": "model",
                    "model_revision": "revision",
                    "dataset": "data",
                    "dataset_revision": "revision",
                    "row_indices": [0],
                    "dataset_sha256": bench.sha256(dataset),
                },
            )
            bench.load_workload(directory)
            dataset.write_text(dataset.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                bench.load_workload(directory)


if __name__ == "__main__":
    unittest.main()
