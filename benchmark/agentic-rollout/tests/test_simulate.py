"""CPU protocol tests using a local HTTP server; no model downloads."""

import builtins
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot as plots
import simulate as bench
from aiohttp import web


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.incremental = False
        self.dp_size = 2
        self.omit_rank = False
        self.failure = None
        self.sessions = {}
        self.closed = []
        self.payloads = []

        async def info(request):
            return web.json_response(
                {
                    "dp_size": self.dp_size,
                    "context_length": 10000,
                    "incremental_streaming_output": self.incremental,
                }
            )

        async def open_session(request):
            payload = await request.json()
            if self.failure == "disabled":
                raise web.HTTPBadRequest(text="Streaming sessions are disabled")
            self.sessions[payload["session_id"]] = ([], None, payload["streaming"])
            return web.json_response(payload["session_id"])

        async def close_session(request):
            self.closed.append((await request.json())["session_id"])
            return web.json_response(None)

        async def generate(request):
            payload = await request.json()
            self.payloads.append(payload)
            ids = payload["input_ids"]
            sid = payload.get("session_params", {}).get("id")
            if sid:
                history, previous, streaming = self.sessions[sid]
                self.assertEqual(payload["session_params"]["rid"], previous)
                ids = history + ids
            rid = str(len(self.payloads))
            count = payload["sampling_params"]["max_new_tokens"]
            output = list(range(1000, 1000 + count))
            if sid:
                self.sessions[sid] = (ids + output, rid, streaming)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for n in (0, 1, count):
                meta = {
                    "id": rid,
                    "completion_tokens": n,
                    "prompt_tokens": len(ids),
                    "dp_rank": payload.get("routed_dp_rank", 0),
                    "cached_tokens": 0,
                    "finish_reason": {"type": "length"} if n == count else None,
                }
                if self.omit_rank:
                    meta.pop("dp_rank")
                if self.failure == "context":
                    meta["prompt_tokens"] += 1
                if self.failure == "rank":
                    meta["dp_rank"] += 1
                if self.failure == "abort" and n == count:
                    meta["finish_reason"] = {"type": "abort"}
                tokens = output[:n]
                if self.incremental and n == count:
                    tokens = output[1:]
                frame = (
                    "data: "
                    + json.dumps(
                        {"meta_info": meta, "output_ids": tokens, "text": "é"},
                        ensure_ascii=False,
                    )
                    + "\r\n\r\n"
                ).encode()
                for byte in frame:
                    await response.write(bytes([byte]))
            if self.failure != "truncated":
                await response.write(b"data: [DONE]\n\n")
            return response

        async def metrics(request):
            return web.Response(text='sglang:num_running_reqs{dp_rank="0"} 1\n')

        app = web.Application()
        for method, path, handler in [
            ("GET", "/server_info", info),
            ("POST", "/open_session", open_session),
            ("POST", "/close_session", close_session),
            ("POST", "/generate", generate),
            ("GET", "/metrics", metrics),
        ]:
            app.router.add_route(method, path, handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{self.runner.addresses[0][1]}"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    def args(self, path, mode, extra=()):
        return bench.parse_args(
            [
                "--base-url",
                self.url,
                "--tokenizer",
                "fake",
                "--output-dir",
                str(path),
                "--mode",
                mode,
                "--conversations",
                "2",
                "--concurrency",
                "1",
                "--turns",
                "3",
                "--initial-tokens",
                "8",
                "--tool-tokens",
                "4",
                "--output-tokens",
                "2",
                "--tool-delay",
                "0",
                "0",
                "--start-spread",
                "0",
            ]
            + list(extra)
        )

    async def test_all_modes_and_stream_formats_preserve_history(self):
        for incremental in (False, True):
            for mode in ("full-history", "ordinary", "streaming"):
                with (
                    self.subTest(incremental=incremental, mode=mode),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    self.incremental = incremental
                    self.sessions.clear()
                    self.closed.clear()
                    self.payloads.clear()
                    path = Path(tmp) / "run"
                    await bench.run(self.args(path, mode), Tokenizer())
                    rows = list(plots.read_rows(path / "requests.jsonl"))
                    self.assertEqual(len(rows), 6)
                    self.assertTrue(
                        all(
                            r["meta_info"]["dp_rank"] == r["conversation"] % 2
                            for r in rows
                        )
                    )
                    self.assertTrue(all("routed_dp_rank" in p for p in self.payloads))
                    self.assertEqual({r["conversation"] for r in rows}, {0, 1})
                    for conversation in (0, 1):
                        self.assertEqual(
                            [
                                r["turn"]
                                for r in rows
                                if r["conversation"] == conversation
                            ],
                            [0, 1, 2],
                        )
                    self.assertTrue(
                        all(r["context_tokens"] == 8 + r["turn"] * 6 for r in rows)
                    )
                    self.assertTrue(
                        all(r["meta_info"]["completion_tokens"] == 2 for r in rows)
                    )
                    self.assertEqual(
                        len(self.closed), 0 if mode == "full-history" else 2
                    )
                    self.assertTrue(
                        all(
                            s[2] == (mode == "streaming")
                            for s in self.sessions.values()
                        )
                    )
                    if mode == "full-history":
                        self.assertTrue(
                            any(
                                p["input_ids"][8:10] == [1000, 1001]
                                for p in self.payloads
                            )
                        )
                    plots.analyze(path, 30)
                    self.assertEqual(
                        json.loads((path / "summary.json").read_text())["status"],
                        "completed",
                    )

    async def test_omitted_rank_requires_verified_single_worker(self):
        self.omit_rank = True
        for size in (1, 2):
            self.dp_size = size
            with self.subTest(dp_size=size), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "run"
                if size == 1:
                    await bench.run(self.args(path, "ordinary"), Tokenizer())
                    rows = list(plots.read_rows(path / "requests.jsonl"))
                    for row in rows:
                        self.assertLessEqual(
                            row["client_wait_started_at"], row["submitted_at"]
                        )
                        if row["turn"]:
                            self.assertLessEqual(
                                row["tool_started_at"], row["tool_completed_at"]
                            )
                            self.assertLessEqual(
                                row["tool_completed_at"], row["client_wait_started_at"]
                            )
                else:
                    with self.assertRaises(builtins.ExceptionGroup):
                        await bench.run(self.args(path, "ordinary"), Tokenizer())

    async def test_disable_dp_sticky_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(
                Path(tmp) / "run", "ordinary", ["--disable-dp-sticky-routing"]
            )
            await bench.run(args, Tokenizer())
            self.assertEqual(len(self.payloads), 6)
            self.assertTrue(all("routed_dp_rank" not in p for p in self.payloads))

    async def test_invalid_stream_fails_and_closes_owned_sessions(self):
        for failure in ("abort", "context", "rank", "truncated"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                self.failure = failure
                self.sessions.clear()
                self.closed.clear()
                path = Path(tmp) / "run"
                with self.assertRaises(builtins.ExceptionGroup):
                    await bench.run(self.args(path, "ordinary"), Tokenizer())
                self.assertEqual(
                    json.loads((path / "manifest.json").read_text())["status"], "failed"
                )
                self.assertEqual(set(self.closed), set(self.sessions))
                self.assertTrue(
                    any("error" in r for r in plots.read_rows(path / "requests.jsonl"))
                )

    async def test_server_rejection_is_preserved(self):
        self.failure = "disabled"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            with self.assertRaises(builtins.ExceptionGroup):
                await bench.run(self.args(path, "streaming"), Tokenizer())
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertIn("Streaming sessions are disabled", manifest["error"])
            self.assertIn("HTTP 400", manifest["error"])

    async def test_timeout_is_failure(self):
        # A timeout while entering the request must not become a successful sample.
        from unittest.mock import AsyncMock, patch

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(bench, "generate", new=AsyncMock(side_effect=TimeoutError)),
        ):
            path = Path(tmp) / "run"
            with self.assertRaises(builtins.ExceptionGroup):
                await bench.run(self.args(path, "ordinary"), Tokenizer())
            self.assertEqual(set(self.closed), set(self.sessions))


class MeasurementTests(unittest.TestCase):
    def test_inputs_are_repeatable_and_distinct(self):
        a = bench.synthetic_tokens(Tokenizer(), 1, 0, 0, 100)
        self.assertEqual(a, bench.synthetic_tokens(Tokenizer(), 1, 0, 0, 100))
        self.assertNotEqual(a, bench.synthetic_tokens(Tokenizer(), 1, 1, 0, 100))
        self.assertEqual(len(a), 100)

    def test_token_weighted_cache_and_missing_data(self):
        rows = [
            {"meta_info": {"prompt_tokens": 100, "cached_tokens": 100}},
            {"meta_info": {"prompt_tokens": 900, "cached_tokens": 0}},
        ]
        self.assertEqual(plots.cache_hit(rows, "total"), 0.1)
        self.assertIsNone(plots.cache_hit(rows, "host"))
        self.assertIsNone(plots.cache_hit([], "total"))

    def test_throughput_uses_arrivals_and_partial_window_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.write_json(
                root / "manifest.json",
                {
                    "status": "completed",
                    "started_at": 0,
                    "finished_at": 1.5,
                },
            )
            row = {
                "turn": 0,
                "context_tokens": 8,
                "submitted_at": 0,
                "first_token_at": 0.2,
                "completed_at": 1.4,
                "events": [[0.2, 1], [1.2, 3]],
                "ttft_s": 0.2,
                "avg_token_time_s": 0.5,
                "meta_info": {"prompt_tokens": 8, "completion_tokens": 3},
            }
            with (root / "requests.jsonl").open("w") as file:
                bench.record(file, row)
            _, windows, _ = plots.analyze(root, 1)
            self.assertEqual([w["output_tokens_s"] for w in windows], [1, 4])
            self.assertIsNone(windows[1]["ttft_p95_s"])

    def test_counter_labels_resets_and_missing_series(self):
        before = plots.metric_values('x_total{rank="0"} 5\nx_total{rank="1"} 8\n')
        after = plots.metric_values('x_total{rank="0"} 7\nx_total{rank="1"} 12\n')
        self.assertEqual(plots.counter_rate(before, after, "x_total", 2), 3)
        self.assertIsNone(plots.counter_rate(after, before, "x_total", 2))
        self.assertIsNone(plots.counter_rate({}, after, "x_total", 2))


if __name__ == "__main__":
    unittest.main()
