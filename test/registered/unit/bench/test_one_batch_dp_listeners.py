"""Unit tests for spreading a one-batch /generate over per-DP-rank listeners."""

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import requests

from sglang.benchmark import one_batch_server
from sglang.benchmark.endpoint import BenchEndpoint
from sglang.benchmark.utils import dp_listener_urls
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

BATCH_SIZE = 5
INPUT_LEN = 4


class _Counters:
    """The node's /metrics counters, which every listener adds to."""

    def __init__(self):
        self.lock = threading.Lock()
        self.prompt_tokens = 0  # a fresh server, its startup warmup skipped
        self.cached_tokens = 0

    def add(self, prompt_tokens, cached_tokens):
        with self.lock:
            self.prompt_tokens += prompt_tokens
            self.cached_tokens += cached_tokens

    def text(self):
        with self.lock:
            return (
                f'sglang:prompt_tokens_total{{model_name="m"}} {self.prompt_tokens}\n'
                f'sglang:cached_tokens_total{{model_name="m"}} {self.cached_tokens}\n'
            )


class _FakeListener:
    """One HTTP listener. A streamed /generate gives prompt j as many output
    tokens as its first token id, so every chunk names its prompt, and hits
    the cache for half of its prompt. A finished /generate reaches the shared
    counters after `metrics_delay`; only `serves_metrics` serves /metrics."""

    def __init__(
        self,
        counters,
        *,
        first_chunk_delay=0.0,
        chunk_delay=0.0,
        keepalives=0,
        status=200,
        metrics_delay=0.0,
        serves_metrics=False,
    ):
        self.first_chunk_delay = first_chunk_delay
        self.chunk_delay = chunk_delay
        self.status = status
        self.server_info = {}
        self.requests = []
        self.completed_streams = 0
        listener = self

        def count(body):
            prompt = sum(len(ids) for ids in body["input_ids"])
            cached = sum(len(ids) // 2 for ids in body["input_ids"]) * body["stream"]
            threading.Timer(metrics_delay, counters.add, (prompt, cached)).start()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                listener.requests.append(("GET", self.path, None))
                if self.path == "/server_info":
                    self._reply(200, listener.server_info)
                elif self.path == "/metrics" and serves_metrics:
                    data = counters.text().encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._reply(404, {})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"null")
                listener.requests.append(("POST", self.path, body))
                if self.path != "/generate":
                    return self._reply(200, {})
                if listener.status != 200:
                    return self._reply(listener.status, {"error": "failed"})
                if not body["stream"]:
                    count(body)
                    return self._reply(200, [{"text": ""}] * len(body["input_ids"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(listener.first_chunk_delay)
                lengths = [ids[0] for ids in body["input_ids"]]
                try:
                    for step in range(1, max(lengths) + 1):
                        for index, n in enumerate(lengths):
                            if step > n:
                                continue
                            finish = {"type": "length"} if step == n else None
                            chunk = {
                                "index": index,
                                "meta_info": {
                                    "completion_tokens": step,
                                    "finish_reason": finish,
                                },
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                        time.sleep(listener.chunk_delay)
                    # SSE comments, one client read block each, before the end.
                    for _ in range(keepalives):
                        self.wfile.write(b": " + b"x" * 508 + b"\n\n")
                        self.wfile.flush()
                        time.sleep(listener.chunk_delay)
                    self.wfile.write(b"data: [DONE]\n\n")
                    listener.completed_streams += 1
                    count(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    def generate_bodies(self, *, stream):
        return [
            body
            for method, path, body in self.requests
            if path == "/generate" and body["stream"] == stream
        ]


class _IndexRecorder(one_batch_server.BatchStreamRecorder):
    """Also keeps each batch index's last completion token count."""

    def __init__(self, batch_size):
        super().__init__(batch_size)
        self.tokens_by_index = [0] * batch_size

    def record_chunk(self, *, index, completion_tokens, finish_type, now):
        super().record_chunk(
            index=index,
            completion_tokens=completion_tokens,
            finish_type=finish_type,
            now=now,
        )
        self.tokens_by_index[index] = completion_tokens


def _server_info(dp_listeners, max_running_requests=64):
    # No DP sizes in the internal state, as a server with per-rank listeners
    # may report it.
    info = {
        "tokenizer_path": "fake-tokenizer",
        "internal_states": [
            {
                "effective_max_running_requests_per_dp": max_running_requests,
                "memory_usage": {"token_capacity": 1_000_000},
                "last_gen_throughput": 1.0,
            }
        ],
    }
    if dp_listeners:
        info["dp_listeners"] = dp_listeners
    return info


class TestDpListenerSplit(CustomTestCase):
    def setUp(self):
        self.counters = _Counters()
        self.listeners = []
        self.recorders = []

    def tearDown(self):
        for listener in self.listeners:
            listener.close()

    def _listener(self, **kwargs):
        listener = _FakeListener(self.counters, **kwargs)
        self.listeners.append(listener)
        return listener

    def _make_recorder(self, batch_size):
        recorder = _IndexRecorder(batch_size)
        self.recorders.append(recorder)
        return recorder

    def _run(self, base, **bench_kwargs):
        """run_benchmark_internal against `base`; prompt g's tokens are g + 1."""
        prompts = [
            SimpleNamespace(prompt=[g + 1] * INPUT_LEN) for g in range(BATCH_SIZE)
        ]
        bench_args = one_batch_server.BenchArgs(
            base_url=base.url,
            batch_size=(BATCH_SIZE,),
            input_len=(INPUT_LEN,),
            output_len=(BATCH_SIZE,),
            skip_warmup=True,
            result_filename="",
            **bench_kwargs,
        )
        with (
            patch.object(
                one_batch_server,
                "acquire_endpoint",
                lambda *args: BenchEndpoint(base_url=base.url),
            ),
            patch.object(
                one_batch_server, "get_tokenizer", lambda path: SimpleNamespace()
            ),
            patch.object(one_batch_server, "get_dataset", lambda *a, **k: prompts),
            patch.object(one_batch_server, "BatchStreamRecorder", self._make_recorder),
        ):
            results, _ = one_batch_server.run_benchmark_internal(
                SimpleNamespace(admin_api_key=None), bench_args
            )
        return results

    def test_dp_listener_urls(self):
        info = {
            "dp_listeners": [
                {"dp_rank": 1, "url": "http://h:31/"},
                {"dp_rank": 0, "url": "http://h:30"},
            ]
        }
        self.assertEqual(dp_listener_urls(info), ["http://h:30", "http://h:31"])
        self.assertEqual(dp_listener_urls({}), [])
        self.assertEqual(
            dp_listener_urls({"dp_listeners": [{"dp_rank": 0, "url": "http://h:30"}]}),
            [],
        )

    def test_sub_batch_payload(self):
        parts = one_batch_server._round_robin_parts(5, 2)
        self.assertEqual(parts, [[0, 2, 4], [1, 3]])
        self.assertEqual(one_batch_server._round_robin_parts(1, 2), [[0]])
        payload = {
            "input_ids": [[0], [1], [2], [3], [4]],
            "image_data": ["i0", "i1", "i2", "i3", "i4"],
            "lora_path": "adapter",
            "sampling_params": {"max_new_tokens": 3},
        }
        sub = one_batch_server._sub_batch_payload(payload, parts[1])
        self.assertEqual(sub["input_ids"], [[1], [3]])
        self.assertEqual(sub["image_data"], ["i1", "i3"])
        self.assertEqual(sub["lora_path"], "adapter")
        self.assertIs(sub["sampling_params"], payload["sampling_params"])
        sub = one_batch_server._sub_batch_payload(
            {**payload, "lora_path": ["a", "b", "c", "d", "e"]}, parts[0]
        )
        self.assertEqual(sub["lora_path"], ["a", "c", "e"])

    def test_batch_spreads_over_listeners(self):
        delay = 0.5
        rank0 = self._listener(first_chunk_delay=delay, serves_metrics=True)
        # Rank 1's requests reach /metrics late: its warmup during the batch,
        # its batch after the batch ends.
        rank1 = self._listener(first_chunk_delay=delay, metrics_delay=0.3)
        # Per rank, 3 running requests hold a batch of 5 spread over 2 ranks.
        rank0.server_info = _server_info(
            [{"dp_rank": 1, "url": rank1.url}, {"dp_rank": 0, "url": rank0.url}],
            max_running_requests=3,
        )
        # Only the base URL is on the proxy bypass list, not the listener
        # addresses the server publishes.
        rank0.url = rank0.url.replace("127.0.0.1", "localhost")
        proxy_env = {"http_proxy": "http://127.0.0.1:9", "no_proxy": "localhost"}
        proxy_env.update({k.upper(): v for k, v in proxy_env.items()})

        snapshots = []
        original_metrics = one_batch_server.get_cache_tokens_from_metrics

        def metrics_snapshot(url):
            snapshots.append(original_metrics(url))
            return snapshots[-1]

        with (
            patch.object(
                one_batch_server, "get_cache_tokens_from_metrics", metrics_snapshot
            ),
            patch.dict(os.environ, proxy_env),
        ):
            (result,) = self._run(rank0, cache_hit_rate=0.5)

        # Prompt i goes to listener i % 2, its cache prefix warmed there first.
        for listener, prompts in ((rank0, [0, 2, 4]), (rank1, [1, 3])):
            (sent,) = listener.generate_bodies(stream=True)
            self.assertEqual(sent["input_ids"], [[g + 1] * INPUT_LEN for g in prompts])
            (warmed,) = listener.generate_bodies(stream=False)
            self.assertEqual(warmed["input_ids"], [[g + 1] * 2 for g in prompts])
        # Control and metrics requests go to the base URL only.
        self.assertEqual({path for _, path, _ in rank1.requests}, {"/generate"})
        self.assertIn(("POST", "/flush_cache", None), rank0.requests)

        # Chunks land under their batch index; timings span both sub-batches,
        # which ran concurrently.
        (recorder,) = self.recorders
        self.assertEqual(recorder.tokens_by_index, [1, 2, 3, 4, 5])
        self.assertEqual(result.total_output_tokens, 15)
        self.assertGreaterEqual(result.last_ttft, delay)
        self.assertLess(result.latency, 2 * delay)
        self.assertAlmostEqual(
            result.input_throughput, BATCH_SIZE * INPUT_LEN / result.last_ttft
        )
        # The before/after /metrics delta brackets exactly the batch, and a
        # wrapper of the metrics read sees just those two snapshots.
        self.assertEqual(result.cache_hit_rate, 0.5)
        self.assertEqual(len(snapshots), 2)

    def test_single_listener_sends_one_request(self):
        base = self._listener(serves_metrics=True)
        base.server_info = _server_info([])

        (result,) = self._run(base)

        (sent,) = base.generate_bodies(stream=True)
        self.assertEqual(
            sent["input_ids"], [[g + 1] * INPUT_LEN for g in range(BATCH_SIZE)]
        )
        self.assertEqual(self.recorders[0].tokens_by_index, [1, 2, 3, 4, 5])
        self.assertEqual(result.total_output_tokens, 15)
        self.assertEqual(result.cache_hit_rate, 0.5)

    def test_failed_sub_batch_fails_the_batch(self):
        # Rank 0 alone would stream for about 30 s.
        rank0 = self._listener(chunk_delay=0.2, keepalives=150)
        rank1 = self._listener(status=500)
        rank0.server_info = _server_info(
            [{"dp_rank": 0, "url": rank0.url}, {"dp_rank": 1, "url": rank1.url}]
        )

        with self.assertRaises(requests.HTTPError):
            self._run(rank0)
        # The failure stopped rank 0's stream instead of waiting for its end.
        self.assertEqual(rank0.completed_streams, 0)


if __name__ == "__main__":
    unittest.main()
