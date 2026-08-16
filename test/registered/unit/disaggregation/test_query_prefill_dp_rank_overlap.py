import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _QueryServer(ThreadingHTTPServer):
    def __init__(self, responses):
        super().__init__(("127.0.0.1", 0), _QueryHandler)
        self.responses = list(responses)
        self.requests = []
        self.events = []
        self.started = threading.Event()


class _QueryHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(size))
        self.server.requests.append(payload)
        status, body, gate = self.server.responses.pop(0)
        self.server.events.append("query-start")
        self.server.started.set()
        if gate is not None:
            gate.wait(timeout=2)
        self.server.events.append("query-complete")
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        pass


class _FakeReceiver:
    def __init__(self, events, room, step):
        self.events = events
        self.room = room
        self.step = step
        self.init_calls = []

    def init(self, rank):
        self.events.append(("receiver-init", self.room, rank, self.step[0]))
        self.init_calls.append(rank)


class TestQueryPrefillDPRankOverlap(CustomTestCase):
    def setUp(self):
        self.executors = []
        self.servers = []

    def tearDown(self):
        for executor in self.executors:
            executor.shutdown(wait=True)
        for server, thread in self.servers:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def _start_server(self, responses):
        server = _QueryServer(responses)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return server

    def _make_queue(self, server, rooms):
        executor = ThreadPoolExecutor(max_workers=2)
        self.executors.append(executor)
        addr = f"127.0.0.1:{server.server_port}"
        manager = SimpleNamespace(
            prefill_info_table={
                addr: SimpleNamespace(dp_size=8, follow_bootstrap_room=False)
            },
            try_ensure_parallel_info=lambda _addr: True,
            _ensure_prefill_recompute_executor=lambda: executor,
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.kv_manager = manager
        queue._prefill_dp_rank_queries = {}
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue._ensure_retry_interval = 0
        queue._max_ensure_retries = 15
        queue.queue = []
        queue.retracted_queue = []
        step = [7]
        decode_reqs = []
        for room in rooms:
            req = SimpleNamespace(
                bootstrap_host="127.0.0.1",
                bootstrap_port=server.server_port,
                bootstrap_room=room,
                disagg_prefill_dp_rank=None,
            )
            receiver = _FakeReceiver(server.events, room, step)
            decode_req = SimpleNamespace(req=req, kv_receiver=receiver)
            decode_reqs.append(decode_req)
        queue.pending_reqs = decode_reqs.copy()
        return queue, decode_reqs, step

    def test_waits_at_original_consume_point_and_initializes_same_step(self):
        release = threading.Event()
        server = self._start_server([(200, {"11": 3}, release)])
        queue, decode_reqs, step = self._make_queue(server, [11])

        queue.prefetch_prefill_dp_rank_queries()
        self.assertTrue(server.started.wait(timeout=1))
        threading.Timer(0.03, release.set).start()
        queue._resolve_pending_reqs()

        self.assertEqual(decode_reqs[0].kv_receiver.init_calls, [3])
        self.assertEqual(server.events[:2], ["query-start", "query-complete"])
        self.assertEqual(server.events[2], ("receiver-init", 11, 3, step[0]))
        self.assertEqual(step[0], 7)
        self.assertEqual(queue.pending_reqs, [])

    def test_multi_request_is_one_authoritative_batch_and_fifo(self):
        server = self._start_server([(200, {"22": 6, "21": 5}, None)])
        queue, decode_reqs, _ = self._make_queue(server, [21, 22])

        queue.prefetch_prefill_dp_rank_queries()
        queue._resolve_pending_reqs()

        self.assertEqual(server.requests, [{"bootstrap_rooms": [21, 22]}])
        self.assertEqual(decode_reqs[0].kv_receiver.init_calls, [5])
        self.assertEqual(decode_reqs[1].kv_receiver.init_calls, [6])
        init_events = [event for event in server.events if isinstance(event, tuple)]
        self.assertEqual([event[1] for event in init_events], [21, 22])

    def test_missing_mapping_retries_without_initializing_early(self):
        server = self._start_server(
            [(200, {"31": 1}, None), (200, {"32": 2}, None)]
        )
        queue, decode_reqs, _ = self._make_queue(server, [31, 32])

        queue.prefetch_prefill_dp_rank_queries()
        queue._resolve_pending_reqs()
        self.assertEqual(decode_reqs[0].kv_receiver.init_calls, [1])
        self.assertEqual(decode_reqs[1].kv_receiver.init_calls, [])
        self.assertEqual(queue.pending_reqs, [decode_reqs[1]])

        queue.prefetch_prefill_dp_rank_queries()
        queue._resolve_pending_reqs()
        self.assertEqual(decode_reqs[1].kv_receiver.init_calls, [2])
        self.assertEqual(
            server.requests,
            [{"bootstrap_rooms": [31, 32]}, {"bootstrap_rooms": [32]}],
        )

    def test_request_appended_after_prefetch_is_consumed_same_step_in_fifo(self):
        server = self._start_server(
            [(200, {"61": 1}, None), (200, {"62": 2}, None)]
        )
        queue, decode_reqs, step = self._make_queue(server, [61])

        queue.prefetch_prefill_dp_rank_queries()
        req = SimpleNamespace(
            bootstrap_host="127.0.0.1",
            bootstrap_port=server.server_port,
            bootstrap_room=62,
            disagg_prefill_dp_rank=None,
        )
        receiver = _FakeReceiver(server.events, 62, step)
        appended = SimpleNamespace(req=req, kv_receiver=receiver)
        queue.pending_reqs.append(appended)

        queue._resolve_pending_reqs()

        self.assertEqual(
            server.requests,
            [{"bootstrap_rooms": [61]}, {"bootstrap_rooms": [62]}],
        )
        init_events = [event for event in server.events if isinstance(event, tuple)]
        self.assertEqual([event[1] for event in init_events], [61, 62])
        self.assertEqual(decode_reqs[0].kv_receiver.init_calls, [1])
        self.assertEqual(receiver.init_calls, [2])
        self.assertEqual(queue.pending_reqs, [])

    def test_http_error_keeps_request_for_original_retry_path(self):
        server = self._start_server([(503, {}, None), (200, {"41": 4}, None)])
        queue, decode_reqs, _ = self._make_queue(server, [41])

        queue.prefetch_prefill_dp_rank_queries()
        queue._resolve_pending_reqs()
        self.assertEqual(decode_reqs[0].kv_receiver.init_calls, [])
        self.assertEqual(queue.pending_reqs, decode_reqs)

        queue.prefetch_prefill_dp_rank_queries()
        queue._resolve_pending_reqs()
        self.assertEqual(decode_reqs[0].kv_receiver.init_calls, [4])
        self.assertEqual(len(server.requests), 2)

    def test_release_cancels_ownership_without_receiver_init(self):
        release = threading.Event()
        server = self._start_server([(200, {"51": 7}, release)])
        queue, decode_reqs, _ = self._make_queue(server, [51])

        queue.prefetch_prefill_dp_rank_queries()
        self.assertTrue(server.started.wait(timeout=1))
        queue.release_memory_occupation()
        self.assertEqual(queue._prefill_dp_rank_queries, {})
        release.set()
        time.sleep(0.02)
        self.assertEqual(decode_reqs[0].kv_receiver.init_calls, [])


if __name__ == "__main__":
    unittest.main()
