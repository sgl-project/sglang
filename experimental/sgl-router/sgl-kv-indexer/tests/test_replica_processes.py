"""Real Rust Indexer/Bridge processes with production Python KV publishers.

This validates transport/recovery, not model inference. Run from repository root:
PYTHONPATH=python .venv/bin/python -m pytest experimental/sgl-router/sgl-kv-indexer/tests/test_replica_processes.py -q
"""

import importlib
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grpc
import pytest

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)
from sglang.srt.disaggregation.load_reporter import register_worker_reporting

ROOT = Path(__file__).resolve().parents[4]
CRATE = ROOT / "experimental/sgl-router/sgl-kv-indexer"
BIN = ROOT / "experimental/sgl-router/target/debug"


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(fn, seconds=20):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        try:
            result = fn()
            if result:
                return result
        except (grpc.RpcError, OSError) as exc:
            last = exc
        time.sleep(0.1)
    raise AssertionError(f"condition timed out; last error: {last}")


class Worker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.live, self.snapshot, self.replay = port(), port(), port()
        self.load = SimpleNamespace(num_running_reqs=0, num_waiting_reqs=0,
            num_used_tokens=0, max_total_num_tokens=100000, max_running_requests=100,
            num_total_tokens=0, num_waiting_uncached_tokens=0)
        self.load_stop = threading.Event()
        descriptor = self.descriptor()
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"kv_events": descriptor, "served_model_name": "model", "model_path": "model", "dp_size": 1}).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.path == "/v1/start_reporting":
                    args = SimpleNamespace(resolved_dict=lambda: {"kv_events_config": json.dumps({
                        "publisher": "zmq", "worker_id": worker.worker_id,
                        "snapshot_endpoint": f"tcp://*:{worker.snapshot}"}), "dp_size": 1})
                    response = asyncio.run(register_worker_reporting(args, body))
                else:
                    response = {"id": worker.worker_id, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": worker.worker_id}, "finish_reason": "stop"}]}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            def log_message(self, *args):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_port}"
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.start()
        self.load_thread = threading.Thread(target=self.report_load, daemon=True)
        self.load_thread.start()

    def report_load(self):
        while not self.load_stop.wait(0.1):
            self.publisher.load_reporter.update(self.load)

    def descriptor(self):
        return {
            "namespace": "default", "worker_id": self.worker_id,
            "model": "model", "hash_schema_version": 1, "is_bigram": False,
            "block_size": 4, "dp_size": 1, "snapshot_versions": [1, 2], "topic": "kv",
            "endpoint_host": "127.0.0.1", "endpoint_port_base": self.live,
            "snapshot_endpoint_host": "127.0.0.1", "snapshot_endpoint_port_base": self.snapshot,
            "replay_endpoint_host": "127.0.0.1", "replay_endpoint_port_base": self.replay,
        }

    def start(self):
        self.publisher = ZmqEventPublisher(
            0, endpoint=f"tcp://*:{self.live}",
            snapshot_endpoint=f"tcp://*:{self.snapshot}",
            replay_endpoint=f"tcp://*:{self.replay}", topic="kv",
            worker_id=self.worker_id, model="model", page_size=4,
        )

    def store(self, hashes):
        self.publisher.publish(KVEventBatch(ts=time.time(), events=[
            BlockStored(hashes, None, [1, 2, 3, 4] * len(hashes), 4, None, "GPU")
        ]))
        self.publisher._event_queue.join()

    def remove(self, hashes):
        self.publisher.publish(KVEventBatch(ts=time.time(), events=[BlockRemoved(hashes, "GPU")]))
        self.publisher._event_queue.join()

    def close(self):
        self.load_stop.set()
        self.load_thread.join()
        self.publisher.shutdown()
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()


@pytest.fixture(scope="module")
def pb(tmp_path_factory):
    generated = tmp_path_factory.mktemp("replica-proto")
    subprocess.run([sys.executable, "-m", "grpc_tools.protoc", f"-I{CRATE / 'proto'}",
                    f"--python_out={generated}", f"--grpc_python_out={generated}",
                    str(CRATE / "proto/kv_indexer.proto")], check=True)
    sys.path.insert(0, str(generated))
    return importlib.import_module("kv_indexer_pb2"), importlib.import_module("kv_indexer_pb2_grpc")


class Pair:
    def __init__(self, directory, workers, pb, name):
        self.endpoint = f"127.0.0.1:{port()}"
        self.config = directory / f"{name}.json"
        self.log = open(directory / f"{name}.log", "w+")
        self.config.write_text(json.dumps({"indexer_endpoint": f"http://{self.endpoint}", "worker_urls": [w.url for w in workers]}))
        self.server = None
        self.bridge = None
        self.start_server()
        self.bridge = subprocess.Popen([BIN / "kv-indexer-bridge"], env={**os.environ, "KV_BRIDGE_CONFIG": str(self.config)}, stdout=self.log, stderr=self.log)
        self.channel = grpc.insecure_channel(self.endpoint)
        self.stub = pb[1].KVReplicaStub(self.channel)

    def start_server(self):
        self.server = subprocess.Popen([BIN / "kv-indexer-server"], env={**os.environ, "KV_INDEXER_LISTEN_ADDR": self.endpoint, "KV_INDEXER_STREAM_LEASE_MS": "2000"}, stdout=self.log, stderr=self.log)

    def close(self):
        for p in (self.bridge, self.server):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
        self.channel.close()
        self.log.close()


def query(pair, pb, workers):
    return pair.stub.MatchPrefix(pb[0].ReplicaPrefixRequest(
        namespace="default", model="model", hash_schema_version=1, page_size=4,
        hashes=[11, 12, 13], eligible_streams=[pb[0].StreamKey(namespace="default", worker_id=w.worker_id, dp_rank=0) for w in workers]
    ), timeout=1)


def test_two_pairs_recover_restart_worker_and_scale_membership(tmp_path, pb):
    workers = [Worker("a"), Worker("b")]
    pairs = []
    try:
        workers[0].store([11, 12, 13])
        workers[1].store([11])
        pairs = [Pair(tmp_path, workers, pb, str(i)) for i in range(2)]
        for pair in pairs:
            ready = wait_for(lambda: (r if (r := query(pair, pb, workers)).complete else None))
            assert {m.worker_id: m.matched_prefix_blocks for m in ready.matches} == {"a": 3, "b": 1}
        workers[0].remove([12])
        for pair in pairs:
            wait_for(lambda: all(m.matched_prefix_blocks == 1 for m in query(pair, pb, workers).matches))
        pairs[0].server.kill()
        pairs[0].server.wait(timeout=5)
        workers[0].store([11, 12, 13])
        pairs[0].start_server()
        recovered = wait_for(lambda: (r if (r := query(pairs[0], pb, workers)).complete and max((m.matched_prefix_blocks for m in r.matches), default=0) == 3 else None))
        assert len(recovered.coverage) == 2
        pairs.append(Pair(tmp_path, workers, pb, "scale-out"))
        wait_for(lambda: query(pairs[2], pb, workers).complete)
        # A publisher restart changes epoch and discards its cache. Both an
        # existing and a newly added replica must forget old placements.
        old_epoch = recovered.coverage[0].worker_epoch
        workers[0].publisher.shutdown()
        workers[0].start()
        workers[0].store([11])
        for pair in pairs:
            r = wait_for(lambda: (r if (r := query(pair, pb, workers)).complete and all(m.matched_prefix_blocks == 1 for m in r.matches) and r.coverage[0].worker_epoch != old_epoch else None))
            assert r.coverage[0].worker_epoch != old_epoch
        # Removing a Worker from a Bridge's desired list terminates its stream
        # renewal; the Indexer must stop claiming coverage after the lease.
        cfg = json.loads(pairs[0].config.read_text())
        cfg["worker_urls"] = [workers[0].url]
        pairs[0].config.write_text(json.dumps(cfg))
        wait_for(lambda: not query(pairs[0], pb, workers).complete)
        assert query(pairs[1], pb, workers).complete
    finally:
        for pair in pairs:
            pair.close()
        for worker in workers:
            worker.close()


def http_json(url, data=None):
    request = Request(url, data=None if data is None else json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def test_router_random_failover_all_down_restart_and_membership(tmp_path, pb):
    workers = [Worker("cached"), Worker("idle")]
    workers[0].load.num_running_reqs = 5
    pairs = []
    router = None
    router_log = open(tmp_path / "router.log", "w+")
    try:
        from tokenizers import Tokenizer
        tokenizer_path = ROOT / "experimental/sgl-router/tests/fixtures/tiny_tokenizer.json"
        prompt = "hello world " * 128
        ids = Tokenizer.from_file(str(tokenizer_path)).encode(prompt).ids
        hashes = []
        previous = ""
        for start in range(0, len(ids) - 3, 4):
            # Worker hash parity is covered by the existing fixture suite.
            digest = hashlib.sha256()
            if previous:
                digest.update(bytes.fromhex(previous))
            for token in ids[start:start + 4]:
                digest.update(int(token).to_bytes(4, "little", signed=False))
            previous = digest.hexdigest()
            hashes.append(int.from_bytes(bytes.fromhex(previous)[:8], "big", signed=True))
        workers[0].store(hashes)
        pairs = [Pair(tmp_path, workers, pb, f"routing-{i}") for i in range(2)]
        endpoints = tmp_path / "indexers.json"
        endpoints.write_text(json.dumps([f"http://{pair.endpoint}" for pair in pairs]))
        router_port, load_port = port(), port()
        url = f"http://127.0.0.1:{router_port}"
        router = subprocess.Popen([BIN / "sgl-router", "--model-id", "model", "--tokenizer-path", str(tokenizer_path),
            "--worker-urls", *[w.url for w in workers], "--policy", "cache_aware", "--cache-prefix-provider", "indexer",
            "--kv-indexer-endpoint", f"@{endpoints}", "--port", str(router_port),
            "--cache-affinity-min-matched-tokens", "0", "--cache-candidate-min-workers", "1", "--cache-candidate-max-workers", "1"],
            env={**os.environ, "SGL_ROUTER_LOAD_LISTEN_ADDR": f"127.0.0.1:{load_port}",
                 "RUST_LOG": "info,sgl_kv_indexer::fleet=debug"}, stdout=router_log, stderr=router_log)
        body = {"model": "model", "prompt": prompt, "max_tokens": 1}
        def request():
            assert router.poll() is None, (tmp_path / "router.log").read_text()
            return http_json(url + "/v1/chat/completions", body)
        def ready_request():
            result = request()
            with urlopen(url + "/metrics", timeout=5) as response:
                metrics = response.read().decode()
            count = next((int(line.split()[1]) for line in metrics.splitlines()
                          if line.startswith("sgl_router_indexer_complete_queries_total ")), 0)
            return count > 0 and result["id"] == "cached"
        wait_for(ready_request)
        assert all(request()["id"] == "cached" for _ in range(30))
        router_log.flush()
        log = (tmp_path / "router.log").read_text()
        for pair in pairs:
            assert pair.endpoint in log, "both randomly selected replicas must be exercised"
        pairs[0].server.kill()
        pairs[0].server.wait(timeout=5)
        assert all(request()["id"] == "cached" for _ in range(10))
        pairs[1].server.kill()
        pairs[1].server.wait(timeout=5)
        assert all(request()["id"] == "idle" for _ in range(5)), "all down must use fresh minimum load"
        pairs[0].start_server()
        wait_for(lambda: request()["id"] == "cached")
        pairs.append(Pair(tmp_path, workers, pb, "routing-scale-out"))
        endpoints.write_text(json.dumps([f"http://{pairs[2].endpoint}"]))
        wait_for(lambda: request()["id"] == "cached")
        time.sleep(2.5)
        pairs[0].server.kill()
        pairs[0].server.wait(timeout=5)
        assert all(request()["id"] == "cached" for _ in range(10))
        endpoints.write_text("[]")
        wait_for(lambda: request()["id"] == "idle")
        # An invalid discovery update preserves the last valid (empty) fleet.
        endpoints.write_text("invalid-json")
        time.sleep(2.1)
        assert request()["id"] == "idle"
    finally:
        if router and router.poll() is None:
            router.terminate()
            router.wait(timeout=10)
        router_log.close()
        for pair in pairs:
            pair.close()
        for worker in workers:
            worker.close()
