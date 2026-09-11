"""Real Rust Indexer/Bridge processes with production Python KV publishers.

Default tests validate transport/recovery with synthetic inference responses.
KV_REPLICA_GPU_TESTS=1 also runs two real CUDA SGLang Workers. Run from repository root:
PYTHONPATH=python .venv/bin/python -m pytest experimental/sgl-router/sgl-kv-indexer/tests/test_replica_processes.py -q
"""

import asyncio
import hashlib
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

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
        self.load = SimpleNamespace(
            num_running_reqs=0,
            num_waiting_reqs=0,
            num_used_tokens=0,
            max_total_num_tokens=100000,
            max_running_requests=100,
            num_total_tokens=0,
            num_waiting_uncached_tokens=0,
        )
        self.load_stop = threading.Event()
        descriptor = self.descriptor()
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "kv_events": descriptor,
                            "served_model_name": "model",
                            "model_path": "model",
                            "dp_size": 1,
                        }
                    ).encode()
                )

            def do_POST(self):
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                )
                if self.path == "/v1/start_reporting":
                    args = SimpleNamespace(
                        resolved_dict=lambda: {
                            "kv_events_config": json.dumps(
                                {
                                    "publisher": "zmq",
                                    "worker_id": worker.worker_id,
                                    "snapshot_endpoint": f"tcp://*:{worker.snapshot}",
                                }
                            ),
                            "dp_size": 1,
                        }
                    )
                    response = asyncio.run(register_worker_reporting(args, body))
                else:
                    response = {
                        "id": worker.worker_id,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": worker.worker_id,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
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
            "namespace": "default",
            "worker_id": self.worker_id,
            "model": "model",
            "hash_schema_version": 1,
            "is_bigram": False,
            "block_size": 4,
            "dp_size": 1,
            "snapshot_versions": [1, 2],
            "topic": "kv",
            "endpoint_host": "127.0.0.1",
            "endpoint_port_base": self.live,
            "snapshot_endpoint_host": "127.0.0.1",
            "snapshot_endpoint_port_base": self.snapshot,
            "replay_endpoint_host": "127.0.0.1",
            "replay_endpoint_port_base": self.replay,
        }

    def start(self):
        self.publisher = ZmqEventPublisher(
            0,
            endpoint=f"tcp://*:{self.live}",
            snapshot_endpoint=f"tcp://*:{self.snapshot}",
            replay_endpoint=f"tcp://*:{self.replay}",
            topic="kv",
            worker_id=self.worker_id,
            model="model",
            page_size=4,
        )

    def store(self, hashes):
        self.publisher.publish(
            KVEventBatch(
                ts=time.time(),
                events=[
                    BlockStored(
                        hashes, None, [1, 2, 3, 4] * len(hashes), 4, None, "GPU"
                    )
                ],
            )
        )
        self.publisher._event_queue.join()

    def remove(self, hashes):
        self.publisher.publish(
            KVEventBatch(ts=time.time(), events=[BlockRemoved(hashes, "GPU")])
        )
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
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{CRATE / 'proto'}",
            f"--python_out={generated}",
            f"--grpc_python_out={generated}",
            str(CRATE / "proto/kv_indexer.proto"),
        ],
        check=True,
    )
    sys.path.insert(0, str(generated))
    return importlib.import_module("kv_indexer_pb2"), importlib.import_module(
        "kv_indexer_pb2_grpc"
    )


class Pair:
    def __init__(self, directory, workers, pb, name, queue_capacity=256):
        self.endpoint = f"127.0.0.1:{port()}"
        self.config = directory / f"{name}.json"
        self.log = open(directory / f"{name}.log", "w+")
        self.config.write_text(
            json.dumps(
                {
                    "indexer_endpoint": f"http://{self.endpoint}",
                    "worker_urls": [w.url for w in workers],
                    "queue_capacity": queue_capacity,
                }
            )
        )
        self.server = None
        self.bridge = None
        self.start_server()
        self.bridge = subprocess.Popen(
            [BIN / "kv-indexer-bridge"],
            env={**os.environ, "KV_BRIDGE_CONFIG": str(self.config)},
            stdout=self.log,
            stderr=self.log,
        )
        self.channel = grpc.insecure_channel(self.endpoint)
        self.stub = pb[1].KVReplicaStub(self.channel)

    def start_server(self):
        self.server = subprocess.Popen(
            [BIN / "kv-indexer-server"],
            env={
                **os.environ,
                "KV_INDEXER_LISTEN_ADDR": self.endpoint,
                "KV_INDEXER_STREAM_LEASE_MS": "2000",
            },
            stdout=self.log,
            stderr=self.log,
        )

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
    return pair.stub.MatchPrefix(
        pb[0].ReplicaPrefixRequest(
            namespace="default",
            model="model",
            hash_schema_version=1,
            page_size=4,
            hashes=[11, 12, 13],
            eligible_streams=[
                pb[0].StreamKey(namespace="default", worker_id=w.worker_id, dp_rank=0)
                for w in workers
            ],
        ),
        timeout=1,
    )


def test_two_pairs_recover_restart_worker_and_scale_membership(tmp_path, pb):
    workers = [Worker("a"), Worker("b")]
    pairs = []
    try:
        workers[0].store([11, 12, 13])
        workers[1].store([11])
        pairs = [Pair(tmp_path, workers, pb, str(i)) for i in range(2)]
        for pair in pairs:
            ready = wait_for(
                lambda: r if (r := query(pair, pb, workers)).complete else None
            )
            assert {m.worker_id: m.matched_prefix_blocks for m in ready.matches} == {
                "a": 3,
                "b": 1,
            }
        workers[0].remove([12])
        for pair in pairs:
            wait_for(
                lambda: all(
                    m.matched_prefix_blocks == 1
                    for m in query(pair, pb, workers).matches
                )
            )
        pairs[0].server.kill()
        pairs[0].server.wait(timeout=5)
        workers[0].store([11, 12, 13])
        pairs[0].start_server()
        recovered = wait_for(
            lambda: (
                r
                if (r := query(pairs[0], pb, workers)).complete
                and max((m.matched_prefix_blocks for m in r.matches), default=0) == 3
                else None
            )
        )
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
            r = wait_for(
                lambda: (
                    r
                    if (r := query(pair, pb, workers)).complete
                    and all(m.matched_prefix_blocks == 1 for m in r.matches)
                    and r.coverage[0].worker_epoch != old_epoch
                    else None
                )
            )
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


def http_json(url, data=None, timeout=5):
    request = Request(
        url,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def test_tail_loss_replay_slow_replica_overflow_and_worker_scale_out(tmp_path, pb):
    import signal

    workers = [Worker("a"), Worker("b")]
    pairs = []
    paused = None
    try:
        workers[0].store([11, 12, 13])
        pairs = [
            Pair(tmp_path, workers, pb, f"stability-{i}", queue_capacity=4)
            for i in range(2)
        ]
        for pair in pairs:
            wait_for(lambda: query(pair, pb, workers).complete)

        # Drop a tail event, with no later live event to reveal the gap.
        # The periodic epoch-fenced replay probe must repair it without a snapshot.
        publisher = workers[0].publisher
        original = publisher._pub

        class DropOnce:
            def send_multipart(self, frames):
                publisher._pub = original

        publisher._pub = DropOnce()
        workers[0].remove([12])
        tail = publisher._next_seq
        for pair in pairs:
            wait_for(
                lambda: (
                    (r := query(pair, pb, workers)).complete
                    and any(
                        m.worker_id == "a" and m.matched_prefix_blocks == 1
                        for m in r.matches
                    )
                )
            )
        assert publisher._next_seq == tail, (
            "tail repair should not request a new barrier/snapshot"
        )

        # A stopped local Indexer fills only its own Bridge queue. The other
        # pair and Worker must keep progressing while the slow replica recovers.
        paused = pairs[0].server
        os.kill(paused.pid, signal.SIGSTOP)
        for _ in range(100):
            workers[0].store([11, 12, 13])
        wait_for(
            lambda: (
                (r := query(pairs[1], pb, workers)).complete
                and any(
                    m.worker_id == "a" and m.matched_prefix_blocks == 3
                    for m in r.matches
                )
            )
        )
        os.kill(paused.pid, signal.SIGCONT)
        paused = None
        wait_for(
            lambda: (
                (r := query(pairs[0], pb, workers)).complete
                and any(
                    m.worker_id == "a" and m.matched_prefix_blocks == 3
                    for m in r.matches
                )
            )
        )
        assert "overflow" in (tmp_path / "stability-0.log").read_text()

        # Horizontal Worker expansion: each existing pair discovers a new stream
        # through its hot-reloaded desired Worker list and recovers preexisting KV.
        workers.append(Worker("c"))
        workers[2].store([11, 12])
        for pair in pairs:
            cfg = json.loads(pair.config.read_text())
            cfg["worker_urls"] = [w.url for w in workers]
            pair.config.write_text(json.dumps(cfg))
            wait_for(
                lambda: (
                    (r := query(pair, pb, workers)).complete
                    and any(
                        m.worker_id == "c" and m.matched_prefix_blocks == 2
                        for m in r.matches
                    )
                )
            )
    finally:
        if paused and paused.poll() is None:
            os.kill(paused.pid, signal.SIGCONT)
        for pair in pairs:
            pair.close()
        for worker in workers:
            worker.close()


def test_router_random_failover_all_down_restart_and_membership(tmp_path, pb):
    workers = [Worker("cached"), Worker("idle")]
    workers[0].load.num_running_reqs = 5
    pairs = []
    router = None
    second_router = None
    router_log = open(tmp_path / "router.log", "w+")
    try:
        from tokenizers import Tokenizer

        tokenizer_path = (
            ROOT / "experimental/sgl-router/tests/fixtures/tiny_tokenizer.json"
        )
        prompt = "hello world " * 128
        ids = Tokenizer.from_file(str(tokenizer_path)).encode(prompt).ids
        hashes = []
        previous = ""
        for start in range(0, len(ids) - 3, 4):
            # Worker hash parity is covered by the existing fixture suite.
            digest = hashlib.sha256()
            if previous:
                digest.update(bytes.fromhex(previous))
            for token in ids[start : start + 4]:
                digest.update(int(token).to_bytes(4, "little", signed=False))
            previous = digest.hexdigest()
            hashes.append(
                int.from_bytes(bytes.fromhex(previous)[:8], "big", signed=True)
            )
        workers[0].store(hashes)
        pairs = [Pair(tmp_path, workers, pb, f"routing-{i}") for i in range(2)]
        endpoints = tmp_path / "indexers.json"
        endpoints.write_text(json.dumps([f"http://{pair.endpoint}" for pair in pairs]))
        router_port, load_port = port(), port()
        url = f"http://127.0.0.1:{router_port}"
        router = subprocess.Popen(
            [
                BIN / "sgl-router",
                "--model-id",
                "model",
                "--tokenizer-path",
                str(tokenizer_path),
                "--worker-urls",
                *[w.url for w in workers],
                "--policy",
                "cache_aware",
                "--cache-prefix-provider",
                "indexer",
                "--kv-indexer-endpoint",
                f"@{endpoints}",
                "--port",
                str(router_port),
                "--cache-affinity-min-matched-tokens",
                "0",
                "--cache-candidate-min-workers",
                "1",
                "--cache-candidate-max-workers",
                "1",
            ],
            env={
                **os.environ,
                "SGL_ROUTER_LOAD_LISTEN_ADDR": f"127.0.0.1:{load_port}",
                "RUST_LOG": "info,sgl_kv_indexer::fleet=debug",
            },
            stdout=router_log,
            stderr=router_log,
        )
        body = {"model": "model", "prompt": prompt, "max_tokens": 1}

        def request():
            assert router.poll() is None, (tmp_path / "router.log").read_text()
            return http_json(url + "/v1/chat/completions", body)

        def ready_request():
            result = request()
            with urlopen(url + "/metrics", timeout=5) as response:
                metrics = response.read().decode()
            count = next(
                (
                    int(line.split()[1])
                    for line in metrics.splitlines()
                    if line.startswith("sgl_router_indexer_complete_queries_total ")
                ),
                0,
            )
            return count > 0 and result["id"] == "cached"

        wait_for(ready_request)
        assert all(request()["id"] == "cached" for _ in range(30))
        # A second Router registers independent Reporter leases and shares the
        # same Indexers, without subscribing to or rebuilding placement itself.
        second_port, second_load_port = port(), port()
        second_args = list(router.args)
        second_args[second_args.index("--port") + 1] = str(second_port)
        second_router = subprocess.Popen(
            second_args,
            env={
                **os.environ,
                "SGL_ROUTER_LOAD_LISTEN_ADDR": f"127.0.0.1:{second_load_port}",
            },
            stdout=router_log,
            stderr=router_log,
        )
        second_url = f"http://127.0.0.1:{second_port}"

        def second_ready():
            result = http_json(second_url + "/v1/chat/completions", body)
            with urlopen(second_url + "/metrics", timeout=5) as response:
                metrics = response.read().decode()
            return result["id"] == "cached" and any(
                line.startswith("sgl_router_indexer_complete_queries_total ")
                and int(line.split()[1]) > 0
                for line in metrics.splitlines()
            )

        wait_for(second_ready)
        assert all(request()["id"] == "cached" for _ in range(5))
        second_router.terminate()
        second_router.wait(timeout=10)
        router_log.flush()
        log = (tmp_path / "router.log").read_text()
        for pair in pairs:
            assert pair.endpoint in log, (
                "both randomly selected replicas must be exercised"
            )
        pairs[0].server.kill()
        pairs[0].server.wait(timeout=5)
        assert all(request()["id"] == "cached" for _ in range(10))
        pairs[1].server.kill()
        pairs[1].server.wait(timeout=5)
        assert all(request()["id"] == "idle" for _ in range(5)), (
            "all down must use fresh minimum load"
        )
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
        if second_router and second_router.poll() is None:
            second_router.terminate()
            second_router.wait(timeout=10)
        if router and router.poll() is None:
            router.terminate()
            router.wait(timeout=10)
        router_log.close()
        for pair in pairs:
            pair.close()
        for worker in workers:
            worker.close()


class RealWorker:
    """A real CUDA SGLang scheduler with a tiny, randomly initialized Qwen2.

    Small weights keep this functional test independent of model downloads and
    leave room for unrelated GPU jobs. This is not an accuracy/performance run.
    """

    def __init__(self, directory, model, rank):
        self.worker_id = f"gpu-worker-{rank}"
        self.url = f"http://127.0.0.1:{port()}"
        self.log_path = directory / f"{self.worker_id}.log"
        self.log = open(self.log_path, "w+")
        config = {
            "publisher": "zmq",
            "worker_id": self.worker_id,
            "model": "model",
            "endpoint": f"tcp://*:{port()}",
            "snapshot_endpoint": f"tcp://*:{port()}",
            "replay_endpoint": f"tcp://*:{port()}",
            "topic": "kv",
        }
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model),
            "--served-model-name",
            "model",
            "--load-format",
            "dummy",
            "--dtype",
            "float16",
            "--host",
            "127.0.0.1",
            "--port",
            self.url.rsplit(":", 1)[1],
            "--mem-fraction-static",
            "0.99",
            "--max-total-tokens",
            "512",
            "--context-length",
            "256",
            "--max-running-requests",
            "4",
            "--disable-cuda-graph",
            "--disable-overlap-schedule",
            "--attention-backend",
            "triton",
            "--kv-events-config",
            json.dumps(config),
            "--random-seed",
            "1",
        ]
        self.process = subprocess.Popen(
            command,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(rank),
                "PYTHONPATH": str(ROOT / "python"),
            },
            stdout=self.log,
            stderr=self.log,
            start_new_session=True,
        )

    def ready(self):
        assert self.process.poll() is None, self.log_path.read_text()[-8000:]
        return http_json(self.url + "/server_info")

    def close(self):
        import signal

        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self.log.close()


@pytest.mark.skipif(
    os.environ.get("KV_REPLICA_GPU_TESTS") != "1",
    reason="requires two CUDA GPUs; set KV_REPLICA_GPU_TESTS=1",
)
def test_real_two_gpu_workers_router_and_recoverable_pairs(tmp_path, pb):
    model = tmp_path / "tiny-qwen2"
    shutil.copytree(CRATE / "tests/fixtures/tiny_qwen2", model)
    shutil.copyfile(
        ROOT / "experimental/sgl-router/tests/fixtures/tiny_tokenizer.json",
        model / "tokenizer.json",
    )
    workers, pairs = [], []
    router = None
    log = open(tmp_path / "gpu-router.log", "w+")
    try:
        for rank in range(2):
            workers.append(RealWorker(tmp_path, model, rank))
        for worker in workers:
            wait_for(worker.ready, seconds=180)
        prompt = "hello world " * 8
        from tokenizers import Tokenizer

        tokens = Tokenizer.from_file(str(model / "tokenizer.json")).encode(prompt).ids
        warm = http_json(
            workers[0].url + "/generate",
            {
                "input_ids": tokens,
                "sampling_params": {
                    "max_new_tokens": 8,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        assert warm["meta_info"]["completion_tokens"] == 8
        # Warm cache predates both replicas: live-only implementations fail.
        pairs = [Pair(tmp_path, workers, pb, f"gpu-pair-{i}") for i in range(2)]
        hashes = []
        prior = b""
        for token in tokens:
            prior = hashlib.sha256(prior + int(token).to_bytes(4, "little")).digest()
            hashes.append(int.from_bytes(prior[:8], "big", signed=True))
        rpc = pb[0].ReplicaPrefixRequest(
            namespace="default",
            model="model",
            hash_schema_version=1,
            page_size=1,
            hashes=hashes,
            eligible_streams=[
                pb[0].StreamKey(namespace="default", worker_id=w.worker_id)
                for w in workers
            ],
        )

        def recovered(pair):
            r = pair.stub.MatchPrefix(rpc, timeout=1)
            return r.complete and any(
                m.worker_id == workers[0].worker_id
                and m.matched_prefix_blocks >= len(tokens) - 1
                for m in r.matches
            )

        for pair in pairs:
            wait_for(lambda: recovered(pair))
        endpoints = tmp_path / "gpu-indexers.json"
        endpoints.write_text(json.dumps([f"http://{pair.endpoint}" for pair in pairs]))
        router_port, load_port = port(), port()
        router_url = f"http://127.0.0.1:{router_port}"
        router = subprocess.Popen(
            [
                BIN / "sgl-router",
                "--model-id",
                "model",
                "--tokenizer-path",
                str(model / "tokenizer.json"),
                "--worker-urls",
                *[w.url for w in workers],
                "--policy",
                "cache_aware",
                "--cache-prefix-provider",
                "indexer",
                "--kv-indexer-endpoint",
                f"@{endpoints}",
                "--port",
                str(router_port),
                "--cache-affinity-min-matched-tokens",
                "0",
                "--cache-candidate-min-workers",
                "1",
                "--cache-candidate-max-workers",
                "1",
            ],
            env={
                **os.environ,
                "SGL_ROUTER_LOAD_LISTEN_ADDR": f"127.0.0.1:{load_port}",
                "RUST_LOG": "info,sgl_kv_indexer::fleet=debug",
            },
            stdout=log,
            stderr=log,
        )
        body = {
            "model": "model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0,
        }

        def request():
            assert router.poll() is None, (tmp_path / "gpu-router.log").read_text()[
                -5000:
            ]
            response = http_json(router_url + "/v1/chat/completions", body, timeout=120)
            assert response["choices"]
            return response

        wait_for(request)
        time.sleep(3)
        for _ in range(8):
            request()
        pairs[0].server.kill()
        pairs[0].server.wait(timeout=5)
        for _ in range(4):
            request()
        pairs[1].server.kill()
        pairs[1].server.wait(timeout=5)
        for _ in range(4):
            request()
        pairs[0].start_server()
        wait_for(lambda: recovered(pairs[0]))
        pairs.append(Pair(tmp_path, workers, pb, "gpu-scale-out"))
        wait_for(lambda: recovered(pairs[2]))
        endpoints.write_text(json.dumps([f"http://{pairs[2].endpoint}"]))
        time.sleep(2.5)
        pairs[0].server.kill()
        pairs[0].server.wait(timeout=5)
        for _ in range(4):
            request()
        with urlopen(router_url + "/metrics", timeout=5) as response:
            metrics = response.read().decode()
        (tmp_path / "gpu-router-metrics.txt").write_text(metrics)
        counts = {
            fields[0]: int(fields[1])
            for line in metrics.splitlines()
            if (fields := line.split()) and fields[0].startswith("sgl_router_indexer_")
        }
        assert counts["sgl_router_indexer_complete_queries_total"] > 0
        assert counts["sgl_router_indexer_fallback_queries_total"] >= 4
    finally:
        if router and router.poll() is None:
            router.terminate()
            router.wait(timeout=10)
        log.close()
        for pair in pairs:
            pair.close()
        for worker in workers:
            worker.close()
