"""Worker-owned, leased gRPC load streams with one latest sample per target."""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import grpc
import msgspec
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.load_report_pb2 import LoadReportAck, LoadSample

logger = logging.getLogger(__name__)


@dataclass
class _Target:
    token: str
    expires: float
    thread: threading.Thread = None


class LoadReporter:
    def __init__(self, namespace, worker_id, rank, epoch):
        self.identity = dict(namespace=namespace, worker_id=worker_id, dp_rank=rank, worker_epoch=epoch)
        self.lock = threading.Lock()
        self.targets = {}
        self.latest = None
        self.sequence = 0
        self.stopped = threading.Event()

    def update(self, load):
        fields = ("num_running_reqs", "num_waiting_reqs", "num_used_tokens",
                  "max_total_num_tokens", "num_waiting_uncached_tokens", "num_total_tokens",
                  "max_running_requests", "total_prefill_uncached_tokens", "total_prefill_busy_us")
        values = {field: int(getattr(load, field, 0)) for field in fields}
        values.update(generation_throughput=float(getattr(load, "gen_throughput", 0)),
                      cache_hit_rate=float(getattr(load, "cache_hit_rate", 0)),
                      utilization=float(getattr(load, "utilization", 0)))
        queues = getattr(load, "queues", None)
        values["num_prefill_reqs"] = int(getattr(queues, "num_prefill_reqs", values["num_waiting_reqs"]))
        values["num_decode_reqs"] = int(getattr(queues, "num_decode_reqs", values["num_running_reqs"]))
        with self.lock:
            self.sequence += 1
            self.latest = (time.monotonic(), self.sequence, values)

    def register(self, target, token, lease_seconds):
        parsed = urlparse(target)
        if parsed.scheme != "http" or not parsed.hostname or not parsed.port or parsed.path not in ("", "/"):
            raise ValueError("load target must be an http://host:port gRPC endpoint")
        if not token or len(token) > 256 or not 1 <= lease_seconds <= 30:
            raise ValueError("invalid reporting token or lease")
        with self.lock:
            current = self.targets.get(target)
            if current is not None and current.token == token and current.thread.is_alive():
                current.expires = time.monotonic() + lease_seconds
                return self.identity
            if len(self.targets) >= 64 and current is None:
                raise ValueError("too many Router reporting targets")
            state = _Target(token=token, expires=time.monotonic() + lease_seconds)
            thread = threading.Thread(target=self._report, args=(target, state), daemon=True,
                                      name="kv-load-reporter")
            state.thread = thread
            self.targets[target] = state
            thread.start()
        return self.identity

    def _active(self, target, state):
        with self.lock:
            return (not self.stopped.is_set() and self.targets.get(target) is state
                    and time.monotonic() < state.expires)

    def _report(self, target, state):
        last_sequence = 0

        def samples():
            nonlocal last_sequence
            while self._active(target, state):
                with self.lock:
                    latest = self.latest
                if latest:
                    measured, sequence, fields = latest
                    age = time.monotonic() - measured
                    if sequence > last_sequence and age < 1:
                        last_sequence = sequence
                        yield LoadSample(**self.identity, **fields, registration_token=state.token,
                                         sequence=sequence, sample_age_ms=int(age * 1000))
                self.stopped.wait(0.1)

        channel = grpc.insecure_channel(urlparse(target).netloc)
        report = channel.stream_unary("/kv_load.v1.LoadMonitor/Report",
                                      request_serializer=LoadSample.SerializeToString,
                                      response_deserializer=LoadReportAck.FromString)
        try:
            while self._active(target, state):
                try:
                    report(samples(), timeout=5)
                except grpc.RpcError:
                    self.stopped.wait(0.2)
        finally:
            channel.close()
            with self.lock:
                if self.targets.get(target) is state:
                    del self.targets[target]

    def close(self):
        self.stopped.set()


async def register_worker_reporting(server_args, body):
    """HTTP control plane fans registration out to the owning scheduler ranks."""
    from sglang.srt.disaggregation.kv_events import KVEventsConfig, ZmqEventPublisher

    raw = server_args.resolved_dict()
    config = KVEventsConfig.from_cli(raw["kv_events_config"])
    if not config.snapshot_endpoint or not config.worker_id:
        raise ValueError("reporting requires snapshot_endpoint and worker_id")
    target, token = body["target"], body["token"]
    lease_seconds = body.get("lease_seconds", 6)
    # Validation also happens inside the publisher before allocating a target.
    command = json.dumps(dict(target=target, token=token, lease_seconds=lease_seconds)).encode()

    async def one(rank):
        endpoint = ZmqEventPublisher.offset_endpoint_port(config.snapshot_endpoint, rank)
        endpoint = endpoint.replace("tcp://*:", "tcp://127.0.0.1:").replace("tcp://0.0.0.0:", "tcp://127.0.0.1:").replace("tcp://[::]:", "tcp://[::1]:")
        socket = zmq.asyncio.Context.instance().socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(endpoint)
            await socket.send_multipart([b"", b"start-reporting-v1", command])
            frames = await asyncio.wait_for(socket.recv_multipart(), timeout=3)
            if len(frames) != 3 or frames[1] != b"registered":
                raise ValueError(f"scheduler rejected reporting: {frames[1:]}")
            return msgspec.msgpack.decode(frames[2])
        finally:
            socket.close()

    ranks = int(raw.get("dp_size", 1))
    if not 1 <= ranks <= 4096:
        raise ValueError("invalid dp_size")
    return {"streams": await asyncio.gather(*(one(rank) for rank in range(ranks))), "lease_seconds": lease_seconds}
