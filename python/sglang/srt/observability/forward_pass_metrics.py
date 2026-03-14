"""
Forward pass metrics for per-iteration scheduler telemetry.

Emits per-iteration scheduling metrics over ZMQ PUB so that external
consumers can observe scheduler behavior in real time without polling
Prometheus.

Uses msgspec.Struct for zero-copy serialization.

Data flow::

    Scheduler process:
        SchedulerMetricsMixin._emit_forward_pass_metrics()
          -> _FpmPublisherThread -> ZMQ PUB (localhost)

    External consumer:
        ZMQ SUB -> deserialize ForwardPassMetrics

"""

from __future__ import annotations

import logging
import queue
import threading
import time
from itertools import count

import msgspec

logger = logging.getLogger(__name__)


class WelfordAccumulator:
    """Welford's online algorithm for count / sum / population-variance.

    Numerically stable single-pass computation.
    """

    __slots__ = ("n", "s", "_mean", "_m2")

    def __init__(self) -> None:
        self.n = 0
        self.s = 0
        self._mean = 0.0
        self._m2 = 0.0

    def add(self, v: int) -> None:
        self.n += 1
        self.s += v
        delta = v - self._mean
        self._mean += delta / self.n
        delta2 = v - self._mean
        self._m2 += delta * delta2

    def variance(self) -> float:
        if self.n == 0:
            return 0.0
        return self._m2 / self.n


class ScheduledRequestMetrics(
    msgspec.Struct,
    frozen=True,
    gc=False,
):
    """Metrics for requests scheduled in this iteration."""

    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    sum_prefill_kv_tokens: int = 0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


class QueuedRequestMetrics(
    msgspec.Struct,
    frozen=True,
    gc=False,
):
    """Metrics for requests waiting in the queue."""

    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


class ForwardPassMetrics(
    msgspec.Struct,
    frozen=True,
    gc=False,
):
    """Per-iteration metrics emitted by the scheduler.

    One message per scheduler iteration (one per forward pass).
    An idle heartbeat (all zeros, wall_time=0) is emitted when the
    engine transitions from active to idle.
    """

    worker_id: str = ""
    dp_rank: int = 0
    wall_time: float = 0.0
    scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()
    queued_requests: QueuedRequestMetrics = QueuedRequestMetrics()


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)


def encode(metrics: ForwardPassMetrics) -> bytes:
    return _encoder.encode(metrics)


def decode(data: bytes) -> ForwardPassMetrics:
    return _decoder.decode(data)


class _FpmPublisherThread:
    """Background thread that serializes and sends ForwardPassMetrics over ZMQ.

    Also emits periodic heartbeats when idle.
    """

    SHUTDOWN_TIMEOUT: float = 1.0
    HEARTBEAT_INTERVAL: float = 1.0

    def __init__(
        self,
        endpoint: str,
        worker_id: str,
        dp_rank: int,
        max_queue_size: int = 10_000,
    ) -> None:
        import zmq

        self._queue: queue.Queue[ForwardPassMetrics | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._seq = count()
        self._worker_id = worker_id
        self._dp_rank = dp_rank

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(endpoint)
        self._zmq = zmq

        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="fpm-zmq-publisher"
        )
        self._thread.start()

    def publish(self, metrics: ForwardPassMetrics) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(metrics)
        except queue.Full:
            pass

    def shutdown(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            self._pub.close(linger=0)
        except Exception:
            pass

    def _run(self) -> None:
        zmq = self._zmq
        topic = b""
        last_publish = time.monotonic()

        while self._running or not self._queue.empty():
            try:
                metrics = self._queue.get(timeout=self.HEARTBEAT_INTERVAL)
                if metrics is None:
                    break
            except queue.Empty:
                if time.monotonic() - last_publish >= self.HEARTBEAT_INTERVAL:
                    metrics = ForwardPassMetrics(
                        worker_id=self._worker_id,
                        dp_rank=self._dp_rank,
                    )
                else:
                    continue

            try:
                payload = encode(metrics)
                seq_bytes = next(self._seq).to_bytes(8, "big")
                self._pub.send_multipart(
                    (topic, seq_bytes, payload), flags=zmq.NOBLOCK
                )
                last_publish = time.monotonic()
            except zmq.Again:
                pass
            except Exception:
                logger.warning("FPM publisher send failed", exc_info=True)
