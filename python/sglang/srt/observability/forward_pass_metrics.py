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
from functools import wraps
from itertools import count
from typing import TYPE_CHECKING, Callable, Optional

import msgspec
from msgspec.structs import replace

if TYPE_CHECKING:
    from sglang.srt.utils.device_timer import DeviceTimer

# Schema version. Must match the consumer (Dynamo's ForwardPassMetrics).
# Bump when the schema changes incompatibly.
FPM_VERSION: int = 1

logger = logging.getLogger(__name__)


class WelfordAccumulator:
    """Welford's online algorithm for count / total / population-variance.

    Numerically stable single-pass computation.
    """

    __slots__ = ("count", "total", "_mean", "_m2")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self._mean = 0.0
        self._m2 = 0.0

    def add(self, v: int) -> None:
        self.count += 1
        self.total += v
        delta = v - self._mean
        self._mean += delta / self.count
        delta2 = v - self._mean
        self._m2 += delta * delta2

    def variance(self) -> float:
        if self.count == 0:
            return 0.0
        return self._m2 / self.count


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
    ``wall_time`` is the iteration duration in seconds.
    An idle heartbeat (all zeros, wall_time=0) is emitted when the
    engine transitions from active to idle.

    Field order must match Dynamo's ``ForwardPassMetrics`` in
    ``dynamo.common.forward_pass_metrics`` — msgspec uses positional
    encoding so any mismatch silently corrupts data.
    """

    version: int = FPM_VERSION
    worker_id: str = ""
    dp_rank: int = 0
    counter_id: int = 0
    wall_time: float = 0.0
    scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()
    queued_requests: QueuedRequestMetrics = QueuedRequestMetrics()


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)


def encode(metrics: ForwardPassMetrics) -> bytes:
    return _encoder.encode(metrics)


def decode(data: bytes) -> ForwardPassMetrics:
    return _decoder.decode(data)


class FpmTiming:
    """First-start to last-end GPU span on one stream, including segment gaps."""

    def __init__(self):
        self.num_intervals = 0
        self._pending = 0
        self._sealed = False
        self._callback = None
        self._first_interval = None
        self._last_interval = None
        self._same_stream = True

    def on_interval_start(self, interval):
        self.num_intervals += 1
        self._pending += 1
        if self._first_interval is None:
            self._first_interval = interval
        elif interval.stream != self._first_interval.stream:
            self._same_stream = False
        self._last_interval = interval

    def on_interval_ready(self, interval):
        self._pending -= 1
        self._notify()

    def seal(self):
        self._sealed = True
        self._notify()

    def when_ready(self, callback: Callable[[Optional[float]], None]):
        self._callback = callback
        self._notify()

    def _notify(self):
        if not self._sealed or self._pending or self._callback is None:
            return
        callback, self._callback = self._callback, None
        if self._first_interval is None:
            callback(0.0)
        elif not self._same_stream:
            callback(None)
        else:
            callback(
                self._first_interval.start_event.elapsed_time(
                    self._last_interval.end_event
                )
                / 1000.0
            )

    def publish_when_ready(self, metrics, publisher):
        # Hold only immutable scalar statistics, never a live batch/request.
        def publish(elapsed):
            if elapsed is None:
                logger.warning(
                    "FPM timing spans multiple CUDA streams; skipping ambiguous span"
                )
                return
            publisher.publish(replace(metrics, wall_time=elapsed))

        self.when_ready(publish)


def wrap_forward_with_fpm(forward: Callable, timer: DeviceTimer) -> Callable:
    """Installed on the scheduler instance only when that rank enables FPM."""

    @wraps(forward)
    def timed_forward(batch, *args, **kwargs):
        # PREBUILT can recursively dispatch an inner idle forward for DP MLP
        # synchronization. Let that real forward own the capture and result.
        if batch.forward_mode.is_prebuilt():
            return forward(batch, *args, **kwargs)
        timing = FpmTiming()
        try:
            with timer.capture(timing):
                result = forward(batch, *args, **kwargs)
        finally:
            timing.seal()
        result.fpm_timing = timing
        return result

    return timed_forward


class _FpmPublisherThread:
    """Background thread that serializes and sends ForwardPassMetrics over ZMQ.

    Heartbeats require an explicit idle state from the scheduler.
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
        self._idle = False

        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(endpoint)
        self._zmq = zmq

        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="fpm-zmq-publisher"
        )
        self._thread.start()

    def set_idle(self, idle: bool) -> None:
        """Set whether the scheduler permits idle heartbeats."""
        self._idle = idle

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
            self._ctx.term()
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
                if (
                    self._idle
                    and time.monotonic() - last_publish >= self.HEARTBEAT_INTERVAL
                ):
                    metrics = ForwardPassMetrics(
                        worker_id=self._worker_id,
                        dp_rank=self._dp_rank,
                    )
                else:
                    continue

            try:
                seq = next(self._seq)
                metrics = msgspec.structs.replace(metrics, counter_id=seq)
                payload = encode(metrics)
                seq_bytes = seq.to_bytes(8, "big")
                self._pub.send_multipart((topic, seq_bytes, payload), flags=zmq.NOBLOCK)
                last_publish = time.monotonic()
            except zmq.Again:
                pass
            except Exception:
                logger.warning("FPM publisher send failed", exc_info=True)
