from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


def _timing_summary(values: list[float]) -> str:
    if not values:
        return "-"
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return (
        f"{sum(ordered) / len(ordered) * 1000:.2f}/"
        f"{ordered[p95_index] * 1000:.2f}/"
        f"{ordered[-1] * 1000:.2f}"
    )


class MooncakeTransferDebugStats:
    """Low-frequency diagnostics for the prefill Mooncake transfer path."""

    def __init__(
        self,
        rank_label: str,
        queue_count: int,
        report_interval: float = 5.0,
    ):
        self.rank_label = rank_label
        self.report_interval = report_interval
        self._lock = threading.Lock()
        self._queue_pending = [0] * queue_count
        self._queue_active = [0] * queue_count
        self._engine_active = 0
        self._engine_active_peak = 0
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._last_report = time.monotonic()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        logger.info(
            "Mooncake transfer debug enabled rank=%s interval=%.1fs queues=%d",
            self.rank_label,
            self.report_interval,
            len(self._queue_pending),
        )
        threading.Thread(
            target=self._report_loop,
            name="MooncakeTransferDebugStats",
            daemon=True,
        ).start()

    def record_enqueue(self, queue_index: int) -> None:
        with self._lock:
            self._queue_pending[queue_index] += 1

    def record_dequeue(self, queue_index: int, queue_wait: float) -> None:
        with self._lock:
            self._queue_pending[queue_index] = max(
                0, self._queue_pending[queue_index] - 1
            )
            self._queue_active[queue_index] += 1
            self._timings["queue"].append(queue_wait)

    def record_chunk_done(
        self,
        queue_index: int,
        duration: float,
        *,
        skipped: bool = False,
        error: bool = False,
    ) -> None:
        with self._lock:
            self._queue_active[queue_index] = max(
                0, self._queue_active[queue_index] - 1
            )
            self._timings["chunk"].append(duration)
            self._counters["chunks"] += 1
            self._counters["skipped"] += int(skipped)
            self._counters["errors"] += int(error)

    def record_kv_send(
        self, duration: float, token_count: int, *, skipped: bool = False
    ) -> None:
        with self._lock:
            self._timings["kv"].append(duration)
            self._counters["tokens"] += token_count
            self._counters["kv_skipped"] += int(skipped)

    def record_state_send(self, duration: float) -> None:
        with self._lock:
            self._timings["state"].append(duration)

    def record_aux_send(self, duration: float) -> None:
        with self._lock:
            self._timings["aux"].append(duration)

    def record_status_sync(self, duration: float) -> None:
        with self._lock:
            self._timings["status"].append(duration)

    def record_engine_start(self) -> None:
        with self._lock:
            self._engine_active += 1
            self._engine_active_peak = max(
                self._engine_active_peak, self._engine_active
            )

    def record_engine_done(
        self,
        duration: float,
        byte_count: int,
        block_count: int,
        *,
        error: bool = False,
    ) -> None:
        with self._lock:
            self._engine_active = max(0, self._engine_active - 1)
            self._timings["engine"].append(duration)
            self._counters["engine_calls"] += 1
            self._counters["engine_bytes"] += byte_count
            self._counters["engine_blocks"] += block_count
            self._counters["engine_errors"] += int(error)

    def _snapshot(self) -> Optional[dict]:
        now = time.monotonic()
        with self._lock:
            interval = now - self._last_report
            self._last_report = now
            counters = dict(self._counters)
            timings = dict(self._timings)
            pending = self._queue_pending.copy()
            active = self._queue_active.copy()
            engine_active = self._engine_active
            engine_active_peak = self._engine_active_peak
            self._counters.clear()
            self._timings.clear()
            self._engine_active_peak = self._engine_active

        if not counters and not any(pending) and not any(active) and engine_active == 0:
            return None
        return {
            "interval": interval,
            "counters": counters,
            "timings": timings,
            "pending": pending,
            "active": active,
            "engine_active": engine_active,
            "engine_active_peak": engine_active_peak,
        }

    def _report_loop(self) -> None:
        while True:
            time.sleep(self.report_interval)
            snapshot = self._snapshot()
            if snapshot is None:
                continue

            counters = snapshot["counters"]
            timings = snapshot["timings"]
            interval = max(snapshot["interval"], 1e-9)
            payload_gbps = counters.get("engine_bytes", 0) / interval / 1e9
            logger.info(
                "Mooncake transfer debug rank=%s interval=%.2fs "
                "chunks=%d tokens=%d engine_calls=%d blocks=%d bytes=%d "
                "payload=%.3fGB/s pending=%s active=%s "
                "engine_active=%d/%d skipped=%d kv_skipped=%d errors=%d/%d "
                "timing_ms(avg/p95/max) queue=%s chunk=%s kv=%s state=%s "
                "aux=%s status=%s engine_sync=%s",
                self.rank_label,
                interval,
                counters.get("chunks", 0),
                counters.get("tokens", 0),
                counters.get("engine_calls", 0),
                counters.get("engine_blocks", 0),
                counters.get("engine_bytes", 0),
                payload_gbps,
                snapshot["pending"],
                snapshot["active"],
                snapshot["engine_active"],
                snapshot["engine_active_peak"],
                counters.get("skipped", 0),
                counters.get("kv_skipped", 0),
                counters.get("errors", 0),
                counters.get("engine_errors", 0),
                _timing_summary(timings.get("queue", [])),
                _timing_summary(timings.get("chunk", [])),
                _timing_summary(timings.get("kv", [])),
                _timing_summary(timings.get("state", [])),
                _timing_summary(timings.get("aux", [])),
                _timing_summary(timings.get("status", [])),
                _timing_summary(timings.get("engine", [])),
            )
