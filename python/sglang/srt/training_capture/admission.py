"""Host pressure and latency feedback; callers serialize state access."""

from __future__ import annotations

from sglang.srt.training_capture.config import AdaptiveCaptureConfig
from sglang.srt.training_capture.latency import CaptureLatency


class CaptureAdmission:
    def __init__(self, ceiling: float, config: AdaptiveCaptureConfig | None):
        self.ceiling = ceiling
        self.config = config
        self.target_ratio = ceiling
        self.next_adjustment = 0.0
        self.pause_until = 0.0
        self.reason = "configured"
        self.occupancy = 0.0
        self.writer_age_seconds = 0.0
        self.decreases = self.recoveries = self.failures = self.pauses = 0
        self.latency = (
            CaptureLatency(config.latency, config.interval_seconds)
            if config is not None and config.latency is not None
            else None
        )

    def _decrease(self, now, reason):
        if now >= self.next_adjustment:
            ratio = max(self.ceiling * 0.01, self.target_ratio * 0.5)
            self.decreases += ratio < self.target_ratio
            self.target_ratio = ratio
            self.next_adjustment = now + self.config.interval_seconds
        self.reason = reason

    def _pause(self, now, reason):
        if now >= self.pause_until:
            self.pauses += 1
        self._decrease(now, reason)
        self.pause_until = max(self.pause_until, now + self.config.cooldown_seconds)

    def failure(self, now, reason):
        if self.config is not None:
            self.failures += 1
            self._pause(now, reason)

    def observe(self, now, *, occupancy, writer_age_seconds):
        self.occupancy = occupancy
        self.writer_age_seconds = writer_age_seconds
        if self.config is None:
            return self.ratio(now)
        if self.latency is not None:
            self.latency.assess(now)
        if writer_age_seconds >= self.config.writer_stall_seconds:
            self._pause(now, "writer_stall")
        elif self.latency is not None and self.latency.blocked:
            if self.latency.state == "breached":
                self._pause(now, self.latency.reason)
            else:
                self.reason = self.latency.reason
        elif now < self.pause_until:
            pass
        elif occupancy >= self.config.high_watermark:
            self._decrease(now, "host_pressure")
        elif (
            occupancy <= self.config.low_watermark
            and now >= self.next_adjustment
            and self.target_ratio < self.ceiling
            and (self.latency is None or self.latency.recovery_ready)
        ):
            self.target_ratio = min(
                self.ceiling, self.target_ratio + self.ceiling * 0.1
            )
            self.next_adjustment = now + self.config.interval_seconds
            self.reason = (
                "recovering" if self.target_ratio < self.ceiling else "configured"
            )
            self.recoveries += 1
        return self.ratio(now)

    def ratio(self, now):
        stalled = (
            self.config is not None
            and self.writer_age_seconds >= self.config.writer_stall_seconds
        )
        latency_blocked = self.latency is not None and self.latency.blocked
        return (
            0.0
            if stalled or latency_blocked or now < self.pause_until
            else self.target_ratio
        )

    def stats(self, now, *, disabled=False):
        latency = self.latency.stats(now) if self.latency is not None else None
        return {
            "adaptive": self.config is not None,
            "configured_ratio": self.ceiling,
            "target_ratio": self.target_ratio,
            "effective_ratio": 0.0 if disabled else self.ratio(now),
            "reason": self.reason,
            "cooldown_remaining_seconds": max(0.0, self.pause_until - now),
            "occupied_fraction": self.occupancy,
            "writer_age_seconds": self.writer_age_seconds,
            "decreases": self.decreases,
            "recoveries": self.recoveries,
            "failures": self.failures,
            "pauses": self.pauses,
            "latency": latency,
        }
