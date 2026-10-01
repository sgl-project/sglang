"""Bounded scheduler latency observations, independent of capture selection."""

from __future__ import annotations

import math
from collections import deque

import msgspec
from sglang.srt.training_capture.config import CaptureLatencyConfig


class CaptureRequestLatency(msgspec.Struct):
    num_output_tokens: int = 0
    last_output_time: float = 0.0
    done: bool = False


def request_latency(req, now):
    """Observe committed CPU output counts, never inspect or synchronize CUDA."""
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    state = req.training_capture_latency
    if state is not None and state.done:
        return None, None
    if isinstance(req.finished_reason, FINISH_ABORT):
        req.training_capture_latency = CaptureRequestLatency(done=True)
        return None, None
    if req.is_retracted:
        return None, None
    num_tokens = (
        req.finished_len if req.finished_len is not None else len(req.output_ids)
    )
    if not num_tokens:
        return None, None
    if state is None:
        state = req.training_capture_latency = CaptureRequestLatency()
    ttft = tpot = None
    if state.num_output_tokens == 0:
        received = req.time_stats.scheduler_recv_time
        if received > 0:
            ttft = now - received
    elif num_tokens > state.num_output_tokens:
        tpot = (now - state.last_output_time) / (num_tokens - state.num_output_tokens)
    if num_tokens > state.num_output_tokens:
        state.num_output_tokens = num_tokens
        state.last_output_time = now
    state.done = req.finished()
    return ttft, tpot


class CaptureLatency:
    def __init__(self, config: CaptureLatencyConfig, interval_seconds: float):
        self.config = config
        self.interval_seconds = interval_seconds
        self.budgets = {
            name: budget
            for name, budget in (
                ("ttft", config.ttft_seconds),
                ("tpot", config.tpot_seconds),
            )
            if budget is not None
        }
        self.samples = {
            name: deque(maxlen=config.max_observations) for name in self.budgets
        }
        self.observation_ct = dict.fromkeys(self.budgets, 0)
        self.quantiles = dict.fromkeys(self.budgets)
        self.invalid_ct = 0
        self.next_assessment = 0.0
        self.blocked = False
        self.recovery_ready = False
        self.ever_ready = False
        self.state = "warming"
        self.reason = "latency_warming"

    def observe(self, now, *, ttft=None, tpot=None):
        for name, value in (("ttft", ttft), ("tpot", tpot)):
            if name not in self.samples or value is None:
                continue
            if not math.isfinite(value) or value < 0:
                self.invalid_ct += 1
                continue
            self.samples[name].append((now, value))
            self.observation_ct[name] += 1

    def assess(self, now):
        if now < self.next_assessment:
            return
        self.next_assessment = now + self.interval_seconds
        for name, samples in self.samples.items():
            while samples and samples[0][0] <= now - self.config.window_seconds:
                samples.popleft()
            self.quantiles[name] = (
                sorted(value for _, value in samples)[
                    math.ceil(self.config.percentile * len(samples)) - 1
                ]
                if len(samples) >= self.config.min_observations
                else None
            )
        ready = all(value is not None for value in self.quantiles.values())
        self.ever_ready |= ready
        self.recovery_ready = ready and all(
            self.quantiles[name] <= budget * self.config.recovery_fraction
            for name, budget in self.budgets.items()
        )
        breach = next(
            (
                name
                for name, budget in self.budgets.items()
                if self.quantiles[name] is not None and self.quantiles[name] > budget
            ),
            None,
        )
        if breach is not None:
            self.blocked = True
            self.state, self.reason = "breached", "latency_" + breach
        elif self.blocked and not self.recovery_ready:
            # Missing/expired observations cannot clear an existing breach.
            if not ready:
                self.state, self.reason = "stale", "latency_stale"
            else:
                self.state, self.reason = "breached", "latency_hysteresis"
        else:
            self.blocked = False
            self.state = (
                "healthy" if ready else "stale" if self.ever_ready else "warming"
            )
            self.reason = "latency_" + self.state

    def stats(self, now):
        self.assess(now)
        return {
            "state": self.state,
            "blocked": self.blocked,
            "recovery_ready": self.recovery_ready,
            "percentile": self.config.percentile,
            "window_seconds": self.config.window_seconds,
            "invalid_observations": self.invalid_ct,
            "metrics": {
                name: {
                    "budget_seconds": budget,
                    "percentile_seconds": self.quantiles[name],
                    "window_observations": len(self.samples[name]),
                    "observations": self.observation_ct[name],
                    "last_observation_age_seconds": (
                        max(0.0, now - self.samples[name][-1][0])
                        if self.samples[name]
                        else None
                    ),
                }
                for name, budget in self.budgets.items()
            },
        }
