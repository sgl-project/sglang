"""Low-overhead observability for DFLASH_CONFIDENCE scheduling decisions."""

from __future__ import annotations

import time
from collections import Counter, deque

import torch

from sglang.srt.environ import envs


class DFlashConfidenceObserver:
    """Accumulates bounded per-process diagnostics without changing scheduling."""

    _HIST_BINS = 1_000

    def __init__(self) -> None:
        self._confidence_hist = torch.zeros(self._HIST_BINS, dtype=torch.int64)
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._verify_batch_sizes: Counter[int] = Counter()
        self._verify_reasons: Counter[str] = Counter()
        self.deferred_requests = 0
        self.deferred_tokens = 0
        self.low_confidence_tokens = 0
        self._sps_enabled = bool(envs.SGLANG_DSPARK_ENABLE_SPS_RECORD.get())
        self._sps_records: deque[dict] = deque(maxlen=200_000)
        self._step_started_at: float | None = None
        self._forward_ct = 0
        self._verify_num_draft_tokens: int | None = None
        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())

    def configure_sps(self, *, verify_num_draft_tokens: int) -> None:
        self._verify_num_draft_tokens = int(verify_num_draft_tokens)

    def begin_step(self) -> None:
        """Start wall-clock timing for one uniform DFLASH2 decode step."""
        if self._sps_enabled:
            self._step_started_at = time.monotonic()

    def finish_step(self, *, batch_size: int, verify_tokens: int) -> None:
        if not self._sps_enabled or self._step_started_at is None:
            return
        elapsed_ms = (time.monotonic() - self._step_started_at) * 1_000.0
        self._sps_records.append(
            {
                "forward_ct": self._forward_ct,
                "num_running_reqs": int(batch_size),
                "num_verify_tokens": int(verify_tokens),
                "step_cpu_ms": elapsed_ms,
            }
        )
        self._forward_ct += 1
        self._step_started_at = None

    def observe(
        self,
        *,
        confidence: torch.Tensor | None,
        verify_lens: torch.Tensor | None,
        reason: str,
        deferred_tokens: int = 0,
        low_confidence_tokens: int = 0,
    ) -> None:
        self._verify_reasons[reason] += 1
        if verify_lens is not None:
            self._verify_batch_sizes[int(verify_lens.sum().item())] += 1
            self.deferred_requests += int(
                (verify_lens < verify_lens.max()).sum().item()
            )
        if confidence is not None:
            values = confidence.detach().float().cpu().reshape(-1).clamp_(0.0, 1.0)
            if values.numel():
                bins = (values * (self._HIST_BINS - 1)).to(torch.long)
                self._confidence_hist.scatter_add_(
                    0, bins, torch.ones_like(bins, dtype=torch.int64)
                )
                self._confidence_sum += float(values.sum())
                self._confidence_count += int(values.numel())
        self.deferred_tokens += int(deferred_tokens)
        self.low_confidence_tokens += int(low_confidence_tokens)

    def clear(self) -> None:
        self.__init__()

    def _quantile(self, q: float) -> float:
        if self._confidence_count == 0:
            return float("nan")
        rank = max(
            0, min(self._confidence_count - 1, int(q * (self._confidence_count - 1)))
        )
        index = int(
            torch.searchsorted(torch.cumsum(self._confidence_hist, 0), rank + 1)
        )
        return index / (self._HIST_BINS - 1)

    def dump(self) -> dict:
        percentiles = {}
        if self._confidence_count:
            percentiles = {
                "mean": self._confidence_sum / self._confidence_count,
                "p50": self._quantile(0.50),
                "p90": self._quantile(0.90),
                "p99": self._quantile(0.99),
            }
        return {
            "algorithm": "DFLASH_CONFIDENCE",
            "confidence": percentiles,
            "verify_batch_size_distribution": dict(self._verify_batch_sizes),
            "verify_reason_counts": dict(self._verify_reasons),
            "deferred_requests": self.deferred_requests,
            "deferred_tokens": self.deferred_tokens,
            "low_confidence_tokens": self.low_confidence_tokens,
            "deferred_wait_ms": {"mean": 0.0, "max": 0.0},
            "mode": "static" if self._sps_enabled else "confidence",
            "verify_num_draft_tokens": self._verify_num_draft_tokens,
            "components": ["core", "step_cpu_time"] if self._sps_enabled else [],
            "simulate_acc_len": (
                self._simulate_acc_len if self._simulate_acc_len > 0 else None
            ),
            "records": list(self._sps_records),
        }
