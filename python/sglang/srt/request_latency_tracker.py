# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per-request latency callback for debugging agentic workloads.

When SGLANG_ENABLE_PER_REQUEST_LATENCY is set, each request tracks the
end-to-end latency broken down by phase: prefill time, decode time, and
transfer time. This is emitted as structured log entries that can be
aggregated offline or streamed to a monitoring backend.

Extended by JoyFuture: initial implementation of the per-request latency
tracking infrastructure.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PhaseLatency:
    """Latency breakdown for a single phase of request processing."""

    phase: str  # "prefill", "decode", "transfer", "sample"
    start_time: float = 0.0
    end_time: float = 0.0
    num_tokens: int = 0

    @property
    def elapsed_ms(self) -> float:
        if self.start_time > 0 and self.end_time > 0:
            return (self.end_time - self.start_time) * 1000.0
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_ms > 0 and self.num_tokens > 0:
            return self.num_tokens / (self.elapsed_ms / 1000.0)
        return 0.0


@dataclass
class RequestLatencyRecord:
    """Complete latency record for a single request lifecycle."""

    request_id: str
    rid: str
    start_time: float = 0.0
    end_time: float = 0.0
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    phases: Dict[str, PhaseLatency] = field(default_factory=dict)
    retracted_count: int = 0
    aborted: bool = False
    abort_reason: Optional[str] = None

    def start_phase(self, phase: str) -> None:
        self.phases[phase] = PhaseLatency(phase=phase, start_time=time.monotonic())

    def end_phase(self, phase: str, num_tokens: int = 0) -> None:
        if phase in self.phases:
            self.phases[phase].end_time = time.monotonic()
            self.phases[phase].num_tokens = num_tokens

    @property
    def total_elapsed_ms(self) -> float:
        if self.start_time > 0 and self.end_time > 0:
            return (self.end_time - self.start_time) * 1000.0
        return 0.0

    @property
    def time_to_first_token_ms(self) -> float:
        prefill = self.phases.get("prefill")
        if prefill and prefill.end_time > 0:
            return (prefill.end_time - self.start_time) * 1000.0
        return 0.0

    def to_log_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "rid": self.rid,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "num_input_tokens": self.num_input_tokens,
            "num_output_tokens": self.num_output_tokens,
            "retracted_count": self.retracted_count,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason or "",
            "phases": {
                name: {
                    "elapsed_ms": round(p.elapsed_ms, 2),
                    "tokens_per_second": round(p.tokens_per_second, 2),
                    "num_tokens": p.num_tokens,
                }
                for name, p in self.phases.items()
            },
        }


class RequestLatencyTracker:
    """Tracks per-request latency across the scheduler pipeline.

    Usage::

        tracker = RequestLatencyTracker(enabled=True)
        record = tracker.start_request(req)
        # ... request goes through prefill, decode, etc ...
        tracker.end_phase(record.rid, "prefill", num_tokens=100)
        tracker.end_phase(record.rid, "decode", num_tokens=50)
        record = tracker.end_request(record.rid)
        if record:
            logger.info(f"Request {record.rid} completed in {record.total_elapsed_ms:.1f}ms")
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._records: Dict[str, RequestLatencyRecord] = {}
        self._completed: List[RequestLatencyRecord] = []

    def start_request(self, req) -> RequestLatencyRecord:
        """Start tracking a new request. Returns the latency record."""
        if not self.enabled:
            return None  # type: ignore
        record = RequestLatencyRecord(
            request_id=getattr(req, "request_id", str(req.rid)),
            rid=str(req.rid),
            start_time=time.monotonic(),
            num_input_tokens=len(getattr(req, "origin_input_ids", [])),
        )
        self._records[record.rid] = record
        return record

    def start_phase(self, rid: str, phase: str) -> None:
        """Mark the start of a processing phase for a request."""
        record = self._records.get(rid)
        if record:
            record.start_phase(phase)

    def end_phase(self, rid: str, phase: str, num_tokens: int = 0) -> None:
        """Mark the end of a processing phase for a request."""
        record = self._records.get(rid)
        if record:
            record.end_phase(phase, num_tokens)

    def record_retraction(self, rid: str) -> None:
        """Record that a request was retracted."""
        record = self._records.get(rid)
        if record:
            record.retracted_count += 1

    def end_request(self, rid: str, aborted: bool = False, abort_reason: str = None) -> Optional[RequestLatencyRecord]:
        """Mark a request as completed and return its latency record."""
        record = self._records.pop(rid, None)
        if record:
            record.end_time = time.monotonic()
            record.aborted = aborted
            record.abort_reason = abort_reason
            self._completed.append(record)
            if self.enabled:
                logger.info(
                    "Request latency: %s",
                    record.to_log_dict(),
                )
            return record
        return None

    def get_recent_records(self, n: int = 10) -> List[RequestLatencyRecord]:
        """Return the N most recently completed latency records."""
        return self._completed[-n:] if self._completed else []

    def get_summary(self) -> dict:
        """Return aggregated latency statistics across all completed requests."""
        if not self._completed:
            return {}
        total = len(self._completed)
        ttfts = [r.time_to_first_token_ms for r in self._completed if not r.aborted]
        totals = [r.total_elapsed_ms for r in self._completed if not r.aborted]
        return {
            "total_requests": total,
            "completed_requests": len([r for r in self._completed if not r.aborted]),
            "aborted_requests": len([r for r in self._completed if r.aborted]),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else 0,
            "avg_total_ms": round(sum(totals) / len(totals), 2) if totals else 0,
            "p50_total_ms": round(sorted(totals)[len(totals) // 2], 2) if totals else 0,
            "p95_total_ms": round(sorted(totals)[int(len(totals) * 0.95)], 2) if totals else 0,
        }
