"""Pure report builder for the embedded load reporter.

Converts a SnapshotView into a pb.LoadReport proto, applying staleness logic
and assigning monotonically increasing sequence IDs.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from sglang.srt.load_reporter.config import WorkerMetadata
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.store import SnapshotView

# ---------------------------------------------------------------------------
# Sequence allocator
# ---------------------------------------------------------------------------


class SequenceAllocator:
    """Allocate process-local monotonically increasing report sequence IDs."""

    def __init__(self) -> None:
        """Initialize the sequence at zero before the first report."""
        self._value = 0

    def next(self) -> int:
        """Return the next positive sequence ID."""
        self._value += 1
        return self._value


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


class ReportBuilder:
    """Convert validated snapshot views into protocol load reports."""

    def __init__(
        self,
        source_instance_id: str,
        stale_after_ms: int,
        sequence: SequenceAllocator,
    ) -> None:
        """Initialize report identity, staleness policy, and sequence allocation."""
        self._source_instance_id = source_instance_id
        self._stale_after_ms = stale_after_ms
        self._sequence = sequence

    def build(
        self,
        view: SnapshotView,
        identity: WorkerMetadata,
        *,
        report_time_unix_ms: int,
    ) -> pb.LoadReport:
        """Build one report from the latest full snapshot."""
        if not view.ranks:
            status = pb.REPORT_STATUS_UNREACHABLE
            error: Optional[str] = view.last_error or "no authoritative rank snapshot"
        else:
            oldest_age_ms = max(
                report_time_unix_ms - rank.snapshot_time_unix_ms for rank in view.ranks
            )
            if oldest_age_ms > self._stale_after_ms:
                status = pb.REPORT_STATUS_STALE
                error = view.last_error or f"load snapshot stale by {oldest_age_ms} ms"
            else:
                status = pb.REPORT_STATUS_HEALTHY
                error = None

        report = pb.LoadReport(
            source_instance_id=self._source_instance_id,
            sequence_id=self._sequence.next(),
            report_time_unix_ms=report_time_unix_ms,
            worker=pb.Worker(
                worker_addr=identity.worker_addr,
                worker_type=identity.worker_type,
            ),
            status=status,
            ranks=[pb.RankLoad(**dataclasses.asdict(rank)) for rank in view.ranks],
        )
        if identity.model is not None:
            report.worker.model = identity.model
        if error is not None:
            report.last_error = error
        return report
