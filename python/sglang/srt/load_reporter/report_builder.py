"""Pure report builder for the embedded load reporter."""

from __future__ import annotations

from typing import Optional, Sequence

import msgspec

from sglang.srt.load_reporter.config import WorkerMetadata
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.snapshot_validation import RankSnapshot

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
    """Convert validated rank tuples into protocol load reports."""

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
        ranks: Sequence[RankSnapshot],
        identity: WorkerMetadata,
        *,
        report_time_unix_ms: int,
    ) -> pb.LoadReport:
        """Build one report from this attempt's validated rank tuple."""
        if not ranks:
            raise ValueError("a healthy/stale report requires at least one rank")
        oldest_age_ms = max(
            report_time_unix_ms - rank.snapshot_time_unix_ms for rank in ranks
        )
        if oldest_age_ms > self._stale_after_ms:
            status = pb.REPORT_STATUS_STALE
            error: Optional[str] = f"load snapshot stale by {oldest_age_ms} ms"
        else:
            status = pb.REPORT_STATUS_HEALTHY
            error = None
        return self._new_report(
            ranks=ranks,
            identity=identity,
            report_time_unix_ms=report_time_unix_ms,
            status=status,
            error=error,
        )

    def build_unreachable(
        self,
        identity: WorkerMetadata,
        *,
        report_time_unix_ms: int,
        error: BaseException | str,
    ) -> pb.LoadReport:
        """Build one empty report describing a failed pull attempt."""
        message = str(error).strip() or "load snapshot unavailable"
        return self._new_report(
            ranks=(),
            identity=identity,
            report_time_unix_ms=report_time_unix_ms,
            status=pb.REPORT_STATUS_UNREACHABLE,
            error=message,
        )

    def _new_report(
        self,
        *,
        ranks: Sequence[RankSnapshot],
        identity: WorkerMetadata,
        report_time_unix_ms: int,
        status: int,
        error: Optional[str],
    ) -> pb.LoadReport:
        """Allocate one protobuf report; success and failure share this path."""
        report = pb.LoadReport(
            source_instance_id=self._source_instance_id,
            sequence_id=self._sequence.next(),
            report_time_unix_ms=report_time_unix_ms,
            worker=pb.Worker(
                worker_addr=identity.worker_addr,
                worker_type=identity.worker_type,
            ),
            status=status,
            ranks=[pb.RankLoad(**msgspec.structs.asdict(rank)) for rank in ranks],
        )
        if identity.model is not None:
            report.worker.model = identity.model
        if error is not None:
            report.last_error = error
        return report
