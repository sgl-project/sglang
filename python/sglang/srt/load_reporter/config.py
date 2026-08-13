"""Load reporter configuration and worker metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

SNAPSHOT_PULL_TIMEOUT_SECONDS = 1.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0
SNAPSHOT_STALE_AFTER_MS = 3000


class LoadReporterConfig(msgspec.Struct, frozen=True):
    """Timing configuration for the load reporter."""

    snapshot_stale_after_ms: int = SNAPSHOT_STALE_AFTER_MS

    @classmethod
    def from_server_args(cls, args: ServerArgs) -> LoadReporterConfig:
        """Build reporter timing configuration from ServerArgs."""
        return cls()


class WorkerMetadata(msgspec.Struct, frozen=True):
    """Stable identity fields reported with every load snapshot."""

    worker_addr: str
    worker_type: int
    model: Optional[str] = None

    @classmethod
    def from_server_args(cls, args: ServerArgs) -> WorkerMetadata:
        """Build worker metadata from server arguments."""
        worker_type = {
            "prefill": pb.WORKER_TYPE_PREFILL,
            "decode": pb.WORKER_TYPE_DECODE,
        }.get(args.disaggregation_mode, pb.WORKER_TYPE_REGULAR)
        worker_addr = f"{args.host}:{args.load_reporter_port}"
        return cls(
            worker_addr=worker_addr,
            worker_type=worker_type,
            model=args.served_model_name,
        )
