"""Frozen configuration structs for the embedded SGLang load reporter.

``LoadReporterConfig`` carries the only timing knob exposed on ``ServerArgs``
(the snapshot stale threshold).  ``WorkerMetadata`` carries the identity fields
that are stable for the lifetime of the worker process.

gRPC transport/lifecycle knobs (connect/close timeout, reconnect backoff,
shutdown timeout) are reporter-internal implementation constants defined in
this module in seconds; they are intentionally not surfaced as CLI arguments.

Both classes are constructed via ``from_server_args`` factory methods so that
callers never reach into ``ServerArgs`` directly after the reporter starts.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

# Reporter-internal implementation constants (seconds). Not CLI arguments.
GRPC_CONNECT_TIMEOUT_SECONDS = 3.0
GRPC_CLOSE_TIMEOUT_SECONDS = 0.5
RECONNECT_INITIAL_SECONDS = 0.25
RECONNECT_MAX_SECONDS = 5.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0


@dataclasses.dataclass(frozen=True, slots=True)
class LoadReporterConfig:
    """Timing configuration for the load reporter derived from ServerArgs."""

    snapshot_stale_after_ms: int

    @classmethod
    def from_server_args(cls, args: ServerArgs) -> LoadReporterConfig:
        """Build reporter timing configuration from server arguments.

        Args:
            args: Resolved SGLang server configuration.

        Returns:
            Frozen load-reporter timing configuration.
        """
        return cls(
            snapshot_stale_after_ms=args.load_reporter_snapshot_stale_after_ms,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerMetadata:
    """Stable identity fields reported with every load snapshot."""

    worker_type: int
    model: Optional[str]
    zone: Optional[str]

    @classmethod
    def from_server_args(cls, args: ServerArgs) -> WorkerMetadata:
        """Build stable worker metadata from server arguments.

        Args:
            args: Resolved SGLang server configuration.

        Returns:
            Frozen worker type, model, and zone metadata.
        """
        worker_type = {
            "prefill": pb.WORKER_TYPE_PREFILL,
            "decode": pb.WORKER_TYPE_DECODE,
        }.get(args.disaggregation_mode, pb.WORKER_TYPE_REGULAR)
        return cls(
            worker_type=worker_type,
            model=args.served_model_name,
            zone=args.load_reporter_zone,
        )
