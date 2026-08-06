"""Frozen configuration structs for the embedded SGLang load reporter.

``LoadReporterConfig`` carries the snapshot stale threshold exposed on
``ServerArgs``. ``WorkerMetadata`` carries identity fields that are stable for
the lifetime of the worker process. Internal lifecycle timeouts are expressed
in seconds and are intentionally not surfaced as CLI arguments.

Both classes are constructed via ``from_server_args`` factory methods so that
callers never reach into ``ServerArgs`` directly after the reporter starts.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

# Reporter-internal implementation constants. Not CLI arguments.
INITIAL_SAMPLE_TIMEOUT_SECONDS = 1.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0
# Maximum accepted age of the oldest rank snapshot before a report is marked
# STALE. Reporter-internal; intentionally not surfaced as a CLI argument.
SNAPSHOT_STALE_AFTER_MS = 3000


@dataclasses.dataclass(frozen=True, slots=True)
class LoadReporterConfig:
    """Timing configuration for the load reporter."""

    snapshot_stale_after_ms: int

    @classmethod
    def from_server_args(cls, args: ServerArgs) -> LoadReporterConfig:
        """Build reporter timing configuration.

        Args:
            args: Resolved SGLang server configuration (unused; the stale
                threshold is a reporter-internal constant).

        Returns:
            Frozen load-reporter timing configuration.
        """
        return cls(
            snapshot_stale_after_ms=SNAPSHOT_STALE_AFTER_MS,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerMetadata:
    """Stable identity fields reported with every load snapshot."""

    worker_addr: str
    worker_type: int
    model: Optional[str]

    @classmethod
    def from_server_args(cls, args: ServerArgs) -> WorkerMetadata:
        """Build stable worker metadata from server arguments.

        Args:
            args: Resolved SGLang server configuration.

        Returns:
            Frozen worker address, type, and model metadata.
        """
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
