"""Load reporter lifecycle management."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


class LoadReporterHandle:
    """Own reporter resources for one process."""

    def __init__(self, runtime: Any, server: Any) -> None:
        """Hold the started runtime and gRPC server until close()."""
        self._runtime = runtime
        self._server = server
        self._close_task: Optional[asyncio.Task[None]] = None

    def update_expected_dp_ranks(self, ranks: Iterable[int]) -> bool:
        """Update the rank-aware source after elastic scaling."""
        return self._runtime.update_expected_dp_ranks(ranks)

    async def close(self) -> None:
        """Tear down reporter resources once."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_impl(), name="load-reporter-handle-close"
            )
        await asyncio.shield(self._close_task)

    async def _close_impl(self) -> None:
        """Run one shared teardown attempt to completion for every caller."""
        try:
            await self._server.stop(grace=None)
        except Exception:
            logger.exception("Load reporter gRPC server stop failed")
        try:
            await self._runtime.close()
        except Exception:
            logger.exception("Load reporter runtime shutdown failed")


async def start_load_reporter(
    server_args: Any,
    snapshot_source: Any,
) -> Optional[LoadReporterHandle]:
    """Start the reporter and return its handle when enabled."""
    if getattr(server_args, "load_reporter_port", None) is None:
        return None

    if snapshot_source is None:
        raise ValueError("snapshot_source is required when load reporter is enabled")

    return await _start_owner(server_args, snapshot_source)


async def _start_owner(
    server_args: Any,
    snapshot_source: Any,
) -> LoadReporterHandle:
    """Start the reporter runtime and gRPC listener on the reporter port."""
    import grpc.aio

    from sglang.srt.load_reporter.runtime import LoadReporterRuntime
    from sglang.srt.load_reporter.service import add_service_to_server

    runtime = LoadReporterRuntime(snapshot_source, server_args)
    try:
        server = grpc.aio.server()
        add_service_to_server(runtime, server)
        server.add_insecure_port(f"{server_args.host}:{server_args.load_reporter_port}")
        await server.start()
    except BaseException:
        # No handle yet; a server that never started needs no stop. Log and
        # re-raise the original failure even if runtime teardown itself fails.
        try:
            await runtime.close()
        except Exception:
            logger.exception("Load reporter runtime shutdown failed")
        raise
    return LoadReporterHandle(runtime=runtime, server=server)
