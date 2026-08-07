"""Load reporter lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


class LoadReporterHandle:
    """Own reporter resources for one worker process."""

    def __init__(self) -> None:
        self._runtime: Optional[Any] = None
        self._server: Optional[Any] = None
        self._notifier: Optional[Any] = None
        self._unbind: Optional[Callable[[], None]] = None
        self._restore: Optional[Callable[[], None]] = None
        self._close_task: Optional[asyncio.Task[None]] = None

    def notify_refresh(self) -> None:
        """Wake the sampler once (router IPC refresh).  No-op without runtime."""
        if self._runtime is not None:
            self._runtime.notify_refresh()

    def update_expected_dp_ranks(self, ranks: Iterable[int]) -> bool:
        """Update the rank-aware source after elastic scaling."""
        if self._runtime is None:
            return False
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
        if self._server is not None:
            try:
                await self._server.stop(grace=None)
            except Exception:
                logger.exception("Load reporter gRPC server stop failed")
        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception:
                logger.exception("Load reporter runtime shutdown failed")
        if self._notifier is not None:
            try:
                await self._notifier.close()
            except Exception:
                logger.exception("Load reporter notifier shutdown failed")
        if self._unbind is not None:
            try:
                self._unbind()
            except Exception:
                logger.exception("Load reporter unbind failed")
        if self._restore is not None:
            try:
                self._restore()
            except Exception:
                logger.exception("Load reporter method restore failed")


async def start_load_reporter(
    server_args: Any,
    snapshot_source: Optional[Any],
    *,
    event_owner: Optional[Any] = None,
    request_lifecycle_method: Optional[str] = None,
) -> Optional[LoadReporterHandle]:
    """Start the reporter and return its handle when enabled."""
    if getattr(server_args, "load_reporter_port", None) is None:
        return None

    if snapshot_source is None:
        return await _start_ipc_worker(server_args, event_owner)
    return await _start_owner(
        server_args, snapshot_source, event_owner, request_lifecycle_method
    )


async def _start_ipc_worker(
    server_args: Any, event_owner: Optional[Any]
) -> Optional[LoadReporterHandle]:
    """Multi-tokenizer HTTP worker: coalesce and forward refresh hints to router."""
    if event_owner is None:
        return None

    from sglang.srt.load_reporter.event_hooks import bind_load_monitor
    from sglang.srt.load_reporter.worker_notifier import LoadReporterRefreshNotifier

    handle = LoadReporterHandle()
    notifier = LoadReporterRefreshNotifier(
        worker_id=f"http-worker-{os.getpid()}",
        send=event_owner._dispatch_to_scheduler,
    )
    handle._notifier = notifier
    await notifier.start()
    handle._unbind = bind_load_monitor(event_owner, notifier.notify)
    return handle


async def _start_owner(
    server_args: Any,
    snapshot_source: Any,
    event_owner: Optional[Any],
    request_lifecycle_method: Optional[str],
) -> LoadReporterHandle:
    """Single-owner path: own a runtime + gRPC listener on the reporter port."""
    import grpc.aio

    from sglang.srt.load_reporter.event_hooks import bind_load_monitor
    from sglang.srt.load_reporter.runtime import LoadReporterRuntime
    from sglang.srt.load_reporter.service import add_service_to_server

    handle = LoadReporterHandle()
    try:
        runtime = LoadReporterRuntime(snapshot_source, server_args)
        handle._runtime = runtime

        server = grpc.aio.server()
        add_service_to_server(runtime, server)
        server.add_insecure_port(f"{server_args.host}:{server_args.load_reporter_port}")
        await server.start()
        handle._server = server

        if event_owner is not None:
            handle._unbind = bind_load_monitor(
                event_owner, lambda reason, count: runtime.notify_refresh()
            )
        if request_lifecycle_method is not None:
            _install_lifecycle_shadow(handle, event_owner, request_lifecycle_method)
    except BaseException:
        await handle.close()
        raise
    return handle


def _install_lifecycle_shadow(
    handle: LoadReporterHandle, owner: Any, method_name: str
) -> None:
    """Wrap one owner method with the request-lifecycle decorator."""
    from sglang.srt.load_reporter.event_hooks import enable_load_monitor

    had_instance_override = method_name in owner.__dict__
    instance_override = owner.__dict__.get(method_name)
    original = getattr(owner, method_name)
    decorated = enable_load_monitor("request_lifecycle")(original)
    setattr(owner, method_name, decorated)

    def _restore() -> None:
        # Only undo our own shadow; never clobber a later replacement.
        if owner.__dict__.get(method_name, None) is decorated:
            if had_instance_override:
                owner.__dict__[method_name] = instance_override
            else:
                del owner.__dict__[method_name]

    handle._restore = _restore
