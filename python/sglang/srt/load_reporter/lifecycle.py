"""Single composition root for the embedded load reporter.

``start_load_reporter`` is the only public bootstrap symbol.  Every serving
mode (HTTP, native gRPC, embedded Engine, multi-tokenizer, standalone SMG RPC)
calls it and only sees ``start`` and ``close``; no reporter-internal type
(runtime, sampler, IPC notifier, gRPC/protobuf) leaks into serving entrypoints.

Path selection
--------------
* ``load_reporter_port is None`` → return ``None`` *before* importing the
  optional gRPC/protobuf stack.  Zero socket, task, or dependency overhead.
* ``snapshot_source is None`` (multi-tokenizer HTTP worker) → install a
  coalescing refresh notifier bound to ``event_owner`` that forwards refresh
  hints to the sole router over IPC.  No gRPC server, no port is bound.
* otherwise (single-owner) → own a ``LoadReporterRuntime`` + a ``grpc.aio``
  server listening on ``host:load_reporter_port``, and optionally bind
  ``event_owner`` so decorator events wake the sampler.

The returned :class:`LoadReporterHandle` owns every resource it created and
tears them down in reverse order on an idempotent ``close()``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


class LoadReporterHandle:
    """Owns the reporter resources for one Worker process.

    Serving entrypoints keep the handle opaque: they call ``close()`` on
    shutdown and, in the multi-tokenizer router only, ``notify_refresh`` /
    ``update_expected_dp_ranks``.  All methods are idempotent and safe to call
    after ``close()``.
    """

    def __init__(self) -> None:
        self._runtime: Optional[Any] = None
        self._server: Optional[Any] = None
        self._notifier: Optional[Any] = None
        self._unbind: Optional[Callable[[], None]] = None
        self._restore: Optional[Callable[[], None]] = None
        self._close_task: Optional[asyncio.Task[None]] = None

    # -- delegation surface (multi-tokenizer router) -------------------------

    def notify_refresh(self) -> None:
        """Wake the sampler once (router IPC refresh).  No-op without runtime."""
        if self._runtime is not None:
            self._runtime.notify_refresh()

    def update_expected_dp_ranks(self, ranks: Iterable[int]) -> bool:
        """Update the rank-aware source after elastic scaling.

        Returns ``False`` when there is no owning runtime (IPC-worker handle)
        or the source did not accept a changed rank set.
        """
        if self._runtime is None:
            return False
        return self._runtime.update_expected_dp_ranks(ranks)

    # -- shutdown ------------------------------------------------------------

    async def close(self) -> None:
        """Idempotent, cancellation-safe teardown.

        The teardown runs once on a shared task shielded from the caller's
        cancellation, so a caller cancelled mid-``close()`` never abandons the
        remaining steps and every subsequent caller awaits the same completion.

        Order: stop accepting Router sessions/reports, stop sampling, close the
        IPC notifier, then unbind the decorator registry callback and restore
        any shadowed bound method.  Each step is guarded so a partially started
        handle (e.g. failed port bind) closes cleanly.
        """
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
    """Start the embedded load reporter for one serving entrypoint.

    Args:
        server_args: Resolved SGLang server configuration.  Only
            ``load_reporter_port`` gates activation.
        snapshot_source: A ``LoadSnapshotSource`` for the owner path, or
            ``None`` for a multi-tokenizer HTTP worker (IPC-forwarding path).
        event_owner: The instance whose decorated ``generate_request`` /
            ``_dispatch_to_scheduler`` should wake the sampler.  ``None`` means
            interval + register-time sampling only.
        request_lifecycle_method: When set (standalone SMG RPC), the named bound
            async-generator method on ``event_owner`` is wrapped at runtime with
            the same ``enable_load_monitor("request_lifecycle")`` decorator and
            installed on that single instance; restored on ``close()``.

    Returns:
        A :class:`LoadReporterHandle` when reporting is enabled, else ``None``.
    """
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
    """Multi-tokenizer HTTP worker: coalesce refresh hints to the sole owner.

    Binds a refresh notifier to ``event_owner`` and forwards its coalesced
    events to the router over the existing scheduler IPC channel.  No gRPC
    server is started and no reporter port is bound.
    """
    if event_owner is None:
        return None

    from sglang.srt.load_reporter.decorator import bind_load_monitor
    from sglang.srt.load_reporter.ipc import LoadReporterRefreshNotifier

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

    from sglang.srt.load_reporter.decorator import bind_load_monitor
    from sglang.srt.load_reporter.runtime import LoadReporterRuntime
    from sglang.srt.load_reporter.service import add_service_to_server

    handle = LoadReporterHandle()
    try:
        runtime = LoadReporterRuntime(snapshot_source, server_args)
        handle._runtime = runtime

        server = grpc.aio.server()
        add_service_to_server(runtime, server)
        # Explicit bind: grpc.aio raises RuntimeError on failure (never a
        # silent random-port fallback), which we surface after cleanup.
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
    """Wrap ``owner.<method_name>`` with the request-lifecycle decorator.

    Installs the decorated callable as an instance attribute on this single
    ``owner`` only — the class method and every other instance are untouched.
    Registers an identity-safe restore that removes the instance shadow only
    while it still resolves to this wrapper.
    """
    from sglang.srt.load_reporter.decorator import enable_load_monitor

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
