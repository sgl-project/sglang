"""gRPC service for inbound Router load-monitor streams."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import grpc
import grpc.aio

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)


def _invalid_argument(message: str) -> pb.StreamError:
    """Build the stable terminal error used for invalid Router input."""
    return pb.StreamError(code="INVALID_ARGUMENT", message=message)


def _validate_timing(
    report_interval_ms: Optional[int] = None,
    lease_ttl_ms: Optional[int] = None,
) -> Optional[pb.StreamError]:
    """Return an error when a present timing field is not positive."""
    if report_interval_ms is not None and report_interval_ms <= 0:
        return _invalid_argument("report_interval_ms must be greater than zero")
    if lease_ttl_ms is not None and lease_ttl_ms <= 0:
        return _invalid_argument("lease_ttl_ms must be greater than zero")
    return None


def _validate_register(request: pb.RegisterRequest) -> Optional[pb.StreamError]:
    """Validate a complete registration before creating a runtime session."""
    if not request.router_id or not request.router_id.strip():
        return _invalid_argument("router_id must be non-empty")
    return _validate_timing(request.report_interval_ms, request.lease_ttl_ms)


def add_service_to_server(runtime: Any, server: grpc.aio.Server) -> None:
    """Register a LoadMonitorService servicer onto server."""
    pb_grpc.add_LoadMonitorServiceServicer_to_server(
        LoadMonitorService(runtime), server
    )


class LoadMonitorService(pb_grpc.LoadMonitorServiceServicer):
    """Serve Router-initiated load-monitor streams."""

    def __init__(self, runtime: Any) -> None:
        """Bind to the given runtime."""
        self._runtime = runtime

    async def Monitor(self, request_iterator: Any, context: Any) -> Any:
        """Handle a bidirectional stream after its register frame."""
        session = None
        read_task = None
        write_task = None
        try:
            first_frame = await self._read_frame(request_iterator, context)
            if first_frame is None:
                return  # stream ended before any frame

            if first_frame.WhichOneof("payload") != "register":
                await self._send(
                    context,
                    pb.WorkerFrame(
                        error=pb.StreamError(
                            code="INVALID_FIRST_FRAME",
                            message="first RouterFrame must be a RegisterRequest",
                        )
                    ),
                )
                return

            reg = first_frame.register
            validation_error = _validate_register(reg)
            if validation_error is not None:
                await self._send(context, pb.WorkerFrame(error=validation_error))
                return

            ack, session = self._runtime.register_session(
                router_id=reg.router_id,
                report_interval_ms=reg.report_interval_ms,
                lease_ttl_ms=reg.lease_ttl_ms,
            )
            await self._send(context, pb.WorkerFrame(registered=ack))

            read_task = asyncio.create_task(
                self._read_loop(request_iterator, session),
                name=f"lr-svc-read-{reg.router_id}",
            )
            write_task = asyncio.create_task(
                self._write_loop(context, session),
                name=f"lr-svc-write-{reg.router_id}",
            )
            done, pending = await asyncio.wait(
                {read_task, write_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            stream_error = None
            if read_task in done and not read_task.cancelled():
                stream_error = read_task.result()
            if stream_error is not None:
                session.stop()
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            if stream_error is not None:
                await self._send(context, pb.WorkerFrame(error=stream_error))
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Unexpected error in Monitor handler")
        finally:
            if session is not None:
                session.stop()
            for t in (read_task, write_task):
                if t is not None and not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

    @staticmethod
    async def _read_frame(request_iterator: Any, context: Any) -> pb.RouterFrame | None:
        """Read one RouterFrame; return None on EOF or context cancel."""
        try:
            return await request_iterator.__anext__()
        except StopAsyncIteration:
            return None
        except Exception:
            return None

    @staticmethod
    async def _send(context: Any, frame: pb.WorkerFrame) -> None:
        """Write one WorkerFrame to the context; swallow any send error."""
        try:
            await context.write(frame)
        except (asyncio.CancelledError, Exception):
            pass

    async def _read_loop(
        self, request_iterator: Any, session: Any
    ) -> Optional[pb.StreamError]:
        """Consume RouterFrames; return a terminal validation error if needed."""
        async for frame in request_iterator:
            which = frame.WhichOneof("payload")
            if which == "keep_alive":
                session.refresh_lease()
            elif which == "register":
                # Re-register on the same stream is a full config update.
                reg = frame.register
                validation_error = _validate_register(reg)
                if validation_error is not None:
                    return validation_error
                session.update_config(
                    report_interval_ms=reg.report_interval_ms,
                    lease_ttl_ms=reg.lease_ttl_ms,
                )
            elif which == "update_config":
                uc = frame.update_config
                interval = (
                    uc.report_interval_ms if uc.HasField("report_interval_ms") else None
                )
                lease = uc.lease_ttl_ms if uc.HasField("lease_ttl_ms") else None
                validation_error = _validate_timing(interval, lease)
                if validation_error is not None:
                    return validation_error
                session.update_config(report_interval_ms=interval, lease_ttl_ms=lease)
            elif which == "stop":
                session.stop()
                return None
        return None

    async def _write_loop(self, context: Any, session: Any) -> None:
        """Drain session queue and write WorkerFrames to the Router."""
        while True:
            item = await session.queue.get()
            if item is None:
                return  # session ended
            await self._send(context, pb.WorkerFrame(report=item))
