# SPDX-License-Identifier: Apache-2.0

"""Public realtime Gateway for Coordinator-routed Denoiser/VAE sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    ProtocolViolation,
    decode_message,
    encode_message,
)
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    SessionAssignment,
    WorkerSlot,
)
from sglang.multimodal_gen.runtime.realtime.gateway import (
    AdmissionQueueFull,
    BoundedAdmissionWaiterGate,
    GatewayOutputRegistry,
    OutputBackpressureError,
    OutputProtocolError,
    OutputRouteClosed,
    build_denoiser_url,
    worker_message_allowed,
    worker_message_type,
)
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    compact_client_trace_event,
    emit_realtime_trace_payload,
    normalize_trace_id,
)


WEBUI_ROOT = Path(__file__).resolve().parents[2] / "apps" / "realtime_webui"
logger = logging.getLogger(__name__)
_IDEMPOTENT_COORDINATOR_RELEASE_REASONS = frozenset(
    {
        "LEASE_LOST",
        "WORKER_LOST",
    }
)


def _parse_ui_config(raw: str) -> dict[str, Any]:
    try:
        config = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("UI config must be valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("UI config must be a JSON object")
    return config


def _log_gateway_trace(trace_id: str, event: str, **fields: Any) -> None:
    now_ms = int(time.time() * 1000)
    payload = {
        "trace_id": trace_id,
        "event": event,
        "server_epoch_ms": now_ms,
        "trace_seq": now_ms * 1000 + time.perf_counter_ns() % 1000,
        **fields,
    }
    emit_realtime_trace_payload(logger, payload)


class CoordinatorClient(Protocol):
    async def health(self) -> dict[str, Any]: ...

    async def admit(self, **request: Any) -> SessionAssignment: ...

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment: ...

    async def release(self, assignment: SessionAssignment) -> None: ...


def _assignment(payload: dict[str, Any]) -> SessionAssignment:
    return SessionAssignment(
        user_id=payload["user_id"],
        session_id=payload["session_id"],
        generation_id=payload["generation_id"],
        token=payload["token"],
        expires_at=float(payload["expires_at"]),
        denoiser=WorkerSlot(**payload["denoiser"]),
        vae=WorkerSlot(**payload["vae"]),
    )


class HTTPCoordinatorClient:
    def __init__(self, base_url: str, *, timeout_s: float = 15.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_s
        )

    @staticmethod
    def _raise_rejection(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            detail = response.json().get("detail", {})
        except (ValueError, AttributeError):
            detail = {}
        reason = detail.get("reason") or f"COORDINATOR_HTTP_{response.status_code}"
        raise CoordinatorRejected(
            reason,
            retry_after_s=detail.get("retry_after_s"),
        )

    async def admit(self, **request: Any) -> SessionAssignment:
        response = await self._client.post("/v1/sessions/admit", json=request)
        self._raise_rejection(response)
        return _assignment(response.json())

    async def health(self) -> dict[str, Any]:
        response = await self._client.get("/healthz")
        response.raise_for_status()
        return response.json()

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        response = await self._client.post(
            "/v1/sessions/renew", json=asdict(assignment)
        )
        self._raise_rejection(response)
        return _assignment(response.json())

    async def release(self, assignment: SessionAssignment) -> None:
        response = await self._client.request(
            "DELETE", "/v1/sessions/release", json=asdict(assignment)
        )
        if response.status_code == 404:
            return
        if response.status_code == 409:
            try:
                detail = response.json().get("detail", {})
            except (ValueError, AttributeError):
                detail = {}
            if detail.get("reason") in _IDEMPOTENT_COORDINATOR_RELEASE_REASONS:
                return
        self._raise_rejection(response)

    async def close(self) -> None:
        await self._client.aclose()


class _BrowserSender:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._lock = asyncio.Lock()

    async def send(self, payload: bytes | str) -> None:
        async with self._lock:
            if isinstance(payload, bytes):
                await self.websocket.send_bytes(payload)
            else:
                await self.websocket.send_text(payload)

    async def error(self, content: str, **fields: Any) -> None:
        await self.send(encode_message("error", content=content, **fields))


def _user_id(websocket: WebSocket) -> str:
    query = websocket.query_params.get("user_id")
    if query:
        return f"query:{query[:240]}"
    header = websocket.headers.get("x-user-id")
    if header:
        return f"header:{header[:240]}"
    client = websocket.client.host if websocket.client else "unknown"
    return f"client:{client}"


async def _receive_browser(websocket: WebSocket) -> bytes | str:
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    if message.get("bytes") is not None:
        return message["bytes"]
    if message.get("text") is not None:
        return message["text"]
    raise WebSocketDisconnect(1002)


async def _cancel_tasks(tasks: set[asyncio.Task]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def create_app(
    coordinator: CoordinatorClient,
    *,
    model_revision: str,
    vae_fingerprint: str,
    internal_output_url: str,
    output_queue_depth: int = 2,
    output_enqueue_timeout_s: float = 1.0,
    output_drain_timeout_s: float = 5.0,
    lease_renew_interval_s: float = 10.0,
    release_grace_s: float = 0.5,
    max_admission_waiters: int = 64,
    connect_factory=connect,
    ui_config: dict[str, Any] | None = None,
    trace_query=None,
) -> FastAPI:
    if release_grace_s < 0:
        raise ValueError("release_grace_s must be non-negative")
    if output_drain_timeout_s <= 0:
        raise ValueError("output_drain_timeout_s must be positive")
    registry = GatewayOutputRegistry(
        queue_depth=output_queue_depth,
        enqueue_timeout_s=output_enqueue_timeout_s,
    )
    admission_gate = BoundedAdmissionWaiterGate(
        max_waiters=max_admission_waiters
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        closer = getattr(coordinator, "close", None)
        if closer is not None:
            await closer()

    app = FastAPI(title="SGLang Realtime Gateway", lifespan=lifespan)
    app.state.output_registry = registry
    app.state.admission_gate = admission_gate

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        try:
            await coordinator.health()
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="coordinator unavailable"
            ) from exc
        return {"status": "ready"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_revision}]}

    @app.get("/v1/realtime_video/traces/{trace_id}")
    async def get_trace(trace_id: str, after: int = 0, limit: int = 220):
        if trace_query is None:
            raise HTTPException(status_code=503, detail="Trace query is not configured")
        try:
            return await trace_query.query(trace_id, after=after, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Trace query failed for trace_id=%s", trace_id)
            raise HTTPException(
                status_code=503, detail="trace query unavailable"
            ) from exc

    @app.post("/v1/realtime_video/traces/{trace_id}/client-events")
    async def post_client_trace(trace_id: str, payload: dict):
        normalized = normalize_trace_id(trace_id, fallback="")
        if not normalized or normalized != trace_id:
            raise HTTPException(status_code=400, detail="invalid trace_id")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list) or len(raw_events) > 64:
            raise HTTPException(status_code=400, detail="events must contain at most 64 items")
        accepted = 0
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            event = compact_client_trace_event(raw_event)
            event["trace_id"] = trace_id
            event["event"] = str(event.pop("name", "client.metric"))[:128]
            event["server_epoch_ms"] = int(time.time() * 1000)
            event["trace_seq"] = (
                event["server_epoch_ms"] * 1000 + int(event.get("seq") or 0) % 1000
            )
            emit_realtime_trace_payload(logger, event)
            accepted += 1
        return {"accepted": accepted}

    @app.get("/runtime-config.js")
    async def runtime_config():
        config = ui_config or {}
        body = f"globalThis.SGLANG_REALTIME_UI_CONFIG = {json.dumps(config)};\n"
        return Response(
            body,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/v1/internal/realtime_output")
    async def realtime_output(websocket: WebSocket):
        await websocket.accept()
        route = None
        session_id = ""
        generation_id = ""
        output_token = ""
        try:
            opened = decode_message(await websocket.receive_bytes())
            if opened.get("type") != "session_output_open":
                raise OutputProtocolError("session_output_open is required")
            session_id = str(opened.get("session_id") or "")
            generation_id = str(opened.get("generation_id") or "")
            output_token = str(opened.get("token") or "")
            route = await registry.bind(
                session_id,
                generation_id,
                token=output_token,
            )
            await websocket.send_bytes(
                encode_message(
                    "session_output_accepted",
                    session_id=session_id,
                    generation_id=generation_id,
                )
            )
            while True:
                wire = await websocket.receive_bytes()
                message = decode_message(wire)
                await route.put(wire)
                if message.get("type") == "media_chunk_complete":
                    await websocket.send_bytes(
                        encode_message(
                            "media_chunk_complete_accepted",
                            session_id=session_id,
                            generation_id=generation_id,
                            request_id=message.get("request_id"),
                            chunk_index=message.get("chunk_index"),
                        )
                    )
        except (WebSocketDisconnect, OutputRouteClosed):
            pass
        except OutputBackpressureError as exc:
            await websocket.close(code=1013, reason=str(exc))
        except (OutputProtocolError, ProtocolViolation) as exc:
            await websocket.close(code=1008, reason=str(exc))
        finally:
            if route is not None:
                await registry.unbind(
                    session_id, generation_id, token=output_token
                )

    @app.websocket("/v1/realtime_video/generate")
    async def generate(websocket: WebSocket):
        await websocket.accept()
        sender = _BrowserSender(websocket)
        session_id = uuid4().hex
        generation_id = uuid4().hex
        trace_id = normalize_trace_id(
            websocket.query_params.get("trace_id"), fallback=session_id
        )
        output_token = secrets.token_urlsafe(32)
        assignment = None
        route = None
        upstream = None
        tasks: set[asyncio.Task] = set()
        expected_last_chunk: int | None = None
        try:
            admitted_at = time.perf_counter()
            _log_gateway_trace(trace_id, "gateway.ws_accepted", session_id=session_id)
            async with admission_gate.waiter():
                assignment = await coordinator.admit(
                    user_id=_user_id(websocket),
                    session_id=session_id,
                    generation_id=generation_id,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    wait_for_capacity=True,
                    trace_id=trace_id,
                )
            _log_gateway_trace(
                trace_id,
                "gateway.coordinator_admit_complete",
                session_id=session_id,
                coordinator_admit_ms=round(
                    (time.perf_counter() - admitted_at) * 1000, 3
                ),
                denoiser_worker_id=assignment.denoiser.worker_id,
                vae_worker_id=assignment.vae.worker_id,
            )
            route = await registry.register(
                session_id, generation_id, token=output_token
            )
            upstream_url = build_denoiser_url(
                assignment.denoiser.endpoint,
                session_id=session_id,
                generation_id=generation_id,
                coordinator_token=assignment.token,
                worker_epoch=assignment.denoiser.worker_epoch,
                vae_url=assignment.vae.endpoint,
                vae_worker_epoch=assignment.vae.worker_epoch,
                output_url=internal_output_url,
                output_token=output_token,
                trace_id=trace_id,
            )
            upstream = await connect_factory(
                upstream_url,
                max_size=None,
                compression=None,
                open_timeout=10,
                close_timeout=2,
                ping_interval=20,
                ping_timeout=20,
            )
            _log_gateway_trace(
                trace_id,
                "gateway.denoiser_connected",
                session_id=session_id,
            )

            async def browser_to_worker():
                nonlocal expected_last_chunk
                try:
                    while True:
                        payload = await _receive_browser(websocket)
                        if isinstance(payload, bytes) and expected_last_chunk is None:
                            try:
                                control = decode_message(payload)
                            except ProtocolViolation:
                                control = None
                            if isinstance(control, dict) and control.get("type") == "init":
                                max_chunks = int(control.get("max_chunks") or 0)
                                if max_chunks > 0:
                                    expected_last_chunk = max_chunks - 1
                        await upstream.send(payload)
                except ConnectionClosedOK:
                    return

            async def worker_to_browser():
                try:
                    while True:
                        wire = await upstream.recv()
                        if isinstance(wire, str):
                            raise ProtocolViolation(
                                "Denoiser control messages must be binary"
                            )
                        if not worker_message_allowed(wire):
                            message_type = worker_message_type(wire)
                            raise ProtocolViolation(
                                f"Denoiser emitted forbidden message: {message_type}"
                            )
                        await sender.send(wire)
                except ConnectionClosedOK:
                    return

            async def output_to_browser():
                while True:
                    wire = await route.get()
                    try:
                        await sender.send(wire)
                    finally:
                        route.task_done()

            async def renew_lease():
                nonlocal assignment
                while True:
                    await asyncio.sleep(lease_renew_interval_s)
                    assignment = await coordinator.renew(assignment)

            browser_input_task = asyncio.create_task(
                browser_to_worker(), name="gateway-browser-input"
            )
            worker_control_task = asyncio.create_task(
                worker_to_browser(), name="gateway-worker-control"
            )
            output_task = asyncio.create_task(
                output_to_browser(), name="gateway-vae-output"
            )
            lease_task = asyncio.create_task(
                renew_lease(), name="gateway-lease-renew"
            )
            tasks = {
                browser_input_task,
                worker_control_task,
                output_task,
                lease_task,
            }
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception
            if route is not None and worker_control_task in done:
                try:
                    if expected_last_chunk is not None:
                        await asyncio.wait_for(
                            route.wait_until_chunk_completed(expected_last_chunk),
                            timeout=output_drain_timeout_s,
                        )
                    else:
                        await asyncio.wait_for(
                            route.wait_until_output_closed(),
                            timeout=output_drain_timeout_s,
                        )
                    await asyncio.wait_for(
                        route.join(), timeout=output_drain_timeout_s
                    )
                except TimeoutError:
                    logger.warning(
                        "Gateway media drain timed out for session_id=%s",
                        session_id,
                    )
        except CoordinatorRejected as exc:
            await sender.error(
                f"realtime admission rejected: {exc.reason}",
                reason=exc.reason,
                retry_after_s=exc.retry_after_s,
            )
            close_code = 1013 if exc.reason == "CAPACITY_EXHAUSTED" else 1008
            await websocket.close(code=close_code, reason=exc.reason)
        except AdmissionQueueFull as exc:
            await sender.error(f"realtime admission rejected: {exc.reason}")
            await websocket.close(code=1013, reason=exc.reason)
        except (WebSocketDisconnect, OutputRouteClosed):
            pass
        except Exception as exc:
            try:
                await sender.error(f"realtime gateway error: {str(exc).splitlines()[0]}")
                await websocket.close(code=1011, reason="gateway session failed")
            except Exception:
                pass
        finally:
            await _cancel_tasks(tasks)
            if upstream is not None:
                await upstream.close()
                if release_grace_s:
                    await asyncio.sleep(release_grace_s)
            if route is not None:
                await registry.unregister(
                    session_id, generation_id, token=output_token
                )
            if assignment is not None:
                try:
                    await coordinator.release(assignment)
                except Exception:
                    logger.exception(
                        "Coordinator release failed for session_id=%s",
                        assignment.session_id,
                    )
            _log_gateway_trace(trace_id, "gateway.session_closed", session_id=session_id)
            try:
                await websocket.close(code=1000)
            except Exception:
                pass

    @app.get("/")
    async def index():
        return FileResponse(WEBUI_ROOT / "index.html")

    app.mount("/", StaticFiles(directory=WEBUI_ROOT), name="realtime-webui")
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vae-fingerprint", default="taew2_2")
    parser.add_argument(
        "--internal-output-url",
        default=os.environ.get("REALTIME_GATEWAY_OUTPUT_URL"),
    )
    parser.add_argument("--output-queue-depth", type=int, default=2)
    parser.add_argument("--output-enqueue-timeout-s", type=float, default=1.0)
    parser.add_argument("--output-drain-timeout-s", type=float, default=5.0)
    parser.add_argument("--lease-renew-interval-s", type=float, default=10.0)
    parser.add_argument("--release-grace-s", type=float, default=0.5)
    parser.add_argument("--max-admission-waiters", type=int, default=64)
    parser.add_argument("--trace-log-group")
    parser.add_argument(
        "--ui-config-json",
        default=os.environ.get("REALTIME_UI_CONFIG_JSON", "{}"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.internal_output_url:
        raise SystemExit("--internal-output-url is required")
    try:
        ui_config = _parse_ui_config(args.ui_config_json)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    coordinator = HTTPCoordinatorClient(args.coordinator_url)
    trace_query = None
    if args.trace_log_group:
        try:
            import boto3
        except ImportError as exc:
            raise SystemExit("boto3 is required for CloudWatch Trace query") from exc
        from sglang.multimodal_gen.runtime.realtime.trace_query import (
            CloudWatchTraceQuery,
        )

        trace_query = CloudWatchTraceQuery(
            boto3.client("logs"), log_group=args.trace_log_group
        )
    app = create_app(
        coordinator,
        model_revision=args.model_revision,
        vae_fingerprint=args.vae_fingerprint,
        internal_output_url=args.internal_output_url,
        output_queue_depth=args.output_queue_depth,
        output_enqueue_timeout_s=args.output_enqueue_timeout_s,
        output_drain_timeout_s=args.output_drain_timeout_s,
        lease_renew_interval_s=args.lease_renew_interval_s,
        release_grace_s=args.release_grace_s,
        max_admission_waiters=args.max_admission_waiters,
        ui_config=ui_config,
        trace_query=trace_query,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
