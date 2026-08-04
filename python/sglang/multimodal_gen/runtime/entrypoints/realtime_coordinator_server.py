# SPDX-License-Identifier: Apache-2.0

"""HTTP entrypoint for the stateless realtime Session Coordinator."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI, HTTPException, Response, status

from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    DynamoDBCoordinatorStore,
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
    SessionAssignment,
    WorkerHeartbeat,
    WorkerSlot,
)
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    log_realtime_trace,
    normalize_trace_id,
)


logger = logging.getLogger(__name__)


def _assignment(payload: dict) -> SessionAssignment:
    return SessionAssignment(
        user_id=payload["user_id"],
        session_id=payload["session_id"],
        generation_id=payload["generation_id"],
        token=payload["token"],
        expires_at=float(payload["expires_at"]),
        denoiser=WorkerSlot(**payload["denoiser"]),
        vae=WorkerSlot(**payload["vae"]),
    )


def _raise_http(exc: CoordinatorRejected) -> None:
    status_code = {
        "USER_SESSION_LIMIT": status.HTTP_409_CONFLICT,
        "CAPACITY_EXHAUSTED": status.HTTP_429_TOO_MANY_REQUESTS,
        "LEASE_LOST": status.HTTP_409_CONFLICT,
    }.get(exc.reason, status.HTTP_400_BAD_REQUEST)
    headers = None
    if exc.retry_after_s is not None:
        headers = {"Retry-After": str(max(1, int(exc.retry_after_s)))}
    raise HTTPException(
        status_code=status_code,
        detail={"reason": exc.reason, "retry_after_s": exc.retry_after_s},
        headers=headers,
    ) from exc


def create_app(coordinator: RealtimeCoordinator) -> FastAPI:
    app = FastAPI(title="SGLang Realtime Coordinator")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/workers/heartbeat", status_code=status.HTTP_204_NO_CONTENT)
    async def heartbeat(payload: dict):
        try:
            await coordinator.heartbeat(WorkerHeartbeat(**payload))
        except (CoordinatorRejected, TypeError, KeyError) as exc:
            if isinstance(exc, CoordinatorRejected):
                _raise_http(exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"reason": "INVALID_WORKER_HEARTBEAT"},
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/v1/sessions/admit")
    async def admit(payload: dict):
        request = dict(payload)
        trace_id = normalize_trace_id(
            request.pop("trace_id", None), fallback=str(request.get("session_id") or "")
        )
        trace_session = SimpleNamespace(
            id=request.get("session_id"),
            generation_id=request.get("generation_id"),
            trace_id=trace_id,
            trace_started_at=time.perf_counter(),
        )
        started_at = time.perf_counter()
        try:
            assignment = await coordinator.admit(**request)
        except CoordinatorRejected as exc:
            log_realtime_trace(
                logger,
                trace_session,
                "coordinator.admit_rejected",
                reason=exc.reason,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            )
            _raise_http(exc)
        log_realtime_trace(
            logger,
            trace_session,
            "coordinator.admit_complete",
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            denoiser_worker_id=assignment.denoiser.worker_id,
            vae_worker_id=assignment.vae.worker_id,
        )
        return asdict(assignment)

    @app.post("/v1/sessions/renew")
    async def renew(payload: dict):
        try:
            assignment = await coordinator.renew(_assignment(payload))
        except (CoordinatorRejected, TypeError, KeyError, ValueError) as exc:
            if isinstance(exc, CoordinatorRejected):
                _raise_http(exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"reason": "INVALID_ASSIGNMENT"},
            ) from exc
        return asdict(assignment)

    @app.delete("/v1/sessions/release", status_code=status.HTTP_204_NO_CONTENT)
    async def release(payload: dict):
        try:
            await coordinator.release(_assignment(payload))
        except (TypeError, KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"reason": "INVALID_ASSIGNMENT"},
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--backend", choices=("memory", "dynamodb"), default="dynamodb")
    parser.add_argument("--table-name")
    parser.add_argument("--region")
    parser.add_argument("--ttl-s", type=float, default=30.0)
    parser.add_argument("--worker-ttl-s", type=float, default=15.0)
    parser.add_argument("--wait-timeout-s", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.backend == "dynamodb":
        if not args.table_name:
            raise SystemExit("--table-name is required for DynamoDB")
        store = DynamoDBCoordinatorStore(
            args.table_name,
            ttl_s=args.ttl_s,
            worker_ttl_s=args.worker_ttl_s,
            region_name=args.region,
        )
    else:
        store = InMemoryCoordinatorStore(
            ttl_s=args.ttl_s,
            worker_ttl_s=args.worker_ttl_s,
        )
    coordinator = RealtimeCoordinator(store, wait_timeout_s=args.wait_timeout_s)
    uvicorn.run(create_app(coordinator), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
