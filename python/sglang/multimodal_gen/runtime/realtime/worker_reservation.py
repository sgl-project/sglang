# SPDX-License-Identifier: Apache-2.0

"""Bounded, epoch-fenced reservations for realtime GPU workers."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal, Mapping
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response, status


WorkerLifecycle = Literal["ready", "draining", "failed"]
_PROCESS_EPOCH = os.environ.get("WORKER_EPOCH") or uuid4().hex
_PROCESS_EPOCH_PATHS: set[str] = set()


def resolve_worker_epoch(value: str | None = None) -> str:
    resolved = value or os.environ.get("WORKER_EPOCH") or _PROCESS_EPOCH
    epoch_file = os.environ.get("WORKER_EPOCH_FILE")
    if epoch_file and not value and not os.environ.get("WORKER_EPOCH"):
        path = Path(epoch_file)
        normalized = str(path.resolve())
        if normalized not in _PROCESS_EPOCH_PATHS:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(f"{resolved}\n")
            os.replace(temporary, path)
            _PROCESS_EPOCH_PATHS.add(normalized)
    return resolved


class WorkerReservationRejected(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class WorkerReservation:
    token: str
    session_id: str
    generation_id: str
    worker_epoch: str
    expires_at: float
    consumed: bool = False
    owner_id: str | None = None
    consumed_at: float | None = None
    runnable: bool = False


class WorkerReservationRegistry:
    """Own worker admission from coordinator reserve through Session close."""

    def __init__(
        self,
        *,
        worker_epoch: str,
        capacity: int,
        clock: Callable[[], float] = time.time,
        load_provider: Callable[[], Mapping[str, int | float]] | None = None,
    ) -> None:
        if not worker_epoch:
            raise ValueError("worker_epoch is required")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.worker_epoch = worker_epoch
        self.capacity = capacity
        self._clock = clock
        self._load_provider = load_provider
        self._lifecycle: WorkerLifecycle = "ready"
        self._drain_deadline: float | None = None
        self._reservations: dict[str, WorkerReservation] = {}
        self._service_time_ms = 0.0
        self._completed_sessions = 0
        self._lock = asyncio.Lock()

    def set_load_provider(
        self,
        load_provider: Callable[[], Mapping[str, int | float]],
    ) -> None:
        self._load_provider = load_provider

    def _expire_locked(self, now: float) -> None:
        for token, reservation in list(self._reservations.items()):
            if not reservation.consumed and reservation.expires_at <= now:
                self._reservations.pop(token, None)
        if (
            self._lifecycle == "draining"
            and self._drain_deadline is not None
            and self._drain_deadline <= now
        ):
            self._lifecycle = "failed"

    def _validate_epoch(self, worker_epoch: str) -> None:
        if not worker_epoch or worker_epoch != self.worker_epoch:
            raise WorkerReservationRejected("WORKER_EPOCH_MISMATCH")

    @staticmethod
    def _validate_identity(
        reservation: WorkerReservation,
        *,
        session_id: str,
        generation_id: str,
    ) -> None:
        if (
            reservation.session_id != session_id
            or reservation.generation_id != generation_id
        ):
            raise WorkerReservationRejected("RESERVATION_IDENTITY_MISMATCH")

    async def reserve(
        self,
        token: str,
        *,
        session_id: str,
        generation_id: str,
        worker_epoch: str,
        ttl_s: float,
    ) -> WorkerReservation:
        if not token or not session_id or not generation_id:
            raise WorkerReservationRejected("INVALID_RESERVATION_IDENTITY")
        if ttl_s <= 0:
            raise WorkerReservationRejected("INVALID_RESERVATION_TTL")
        self._validate_epoch(worker_epoch)
        async with self._lock:
            now = self._clock()
            self._expire_locked(now)
            current = self._reservations.get(token)
            if current is not None:
                self._validate_identity(
                    current,
                    session_id=session_id,
                    generation_id=generation_id,
                )
                return current
            if self._lifecycle == "draining":
                raise WorkerReservationRejected("WORKER_DRAINING")
            if self._lifecycle == "failed":
                raise WorkerReservationRejected("WORKER_FAILED")
            if len(self._reservations) >= self.capacity:
                raise WorkerReservationRejected("WORKER_CAPACITY_EXHAUSTED")
            reservation = WorkerReservation(
                token=token,
                session_id=session_id,
                generation_id=generation_id,
                worker_epoch=worker_epoch,
                expires_at=now + ttl_s,
            )
            self._reservations[token] = reservation
            return reservation

    async def consume(
        self,
        token: str,
        *,
        session_id: str,
        generation_id: str,
        worker_epoch: str,
        owner_id: str,
    ) -> WorkerReservation:
        if not owner_id:
            raise WorkerReservationRejected("INVALID_RESERVATION_OWNER")
        self._validate_epoch(worker_epoch)
        async with self._lock:
            self._expire_locked(self._clock())
            reservation = self._reservations.get(token)
            if reservation is None:
                raise WorkerReservationRejected("RESERVATION_NOT_FOUND")
            self._validate_identity(
                reservation,
                session_id=session_id,
                generation_id=generation_id,
            )
            if reservation.consumed:
                raise WorkerReservationRejected("RESERVATION_ALREADY_CONSUMED")
            reservation = replace(
                reservation,
                consumed=True,
                owner_id=owner_id,
                consumed_at=self._clock(),
                runnable=False,
            )
            self._reservations[token] = reservation
            return reservation

    @staticmethod
    def _validate_owner(
        reservation: WorkerReservation,
        *,
        owner_id: str | None,
    ) -> None:
        if reservation.consumed and (
            not owner_id or reservation.owner_id != owner_id
        ):
            raise WorkerReservationRejected("RESERVATION_OWNER_MISMATCH")

    async def mark_runnable(self, token: str, *, owner_id: str) -> None:
        async with self._lock:
            reservation = self._reservations.get(token)
            if reservation is None:
                raise WorkerReservationRejected("RESERVATION_NOT_FOUND")
            self._validate_owner(reservation, owner_id=owner_id)
            self._reservations[token] = replace(reservation, runnable=True)

    async def mark_blocked(self, token: str, *, owner_id: str) -> None:
        async with self._lock:
            reservation = self._reservations.get(token)
            if reservation is None:
                raise WorkerReservationRejected("RESERVATION_NOT_FOUND")
            self._validate_owner(reservation, owner_id=owner_id)
            self._reservations[token] = replace(reservation, runnable=False)

    async def release(self, token: str, *, owner_id: str | None = None) -> None:
        async with self._lock:
            reservation = self._reservations.get(token)
            if reservation is None:
                return
            self._validate_owner(reservation, owner_id=owner_id)
            self._reservations.pop(token, None)
            if reservation.consumed_at is not None:
                elapsed_ms = max(0.0, (self._clock() - reservation.consumed_at) * 1000)
                self._completed_sessions += 1
                alpha = 1.0 / min(self._completed_sessions, 16)
                self._service_time_ms += alpha * (
                    elapsed_ms - self._service_time_ms
                )

    async def drain(self, deadline: float) -> None:
        if deadline <= 0:
            raise ValueError("drain deadline must be positive")
        async with self._lock:
            self._drain_deadline = deadline
            self._lifecycle = (
                "draining" if deadline > self._clock() else "failed"
            )

    async def snapshot(self) -> dict[str, str | int | float | None]:
        async with self._lock:
            self._expire_locked(self._clock())
            active_sessions = sum(
                reservation.consumed
                for reservation in self._reservations.values()
            )
            reserved_sessions = len(self._reservations) - active_sessions
            load = dict(self._load_provider() if self._load_provider else {})
            intrinsic_runnable = sum(
                reservation.consumed and reservation.runnable
                for reservation in self._reservations.values()
            )
            intrinsic_blocked = active_sessions - intrinsic_runnable
            runnable_sessions = max(
                0, int(load.get("runnable_sessions", intrinsic_runnable))
            )
            blocked_sessions = max(
                0, int(load.get("blocked_sessions", intrinsic_blocked))
            )
            queue_depth = max(0, int(load.get("queue_depth", reserved_sessions)))
            service_time_ms = max(
                0.0, float(load.get("service_time_ms", self._service_time_ms))
            )
            return {
                "worker_epoch": self.worker_epoch,
                "lifecycle": self._lifecycle,
                "drain_deadline": self._drain_deadline,
                "capacity": self.capacity,
                "active_sessions": active_sessions,
                "reserved_sessions": reserved_sessions,
                "runnable_sessions": runnable_sessions,
                "blocked_sessions": blocked_sessions,
                "queue_depth": queue_depth,
                "service_time_ms": service_time_ms,
                "normalized_load": (
                    active_sessions + reserved_sessions
                )
                / self.capacity,
            }


def install_worker_reservation_routes(
    app: FastAPI,
    registry: WorkerReservationRegistry,
) -> None:
    app.state.worker_reservations = registry

    @app.get("/v1/realtime_worker/state")
    async def realtime_worker_state():
        return await registry.snapshot()

    @app.post(
        "/v1/realtime_worker/reservations",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def reserve_realtime_worker(payload: dict):
        try:
            await registry.reserve(
                str(payload.get("token") or ""),
                session_id=str(payload.get("session_id") or ""),
                generation_id=str(payload.get("generation_id") or ""),
                worker_epoch=str(payload.get("worker_epoch") or ""),
                ttl_s=float(payload.get("ttl_s") or 0),
            )
        except (WorkerReservationRejected, TypeError, ValueError) as exc:
            reason = (
                exc.reason
                if isinstance(exc, WorkerReservationRejected)
                else "INVALID_RESERVATION"
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": reason},
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.delete(
        "/v1/realtime_worker/reservations/{token}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def release_realtime_worker(token: str):
        try:
            await registry.release(token)
        except WorkerReservationRejected as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": exc.reason},
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post(
        "/v1/realtime_worker/drain",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def drain_realtime_worker(payload: dict):
        try:
            await registry.drain(float(payload.get("deadline") or 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"reason": "INVALID_DRAIN_DEADLINE"},
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)
