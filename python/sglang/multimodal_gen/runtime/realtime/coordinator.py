# SPDX-License-Identifier: Apache-2.0

"""Production realtime worker coordination and fenced Session leases."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Mapping, Protocol
from uuid import uuid4


WorkerRole = Literal["denoiser", "vae"]
WorkerLifecycle = Literal["ready", "draining", "failed"]
logger = logging.getLogger(__name__)
_IDEMPOTENT_WORKER_RELEASE_REASONS = frozenset(
    {
        "RESERVATION_NOT_FOUND",
        "RESERVATION_OWNER_MISMATCH",
    }
)


class CoordinatorRejected(RuntimeError):
    def __init__(self, reason: str, *, retry_after_s: float | None = None) -> None:
        self.reason = reason
        self.retry_after_s = retry_after_s
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class WorkerHeartbeat:
    worker_id: str
    role: WorkerRole
    endpoint: str
    az: str
    capacity: int
    model_revision: str
    vae_fingerprint: str
    worker_epoch: str = ""
    lifecycle: WorkerLifecycle = "ready"
    active_sessions: int = 0
    runnable_sessions: int = 0
    blocked_sessions: int = 0
    queue_depth: int = 0
    service_time_ms: float = 0.0
    reservation_endpoint: str = ""
    drain_deadline: float | None = None


@dataclass(frozen=True, slots=True)
class WorkerSlot:
    worker_id: str
    role: WorkerRole
    endpoint: str
    az: str
    slot_index: int
    model_revision: str
    vae_fingerprint: str
    worker_epoch: str = ""
    lifecycle: WorkerLifecycle = "ready"
    active_sessions: int = 0
    runnable_sessions: int = 0
    blocked_sessions: int = 0
    queue_depth: int = 0
    service_time_ms: float = 0.0
    reservation_endpoint: str = ""
    drain_deadline: float | None = None
    capacity: int = 1


@dataclass(frozen=True, slots=True)
class SessionAssignment:
    user_id: str
    session_id: str
    generation_id: str
    token: str
    expires_at: float
    denoiser: WorkerSlot
    vae: WorkerSlot


class CoordinatorStore(Protocol):
    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None: ...

    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        excluded_workers: frozenset[tuple[WorkerRole, str]] = frozenset(),
    ) -> SessionAssignment: ...

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment: ...

    async def release(self, assignment: SessionAssignment) -> None: ...

    async def waiting_started(self, waiter_id: str) -> None: ...

    async def waiting_finished(self, waiter_id: str) -> None: ...

    async def capacity_snapshot(self) -> dict[str, Any]: ...


class WorkerReservationClient(Protocol):
    async def reserve(
        self,
        slot: WorkerSlot,
        *,
        token: str,
        session_id: str,
        generation_id: str,
        ttl_s: float,
    ) -> None: ...

    async def release(self, slot: WorkerSlot, *, token: str) -> None: ...


class HTTPWorkerReservationClient:
    def __init__(
        self,
        *,
        timeout_s: float = 2.0,
        client: Any | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def reserve(
        self,
        slot: WorkerSlot,
        *,
        token: str,
        session_id: str,
        generation_id: str,
        ttl_s: float,
    ) -> None:
        response = await self._get_client().post(
            f"{slot.reservation_endpoint.rstrip('/')}/reservations",
            json={
                "token": token,
                "session_id": session_id,
                "generation_id": generation_id,
                "worker_epoch": slot.worker_epoch,
                "ttl_s": ttl_s,
            },
        )
        response.raise_for_status()

    async def release(self, slot: WorkerSlot, *, token: str) -> None:
        response = await self._get_client().delete(
            f"{slot.reservation_endpoint.rstrip('/')}/reservations/{token}"
        )
        if response.status_code == 404:
            return
        if response.status_code == 409:
            try:
                detail = response.json().get("detail", {})
            except (ValueError, AttributeError):
                detail = {}
            if detail.get("reason") in _IDEMPOTENT_WORKER_RELEASE_REASONS:
                return
        response.raise_for_status()

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


@dataclass(slots=True)
class _WorkerState:
    heartbeat: WorkerHeartbeat
    updated_at: float


class InMemoryCoordinatorStore:
    """Atomic reference store used by tests and single-process development.

    A single condition protects user leases and both worker-slot reservations.
    An assignment is committed only after a compatible Denoiser/VAE pair is
    available, so a failed admission cannot leak a partial GPU reservation.
    """

    def __init__(
        self,
        *,
        ttl_s: float,
        worker_ttl_s: float,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        if worker_ttl_s <= 0:
            raise ValueError("worker_ttl_s must be positive")
        self.ttl_s = ttl_s
        self.worker_ttl_s = worker_ttl_s
        self._clock = clock
        self._wall_clock = wall_clock
        self._workers: dict[str, _WorkerState] = {}
        self._assignments_by_user: dict[str, SessionAssignment] = {}
        self._assignments_by_token: dict[str, SessionAssignment] = {}
        self._slot_tokens: dict[tuple[WorkerRole, str, int], str] = {}
        self._waiting: set[str] = set()
        self._condition = asyncio.Condition()

    @staticmethod
    def _validate_heartbeat(heartbeat: WorkerHeartbeat) -> None:
        if heartbeat.role not in ("denoiser", "vae"):
            raise CoordinatorRejected("INVALID_WORKER_ROLE")
        if not heartbeat.worker_id or not heartbeat.endpoint or not heartbeat.az:
            raise CoordinatorRejected("INVALID_WORKER_IDENTITY")
        if not heartbeat.worker_epoch or not heartbeat.reservation_endpoint:
            raise CoordinatorRejected("INVALID_WORKER_IDENTITY")
        if heartbeat.capacity < 1:
            raise CoordinatorRejected("INVALID_WORKER_CAPACITY")
        if heartbeat.lifecycle not in ("ready", "draining", "failed"):
            raise CoordinatorRejected("INVALID_WORKER_LIFECYCLE")
        if any(
            value < 0
            for value in (
                heartbeat.active_sessions,
                heartbeat.runnable_sessions,
                heartbeat.blocked_sessions,
                heartbeat.queue_depth,
                heartbeat.service_time_ms,
            )
        ):
            raise CoordinatorRejected("INVALID_WORKER_LOAD")

    def _release_locked(self, assignment: SessionAssignment) -> None:
        current = self._assignments_by_token.get(assignment.token)
        if current is None:
            return
        self._assignments_by_token.pop(assignment.token, None)
        if self._assignments_by_user.get(current.user_id) == current:
            self._assignments_by_user.pop(current.user_id, None)
        for slot in (current.denoiser, current.vae):
            key = (slot.role, slot.worker_id, slot.slot_index)
            if self._slot_tokens.get(key) == current.token:
                self._slot_tokens.pop(key, None)

    def _expire_locked(self, now: float) -> bool:
        expired = [
            assignment
            for assignment in self._assignments_by_token.values()
            if assignment.expires_at <= now
        ]
        for assignment in expired:
            self._release_locked(assignment)
        return bool(expired)

    def _active_workers_locked(
        self,
        *,
        role: WorkerRole,
        now: float,
        model_revision: str,
        vae_fingerprint: str,
        excluded_workers: frozenset[tuple[WorkerRole, str]],
    ) -> list[_WorkerState]:
        workers = []
        for state in self._workers.values():
            heartbeat = state.heartbeat
            if heartbeat.role != role:
                continue
            if (heartbeat.role, heartbeat.worker_id) in excluded_workers:
                continue
            if state.updated_at + self.worker_ttl_s <= now:
                continue
            if heartbeat.lifecycle != "ready":
                continue
            if role == "denoiser" and heartbeat.model_revision != model_revision:
                continue
            if role == "vae" and heartbeat.vae_fingerprint != vae_fingerprint:
                continue
            workers.append(state)
        return workers

    def _free_slots_locked(
        self, workers: list[_WorkerState], *, identity: str
    ) -> list[WorkerSlot]:
        slots: list[WorkerSlot] = []
        for state in workers:
            heartbeat = state.heartbeat
            for slot_index in range(heartbeat.capacity):
                key = (heartbeat.role, heartbeat.worker_id, slot_index)
                if key in self._slot_tokens:
                    continue
                slots.append(
                    WorkerSlot(
                        worker_id=heartbeat.worker_id,
                        role=heartbeat.role,
                        endpoint=heartbeat.endpoint,
                        az=heartbeat.az,
                        slot_index=slot_index,
                        model_revision=heartbeat.model_revision,
                        vae_fingerprint=heartbeat.vae_fingerprint,
                        worker_epoch=heartbeat.worker_epoch,
                        lifecycle=heartbeat.lifecycle,
                        active_sessions=heartbeat.active_sessions,
                        runnable_sessions=heartbeat.runnable_sessions,
                        blocked_sessions=heartbeat.blocked_sessions,
                        queue_depth=heartbeat.queue_depth,
                        service_time_ms=heartbeat.service_time_ms,
                        reservation_endpoint=heartbeat.reservation_endpoint,
                        drain_deadline=heartbeat.drain_deadline,
                        capacity=heartbeat.capacity,
                    )
                )
        slots.sort(
            key=lambda slot: (
                slot.active_sessions / slot.capacity,
                slot.queue_depth,
                slot.service_time_ms,
                hashlib.sha256(
                    f"{identity}:{slot.role}:{slot.worker_id}".encode()
                ).digest(),
                slot.worker_id,
                slot.slot_index,
            )
        )
        return slots

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        self._validate_heartbeat(heartbeat)
        async with self._condition:
            self._workers[heartbeat.worker_id] = _WorkerState(
                heartbeat=heartbeat,
                updated_at=self._clock(),
            )
            self._condition.notify_all()

    async def waiting_started(self, waiter_id: str) -> None:
        async with self._condition:
            self._waiting.add(waiter_id)

    async def waiting_finished(self, waiter_id: str) -> None:
        async with self._condition:
            self._waiting.discard(waiter_id)

    async def capacity_snapshot(self) -> dict[str, Any]:
        async with self._condition:
            now = self._clock()
            roles = {
                role: {
                    "waiting_sessions": len(self._waiting),
                    "active_sessions": 0,
                    "queued_sessions": 0,
                    "free_slots": 0,
                    "draining_workers": 0,
                }
                for role in ("denoiser", "vae")
            }
            for state in self._workers.values():
                if state.updated_at + self.worker_ttl_s <= now:
                    continue
                heartbeat = state.heartbeat
                role = roles[heartbeat.role]
                role["active_sessions"] += heartbeat.active_sessions
                role["queued_sessions"] += heartbeat.queue_depth
                if heartbeat.lifecycle == "draining":
                    role["draining_workers"] += 1
                elif heartbeat.lifecycle == "ready":
                    role["free_slots"] += max(
                        0, heartbeat.capacity - heartbeat.active_sessions
                    )
            return {"observed_at": self._wall_clock(), "roles": roles}

    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        excluded_workers: frozenset[tuple[WorkerRole, str]] = frozenset(),
    ) -> SessionAssignment:
        if not all(
            (user_id, session_id, generation_id, model_revision, vae_fingerprint)
        ):
            raise CoordinatorRejected("INVALID_SESSION_IDENTITY")
        async with self._condition:
            now = self._clock()
            identity = f"{user_id}:{session_id}:{generation_id}"
            self._expire_locked(now)
            if user_id in self._assignments_by_user:
                raise CoordinatorRejected("USER_SESSION_LIMIT")

            denoisers = self._free_slots_locked(
                self._active_workers_locked(
                    role="denoiser",
                    now=now,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    excluded_workers=excluded_workers,
                ),
                identity=identity,
            )
            vaes = self._free_slots_locked(
                self._active_workers_locked(
                    role="vae",
                    now=now,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                    excluded_workers=excluded_workers,
                ),
                identity=identity,
            )
            if not denoisers or not vaes:
                raise CoordinatorRejected("CAPACITY_EXHAUSTED", retry_after_s=0.1)

            denoiser = denoisers[0]
            vae = min(
                vaes,
                key=lambda slot: (
                    slot.az != denoiser.az,
                    slot.active_sessions / slot.capacity,
                    slot.queue_depth,
                    slot.service_time_ms,
                    hashlib.sha256(
                        f"{identity}:vae:{slot.worker_id}".encode()
                    ).digest(),
                    slot.worker_id,
                    slot.slot_index,
                ),
            )
            token = uuid4().hex
            assignment = SessionAssignment(
                user_id=user_id,
                session_id=session_id,
                generation_id=generation_id,
                token=token,
                expires_at=now + self.ttl_s,
                denoiser=denoiser,
                vae=vae,
            )
            self._assignments_by_user[user_id] = assignment
            self._assignments_by_token[token] = assignment
            for slot in (denoiser, vae):
                self._slot_tokens[(slot.role, slot.worker_id, slot.slot_index)] = token
            return assignment

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        async with self._condition:
            now = self._clock()
            self._expire_locked(now)
            current = self._assignments_by_token.get(assignment.token)
            if current is None or current != assignment:
                raise CoordinatorRejected("LEASE_LOST")
            for slot in (current.denoiser, current.vae):
                state = self._workers.get(slot.worker_id)
                if (
                    state is None
                    or state.updated_at + self.worker_ttl_s <= now
                    or state.heartbeat.worker_epoch != slot.worker_epoch
                    or state.heartbeat.lifecycle == "failed"
                    or (
                        state.heartbeat.lifecycle == "draining"
                        and state.heartbeat.drain_deadline is not None
                        and state.heartbeat.drain_deadline <= self._wall_clock()
                    )
                ):
                    raise CoordinatorRejected("WORKER_LOST")
            renewed = replace(current, expires_at=now + self.ttl_s)
            self._assignments_by_token[current.token] = renewed
            self._assignments_by_user[current.user_id] = renewed
            return renewed

    async def release(self, assignment: SessionAssignment) -> None:
        async with self._condition:
            self._release_locked(assignment)
            self._condition.notify_all()

    async def wait_for_change(self, timeout_s: float) -> None:
        async with self._condition:
            try:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout_s)
            except TimeoutError:
                pass


class DynamoDBCoordinatorStore:
    """Multi-replica Coordinator state backed by DynamoDB transactions.

    The table uses ``pk`` and ``sk`` string keys plus an ``allocation-index``
    GSI whose partition key is ``allocation_key`` and sort key is
    ``allocation_sort``. Worker heartbeats maintain allocatable slot records;
    admission atomically fences the user, Session, Denoiser slot, and VAE slot.
    """

    def __init__(
        self,
        table_name: str,
        *,
        ttl_s: float,
        worker_ttl_s: float,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
        candidate_limit: int = 24,
        capacity_limits: Mapping[WorkerRole, int] | None = None,
        wall_clock: Callable[[], float] = time.time,
        lease_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not table_name:
            raise ValueError("table_name is required")
        if ttl_s <= 0 or worker_ttl_s <= 0:
            raise ValueError("lease TTLs must be positive")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if capacity_limits and any(limit < 1 for limit in capacity_limits.values()):
            raise ValueError("worker capacity limits must be positive")
        self.table_name = table_name
        self.ttl_s = ttl_s
        self.worker_ttl_s = worker_ttl_s
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self.candidate_limit = candidate_limit
        self.capacity_limits = dict(capacity_limits or {})
        self._client = client
        self._wall_clock = wall_clock
        self._lease_clock = lease_clock

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required for DynamoDB Coordinator state"
                ) from exc
            self._client = boto3.client(
                "dynamodb",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
            )
        return self._client

    @staticmethod
    def _slot_pk(role: WorkerRole, worker_id: str, slot_index: int) -> str:
        return f"SLOT#{role}#{worker_id}#{slot_index:04d}"

    @staticmethod
    def _read_s(item: dict, key: str) -> str:
        return item[key]["S"]

    @staticmethod
    def _read_n(item: dict, key: str) -> int:
        return int(item[key]["N"])

    @staticmethod
    def _read_optional_s(item: dict, key: str, default: str = "") -> str:
        return item.get(key, {}).get("S", default)

    @staticmethod
    def _read_optional_n(
        item: dict, key: str, default: int | float = 0
    ) -> int | float:
        value = item.get(key, {}).get("N")
        if value is None:
            return default
        return float(value) if "." in value else int(value)

    @staticmethod
    def _is_active_item(item: dict | None, now_epoch: int) -> bool:
        if not item:
            return False
        expires = item.get("lease_expires_at", {}).get("N")
        return expires is not None and int(expires) > now_epoch

    def _allocation_key(
        self,
        role: WorkerRole,
        *,
        model_revision: str,
        vae_fingerprint: str,
    ) -> str:
        if role == "denoiser":
            return f"DENOISER#{model_revision}"
        return f"VAE#{vae_fingerprint}"

    def _slot_from_item(self, item: dict) -> WorkerSlot:
        return WorkerSlot(
            worker_id=self._read_s(item, "worker_id"),
            role=self._read_s(item, "role"),
            endpoint=self._read_s(item, "endpoint"),
            az=self._read_s(item, "az"),
            slot_index=self._read_n(item, "slot_index"),
            model_revision=self._read_s(item, "model_revision"),
            vae_fingerprint=self._read_s(item, "vae_fingerprint"),
            worker_epoch=self._read_optional_s(item, "worker_epoch"),
            lifecycle=self._read_optional_s(item, "lifecycle", "ready"),
            active_sessions=int(
                self._read_optional_n(item, "active_sessions")
            ),
            runnable_sessions=int(
                self._read_optional_n(item, "runnable_sessions")
            ),
            blocked_sessions=int(
                self._read_optional_n(item, "blocked_sessions")
            ),
            queue_depth=int(self._read_optional_n(item, "queue_depth")),
            service_time_ms=float(
                self._read_optional_n(item, "service_time_ms")
            ),
            reservation_endpoint=self._read_optional_s(
                item, "reservation_endpoint"
            ),
            drain_deadline=(
                float(item["drain_deadline"]["N"])
                if "drain_deadline" in item
                else None
            ),
            capacity=int(self._read_optional_n(item, "capacity", 1)),
        )

    @staticmethod
    def _candidate_pairs(
        denoisers: list[WorkerSlot],
        vaes: list[WorkerSlot],
        *,
        identity: str,
    ) -> list[tuple[WorkerSlot, WorkerSlot]]:
        if not denoisers or not vaes:
            return []

        def available_slots_by_worker(
            slots: list[WorkerSlot],
        ) -> dict[str, int]:
            available: dict[str, int] = {}
            for slot in slots:
                available[slot.worker_id] = available.get(slot.worker_id, 0) + 1
            return available

        denoiser_free = available_slots_by_worker(denoisers)
        vae_free = available_slots_by_worker(vaes)

        def order(
            slot: WorkerSlot,
            salt: str,
            available: dict[str, int],
        ) -> tuple:
            inferred_active = max(
                0,
                slot.capacity - available.get(slot.worker_id, 0),
            )
            effective_active = max(slot.active_sessions, inferred_active)
            spread = hashlib.sha256(
                f"{salt}:{slot.worker_id}".encode()
            ).digest()
            return (
                effective_active / slot.capacity,
                slot.queue_depth,
                slot.service_time_ms,
                slot.slot_index,
                spread,
                slot.worker_id,
            )

        ordered_denoisers = sorted(
            denoisers,
            key=lambda slot: order(slot, "denoiser", denoiser_free),
        )
        available_vaes = list(vaes)
        pairs: list[tuple[WorkerSlot, WorkerSlot]] = []
        for denoiser in ordered_denoisers:
            if not available_vaes:
                break
            vae_index = min(
                range(len(available_vaes)),
                key=lambda index: (
                    available_vaes[index].az != denoiser.az,
                    *order(available_vaes[index], "vae", vae_free),
                ),
            )
            pairs.append((denoiser, available_vaes.pop(vae_index)))
        return pairs

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        InMemoryCoordinatorStore._validate_heartbeat(heartbeat)
        await asyncio.to_thread(self._heartbeat_sync, heartbeat)

    def _heartbeat_sync(self, heartbeat: WorkerHeartbeat) -> None:
        capacity_limit = self.capacity_limits.get(heartbeat.role)
        if capacity_limit is not None and heartbeat.capacity > capacity_limit:
            heartbeat = replace(heartbeat, capacity=capacity_limit)
        client = self._get_client()
        now_epoch = int(self._wall_clock())
        heartbeat_expires = now_epoch + max(1, int(self.worker_ttl_s))
        client.put_item(
            TableName=self.table_name,
            Item={
                "pk": {"S": f"WORKER#{heartbeat.worker_id}"},
                "sk": {"S": "HEARTBEAT"},
                "item_type": {"S": "worker"},
                "role": {"S": heartbeat.role},
                "endpoint": {"S": heartbeat.endpoint},
                "az": {"S": heartbeat.az},
                "capacity": {"N": str(heartbeat.capacity)},
                "model_revision": {"S": heartbeat.model_revision},
                "vae_fingerprint": {"S": heartbeat.vae_fingerprint},
                "worker_epoch": {"S": heartbeat.worker_epoch},
                "lifecycle": {"S": heartbeat.lifecycle},
                "active_sessions": {"N": str(heartbeat.active_sessions)},
                "runnable_sessions": {"N": str(heartbeat.runnable_sessions)},
                "blocked_sessions": {"N": str(heartbeat.blocked_sessions)},
                "queue_depth": {"N": str(heartbeat.queue_depth)},
                "service_time_ms": {"N": str(heartbeat.service_time_ms)},
                "reservation_endpoint": {"S": heartbeat.reservation_endpoint},
                "heartbeat_expires_at": {"N": str(heartbeat_expires)},
                "allocation_key": {"S": f"CAPACITY#{heartbeat.role}"},
                "allocation_sort": {"S": f"worker#{heartbeat.worker_id}"},
                "ttl": {"N": str(heartbeat_expires + 86400)},
                **(
                    {
                        "drain_deadline": {
                            "N": str(heartbeat.drain_deadline)
                        }
                    }
                    if heartbeat.drain_deadline is not None
                    else {}
                ),
            },
        )
        allocation_key = self._allocation_key(
            heartbeat.role,
            model_revision=heartbeat.model_revision,
            vae_fingerprint=heartbeat.vae_fingerprint,
        )
        for slot_index in range(heartbeat.capacity):
            update = {
                "TableName": self.table_name,
                "Key": {
                    "pk": {
                        "S": self._slot_pk(
                            heartbeat.role, heartbeat.worker_id, slot_index
                        )
                    },
                    "sk": {"S": "LEASE"},
                },
                "UpdateExpression": (
                    "SET item_type = :item_type, #role = :role, "
                    "worker_id = :worker_id, endpoint = :endpoint, az = :az, "
                    "slot_index = :slot_index, model_revision = :model_revision, "
                    "vae_fingerprint = :vae_fingerprint, "
                    "worker_epoch = :worker_epoch, lifecycle = :lifecycle, "
                    "#capacity = :capacity, active_sessions = :active_sessions, "
                    "runnable_sessions = :runnable_sessions, "
                    "blocked_sessions = :blocked_sessions, "
                    "queue_depth = :queue_depth, service_time_ms = :service_time_ms, "
                    "reservation_endpoint = :reservation_endpoint, "
                    "heartbeat_expires_at = :heartbeat_expires, "
                    "allocation_key = :allocation_key, "
                    "allocation_sort = :allocation_sort, #ttl = :ttl"
                ),
                "ExpressionAttributeNames": {
                    "#capacity": "capacity",
                    "#role": "role",
                    "#ttl": "ttl",
                },
                "ExpressionAttributeValues": {
                    ":item_type": {"S": "worker_slot"},
                    ":role": {"S": heartbeat.role},
                    ":worker_id": {"S": heartbeat.worker_id},
                    ":endpoint": {"S": heartbeat.endpoint},
                    ":az": {"S": heartbeat.az},
                    ":slot_index": {"N": str(slot_index)},
                    ":model_revision": {"S": heartbeat.model_revision},
                    ":vae_fingerprint": {"S": heartbeat.vae_fingerprint},
                    ":worker_epoch": {"S": heartbeat.worker_epoch},
                    ":lifecycle": {"S": heartbeat.lifecycle},
                    ":capacity": {"N": str(heartbeat.capacity)},
                    ":active_sessions": {"N": str(heartbeat.active_sessions)},
                    ":runnable_sessions": {"N": str(heartbeat.runnable_sessions)},
                    ":blocked_sessions": {"N": str(heartbeat.blocked_sessions)},
                    ":queue_depth": {"N": str(heartbeat.queue_depth)},
                    ":service_time_ms": {"N": str(heartbeat.service_time_ms)},
                    ":reservation_endpoint": {"S": heartbeat.reservation_endpoint},
                    ":heartbeat_expires": {"N": str(heartbeat_expires)},
                    ":allocation_key": {"S": allocation_key},
                    ":allocation_sort": {
                        "S": f"{heartbeat.az}#{heartbeat.worker_id}#{slot_index:04d}"
                    },
                    ":ttl": {"N": str(heartbeat_expires + 86400)},
                },
            }
            for attempt in range(3):
                try:
                    client.update_item(**update)
                    break
                except client.exceptions.TransactionConflictException:
                    if attempt == 2:
                        raise
                    time.sleep(0.01 * (2**attempt))

    async def waiting_started(self, waiter_id: str) -> None:
        await asyncio.to_thread(self._waiting_started_sync, waiter_id)

    def _waiting_started_sync(self, waiter_id: str) -> None:
        if not waiter_id:
            raise ValueError("waiter_id is required")
        client = self._get_client()
        now_epoch = int(self._wall_clock())
        expires_epoch = now_epoch + max(30, int(self.worker_ttl_s * 2))
        for role in ("denoiser", "vae"):
            client.put_item(
                TableName=self.table_name,
                Item={
                    "pk": {"S": f"CAPACITY_DEMAND#{role}#{waiter_id}"},
                    "sk": {"S": "WAIT"},
                    "item_type": {"S": "capacity_demand"},
                    "role": {"S": role},
                    "allocation_key": {"S": f"CAPACITY#{role}"},
                    "allocation_sort": {"S": f"demand#{waiter_id}"},
                    "demand_expires_at": {"N": str(expires_epoch)},
                    "ttl": {"N": str(expires_epoch + 3600)},
                },
            )

    async def waiting_finished(self, waiter_id: str) -> None:
        await asyncio.to_thread(self._waiting_finished_sync, waiter_id)

    def _waiting_finished_sync(self, waiter_id: str) -> None:
        client = self._get_client()
        for role in ("denoiser", "vae"):
            client.delete_item(
                TableName=self.table_name,
                Key={
                    "pk": {"S": f"CAPACITY_DEMAND#{role}#{waiter_id}"},
                    "sk": {"S": "WAIT"},
                },
            )

    async def capacity_snapshot(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._capacity_snapshot_sync)

    def _capacity_snapshot_sync(self) -> dict[str, Any]:
        client = self._get_client()
        now_epoch = int(self._wall_clock())
        roles: dict[str, dict[str, int]] = {}
        for role in ("denoiser", "vae"):
            values = {
                "waiting_sessions": 0,
                "active_sessions": 0,
                "queued_sessions": 0,
                "free_slots": 0,
                "draining_workers": 0,
            }
            query = {
                "TableName": self.table_name,
                "IndexName": "allocation-index",
                "KeyConditionExpression": "allocation_key = :allocation",
                "ExpressionAttributeValues": {
                    ":allocation": {"S": f"CAPACITY#{role}"}
                },
            }
            while True:
                response = client.query(**query)
                for item in response.get("Items", []):
                    item_type = self._read_optional_s(item, "item_type")
                    if item_type == "capacity_demand":
                        if int(
                            self._read_optional_n(item, "demand_expires_at")
                        ) > now_epoch:
                            values["waiting_sessions"] += 1
                        continue
                    if item_type != "worker" or int(
                        self._read_optional_n(item, "heartbeat_expires_at")
                    ) <= now_epoch:
                        continue
                    active = int(self._read_optional_n(item, "active_sessions"))
                    values["active_sessions"] += active
                    values["queued_sessions"] += int(
                        self._read_optional_n(item, "queue_depth")
                    )
                    lifecycle = self._read_optional_s(
                        item, "lifecycle", "ready"
                    )
                    if lifecycle == "draining":
                        values["draining_workers"] += 1
                    elif lifecycle == "ready":
                        values["free_slots"] += max(
                            0,
                            int(self._read_optional_n(item, "capacity", 1))
                            - active,
                        )
                last_evaluated_key = response.get("LastEvaluatedKey")
                if not last_evaluated_key:
                    break
                query["ExclusiveStartKey"] = last_evaluated_key
            roles[role] = values
        return {"observed_at": now_epoch, "roles": roles}

    def _query_slots_sync(
        self,
        role: WorkerRole,
        *,
        model_revision: str,
        vae_fingerprint: str,
        now_epoch: int,
    ) -> list[WorkerSlot]:
        query = {
            "TableName": self.table_name,
            "IndexName": "allocation-index",
            "KeyConditionExpression": "allocation_key = :allocation",
            "FilterExpression": (
                "heartbeat_expires_at > :now AND lifecycle = :ready AND "
                "(attribute_not_exists(lease_token) OR lease_expires_at <= :now)"
            ),
            "ExpressionAttributeValues": {
                ":allocation": {
                    "S": self._allocation_key(
                        role,
                        model_revision=model_revision,
                        vae_fingerprint=vae_fingerprint,
                    )
                },
                ":now": {"N": str(now_epoch)},
                ":ready": {"S": "ready"},
            },
            "Limit": self.candidate_limit,
        }
        slots: list[WorkerSlot] = []
        while len(slots) < self.candidate_limit:
            response = self._get_client().query(**query)
            slots.extend(
                self._slot_from_item(item)
                for item in response.get("Items", [])
            )
            last_evaluated_key = response.get("LastEvaluatedKey")
            if not last_evaluated_key:
                break
            query["ExclusiveStartKey"] = last_evaluated_key
        return slots[: self.candidate_limit]

    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        excluded_workers: frozenset[tuple[WorkerRole, str]] = frozenset(),
    ) -> SessionAssignment:
        return await asyncio.to_thread(
            self._acquire_sync,
            user_id=user_id,
            session_id=session_id,
            generation_id=generation_id,
            model_revision=model_revision,
            vae_fingerprint=vae_fingerprint,
            excluded_workers=excluded_workers,
        )

    def _acquire_sync(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        excluded_workers: frozenset[tuple[WorkerRole, str]] = frozenset(),
    ) -> SessionAssignment:
        if not all(
            (user_id, session_id, generation_id, model_revision, vae_fingerprint)
        ):
            raise CoordinatorRejected("INVALID_SESSION_IDENTITY")
        client = self._get_client()
        for query_round in range(4):
            now_epoch = int(self._wall_clock())
            expires_epoch = now_epoch + max(1, int(self.ttl_s))
            denoisers = self._query_slots_sync(
                "denoiser",
                model_revision=model_revision,
                vae_fingerprint=vae_fingerprint,
                now_epoch=now_epoch,
            )
            vaes = self._query_slots_sync(
                "vae",
                model_revision=model_revision,
                vae_fingerprint=vae_fingerprint,
                now_epoch=now_epoch,
            )
            denoisers = [
                slot
                for slot in denoisers
                if (slot.role, slot.worker_id) not in excluded_workers
            ]
            vaes = [
                slot
                for slot in vaes
                if (slot.role, slot.worker_id) not in excluded_workers
            ]
            pairs = self._candidate_pairs(
                denoisers,
                vaes,
                identity=(
                    f"{user_id}:{session_id}:{generation_id}:{query_round}"
                ),
            )
            if not pairs:
                break
            for denoiser, vae in pairs:
                assignment = self._try_acquire_pair_sync(
                    client,
                    user_id=user_id,
                    session_id=session_id,
                    generation_id=generation_id,
                    denoiser=denoiser,
                    vae=vae,
                    now_epoch=now_epoch,
                    expires_epoch=expires_epoch,
                )
                if assignment is not None:
                    return assignment
            if query_round < 3:
                time.sleep(0.005 * (2**query_round))
        raise CoordinatorRejected("CAPACITY_EXHAUSTED", retry_after_s=0.25)

    def _try_acquire_pair_sync(
        self,
        client: Any,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        denoiser: WorkerSlot,
        vae: WorkerSlot,
        now_epoch: int,
        expires_epoch: int,
    ) -> SessionAssignment | None:
        token = uuid4().hex
        identity = {
            "user_id": {"S": user_id},
            "session_id": {"S": session_id},
            "generation_id": {"S": generation_id},
            "lease_token": {"S": token},
            "lease_expires_at": {"N": str(expires_epoch)},
            "ttl": {"N": str(expires_epoch + 86400)},
        }
        values = {
            ":token": {"S": token},
            ":user": {"S": user_id},
            ":session": {"S": session_id},
            ":generation": {"S": generation_id},
            ":now": {"N": str(now_epoch)},
            ":expires": {"N": str(expires_epoch)},
            ":ttl": {"N": str(expires_epoch + 86400)},
        }
        slot_updates = []
        for slot in (denoiser, vae):
            slot_values = {
                **values,
                ":worker_epoch": {"S": slot.worker_epoch},
                ":ready": {"S": "ready"},
            }
            slot_updates.append(
                {
                    "Update": {
                        "TableName": self.table_name,
                        "Key": {
                            "pk": {
                                "S": self._slot_pk(
                                    slot.role, slot.worker_id, slot.slot_index
                                )
                            },
                            "sk": {"S": "LEASE"},
                        },
                        "UpdateExpression": (
                            "SET lease_token = :token, user_id = :user, "
                            "session_id = :session, generation_id = :generation, "
                            "lease_expires_at = :expires, #ttl = :ttl"
                        ),
                        "ConditionExpression": (
                            "heartbeat_expires_at > :now AND "
                            "worker_epoch = :worker_epoch AND lifecycle = :ready AND "
                            "(attribute_not_exists(lease_token) OR "
                            "lease_expires_at <= :now)"
                        ),
                        "ExpressionAttributeNames": {"#ttl": "ttl"},
                        "ExpressionAttributeValues": slot_values,
                    }
                }
            )
        session_item = {
            **identity,
            "pk": {"S": f"SESSION#{session_id}"},
            "sk": {"S": "ASSIGNMENT"},
            "item_type": {"S": "session_assignment"},
            "denoiser_worker_id": {"S": denoiser.worker_id},
            "denoiser_slot": {"N": str(denoiser.slot_index)},
            "denoiser_endpoint": {"S": denoiser.endpoint},
            "denoiser_worker_epoch": {"S": denoiser.worker_epoch},
            "vae_worker_id": {"S": vae.worker_id},
            "vae_slot": {"N": str(vae.slot_index)},
            "vae_endpoint": {"S": vae.endpoint},
            "vae_worker_epoch": {"S": vae.worker_epoch},
        }
        try:
            client.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": {
                                **identity,
                                "pk": {"S": f"USER#{user_id}"},
                                "sk": {"S": "LEASE"},
                                "item_type": {"S": "user_lease"},
                            },
                            "ConditionExpression": (
                                "attribute_not_exists(lease_token) OR "
                                "lease_expires_at <= :now"
                            ),
                            "ExpressionAttributeValues": {
                                ":now": {"N": str(now_epoch)}
                            },
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": session_item,
                            "ConditionExpression": (
                                "attribute_not_exists(lease_token) OR "
                                "lease_expires_at <= :now"
                            ),
                            "ExpressionAttributeValues": {
                                ":now": {"N": str(now_epoch)}
                            },
                        }
                    },
                    *slot_updates,
                ]
            )
        except client.exceptions.TransactionCanceledException as exc:
            getter = getattr(client, "get_item", None)
            if getter is not None:
                current_user = getter(
                    TableName=self.table_name,
                    Key={
                        "pk": {"S": f"USER#{user_id}"},
                        "sk": {"S": "LEASE"},
                    },
                    ConsistentRead=True,
                ).get("Item")
                if self._is_active_item(current_user, now_epoch):
                    raise CoordinatorRejected("USER_SESSION_LIMIT") from exc
            return None
        return SessionAssignment(
            user_id=user_id,
            session_id=session_id,
            generation_id=generation_id,
            token=token,
            expires_at=self._lease_clock() + self.ttl_s,
            denoiser=denoiser,
            vae=vae,
        )

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        return await asyncio.to_thread(self._renew_sync, assignment)

    def _renew_cancellation_reason(
        self,
        client: Any,
        assignment: SessionAssignment,
        keys: list[tuple[str, str]],
        *,
        now_epoch: int,
    ) -> str | None:
        getter = getattr(client, "get_item", None)
        if getter is None:
            return "LEASE_LOST"
        for slot in (assignment.denoiser, assignment.vae):
            item = getter(
                TableName=self.table_name,
                Key={
                    "pk": {"S": f"WORKER#{slot.worker_id}"},
                    "sk": {"S": "HEARTBEAT"},
                },
                ConsistentRead=True,
            ).get("Item")
            if (
                not item
                or item.get("worker_epoch", {}).get("S") != slot.worker_epoch
                or int(item.get("heartbeat_expires_at", {}).get("N", "0"))
                <= now_epoch
                or item.get("lifecycle", {}).get("S", "ready") == "failed"
                or (
                    item.get("lifecycle", {}).get("S") == "draining"
                    and float(item.get("drain_deadline", {}).get("N", "inf"))
                    <= now_epoch
                )
            ):
                return "WORKER_LOST"
        for pk, sk in keys:
            item = getter(
                TableName=self.table_name,
                Key={"pk": {"S": pk}, "sk": {"S": sk}},
                ConsistentRead=True,
            ).get("Item")
            if item.get("lease_token", {}).get("S") != assignment.token:
                return "LEASE_LOST"
        return None

    def _renew_sync(self, assignment: SessionAssignment) -> SessionAssignment:
        client = self._get_client()
        now_epoch = int(self._wall_clock())
        expires_epoch = now_epoch + max(1, int(self.ttl_s))
        values = {
            ":token": {"S": assignment.token},
            ":expires": {"N": str(expires_epoch)},
            ":ttl": {"N": str(expires_epoch + 86400)},
        }
        keys = [
            (f"USER#{assignment.user_id}", "LEASE"),
            (f"SESSION#{assignment.session_id}", "ASSIGNMENT"),
            (
                self._slot_pk(
                    assignment.denoiser.role,
                    assignment.denoiser.worker_id,
                    assignment.denoiser.slot_index,
                ),
                "LEASE",
            ),
            (
                self._slot_pk(
                    assignment.vae.role,
                    assignment.vae.worker_id,
                    assignment.vae.slot_index,
                ),
                "LEASE",
            ),
        ]
        max_attempts = 6
        for attempt in range(max_attempts):
            worker_checks = []
            for slot in (assignment.denoiser, assignment.vae):
                worker_checks.append(
                    {
                        "ConditionCheck": {
                            "TableName": self.table_name,
                            "Key": {
                                "pk": {"S": f"WORKER#{slot.worker_id}"},
                                "sk": {"S": "HEARTBEAT"},
                            },
                            "ConditionExpression": (
                                "worker_epoch = :worker_epoch AND "
                                "heartbeat_expires_at > :now AND "
                                "(#lifecycle = :ready OR "
                                "(#lifecycle = :draining AND "
                                "(attribute_not_exists(drain_deadline) OR "
                                "drain_deadline > :now)))"
                            ),
                            "ExpressionAttributeNames": {
                                "#lifecycle": "lifecycle"
                            },
                            "ExpressionAttributeValues": {
                                ":worker_epoch": {"S": slot.worker_epoch},
                                ":now": {"N": str(now_epoch)},
                                ":ready": {"S": "ready"},
                                ":draining": {"S": "draining"},
                            },
                        }
                    }
                )
            try:
                client.transact_write_items(
                    TransactItems=worker_checks
                    + [
                        {
                            "Update": {
                                "TableName": self.table_name,
                                "Key": {"pk": {"S": pk}, "sk": {"S": sk}},
                                "UpdateExpression": (
                                    "SET lease_expires_at = :expires, #ttl = :ttl"
                                ),
                                "ConditionExpression": "lease_token = :token",
                                "ExpressionAttributeNames": {"#ttl": "ttl"},
                                "ExpressionAttributeValues": values,
                            }
                        }
                        for pk, sk in keys
                    ]
                )
                break
            except client.exceptions.TransactionCanceledException as exc:
                reason = self._renew_cancellation_reason(
                    client,
                    assignment,
                    keys,
                    now_epoch=now_epoch,
                )
                if reason is not None:
                    raise CoordinatorRejected(reason) from exc
                if attempt == max_attempts - 1:
                    raise CoordinatorRejected("LEASE_RENEW_CONFLICT") from exc
                token_jitter = int(
                    hashlib.sha256(assignment.token.encode()).hexdigest()[:2], 16
                ) / 255
                time.sleep(0.01 * (2**attempt) * (1 + token_jitter))
        return replace(
            assignment, expires_at=self._lease_clock() + self.ttl_s
        )

    async def release(self, assignment: SessionAssignment) -> None:
        await asyncio.to_thread(self._release_sync, assignment)

    def _release_sync(self, assignment: SessionAssignment) -> None:
        client = self._get_client()
        token = {":token": {"S": assignment.token}}
        try:
            client.transact_write_items(
                TransactItems=[
                    {
                        "Delete": {
                            "TableName": self.table_name,
                            "Key": {
                                "pk": {"S": f"USER#{assignment.user_id}"},
                                "sk": {"S": "LEASE"},
                            },
                            "ConditionExpression": "lease_token = :token",
                            "ExpressionAttributeValues": token,
                        }
                    },
                    {
                        "Delete": {
                            "TableName": self.table_name,
                            "Key": {
                                "pk": {"S": f"SESSION#{assignment.session_id}"},
                                "sk": {"S": "ASSIGNMENT"},
                            },
                            "ConditionExpression": "lease_token = :token",
                            "ExpressionAttributeValues": token,
                        }
                    },
                    *[
                        {
                            "Update": {
                                "TableName": self.table_name,
                                "Key": {
                                    "pk": {
                                        "S": self._slot_pk(
                                            slot.role,
                                            slot.worker_id,
                                            slot.slot_index,
                                        )
                                    },
                                    "sk": {"S": "LEASE"},
                                },
                                "UpdateExpression": (
                                    "REMOVE lease_token, user_id, session_id, "
                                    "generation_id, lease_expires_at"
                                ),
                                "ConditionExpression": "lease_token = :token",
                                "ExpressionAttributeValues": token,
                            }
                        }
                        for slot in (assignment.denoiser, assignment.vae)
                    ],
                ]
            )
        except client.exceptions.TransactionCanceledException:
            return


class RealtimeCoordinator:
    def __init__(
        self,
        store: CoordinatorStore,
        *,
        wait_timeout_s: float = 10.0,
        reservation_client: WorkerReservationClient | None = None,
    ) -> None:
        self.store = store
        self.wait_timeout_s = max(0.0, wait_timeout_s)
        self.reservation_client = reservation_client
        self._late_cleanup_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    async def _retry_cleanup(self, operation, *, label: str) -> None:
        last_error: BaseException | None = None
        for attempt in range(1, 4):
            try:
                await operation()
                return
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                last_error = exc
                logger.warning(
                    "Coordinator cleanup attempt %s/3 failed for %s: %s",
                    attempt,
                    label,
                    exc,
                )
                await asyncio.sleep(0)
        assert last_error is not None
        logger.error(
            "Coordinator cleanup exhausted retries for %s",
            label,
            exc_info=last_error,
        )
        raise RuntimeError(f"coordinator cleanup failed for {label}") from last_error

    async def _compensate_assignment(
        self,
        assignment: SessionAssignment,
        *,
        strict_worker_cleanup: bool = True,
    ) -> None:
        worker_failures: list[BaseException] = []
        failures: list[BaseException] = []
        if self.reservation_client is not None:
            results = await asyncio.gather(
                *(
                    self._retry_cleanup(
                        lambda slot=slot: self.reservation_client.release(
                            slot, token=assignment.token
                        ),
                        label=f"worker:{slot.role}:{slot.worker_id}",
                    )
                    for slot in (assignment.denoiser, assignment.vae)
                ),
                return_exceptions=True,
            )
            worker_failures.extend(
                result for result in results if isinstance(result, BaseException)
            )
        try:
            await self._retry_cleanup(
                lambda: self.store.release(assignment),
                label=f"store:{assignment.session_id}:{assignment.generation_id}",
            )
        except BaseException as exc:
            failures.append(exc)
        if strict_worker_cleanup:
            failures.extend(worker_failures)
        elif worker_failures:
            logger.warning(
                "Coordinator ignored %s worker release cleanup failure(s) for "
                "session_id=%s generation_id=%s",
                len(worker_failures),
                assignment.session_id,
                assignment.generation_id,
            )
        if failures:
            raise RuntimeError(
                f"assignment compensation failed ({len(failures)} operation(s))"
            ) from failures[0]

    async def _finish_compensation(self, assignment: SessionAssignment) -> None:
        cleanup = asyncio.create_task(self._compensate_assignment(assignment))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        await cleanup

    async def _release_late_acquire(self, acquire_task: asyncio.Task) -> None:
        try:
            assignment = await acquire_task
        except BaseException:
            return
        try:
            await self._compensate_assignment(assignment)
        except BaseException:
            logger.exception("Failed to compensate a late Coordinator acquire")

    def _track_late_acquire(self, acquire_task: asyncio.Task) -> None:
        cleanup = asyncio.create_task(self._release_late_acquire(acquire_task))
        self._late_cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._late_cleanup_tasks.discard)

    async def _acquire_before_deadline(
        self,
        *,
        deadline: float | None,
        **request: Any,
    ) -> SessionAssignment:
        acquire_task = asyncio.create_task(self.store.acquire(**request))
        try:
            remaining = self._remaining(deadline)
            if remaining is None:
                return await acquire_task
            if remaining <= 0:
                self._track_late_acquire(acquire_task)
                raise TimeoutError
            done, _ = await asyncio.wait((acquire_task,), timeout=remaining)
            if acquire_task not in done:
                self._track_late_acquire(acquire_task)
                raise TimeoutError
            return acquire_task.result()
        except asyncio.CancelledError:
            self._track_late_acquire(acquire_task)
            raise

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        await self.store.heartbeat(heartbeat)

    async def capacity_snapshot(self) -> dict[str, Any]:
        return await self.store.capacity_snapshot()

    async def admit(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
        wait_for_capacity: bool = True,
    ) -> SessionAssignment:
        deadline = (
            time.monotonic() + self.wait_timeout_s
            if self.wait_timeout_s > 0
            else None
        )
        excluded_workers: set[tuple[WorkerRole, str]] = set()
        waiting_id: str | None = None
        try:
            while True:
                try:
                    assignment = await self._acquire_before_deadline(
                        deadline=deadline,
                        user_id=user_id,
                        session_id=session_id,
                        generation_id=generation_id,
                        model_revision=model_revision,
                        vae_fingerprint=vae_fingerprint,
                        excluded_workers=frozenset(excluded_workers),
                    )
                except TimeoutError as exc:
                    raise CoordinatorRejected(
                        "CAPACITY_EXHAUSTED", retry_after_s=0.1
                    ) from exc
                except CoordinatorRejected as exc:
                    if exc.reason != "CAPACITY_EXHAUSTED" or not wait_for_capacity:
                        raise
                    remaining = self._remaining(deadline)
                    if remaining is None:
                        remaining = 0.0
                    if remaining <= 0:
                        raise CoordinatorRejected(
                            "CAPACITY_EXHAUSTED", retry_after_s=exc.retry_after_s
                        ) from exc
                    if waiting_id is None:
                        waiting_id = uuid4().hex
                        await self.store.waiting_started(waiting_id)
                    wait_s = min(remaining, exc.retry_after_s or 0.1)
                    waiter = getattr(self.store, "wait_for_change", None)
                    if waiter is None:
                        await asyncio.sleep(wait_s)
                    else:
                        await waiter(wait_s)
                    continue

                if self.reservation_client is None:
                    return assignment
                failed_slot = None
                ttl_s = max(0.001, assignment.expires_at - time.monotonic())
                try:
                    for slot in (assignment.denoiser, assignment.vae):
                        failed_slot = slot
                        reserve = self.reservation_client.reserve(
                            slot,
                            token=assignment.token,
                            session_id=assignment.session_id,
                            generation_id=assignment.generation_id,
                            ttl_s=ttl_s,
                        )
                        remaining = self._remaining(deadline)
                        if remaining is None:
                            await reserve
                        elif remaining <= 0:
                            reserve.close()
                            raise TimeoutError
                        else:
                            await asyncio.wait_for(reserve, timeout=remaining)
                except BaseException as exc:
                    cleanup_error = None
                    try:
                        await self._finish_compensation(assignment)
                    except BaseException as cleanup_exc:
                        cleanup_error = cleanup_exc
                        logger.exception(
                            "Coordinator failed to compensate rejected assignment"
                        )
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    if cleanup_error is not None:
                        raise CoordinatorRejected(
                            "COORDINATOR_CLEANUP_FAILED", retry_after_s=1.0
                        ) from cleanup_error
                    if failed_slot is not None:
                        excluded_workers.add(
                            (failed_slot.role, failed_slot.worker_id)
                        )
                    if (
                        not wait_for_capacity
                        or deadline is None
                        or time.monotonic() >= deadline
                    ):
                        raise CoordinatorRejected(
                            "CAPACITY_EXHAUSTED", retry_after_s=0.1
                        ) from exc
                    continue
                return assignment
        finally:
            if waiting_id is not None:
                await asyncio.gather(
                    self.store.waiting_finished(waiting_id),
                    return_exceptions=True,
                )

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        return await self.store.renew(assignment)

    async def release(self, assignment: SessionAssignment) -> None:
        await self._compensate_assignment(
            assignment,
            strict_worker_cleanup=False,
        )

    async def close(self) -> None:
        if self._late_cleanup_tasks:
            await asyncio.gather(*self._late_cleanup_tasks, return_exceptions=True)
        closer = getattr(self.reservation_client, "close", None)
        if closer is not None:
            await closer()
