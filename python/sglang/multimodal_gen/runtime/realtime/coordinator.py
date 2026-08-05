# SPDX-License-Identifier: Apache-2.0

"""Production realtime worker coordination and fenced Session leases."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Protocol
from uuid import uuid4


WorkerRole = Literal["denoiser", "vae"]


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


@dataclass(frozen=True, slots=True)
class WorkerSlot:
    worker_id: str
    role: WorkerRole
    endpoint: str
    az: str
    slot_index: int
    model_revision: str
    vae_fingerprint: str


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
    ) -> SessionAssignment: ...

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment: ...

    async def release(self, assignment: SessionAssignment) -> None: ...


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
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        if worker_ttl_s <= 0:
            raise ValueError("worker_ttl_s must be positive")
        self.ttl_s = ttl_s
        self.worker_ttl_s = worker_ttl_s
        self._clock = clock
        self._workers: dict[str, _WorkerState] = {}
        self._assignments_by_user: dict[str, SessionAssignment] = {}
        self._assignments_by_token: dict[str, SessionAssignment] = {}
        self._slot_tokens: dict[tuple[WorkerRole, str, int], str] = {}
        self._condition = asyncio.Condition()

    @staticmethod
    def _validate_heartbeat(heartbeat: WorkerHeartbeat) -> None:
        if heartbeat.role not in ("denoiser", "vae"):
            raise CoordinatorRejected("INVALID_WORKER_ROLE")
        if not heartbeat.worker_id or not heartbeat.endpoint or not heartbeat.az:
            raise CoordinatorRejected("INVALID_WORKER_IDENTITY")
        if heartbeat.capacity < 1:
            raise CoordinatorRejected("INVALID_WORKER_CAPACITY")

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
    ) -> list[_WorkerState]:
        workers = []
        for state in self._workers.values():
            heartbeat = state.heartbeat
            if heartbeat.role != role:
                continue
            if state.updated_at + self.worker_ttl_s <= now:
                continue
            if role == "denoiser" and heartbeat.model_revision != model_revision:
                continue
            if role == "vae" and heartbeat.vae_fingerprint != vae_fingerprint:
                continue
            workers.append(state)
        workers.sort(key=lambda state: state.heartbeat.worker_id)
        return workers

    def _free_slots_locked(
        self, workers: list[_WorkerState]
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

    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
    ) -> SessionAssignment:
        if not all(
            (user_id, session_id, generation_id, model_revision, vae_fingerprint)
        ):
            raise CoordinatorRejected("INVALID_SESSION_IDENTITY")
        async with self._condition:
            now = self._clock()
            self._expire_locked(now)
            if user_id in self._assignments_by_user:
                raise CoordinatorRejected("USER_SESSION_LIMIT")

            denoisers = self._free_slots_locked(
                self._active_workers_locked(
                    role="denoiser",
                    now=now,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                )
            )
            vaes = self._free_slots_locked(
                self._active_workers_locked(
                    role="vae",
                    now=now,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                )
            )
            if not denoisers or not vaes:
                raise CoordinatorRejected("CAPACITY_EXHAUSTED", retry_after_s=0.1)

            denoiser = denoisers[0]
            vae = min(
                vaes,
                key=lambda slot: (
                    slot.az != denoiser.az,
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
        wall_clock: Callable[[], float] = time.time,
        lease_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not table_name:
            raise ValueError("table_name is required")
        if ttl_s <= 0 or worker_ttl_s <= 0:
            raise ValueError("lease TTLs must be positive")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.table_name = table_name
        self.ttl_s = ttl_s
        self.worker_ttl_s = worker_ttl_s
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self.candidate_limit = candidate_limit
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
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        denoiser_offset = int.from_bytes(digest[:8], "big") % len(denoisers)
        vae_offset = int.from_bytes(digest[8:16], "big") % len(vaes)
        ordered_denoisers = (
            denoisers[denoiser_offset:] + denoisers[:denoiser_offset]
        )
        available_vaes = vaes[vae_offset:] + vaes[:vae_offset]
        pairs: list[tuple[WorkerSlot, WorkerSlot]] = []
        for denoiser in ordered_denoisers:
            if not available_vaes:
                break
            vae_index = next(
                (
                    index
                    for index, vae in enumerate(available_vaes)
                    if vae.az == denoiser.az
                ),
                0,
            )
            pairs.append((denoiser, available_vaes.pop(vae_index)))
        return pairs

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        InMemoryCoordinatorStore._validate_heartbeat(heartbeat)
        await asyncio.to_thread(self._heartbeat_sync, heartbeat)

    def _heartbeat_sync(self, heartbeat: WorkerHeartbeat) -> None:
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
                "heartbeat_expires_at": {"N": str(heartbeat_expires)},
                "ttl": {"N": str(heartbeat_expires + 86400)},
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
                    "heartbeat_expires_at = :heartbeat_expires, "
                    "allocation_key = :allocation_key, "
                    "allocation_sort = :allocation_sort, #ttl = :ttl"
                ),
                "ExpressionAttributeNames": {
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
                "heartbeat_expires_at > :now AND "
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
    ) -> SessionAssignment:
        return await asyncio.to_thread(
            self._acquire_sync,
            user_id=user_id,
            session_id=session_id,
            generation_id=generation_id,
            model_revision=model_revision,
            vae_fingerprint=vae_fingerprint,
        )

    def _acquire_sync(
        self,
        *,
        user_id: str,
        session_id: str,
        generation_id: str,
        model_revision: str,
        vae_fingerprint: str,
    ) -> SessionAssignment:
        if not all(
            (user_id, session_id, generation_id, model_revision, vae_fingerprint)
        ):
            raise CoordinatorRejected("INVALID_SESSION_IDENTITY")
        client = self._get_client()
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
        pairs = self._candidate_pairs(
            denoisers,
            vaes,
            identity=f"{user_id}:{session_id}:{generation_id}",
        )
        if not pairs:
            raise CoordinatorRejected("CAPACITY_EXHAUSTED", retry_after_s=0.25)

        for denoiser, vae in pairs:
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
                                "(attribute_not_exists(lease_token) OR "
                                "lease_expires_at <= :now)"
                            ),
                            "ExpressionAttributeNames": {"#ttl": "ttl"},
                            "ExpressionAttributeValues": values,
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
                "vae_worker_id": {"S": vae.worker_id},
                "vae_slot": {"N": str(vae.slot_index)},
                "vae_endpoint": {"S": vae.endpoint},
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
                continue
            return SessionAssignment(
                user_id=user_id,
                session_id=session_id,
                generation_id=generation_id,
                token=token,
                expires_at=self._lease_clock() + self.ttl_s,
                denoiser=denoiser,
                vae=vae,
            )
        raise CoordinatorRejected("CAPACITY_EXHAUSTED", retry_after_s=0.25)

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        return await asyncio.to_thread(self._renew_sync, assignment)

    def _renew_sync(self, assignment: SessionAssignment) -> SessionAssignment:
        client = self._get_client()
        expires_epoch = int(self._wall_clock()) + max(1, int(self.ttl_s))
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
        try:
            client.transact_write_items(
                TransactItems=[
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
        except client.exceptions.TransactionCanceledException as exc:
            raise CoordinatorRejected("LEASE_LOST") from exc
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
    ) -> None:
        self.store = store
        self.wait_timeout_s = max(0.0, wait_timeout_s)

    async def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        await self.store.heartbeat(heartbeat)

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
        deadline = time.monotonic() + self.wait_timeout_s
        while True:
            try:
                return await self.store.acquire(
                    user_id=user_id,
                    session_id=session_id,
                    generation_id=generation_id,
                    model_revision=model_revision,
                    vae_fingerprint=vae_fingerprint,
                )
            except CoordinatorRejected as exc:
                if exc.reason != "CAPACITY_EXHAUSTED" or not wait_for_capacity:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CoordinatorRejected(
                        "CAPACITY_EXHAUSTED", retry_after_s=exc.retry_after_s
                    ) from exc
                wait_s = min(remaining, exc.retry_after_s or 0.1)
                waiter = getattr(self.store, "wait_for_change", None)
                if waiter is None:
                    await asyncio.sleep(wait_s)
                else:
                    await waiter(wait_s)

    async def renew(self, assignment: SessionAssignment) -> SessionAssignment:
        return await self.store.renew(assignment)

    async def release(self, assignment: SessionAssignment) -> None:
        await self.store.release(assignment)
