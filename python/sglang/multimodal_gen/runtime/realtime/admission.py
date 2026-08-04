# SPDX-License-Identifier: Apache-2.0

"""Strict admission leases for realtime generation sessions."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4


class AdmissionRejected(RuntimeError):
    def __init__(self, reason: str, *, retry_after_s: float | None = None) -> None:
        self.reason = reason
        self.retry_after_s = retry_after_s
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SessionLease:
    user_id: str
    session_id: str
    generation_id: str
    token: str
    expires_at: float
    capacity_slot: str | None = None


class SessionLeaseStore(Protocol):
    async def acquire(
        self, user_id: str, session_id: str, generation_id: str
    ) -> SessionLease: ...

    async def renew(self, lease: SessionLease) -> SessionLease: ...

    async def release(self, lease: SessionLease) -> None: ...


class InMemorySessionLeaseStore:
    """Process-local strict leases for a single Gateway replica."""

    def __init__(self, max_active_sessions: int, ttl_s: float) -> None:
        if max_active_sessions < 1:
            raise ValueError("max_active_sessions must be positive")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self.max_active_sessions = max_active_sessions
        self.ttl_s = ttl_s
        self._by_user: dict[str, SessionLease] = {}
        self._condition = asyncio.Condition()

    def _expire_locked(self, now: float) -> bool:
        expired = [
            user_id
            for user_id, lease in self._by_user.items()
            if lease.expires_at <= now
        ]
        for user_id in expired:
            self._by_user.pop(user_id, None)
        return bool(expired)

    async def acquire(
        self, user_id: str, session_id: str, generation_id: str
    ) -> SessionLease:
        if not user_id or not session_id or not generation_id:
            raise AdmissionRejected("INVALID_SESSION_IDENTITY")
        async with self._condition:
            self._expire_locked(time.monotonic())
            if user_id in self._by_user:
                raise AdmissionRejected("USER_SESSION_LIMIT")
            if len(self._by_user) >= self.max_active_sessions:
                raise AdmissionRejected("CAPACITY_EXHAUSTED", retry_after_s=0.1)
            lease = SessionLease(
                user_id=user_id,
                session_id=session_id,
                generation_id=generation_id,
                token=uuid4().hex,
                expires_at=time.monotonic() + self.ttl_s,
            )
            self._by_user[user_id] = lease
            return lease

    async def renew(self, lease: SessionLease) -> SessionLease:
        async with self._condition:
            self._expire_locked(time.monotonic())
            current = self._by_user.get(lease.user_id)
            if current is None or current.token != lease.token:
                raise AdmissionRejected("LEASE_LOST")
            renewed = SessionLease(
                user_id=lease.user_id,
                session_id=lease.session_id,
                generation_id=lease.generation_id,
                token=lease.token,
                expires_at=time.monotonic() + self.ttl_s,
            )
            self._by_user[lease.user_id] = renewed
            return renewed

    async def release(self, lease: SessionLease) -> None:
        async with self._condition:
            current = self._by_user.get(lease.user_id)
            if current is not None and current.token == lease.token:
                self._by_user.pop(lease.user_id, None)
                self._condition.notify_all()

    async def active_count(self) -> int:
        async with self._condition:
            expired = self._expire_locked(time.monotonic())
            if expired:
                self._condition.notify_all()
            return len(self._by_user)

    async def wait_for_change(self, timeout_s: float) -> None:
        async with self._condition:
            try:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout_s)
            except TimeoutError:
                pass


class RealtimeAdmissionController:
    def __init__(
        self,
        store: SessionLeaseStore,
        *,
        wait_timeout_s: float = 10.0,
    ) -> None:
        self.store = store
        self.wait_timeout_s = max(0.0, wait_timeout_s)

    async def admit(
        self, user_id: str, session_id: str, generation_id: str
    ) -> SessionLease:
        deadline = time.monotonic() + self.wait_timeout_s
        while True:
            try:
                return await self.store.acquire(user_id, session_id, generation_id)
            except AdmissionRejected as exc:
                if exc.reason != "CAPACITY_EXHAUSTED":
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdmissionRejected(
                        "CAPACITY_EXHAUSTED", retry_after_s=exc.retry_after_s
                    ) from exc
                wait_s = min(remaining, exc.retry_after_s or 0.1)
                waiter = getattr(self.store, "wait_for_change", None)
                if waiter is None:
                    await asyncio.sleep(wait_s)
                else:
                    await waiter(wait_s)

    async def renew(self, lease: SessionLease) -> SessionLease:
        return await self.store.renew(lease)

    async def release(self, lease: SessionLease) -> None:
        await self.store.release(lease)


class DynamoDBSessionLeaseStore:
    """Optional multi-Gateway store backed by expiring capacity-slot leases.

    The table must use a string partition key named ``lease_key``. This adapter is
    intentionally lazy so boto3 is not required for the default in-memory mode.
    """

    def __init__(
        self,
        table_name: str,
        *,
        max_active_sessions: int,
        ttl_s: float,
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        if not table_name:
            raise ValueError("table_name is required")
        if max_active_sessions < 1:
            raise ValueError("max_active_sessions must be positive")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self.table_name = table_name
        self.max_active_sessions = max_active_sessions
        self.ttl_s = ttl_s
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required for DynamoDB realtime admission"
                ) from exc
            self._client = boto3.client(
                "dynamodb",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
            )
        return self._client

    @staticmethod
    def _user_key(user_id: str) -> str:
        return f"USER#{user_id}"

    @staticmethod
    def _slot_key(index: int) -> str:
        return f"CAPACITY#{index:06d}"

    @staticmethod
    def _is_active_item(item: dict | None, now_epoch: int) -> bool:
        if not item:
            return False
        expires_at = item.get("expires_at", {}).get("N")
        return expires_at is not None and int(expires_at) > now_epoch

    async def acquire(
        self, user_id: str, session_id: str, generation_id: str
    ) -> SessionLease:
        return await asyncio.to_thread(
            self._acquire_sync, user_id, session_id, generation_id
        )

    def _acquire_sync(
        self, user_id: str, session_id: str, generation_id: str
    ) -> SessionLease:
        if not user_id or not session_id or not generation_id:
            raise AdmissionRejected("INVALID_SESSION_IDENTITY")
        now_epoch = int(time.time())
        expires_epoch = now_epoch + max(1, int(self.ttl_s))
        token = uuid4().hex
        client = self._get_client()
        user_key = self._user_key(user_id)

        existing = client.get_item(
            TableName=self.table_name,
            Key={"lease_key": {"S": user_key}},
            ConsistentRead=True,
        ).get("Item")
        if self._is_active_item(existing, now_epoch):
            raise AdmissionRejected("USER_SESSION_LIMIT")

        for slot_index in range(self.max_active_sessions):
            slot_key = self._slot_key(slot_index)
            common_values = {":now": {"N": str(now_epoch)}}
            try:
                client.transact_write_items(
                    TransactItems=[
                        {
                            "Put": {
                                "TableName": self.table_name,
                                "Item": {
                                    "lease_key": {"S": slot_key},
                                    "user_id": {"S": user_id},
                                    "session_id": {"S": session_id},
                                    "generation_id": {"S": generation_id},
                                    "token": {"S": token},
                                    "expires_at": {"N": str(expires_epoch)},
                                },
                                "ConditionExpression": (
                                    "attribute_not_exists(#token) OR expires_at <= :now"
                                ),
                                "ExpressionAttributeNames": {"#token": "token"},
                                "ExpressionAttributeValues": common_values,
                            }
                        },
                        {
                            "Put": {
                                "TableName": self.table_name,
                                "Item": {
                                    "lease_key": {"S": user_key},
                                    "user_id": {"S": user_id},
                                    "session_id": {"S": session_id},
                                    "generation_id": {"S": generation_id},
                                    "capacity_slot": {"S": slot_key},
                                    "token": {"S": token},
                                    "expires_at": {"N": str(expires_epoch)},
                                },
                                "ConditionExpression": (
                                    "attribute_not_exists(#token) OR expires_at <= :now"
                                ),
                                "ExpressionAttributeNames": {"#token": "token"},
                                "ExpressionAttributeValues": common_values,
                            }
                        },
                    ]
                )
            except client.exceptions.TransactionCanceledException as exc:
                existing = client.get_item(
                    TableName=self.table_name,
                    Key={"lease_key": {"S": user_key}},
                    ConsistentRead=True,
                ).get("Item")
                if self._is_active_item(existing, now_epoch):
                    raise AdmissionRejected("USER_SESSION_LIMIT") from exc
                continue
            return SessionLease(
                user_id=user_id,
                session_id=session_id,
                generation_id=generation_id,
                token=token,
                expires_at=time.monotonic() + self.ttl_s,
                capacity_slot=slot_key,
            )

        raise AdmissionRejected("CAPACITY_EXHAUSTED", retry_after_s=1.0)

    async def renew(self, lease: SessionLease) -> SessionLease:
        return await asyncio.to_thread(self._renew_sync, lease)

    def _renew_sync(self, lease: SessionLease) -> SessionLease:
        if lease.capacity_slot is None:
            raise AdmissionRejected("LEASE_LOST")
        expires_epoch = int(time.time()) + max(1, int(self.ttl_s))
        client = self._get_client()
        try:
            values = {
                ":token": {"S": lease.token},
                ":expires": {"N": str(expires_epoch)},
            }
            client.transact_write_items(
                TransactItems=[
                    {
                        "Update": {
                            "TableName": self.table_name,
                            "Key": {
                                "lease_key": {"S": self._user_key(lease.user_id)}
                            },
                            "UpdateExpression": "SET expires_at = :expires",
                            "ConditionExpression": "#token = :token",
                            "ExpressionAttributeNames": {"#token": "token"},
                            "ExpressionAttributeValues": values,
                        }
                    },
                    {
                        "Update": {
                            "TableName": self.table_name,
                            "Key": {"lease_key": {"S": lease.capacity_slot}},
                            "UpdateExpression": "SET expires_at = :expires",
                            "ConditionExpression": "#token = :token",
                            "ExpressionAttributeNames": {"#token": "token"},
                            "ExpressionAttributeValues": values,
                        }
                    },
                ]
            )
        except client.exceptions.TransactionCanceledException as exc:
            raise AdmissionRejected("LEASE_LOST") from exc
        return SessionLease(
            user_id=lease.user_id,
            session_id=lease.session_id,
            generation_id=lease.generation_id,
            token=lease.token,
            expires_at=time.monotonic() + self.ttl_s,
            capacity_slot=lease.capacity_slot,
        )

    async def release(self, lease: SessionLease) -> None:
        await asyncio.to_thread(self._release_sync, lease)

    def _release_sync(self, lease: SessionLease) -> None:
        if lease.capacity_slot is None:
            return
        client = self._get_client()
        try:
            client.transact_write_items(
                TransactItems=[
                    {
                        "Delete": {
                            "TableName": self.table_name,
                            "Key": {"lease_key": {"S": self._user_key(lease.user_id)}},
                            "ConditionExpression": "#token = :token",
                            "ExpressionAttributeNames": {"#token": "token"},
                            "ExpressionAttributeValues": {
                                ":token": {"S": lease.token}
                            },
                        }
                    },
                    {
                        "Delete": {
                            "TableName": self.table_name,
                            "Key": {"lease_key": {"S": lease.capacity_slot}},
                            "ConditionExpression": "#token = :token",
                            "ExpressionAttributeNames": {"#token": "token"},
                            "ExpressionAttributeValues": {
                                ":token": {"S": lease.token}
                            },
                        }
                    },
                ]
            )
        except client.exceptions.TransactionCanceledException:
            # Release is deliberately idempotent; a lost/expired token owns nothing.
            return
