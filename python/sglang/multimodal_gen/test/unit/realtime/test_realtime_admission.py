# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.admission import (
    AdmissionRejected,
    DynamoDBSessionLeaseStore,
    InMemorySessionLeaseStore,
    RealtimeAdmissionController,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)


def test_one_active_session_per_user_and_idempotent_release():
    async def scenario():
        store = InMemorySessionLeaseStore(max_active_sessions=2, ttl_s=60)
        lease = await store.acquire("u1", "s1", "g1")

        with pytest.raises(AdmissionRejected, match="USER_SESSION_LIMIT"):
            await store.acquire("u1", "s2", "g2")

        await store.release(lease)
        await store.release(lease)
        assert await store.active_count() == 0

    asyncio.run(scenario())


def test_expired_lease_is_reclaimed():
    async def scenario():
        store = InMemorySessionLeaseStore(max_active_sessions=1, ttl_s=0.01)
        await store.acquire("u1", "s1", "g1")
        await asyncio.sleep(0.02)

        lease = await store.acquire("u2", "s2", "g2")

        assert lease.user_id == "u2"
        assert await store.active_count() == 1

    asyncio.run(scenario())


def test_controller_waits_for_capacity_but_not_same_user_limit():
    async def scenario():
        store = InMemorySessionLeaseStore(max_active_sessions=1, ttl_s=60)
        controller = RealtimeAdmissionController(store, wait_timeout_s=0.2)
        first = await controller.admit("u1", "s1", "g1")

        waiter = asyncio.create_task(controller.admit("u2", "s2", "g2"))
        await asyncio.sleep(0.02)
        await controller.release(first)

        second = await waiter
        assert second.user_id == "u2"

        with pytest.raises(AdmissionRejected, match="USER_SESSION_LIMIT"):
            await controller.admit("u2", "s3", "g3")

        await controller.release(second)

    asyncio.run(scenario())


def test_different_users_run_concurrently_but_same_user_is_rejected():
    async def scenario():
        controller = RealtimeAdmissionController(
            InMemorySessionLeaseStore(max_active_sessions=2, ttl_s=60)
        )
        first = await controller.admit("u1", "s1", "g1")
        second = await controller.admit("u2", "s2", "g2")
        with pytest.raises(AdmissionRejected, match="USER_SESSION_LIMIT"):
            await controller.admit("u1", "s3", "g3")
        await controller.release(first)
        await controller.release(second)

    asyncio.run(scenario())


def test_chunk_snapshots_latest_action_and_prompt_versions():
    session = GenerateSession(max_inflight_chunks=2)
    session.mark_event_version("camera_actions")
    first = session.new_chunk()
    session.mark_event_version("prompt")
    session.mark_event_version("camera_actions")
    second = session.new_chunk()

    assert (first.action_version, first.prompt_version) == (1, 0)
    assert (second.action_version, second.prompt_version) == (2, 1)


def test_dynamodb_capacity_slot_is_reclaimed_after_gateway_lease_expires(monkeypatch):
    class TransactionCanceledException(Exception):
        pass

    class FakeDynamoClient:
        exceptions = SimpleNamespace(
            TransactionCanceledException=TransactionCanceledException,
            ConditionalCheckFailedException=TransactionCanceledException,
        )

        def __init__(self):
            self.items = {}

        def get_item(self, *, TableName, Key, ConsistentRead):
            del TableName, ConsistentRead
            item = self.items.get(Key["lease_key"]["S"])
            return {"Item": deepcopy(item)} if item is not None else {}

        def transact_write_items(self, *, TransactItems):
            staged = deepcopy(self.items)
            try:
                for operation in TransactItems:
                    if "Put" in operation:
                        request = operation["Put"]
                        item = deepcopy(request["Item"])
                        key = item["lease_key"]["S"]
                        existing = staged.get(key)
                        now = int(
                            request["ExpressionAttributeValues"][":now"]["N"]
                        )
                        if (
                            existing is not None
                            and int(existing["expires_at"]["N"]) > now
                        ):
                            raise TransactionCanceledException
                        staged[key] = item
                    elif "Update" in operation:
                        request = operation["Update"]
                        key = request["Key"]["lease_key"]["S"]
                        existing = staged.get(key)
                        token = request["ExpressionAttributeValues"][":token"]["S"]
                        if existing is None or existing["token"]["S"] != token:
                            raise TransactionCanceledException
                        existing["expires_at"] = deepcopy(
                            request["ExpressionAttributeValues"][":expires"]
                        )
                    elif "Delete" in operation:
                        request = operation["Delete"]
                        key = request["Key"]["lease_key"]["S"]
                        existing = staged.get(key)
                        token = request["ExpressionAttributeValues"][":token"]["S"]
                        if existing is None or existing["token"]["S"] != token:
                            raise TransactionCanceledException
                        staged.pop(key)
                    else:
                        raise AssertionError("capacity leases must use fixed slot items")
            except TransactionCanceledException:
                raise
            self.items = staged

        def update_item(self, **kwargs):
            raise AssertionError(f"renew must update user and slot atomically: {kwargs}")

    now = [1_000]
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.realtime.admission.time.time",
        lambda: now[0],
    )
    client = FakeDynamoClient()
    store = DynamoDBSessionLeaseStore(
        "leases",
        max_active_sessions=1,
        ttl_s=10,
    )
    store._client = client

    first = store._acquire_sync("user-a", "session-a", "generation-a")
    assert first.capacity_slot == "CAPACITY#000000"
    with pytest.raises(AdmissionRejected, match="CAPACITY_EXHAUSTED"):
        store._acquire_sync("user-b", "session-b", "generation-b")

    now[0] = 1_011
    second = store._acquire_sync("user-b", "session-b", "generation-b")
    assert second.capacity_slot == first.capacity_slot
    store._release_sync(first)
    assert client.items[second.capacity_slot]["token"]["S"] == second.token

    renewed = store._renew_sync(second)
    assert renewed.capacity_slot == second.capacity_slot
    store._release_sync(renewed)
    assert second.capacity_slot not in client.items
