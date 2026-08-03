# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from sglang.multimodal_gen.runtime.realtime.admission import (
    AdmissionRejected,
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
