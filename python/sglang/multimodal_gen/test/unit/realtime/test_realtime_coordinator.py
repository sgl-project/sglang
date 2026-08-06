# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

import pytest

from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    DynamoDBCoordinatorStore,
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
    SessionAssignment,
    WorkerHeartbeat,
    WorkerSlot,
)


def _heartbeat(
    worker_id: str,
    role: str,
    *,
    capacity: int = 1,
    az: str = "us-east-2a",
    model_revision: str = "minwm-r1",
    vae_fingerprint: str = "taew2_2",
    worker_epoch: str = "epoch-a",
    lifecycle: str = "ready",
    active_sessions: int = 0,
    queue_depth: int = 0,
    service_time_ms: float = 0,
    drain_deadline: float | None = None,
):
    return WorkerHeartbeat(
        worker_id=worker_id,
        role=role,
        endpoint=f"ws://{worker_id}.cluster.local/generate",
        az=az,
        capacity=capacity,
        model_revision=model_revision,
        vae_fingerprint=vae_fingerprint,
        worker_epoch=worker_epoch,
        lifecycle=lifecycle,
        active_sessions=active_sessions,
        runnable_sessions=active_sessions,
        blocked_sessions=0,
        queue_depth=queue_depth,
        service_time_ms=service_time_ms,
        reservation_endpoint=f"http://{worker_id}.cluster.local/v1/realtime_worker",
        drain_deadline=drain_deadline,
    )


def test_coordinator_atomically_pairs_compatible_worker_slots():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        coordinator = RealtimeCoordinator(store, wait_timeout_s=0)
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser", capacity=2))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae", capacity=2))

        assignment = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.denoiser.worker_id == "denoiser-a"
        assert assignment.vae.worker_id == "vae-a"
        assert assignment.denoiser.slot_index == 0
        assert assignment.vae.slot_index == 0
        assert assignment.token
        return coordinator, assignment

    coordinator, assignment = asyncio.run(run())
    assert coordinator is not None
    assert assignment.session_id == "session-a"


def test_capacity_snapshot_combines_waiting_load_free_slots_and_drain_state():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        coordinator = RealtimeCoordinator(store, wait_timeout_s=0)
        await coordinator.heartbeat(
            _heartbeat(
                "denoiser-ready",
                "denoiser",
                capacity=4,
                active_sessions=3,
                queue_depth=2,
                service_time_ms=450,
            )
        )
        await coordinator.heartbeat(
            _heartbeat(
                "denoiser-draining",
                "denoiser",
                capacity=4,
                lifecycle="draining",
                active_sessions=1,
            )
        )
        await coordinator.heartbeat(
            _heartbeat(
                "vae-ready",
                "vae",
                capacity=16,
                active_sessions=4,
                queue_depth=1,
                service_time_ms=30,
            )
        )
        await store.waiting_started("waiter-a")

        snapshot = await coordinator.capacity_snapshot()
        await store.waiting_finished("waiter-a")
        after = await coordinator.capacity_snapshot()

        assert snapshot["roles"]["denoiser"] == {
            "waiting_sessions": 1,
            "active_sessions": 4,
            "queued_sessions": 2,
            "free_slots": 1,
            "draining_workers": 1,
        }
        assert snapshot["roles"]["vae"] == {
            "waiting_sessions": 1,
            "active_sessions": 4,
            "queued_sessions": 1,
            "free_slots": 12,
            "draining_workers": 0,
        }
        assert after["roles"]["denoiser"]["waiting_sessions"] == 0

    asyncio.run(run())


def test_dynamodb_capacity_snapshot_uses_shared_ttl_demand_records():
    class FakeClient:
        def __init__(self):
            self.puts = []
            self.deletes = []
            self.queries = []

        def put_item(self, **kwargs):
            self.puts.append(kwargs)

        def delete_item(self, **kwargs):
            self.deletes.append(kwargs)

        def query(self, **kwargs):
            self.queries.append(kwargs)
            role = kwargs["ExpressionAttributeValues"][":allocation"]["S"].split(
                "#", 1
            )[1]
            capacity = 4 if role == "denoiser" else 16
            active = 3 if role == "denoiser" else 4
            return {
                "Items": [
                    {
                        "item_type": {"S": "worker"},
                        "role": {"S": role},
                        "worker_id": {"S": f"{role}-a"},
                        "lifecycle": {"S": "ready"},
                        "capacity": {"N": str(capacity)},
                        "active_sessions": {"N": str(active)},
                        "queue_depth": {"N": "1"},
                        "heartbeat_expires_at": {"N": "200"},
                    },
                    {
                        "item_type": {"S": "capacity_demand"},
                        "demand_expires_at": {"N": "160"},
                    },
                ]
            }

    async def run():
        client = FakeClient()
        store = DynamoDBCoordinatorStore(
            "minwm-realtime-coordinator",
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 100,
            client=client,
        )
        await store.waiting_started("waiter-a")
        snapshot = await store.capacity_snapshot()
        await store.waiting_finished("waiter-a")

        assert len(client.puts) == 2
        assert {
            item["Item"]["allocation_key"]["S"] for item in client.puts
        } == {"CAPACITY#denoiser", "CAPACITY#vae"}
        assert len(client.queries) == 2
        assert snapshot["roles"]["denoiser"]["waiting_sessions"] == 1
        assert snapshot["roles"]["denoiser"]["free_slots"] == 1
        assert snapshot["roles"]["vae"]["free_slots"] == 12
        assert len(client.deletes) == 2

    asyncio.run(run())


def test_coordinator_rejects_second_session_for_the_same_user():
    async def run():
        coordinator = RealtimeCoordinator(
            InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30),
            wait_timeout_s=0,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser", capacity=2))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae", capacity=2))
        await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        with pytest.raises(CoordinatorRejected, match="USER_SESSION_LIMIT"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-b",
                generation_id="generation-b",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )

    asyncio.run(run())


def test_coordinator_does_not_leak_a_partial_worker_reservation():
    async def run():
        coordinator = RealtimeCoordinator(
            InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30),
            wait_timeout_s=0,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )

        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        assignment = await coordinator.admit(
            user_id="user-b",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert assignment.denoiser.slot_index == 0

    asyncio.run(run())


def test_coordinator_prefers_same_az_and_filters_incompatible_workers():
    async def run():
        coordinator = RealtimeCoordinator(
            InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30),
            wait_timeout_s=0,
        )
        await coordinator.heartbeat(
            _heartbeat("denoiser-a", "denoiser", az="us-east-2a")
        )
        await coordinator.heartbeat(
            _heartbeat("vae-wrong", "vae", az="us-east-2a", vae_fingerprint="wrong")
        )
        await coordinator.heartbeat(
            _heartbeat("vae-cross-az", "vae", az="us-east-2b")
        )
        await coordinator.heartbeat(
            _heartbeat("vae-same-az", "vae", az="us-east-2a")
        )

        assignment = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert assignment.vae.worker_id == "vae-same-az"

    asyncio.run(run())


def test_coordinator_excludes_draining_workers_from_new_allocations():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        await store.heartbeat(
            _heartbeat("denoiser-draining", "denoiser", lifecycle="draining")
        )
        await store.heartbeat(_heartbeat("denoiser-ready", "denoiser"))
        await store.heartbeat(_heartbeat("vae-ready", "vae"))

        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.denoiser.worker_id == "denoiser-ready"

    asyncio.run(run())


def test_coordinator_routes_to_lower_normalized_load_queue_and_service_time():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        await store.heartbeat(
            _heartbeat(
                "denoiser-loaded",
                "denoiser",
                capacity=4,
                active_sessions=3,
                queue_depth=2,
                service_time_ms=20,
            )
        )
        await store.heartbeat(
            _heartbeat(
                "denoiser-light",
                "denoiser",
                capacity=4,
                active_sessions=1,
                queue_depth=0,
                service_time_ms=5,
            )
        )
        await store.heartbeat(_heartbeat("vae-a", "vae", capacity=4))

        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.denoiser.worker_id == "denoiser-light"

    asyncio.run(run())


def test_coordinator_waiting_admission_wakes_when_assignment_is_released():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        coordinator = RealtimeCoordinator(store, wait_timeout_s=1)
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        first = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        waiting = asyncio.create_task(
            coordinator.admit(
                user_id="user-b",
                session_id="session-b",
                generation_id="generation-b",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )
        )
        await asyncio.sleep(0.01)
        assert not waiting.done()
        assert (
            (await coordinator.capacity_snapshot())["roles"]["denoiser"][
                "waiting_sessions"
            ]
            == 1
        )

        await coordinator.release(first)
        second = await asyncio.wait_for(waiting, timeout=0.5)
        assert second.session_id == "session-b"
        assert (
            (await coordinator.capacity_snapshot())["roles"]["denoiser"][
                "waiting_sessions"
            ]
            == 0
        )

    asyncio.run(run())


def test_coordinator_renew_fences_worker_restart_and_heartbeat_loss():
    async def run():
        now = [100.0]
        store = InMemoryCoordinatorStore(
            ttl_s=60,
            worker_ttl_s=5,
            clock=lambda: now[0],
        )
        await store.heartbeat(
            _heartbeat("denoiser-a", "denoiser", worker_epoch="epoch-old")
        )
        await store.heartbeat(_heartbeat("vae-a", "vae"))
        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        await store.heartbeat(
            _heartbeat("denoiser-a", "denoiser", worker_epoch="epoch-new")
        )
        with pytest.raises(CoordinatorRejected, match="WORKER_LOST"):
            await store.renew(assignment)

        await store.heartbeat(
            _heartbeat("denoiser-a", "denoiser", worker_epoch="epoch-old")
        )
        now[0] = 106.0
        with pytest.raises(CoordinatorRejected, match="WORKER_LOST"):
            await store.renew(assignment)

    asyncio.run(run())


def test_coordinator_allows_draining_worker_renew_only_before_deadline():
    async def run():
        now = [100.0]
        wall_now = [1_000.0]
        store = InMemoryCoordinatorStore(
            ttl_s=60,
            worker_ttl_s=30,
            clock=lambda: now[0],
            wall_clock=lambda: wall_now[0],
        )
        await store.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await store.heartbeat(_heartbeat("vae-a", "vae"))
        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        await store.heartbeat(
            _heartbeat(
                "denoiser-a",
                "denoiser",
                lifecycle="draining",
                drain_deadline=1_010.0,
            )
        )

        assignment = await store.renew(assignment)
        wall_now[0] = 1_011.0
        with pytest.raises(CoordinatorRejected, match="WORKER_LOST"):
            await store.renew(assignment)

    asyncio.run(run())


def test_coordinator_partial_worker_reserve_rolls_back_and_retries_another_pair():
    class ReservationClient:
        def __init__(self):
            self.reserved = []
            self.released = []

        async def reserve(self, slot, **identity):
            self.reserved.append((slot.worker_id, identity["token"]))
            if slot.worker_id == "vae-bad":
                raise RuntimeError("worker rejected reservation")

        async def release(self, slot, *, token):
            self.released.append((slot.worker_id, token))

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        reservations = ReservationClient()
        coordinator = RealtimeCoordinator(
            store,
            wait_timeout_s=1,
            reservation_client=reservations,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(
            _heartbeat("vae-bad", "vae", service_time_ms=0)
        )
        await coordinator.heartbeat(
            _heartbeat("vae-good", "vae", service_time_ms=1)
        )

        assignment = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.vae.worker_id == "vae-good"
        failed_token = reservations.reserved[0][1]
        assert ("denoiser-a", failed_token) in reservations.released
        assert ("vae-bad", failed_token) in reservations.released
        assert assignment.token != failed_token

    asyncio.run(run())


def test_coordinator_expires_stale_workers_and_reclaims_expired_assignments():
    async def run():
        now = [100.0]
        store = InMemoryCoordinatorStore(
            ttl_s=5,
            worker_ttl_s=3,
            clock=lambda: now[0],
        )
        coordinator = RealtimeCoordinator(store, wait_timeout_s=0)
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        first = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        now[0] = 104.0
        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-b",
                session_id="session-b",
                generation_id="generation-b",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )

        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        now[0] = 106.0
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        second = await coordinator.admit(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert second.token != first.token

    asyncio.run(run())


def test_coordinator_renew_and_release_are_fenced_and_idempotent():
    async def run():
        coordinator = RealtimeCoordinator(
            InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30),
            wait_timeout_s=0,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        assignment = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        renewed = await coordinator.renew(assignment)
        assert renewed.token == assignment.token
        assert renewed.expires_at >= assignment.expires_at

        await coordinator.release(renewed)
        await coordinator.release(renewed)
        replacement = await coordinator.admit(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"

    asyncio.run(run())


def test_dynamodb_coordinator_admission_is_one_four_item_transaction():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.transactions = []

        def query(self, **kwargs):
            allocation_key = kwargs["ExpressionAttributeValues"][":allocation"]["S"]
            if allocation_key.startswith("DENOISER#"):
                role = "denoiser"
                worker_id = "denoiser-a"
                endpoint = "ws://denoiser-a/generate"
            else:
                role = "vae"
                worker_id = "vae-a"
                endpoint = "ws://vae-a/decode"
            return {
                "Items": [
                    {
                        "pk": {"S": f"SLOT#{role}#{worker_id}#0000"},
                        "sk": {"S": "LEASE"},
                        "role": {"S": role},
                        "worker_id": {"S": worker_id},
                        "endpoint": {"S": endpoint},
                        "az": {"S": "us-east-2a"},
                        "slot_index": {"N": "0"},
                        "model_revision": {"S": "minwm-r1"},
                        "vae_fingerprint": {"S": "taew2_2"},
                        "heartbeat_expires_at": {"N": "9999999999"},
                    }
                ]
            }

        def transact_write_items(self, *, TransactItems):
            self.transactions.append(TransactItems)

    async def run():
        client = FakeClient()
        store = DynamoDBCoordinatorStore(
            "minwm-realtime-coordinator",
            ttl_s=60,
            worker_ttl_s=30,
            client=client,
        )
        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert assignment.denoiser.worker_id == "denoiser-a"
        assert assignment.vae.worker_id == "vae-a"
        assert len(client.transactions) == 1
        transaction = client.transactions[0]
        assert len(transaction) == 4
        keys = {
            item["Put"]["Item"]["pk"]["S"]
            if "Put" in item
            else item["Update"]["Key"]["pk"]["S"]
            for item in transaction
        }
        assert keys == {
            "USER#user-a",
            "SESSION#session-a",
            "SLOT#denoiser#denoiser-a#0000",
            "SLOT#vae#vae-a#0000",
        }

    asyncio.run(run())


def test_dynamodb_slot_query_paginates_past_filtered_stale_slots():
    class FakeClient:
        def __init__(self):
            self.queries = []

        def query(self, **kwargs):
            self.queries.append(kwargs)
            if "ExclusiveStartKey" not in kwargs:
                return {
                    "Items": [],
                    "LastEvaluatedKey": {
                        "allocation_key": {"S": "DENOISER#minwm-r1"},
                        "allocation_sort": {"S": "stale-worker#0001"},
                    },
                }
            return {
                "Items": [
                    {
                        "pk": {"S": "SLOT#denoiser#denoiser-live#0000"},
                        "sk": {"S": "LEASE"},
                        "role": {"S": "denoiser"},
                        "worker_id": {"S": "denoiser-live"},
                        "endpoint": {"S": "ws://denoiser-live/generate"},
                        "az": {"S": "us-east-2a"},
                        "slot_index": {"N": "0"},
                        "model_revision": {"S": "minwm-r1"},
                        "vae_fingerprint": {"S": "taew2_2"},
                        "heartbeat_expires_at": {"N": "9999999999"},
                    }
                ]
            }

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
        candidate_limit=2,
    )

    slots = store._query_slots_sync(
        "denoiser",
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        now_epoch=100,
    )

    assert [slot.worker_id for slot in slots] == ["denoiser-live"]
    assert len(client.queries) == 2
    assert client.queries[1]["ExclusiveStartKey"] == {
        "allocation_key": {"S": "DENOISER#minwm-r1"},
        "allocation_sort": {"S": "stale-worker#0001"},
    }


def test_dynamodb_candidate_pairing_spreads_each_burst_in_stable_order():
    denoisers = [
        WorkerSlot(
            worker_id=f"denoiser-{index}",
            role="denoiser",
            endpoint=f"ws://denoiser-{index}/generate",
            az="us-east-2a",
            slot_index=0,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        for index in range(8)
    ]
    vaes = [
        WorkerSlot(
            worker_id=f"vae-{index}",
            role="vae",
            endpoint=f"ws://vae-{index}/decode",
            az="us-east-2a",
            slot_index=0,
            model_revision="all",
            vae_fingerprint="taew2_2",
        )
        for index in range(8)
    ]

    first_denoisers = set()
    for index in range(32):
        pairs = DynamoDBCoordinatorStore._candidate_pairs(
            denoisers,
            vaes,
            identity=f"user-{index}:session-{index}:generation-{index}",
        )
        assert len(pairs) == 8
        assert len({pair[0].worker_id for pair in pairs}) == 8
        assert len({pair[1].worker_id for pair in pairs}) == 8
        first_denoisers.add(pairs[0][0].worker_id)

    assert len(first_denoisers) == 1


def test_dynamodb_candidate_pairing_prefers_worker_with_more_free_slots():
    def slots(worker_id: str, free_slots: int) -> list[WorkerSlot]:
        return [
            WorkerSlot(
                worker_id=worker_id,
                role="denoiser",
                endpoint=f"ws://{worker_id}/generate",
                az="us-east-2a",
                slot_index=index,
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
                active_sessions=0,
                capacity=4,
            )
            for index in range(free_slots)
        ]

    denoisers = slots("denoiser-mostly-busy", 1) + slots(
        "denoiser-idle", 4
    )
    vaes = [
        WorkerSlot(
            worker_id="vae-a",
            role="vae",
            endpoint="ws://vae-a/decode",
            az="us-east-2a",
            slot_index=0,
            model_revision="all",
            vae_fingerprint="taew2_2",
        )
    ]

    pairs = DynamoDBCoordinatorStore._candidate_pairs(
        denoisers,
        vaes,
        identity="user-1:session-1:generation-1",
    )

    assert pairs[0][0].worker_id == "denoiser-idle"


def test_dynamodb_candidate_pairing_exhausts_worker_layer_before_next_slot():
    denoisers = [
        WorkerSlot(
            worker_id=f"denoiser-{worker_index}",
            role="denoiser",
            endpoint=f"ws://denoiser-{worker_index}/generate",
            az="us-east-2a",
            slot_index=slot_index,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            capacity=4,
        )
        for worker_index in range(8)
        for slot_index in range(4)
    ]
    vaes = [
        WorkerSlot(
            worker_id="vae-a",
            role="vae",
            endpoint="ws://vae-a/decode",
            az="us-east-2a",
            slot_index=slot_index,
            model_revision="all",
            vae_fingerprint="taew2_2",
            capacity=16,
        )
        for slot_index in range(16)
    ]

    pairs = DynamoDBCoordinatorStore._candidate_pairs(
        denoisers,
        vaes,
        identity="burst-a",
    )

    first_layer = [pair[0] for pair in pairs[:8]]
    assert {slot.worker_id for slot in first_layer} == {
        f"denoiser-{index}" for index in range(8)
    }
    assert {slot.slot_index for slot in first_layer} == {0}

    competing_pairs = DynamoDBCoordinatorStore._candidate_pairs(
        denoisers,
        vaes,
        identity="burst-b",
    )
    assert [
        (pair[0].worker_id, pair[0].slot_index, pair[1].slot_index)
        for pair in competing_pairs
    ] == [
        (pair[0].worker_id, pair[0].slot_index, pair[1].slot_index)
        for pair in pairs
    ]


def test_dynamodb_admission_requeries_after_a_stale_candidate_snapshot():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.query_counts = {"denoiser": 0, "vae": 0}
            self.transactions = 0

        @staticmethod
        def _slot(role, index):
            worker_id = f"{role}-{index}"
            return {
                "pk": {"S": f"SLOT#{role}#{worker_id}#{index:04d}"},
                "sk": {"S": "LEASE"},
                "role": {"S": role},
                "worker_id": {"S": worker_id},
                "endpoint": {"S": f"ws://{worker_id}/generate"},
                "az": {"S": "us-east-2a"},
                "slot_index": {"N": str(index)},
                "model_revision": {"S": "minwm-r1"},
                "vae_fingerprint": {"S": "taew2_2"},
                "heartbeat_expires_at": {"N": "9999999999"},
            }

        def query(self, **kwargs):
            allocation_key = kwargs["ExpressionAttributeValues"][":allocation"]["S"]
            role = "denoiser" if allocation_key.startswith("DENOISER#") else "vae"
            self.query_counts[role] += 1
            indices = [0, 1] if self.query_counts[role] == 1 else [1]
            return {"Items": [self._slot(role, index) for index in indices]}

        def transact_write_items(self, *, TransactItems):
            self.transactions += 1
            if self.transactions <= 2:
                raise TransactionCanceledException("candidate was leased concurrently")

    async def run():
        client = FakeClient()
        store = DynamoDBCoordinatorStore(
            "minwm-realtime-coordinator",
            ttl_s=60,
            worker_ttl_s=30,
            client=client,
        )
        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.denoiser.worker_id == "denoiser-1"
        assert assignment.vae.worker_id == "vae-1"
        assert client.query_counts == {"denoiser": 2, "vae": 2}
        assert client.transactions == 3

    asyncio.run(run())


def test_dynamodb_heartbeat_retries_a_transient_transaction_conflict():
    class TransactionConflictException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionConflictException = TransactionConflictException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.slot_updates = 0

        def put_item(self, **kwargs):
            return None

        def update_item(self, **kwargs):
            self.slot_updates += 1
            if self.slot_updates == 1:
                raise TransactionConflictException("transaction in progress")

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
    )

    store._heartbeat_sync(_heartbeat("denoiser-a", "denoiser"))

    assert client.slot_updates == 2


def test_dynamodb_heartbeat_clamps_advertised_denoiser_capacity():
    class TransactionConflictException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionConflictException = TransactionConflictException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.worker_item = None
            self.slot_updates = []

        def put_item(self, *, Item, **_kwargs):
            self.worker_item = Item

        def update_item(self, **kwargs):
            self.slot_updates.append(kwargs)

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
        capacity_limits={"denoiser": 1},
    )

    store._heartbeat_sync(_heartbeat("denoiser-a", "denoiser", capacity=4))

    assert client.worker_item["capacity"] == {"N": "1"}
    assert len(client.slot_updates) == 1
    assert client.slot_updates[0]["ExpressionAttributeValues"][":capacity"] == {
        "N": "1"
    }


def test_dynamodb_heartbeat_persists_epoch_lifecycle_and_worker_load():
    class TransactionConflictException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionConflictException = TransactionConflictException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.worker_item = None
            self.slot_update = None

        def put_item(self, **kwargs):
            self.worker_item = kwargs["Item"]

        def update_item(self, **kwargs):
            self.slot_update = kwargs

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
    )
    heartbeat = _heartbeat(
        "denoiser-a",
        "denoiser",
        active_sessions=2,
        queue_depth=3,
        service_time_ms=12.5,
    )

    store._heartbeat_sync(heartbeat)

    assert client.worker_item["worker_epoch"] == {"S": "epoch-a"}
    assert client.worker_item["lifecycle"] == {"S": "ready"}
    assert client.worker_item["active_sessions"] == {"N": "2"}
    values = client.slot_update["ExpressionAttributeValues"]
    names = client.slot_update["ExpressionAttributeNames"]
    assert names["#capacity"] == "capacity"
    assert "#capacity = :capacity" in client.slot_update["UpdateExpression"]
    assert values[":worker_epoch"] == {"S": "epoch-a"}
    assert values[":queue_depth"] == {"N": "3"}
    assert values[":service_time_ms"] == {"N": "12.5"}


def test_dynamodb_renew_condition_checks_current_worker_epochs_and_expiry():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.transaction = None

        def transact_write_items(self, *, TransactItems):
            self.transaction = TransactItems

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
    )
    assignment = SessionAssignment(
        user_id="user-a",
        session_id="session-a",
        generation_id="generation-a",
        token="token-a",
        expires_at=time.monotonic() + 30,
        denoiser=WorkerSlot(
            worker_id="denoiser-a",
            role="denoiser",
            endpoint="ws://denoiser-a/generate",
            az="us-east-2a",
            slot_index=0,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            worker_epoch="denoiser-epoch",
        ),
        vae=WorkerSlot(
            worker_id="vae-a",
            role="vae",
            endpoint="ws://vae-a/decode",
            az="us-east-2a",
            slot_index=0,
            model_revision="all",
            vae_fingerprint="taew2_2",
            worker_epoch="vae-epoch",
        ),
    )

    store._renew_sync(assignment)

    checks = [item["ConditionCheck"] for item in client.transaction if "ConditionCheck" in item]
    assert len(checks) == 2
    assert {check["Key"]["pk"]["S"] for check in checks} == {
        "WORKER#denoiser-a",
        "WORKER#vae-a",
    }
    epochs = {
        check["ExpressionAttributeValues"][":worker_epoch"]["S"]
        for check in checks
    }
    assert epochs == {"denoiser-epoch", "vae-epoch"}
    assert all("heartbeat_expires_at > :now" in check["ConditionExpression"] for check in checks)


def test_dynamodb_renew_retries_a_transient_transaction_conflict():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.transactions = 0

        def transact_write_items(self, **_kwargs):
            self.transactions += 1
            if self.transactions < 5:
                raise TransactionCanceledException("heartbeat write conflict")

        def get_item(self, *, Key, **_kwargs):
            pk = Key["pk"]["S"]
            if pk.startswith("WORKER#"):
                worker_id = pk.removeprefix("WORKER#")
                return {
                    "Item": {
                        "worker_epoch": {"S": f"{worker_id}-epoch"},
                        "heartbeat_expires_at": {"N": "9999999999"},
                        "lifecycle": {"S": "ready"},
                    }
                }
            return {"Item": {"lease_token": {"S": "token-a"}}}

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
    )
    assignment = SessionAssignment(
        user_id="user-a",
        session_id="session-a",
        generation_id="generation-a",
        token="token-a",
        expires_at=time.monotonic() + 30,
        denoiser=WorkerSlot(
            "denoiser-a",
            "denoiser",
            "ws://denoiser-a/generate",
            "us-east-2a",
            0,
            "minwm-r1",
            "taew2_2",
            worker_epoch="denoiser-a-epoch",
        ),
        vae=WorkerSlot(
            "vae-a",
            "vae",
            "ws://vae-a/decode",
            "us-east-2a",
            0,
            "all",
            "taew2_2",
            worker_epoch="vae-a-epoch",
        ),
    )

    renewed = store._renew_sync(assignment)

    assert renewed.expires_at > assignment.expires_at
    assert client.transactions == 5


def test_dynamodb_renew_classifies_failed_worker_as_worker_lost():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def transact_write_items(self, **_kwargs):
            raise TransactionCanceledException("worker failed")

        def get_item(self, *, Key, **_kwargs):
            worker_id = Key["pk"]["S"].removeprefix("WORKER#")
            return {
                "Item": {
                    "worker_epoch": {"S": f"{worker_id}-epoch"},
                    "heartbeat_expires_at": {"N": "9999999999"},
                    "lifecycle": {"S": "failed"},
                }
            }

    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=FakeClient(),
    )
    assignment = SessionAssignment(
        user_id="user-a",
        session_id="session-a",
        generation_id="generation-a",
        token="token-a",
        expires_at=time.monotonic() + 30,
        denoiser=WorkerSlot(
            "denoiser-a",
            "denoiser",
            "ws://denoiser-a/generate",
            "us-east-2a",
            0,
            "minwm-r1",
            "taew2_2",
            worker_epoch="denoiser-a-epoch",
        ),
        vae=WorkerSlot(
            "vae-a",
            "vae",
            "ws://vae-a/decode",
            "us-east-2a",
            0,
            "all",
            "taew2_2",
            worker_epoch="vae-a-epoch",
        ),
    )

    with pytest.raises(CoordinatorRejected, match="WORKER_LOST"):
        store._renew_sync(assignment)


def test_coordinator_cancellation_compensates_assignment_and_partial_reservations():
    class ReservationClient:
        def __init__(self):
            self.vae_started = asyncio.Event()
            self.never = asyncio.Event()
            self.released = []

        async def reserve(self, slot, **_identity):
            if slot.role == "vae":
                self.vae_started.set()
                await self.never.wait()

        async def release(self, slot, *, token):
            self.released.append((slot.role, token))

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        reservations = ReservationClient()
        coordinator = RealtimeCoordinator(
            store,
            wait_timeout_s=5,
            reservation_client=reservations,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))

        task = asyncio.create_task(
            coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )
        )
        await reservations.vae_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert {role for role, _token in reservations.released} == {
            "denoiser",
            "vae",
        }
        replacement = await store.acquire(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"
        assert (await coordinator.capacity_snapshot())["roles"]["denoiser"][
            "waiting_sessions"
        ] == 0

    asyncio.run(run())


def test_coordinator_compensation_retries_transient_worker_release_failures():
    class ReservationClient:
        def __init__(self):
            self.release_attempts = {"denoiser": 0, "vae": 0}

        async def reserve(self, slot, **_identity):
            if slot.role == "vae":
                raise RuntimeError("reserve failed")

        async def release(self, slot, *, token):
            del token
            self.release_attempts[slot.role] += 1
            if slot.role == "denoiser" and self.release_attempts[slot.role] < 3:
                raise RuntimeError("transient release failure")

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        reservations = ReservationClient()
        coordinator = RealtimeCoordinator(
            store,
            wait_timeout_s=0.25,
            reservation_client=reservations,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))

        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
                wait_for_capacity=False,
            )

        assert reservations.release_attempts["denoiser"] == 3
        assert reservations.release_attempts["vae"] == 1
        replacement = await store.acquire(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"

    asyncio.run(run())


def test_coordinator_deadline_covers_worker_reserve_and_compensates_assignment():
    class ReservationClient:
        def __init__(self):
            self.released = []

        async def reserve(self, slot, **_identity):
            if slot.role == "vae":
                await asyncio.sleep(1)

        async def release(self, slot, *, token):
            self.released.append((slot.role, token))

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        reservations = ReservationClient()
        coordinator = RealtimeCoordinator(
            store,
            wait_timeout_s=0.05,
            reservation_client=reservations,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))

        started = time.monotonic()
        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        assert {role for role, _token in reservations.released} == {
            "denoiser",
            "vae",
        }
        replacement = await store.acquire(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"

    asyncio.run(run())


def test_coordinator_deadline_returns_before_slow_acquire_and_cleans_late_commit():
    class SlowCommitStore(InMemoryCoordinatorStore):
        async def acquire(self, **request):
            await asyncio.sleep(0.05)
            return await super().acquire(**request)

    async def run():
        store = SlowCommitStore(ttl_s=60, worker_ttl_s=30)
        coordinator = RealtimeCoordinator(store, wait_timeout_s=0.01)
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))

        started = time.monotonic()
        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )
        assert time.monotonic() - started < 0.04

        await asyncio.sleep(0.08)
        replacement = await store.acquire(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"

    asyncio.run(run())
