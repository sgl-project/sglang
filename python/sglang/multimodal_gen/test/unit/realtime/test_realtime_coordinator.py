# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    DynamoDBCoordinatorStore,
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
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
):
    return WorkerHeartbeat(
        worker_id=worker_id,
        role=role,
        endpoint=f"ws://{worker_id}.cluster.local/generate",
        az=az,
        capacity=capacity,
        model_revision=model_revision,
        vae_fingerprint=vae_fingerprint,
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


def test_dynamodb_candidate_pairing_spreads_bursts_across_workers():
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

    assert first_denoisers == {slot.worker_id for slot in denoisers}


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
