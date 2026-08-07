# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_video_api import (
    _GatewayManagedConfig,
    _consume_gateway_reservation,
    _release_gateway_reservation,
)
from sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server import create_app
from sglang.multimodal_gen.runtime.entrypoints import realtime_vae_server
from sglang.multimodal_gen.runtime.entrypoints.http_server import (
    create_app as create_denoiser_app,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    decode_message,
    encode_message,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import AsyncVAEWorker
from sglang.multimodal_gen.runtime.realtime.worker_reservation import (
    WorkerReservationRegistry,
    WorkerReservationRejected,
    install_worker_reservation_routes,
    resolve_worker_epoch,
)


def test_worker_reservation_is_idempotent_bounded_and_released():
    async def run():
        registry = WorkerReservationRegistry(
            worker_epoch="epoch-a",
            capacity=1,
        )

        first = await registry.reserve(
            "token-a",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch="epoch-a",
            ttl_s=30,
        )
        duplicate = await registry.reserve(
            "token-a",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch="epoch-a",
            ttl_s=30,
        )
        assert duplicate == first

        with pytest.raises(WorkerReservationRejected, match="WORKER_CAPACITY_EXHAUSTED"):
            await registry.reserve(
                "token-b",
                session_id="session-b",
                generation_id="generation-b",
                worker_epoch="epoch-a",
                ttl_s=30,
            )

        await registry.consume(
            "token-a",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch="epoch-a",
            owner_id="connection-a",
        )
        with pytest.raises(
            WorkerReservationRejected, match="RESERVATION_ALREADY_CONSUMED"
        ):
            await registry.consume(
                "token-a",
                session_id="session-a",
                generation_id="generation-a",
                worker_epoch="epoch-a",
                owner_id="connection-b",
            )
        snapshot = await registry.snapshot()
        assert snapshot["active_sessions"] == 1
        assert snapshot["reserved_sessions"] == 0

        with pytest.raises(
            WorkerReservationRejected, match="RESERVATION_OWNER_MISMATCH"
        ):
            await registry.release("token-a", owner_id="connection-b")
        assert (await registry.snapshot())["active_sessions"] == 1

        await registry.release("token-a", owner_id="connection-a")
        await registry.release("token-a", owner_id="connection-a")
        assert (await registry.snapshot())["active_sessions"] == 0

    asyncio.run(run())


def test_worker_process_publishes_its_epoch_to_the_shared_file(
    tmp_path, monkeypatch
):
    epoch_file = tmp_path / "worker-epoch"
    monkeypatch.delenv("WORKER_EPOCH", raising=False)
    monkeypatch.setenv("WORKER_EPOCH_FILE", str(epoch_file))

    epoch = resolve_worker_epoch()

    assert epoch
    assert epoch_file.read_text().strip() == epoch


def test_worker_reservation_http_routes_expose_state_release_and_drain():
    app = FastAPI()
    registry = WorkerReservationRegistry(worker_epoch="epoch-a", capacity=2)
    install_worker_reservation_routes(app, registry)
    client = TestClient(app)

    response = client.post(
        "/v1/realtime_worker/reservations",
        json={
            "token": "token-a",
            "session_id": "session-a",
            "generation_id": "generation-a",
            "worker_epoch": "epoch-a",
            "ttl_s": 30,
        },
    )
    assert response.status_code == 204
    assert client.get("/v1/realtime_worker/state").json()["reserved_sessions"] == 1

    released = client.delete("/v1/realtime_worker/reservations/token-a")
    assert released.status_code == 204
    assert client.delete("/v1/realtime_worker/reservations/token-a").status_code == 204

    drained = client.post(
        "/v1/realtime_worker/drain",
        json={"deadline": 9_999_999_999},
    )
    assert drained.status_code == 204
    assert client.get("/v1/realtime_worker/state").json()["lifecycle"] == "draining"


def test_vae_session_open_consumes_and_close_releases_reservation():
    class Engine:
        @staticmethod
        def create_decoder(_identity):
            return SimpleNamespace(reset=lambda: None)

    registry = WorkerReservationRegistry(worker_epoch="epoch-a", capacity=1)
    worker = AsyncVAEWorker(Engine(), max_sessions=1)
    app = create_app(
        worker,
        max_message_bytes=1024 * 1024,
        reservation_registry=registry,
    )

    with TestClient(app) as client:
        assert client.post(
            "/v1/realtime_worker/reservations",
            json={
                "token": "token-a",
                "session_id": "session-a",
                "generation_id": "generation-a",
                "worker_epoch": "epoch-a",
                "ttl_s": 30,
            },
        ).status_code == 204
        with client.websocket_connect("/v1/realtime_vae/decode") as websocket:
            websocket.send_bytes(
                encode_message(
                    "session_open",
                    session_id="session-a",
                    generation_id="generation-a",
                    coordinator_token="token-a",
                    worker_epoch="epoch-a",
                    output_format="webp",
                    quality=90,
                )
            )
            assert decode_message(websocket.receive_bytes())["type"] == (
                "session_accepted"
            )
            state = client.get("/v1/realtime_worker/state").json()
            assert state["active_sessions"] == 1
            assert state["reserved_sessions"] == 0
        assert client.get("/v1/realtime_worker/state").json()["active_sessions"] == 0


def test_vae_replay_connection_cannot_release_the_original_session():
    class Engine:
        @staticmethod
        def create_decoder(_identity):
            return SimpleNamespace(reset=lambda: None)

    registry = WorkerReservationRegistry(worker_epoch="epoch-a", capacity=1)
    app = create_app(
        AsyncVAEWorker(Engine(), max_sessions=1),
        max_message_bytes=1024 * 1024,
        reservation_registry=registry,
    )
    opened = encode_message(
        "session_open",
        session_id="session-a",
        generation_id="generation-a",
        coordinator_token="token-a",
        worker_epoch="epoch-a",
        output_format="webp",
        quality=90,
    )

    with TestClient(app) as client:
        assert client.post(
            "/v1/realtime_worker/reservations",
            json={
                "token": "token-a",
                "session_id": "session-a",
                "generation_id": "generation-a",
                "worker_epoch": "epoch-a",
                "ttl_s": 30,
            },
        ).status_code == 204
        with client.websocket_connect("/v1/realtime_vae/decode") as original:
            original.send_bytes(opened)
            assert decode_message(original.receive_bytes())["type"] == "session_accepted"
            with client.websocket_connect("/v1/realtime_vae/decode") as replay:
                replay.send_bytes(opened)
                error = decode_message(replay.receive_bytes())
                assert error["type"] == "error"
                assert "RESERVATION_ALREADY_CONSUMED" in error["message"]
            assert client.get("/v1/realtime_worker/state").json()[
                "active_sessions"
            ] == 1
        assert client.get("/v1/realtime_worker/state").json()["active_sessions"] == 0


def test_vae_direct_output_sends_authoritative_media_completion(monkeypatch):
    direct_messages = []

    class OutputClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def open(self):
            pass

        async def send(self, wire):
            direct_messages.append(decode_message(wire))

        async def close(self):
            pass

    class Worker:
        active_sessions = 0
        max_sessions = 1

        async def open(self, _opened):
            pass

        async def submit(
            self,
            _header,
            _latents,
            *,
            on_frame_batch,
            on_decode_started,
        ):
            await on_decode_started()
            await on_frame_batch(
                SimpleNamespace(
                    payloads=(b"webp",),
                    content_type="image/webp",
                    width=8,
                    height=8,
                    num_frames=1,
                    frame_batch_index=0,
                    is_final=False,
                    encode_ms=1.0,
                )
            )
            future = asyncio.get_running_loop().create_future()
            future.set_result(
                SimpleNamespace(
                    num_frames=1,
                    queue_wait_ms=1.0,
                    decode_ms=2.0,
                    encode_ms=3.0,
                )
            )
            return future

        async def close(self, *_identity):
            pass

        async def close_all(self):
            pass

    monkeypatch.setattr(realtime_vae_server, "GatewayOutputClient", OutputClient)
    app = create_app(Worker(), max_message_bytes=1024 * 1024)
    payload = (
        torch.zeros(1, 48, 1, 1, 1, dtype=torch.bfloat16)
        .view(torch.uint16)
        .numpy()
        .tobytes()
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_vae/decode") as websocket:
            websocket.send_bytes(
                encode_message(
                    "session_open",
                    session_id="session-a",
                    generation_id="generation-a",
                    output_format="webp",
                    quality=90,
                    output_url="ws://gateway/output",
                    output_token="output-token",
                )
            )
            assert decode_message(websocket.receive_bytes())["type"] == (
                "session_accepted"
            )
            websocket.send_bytes(
                encode_message(
                    "latent_chunk",
                    header={
                        "session_id": "session-a",
                        "generation_id": "generation-a",
                        "request_id": "request-0",
                        "chunk_index": 0,
                        "dtype": "bfloat16",
                        "shape": [1, 48, 1, 1, 1],
                        "byte_length": len(payload),
                        "checksum": __import__("hashlib").sha256(payload).hexdigest(),
                    },
                    payload=payload,
                )
            )
            assert decode_message(websocket.receive_bytes())["type"] == (
                "latent_accepted"
            )
            assert decode_message(websocket.receive_bytes())["type"] == (
                "chunk_complete"
            )

    assert [message["type"] for message in direct_messages] == [
        "frame_batch",
        "media_chunk_complete",
    ]
    assert direct_messages[-1]["chunk_index"] == 0


def test_denoiser_gateway_open_consumes_matching_reservation():
    async def run():
        registry = WorkerReservationRegistry(worker_epoch="epoch-a", capacity=1)
        await registry.reserve(
            "token-a",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch="epoch-a",
            ttl_s=30,
        )
        websocket = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(worker_reservations=registry))
        )
        config = _GatewayManagedConfig(
            session_id="session-a",
            generation_id="generation-a",
            coordinator_token="token-a",
            worker_epoch="epoch-a",
            vae_worker_url="ws://vae/decode",
            vae_worker_epoch="vae-epoch",
            output_url="ws://gateway/output",
            output_token="output-token",
        )

        await _consume_gateway_reservation(
            websocket,
            config,
            owner_id="connection-a",
        )
        assert (await registry.snapshot())["active_sessions"] == 1
        await _release_gateway_reservation(
            registry,
            config,
            owner_id="connection-a",
        )
        assert (await registry.snapshot())["active_sessions"] == 0

    asyncio.run(run())


def test_denoiser_fastapi_exposes_worker_reservation_control_plane(monkeypatch):
    monkeypatch.setenv("WORKER_EPOCH", "denoiser-epoch")
    app = create_denoiser_app(
        SimpleNamespace(realtime_max_sessions_per_worker=4)
    )

    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert {
        "/v1/realtime_worker/state",
        "/v1/realtime_worker/reservations",
        "/v1/realtime_worker/reservations/{token}",
        "/v1/realtime_worker/drain",
    } <= paths
    snapshot = asyncio.run(app.state.worker_reservations.snapshot())
    assert snapshot["worker_epoch"] == "denoiser-epoch"
    assert snapshot["capacity"] == 4


def test_worker_reservation_consume_rejects_stale_epoch_and_identity():
    async def run():
        registry = WorkerReservationRegistry(worker_epoch="epoch-new", capacity=2)
        await registry.reserve(
            "token-a",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch="epoch-new",
            ttl_s=30,
        )

        with pytest.raises(WorkerReservationRejected, match="WORKER_EPOCH_MISMATCH"):
            await registry.consume(
                "token-a",
                session_id="session-a",
                generation_id="generation-a",
                worker_epoch="epoch-old",
                owner_id="connection-a",
            )
        with pytest.raises(WorkerReservationRejected, match="RESERVATION_IDENTITY_MISMATCH"):
            await registry.consume(
                "token-a",
                session_id="session-other",
                generation_id="generation-a",
                worker_epoch="epoch-new",
                owner_id="connection-a",
            )

    asyncio.run(run())


def test_worker_reservation_drain_rejects_new_work_and_snapshot_reports_load():
    async def run():
        now = [100.0]
        registry = WorkerReservationRegistry(
            worker_epoch="epoch-a",
            capacity=4,
            clock=lambda: now[0],
            load_provider=lambda: {
                "runnable_sessions": 1,
                "blocked_sessions": 2,
                "queue_depth": 3,
                "service_time_ms": 12.5,
            },
        )
        await registry.reserve(
            "expired",
            session_id="session-expired",
            generation_id="generation-expired",
            worker_epoch="epoch-a",
            ttl_s=1,
        )
        now[0] = 102.0
        await registry.reserve(
            "active",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch="epoch-a",
            ttl_s=30,
        )
        await registry.consume(
            "active",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch="epoch-a",
            owner_id="connection-a",
        )
        await registry.drain(deadline=120.0)

        with pytest.raises(WorkerReservationRejected, match="WORKER_DRAINING"):
            await registry.reserve(
                "token-b",
                session_id="session-b",
                generation_id="generation-b",
                worker_epoch="epoch-a",
                ttl_s=30,
            )

        snapshot = await registry.snapshot()
        assert snapshot == {
            "worker_epoch": "epoch-a",
            "lifecycle": "draining",
            "drain_deadline": 120.0,
            "capacity": 4,
            "active_sessions": 1,
            "reserved_sessions": 0,
            "runnable_sessions": 1,
            "blocked_sessions": 2,
            "queue_depth": 3,
            "service_time_ms": 12.5,
            "normalized_load": 0.25,
        }

    asyncio.run(run())


def test_worker_runtime_load_tracks_pending_blocked_runnable_and_service_time():
    async def run():
        now = [100.0]
        registry = WorkerReservationRegistry(
            worker_epoch="epoch-a",
            capacity=2,
            clock=lambda: now[0],
        )
        await registry.reserve(
            "token-a",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch="epoch-a",
            ttl_s=30,
        )
        pending = await registry.snapshot()
        assert pending["queue_depth"] == 1
        assert pending["active_sessions"] == 0

        await registry.consume(
            "token-a",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch="epoch-a",
            owner_id="connection-a",
        )
        blocked = await registry.snapshot()
        assert blocked["active_sessions"] == 1
        assert blocked["blocked_sessions"] == 1
        assert blocked["runnable_sessions"] == 0
        assert blocked["queue_depth"] == 0

        await registry.mark_runnable("token-a", owner_id="connection-a")
        runnable = await registry.snapshot()
        assert runnable["runnable_sessions"] == 1
        assert runnable["blocked_sessions"] == 0

        now[0] = 100.125
        await registry.release("token-a", owner_id="connection-a")
        completed = await registry.snapshot()
        assert completed["active_sessions"] == 0
        assert completed["service_time_ms"] == pytest.approx(125.0)

    asyncio.run(run())


def test_denoiser_registry_reports_real_session_load_instead_of_zero_defaults():
    async def run():
        app = create_denoiser_app(
            SimpleNamespace(realtime_max_sessions_per_worker=2)
        )
        registry = app.state.worker_reservations
        await registry.reserve(
            "token-a",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch=registry.worker_epoch,
            ttl_s=30,
        )
        await registry.consume(
            "token-a",
            session_id="session-a",
            generation_id="generation-a",
            worker_epoch=registry.worker_epoch,
            owner_id="connection-a",
        )
        await registry.mark_runnable("token-a", owner_id="connection-a")

        snapshot = await registry.snapshot()
        assert snapshot["active_sessions"] == 1
        assert snapshot["runnable_sessions"] == 1
        assert snapshot["blocked_sessions"] == 0
        assert snapshot["queue_depth"] == 0

    asyncio.run(run())
