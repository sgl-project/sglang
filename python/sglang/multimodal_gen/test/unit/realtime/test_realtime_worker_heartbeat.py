import asyncio

import pytest

from sglang.multimodal_gen.runtime.realtime.worker_heartbeat import (
    WorkerHeartbeatReporter,
    discover_kubernetes_node_az,
)


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def raise_for_status(self):
        if not self.is_success:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class _Client:
    def __init__(self, *, health_status=200, state_payload=None):
        self.health_status = health_status
        self.state_payload = state_payload or {
            "worker_epoch": "epoch-a",
            "lifecycle": "ready",
            "active_sessions": 2,
            "runnable_sessions": 1,
            "blocked_sessions": 1,
            "queue_depth": 3,
            "service_time_ms": 12.5,
        }
        self.posts = []

    async def get(self, url, **kwargs):
        if url.endswith("/state"):
            return _Response(200, self.state_payload)
        return _Response(self.health_status)

    async def post(self, url, *, json):
        self.posts.append((url, json))
        return _Response(204)


def test_worker_registers_capacity_only_after_local_health_is_ready():
    asyncio.run(_test_worker_registers_capacity_only_after_local_health_is_ready())


async def _test_worker_registers_capacity_only_after_local_health_is_ready():
    client = _Client(health_status=503)
    reporter = WorkerHeartbeatReporter(
        client,
        coordinator_url="http://coordinator:18081",
        health_url="http://127.0.0.1:30000/health",
        state_url="http://127.0.0.1:30000/v1/realtime_worker/state",
        worker_id="pod-123",
        worker_epoch="epoch-a",
        role="denoiser",
        endpoint="ws://10.0.0.7:30000/v1/realtime_video/generate",
        reservation_endpoint="http://10.0.0.7:30000/v1/realtime_worker",
        az="us-east-2a",
        capacity=4,
        model_revision="model-sha",
        vae_fingerprint="taew2_2",
    )

    assert await reporter.heartbeat_once() is False
    assert client.posts == []

    client.health_status = 200
    assert await reporter.heartbeat_once() is True
    assert client.posts == [
        (
            "http://coordinator:18081/v1/workers/heartbeat",
            {
                "worker_id": "pod-123",
                "worker_epoch": "epoch-a",
                "role": "denoiser",
                "endpoint": "ws://10.0.0.7:30000/v1/realtime_video/generate",
                "reservation_endpoint": (
                    "http://10.0.0.7:30000/v1/realtime_worker"
                ),
                "az": "us-east-2a",
                "capacity": 4,
                "model_revision": "model-sha",
                "vae_fingerprint": "taew2_2",
                "lifecycle": "ready",
                "active_sessions": 2,
                "runnable_sessions": 1,
                "blocked_sessions": 1,
                "queue_depth": 3,
                "service_time_ms": 12.5,
            },
        )
    ]


def test_worker_rejects_public_or_malformed_endpoints():
    with pytest.raises(ValueError, match="WebSocket endpoint"):
        WorkerHeartbeatReporter(
            _Client(),
            coordinator_url="http://coordinator:18081",
            health_url="http://127.0.0.1:30000/health",
            state_url="http://127.0.0.1:30000/v1/realtime_worker/state",
            worker_id="pod-123",
            worker_epoch="epoch-a",
            role="denoiser",
            endpoint="https://public.example.com/generate",
            reservation_endpoint="http://10.0.0.7:30000/v1/realtime_worker",
            az="us-east-2a",
            capacity=1,
            model_revision="model-sha",
            vae_fingerprint="taew2_2",
        )


def test_worker_discovers_real_az_from_its_kubernetes_node():
    class Client:
        async def get(self, url, **kwargs):
            assert url == "https://kubernetes.default.svc/api/v1/nodes/ip-10-0-0-7"
            assert kwargs["headers"] == {"Authorization": "Bearer token"}
            return _Response(
                200,
                {
                    "metadata": {
                        "labels": {"topology.kubernetes.io/zone": "us-east-2b"}
                    }
                },
            )

    assert asyncio.run(
        discover_kubernetes_node_az(
            Client(),
            api_url="https://kubernetes.default.svc",
            node_name="ip-10-0-0-7",
            token="token",
        )
    ) == "us-east-2b"


def test_heartbeat_follows_process_epoch_file_across_worker_restart(tmp_path):
    async def run():
        epoch_file = tmp_path / "worker-epoch"
        epoch_file.write_text("epoch-a\n")
        client = _Client()
        reporter = WorkerHeartbeatReporter(
            client,
            coordinator_url="http://coordinator:18081",
            health_url="http://127.0.0.1:30000/health",
            state_url="http://127.0.0.1:30000/v1/realtime_worker/state",
            worker_id="pod-123",
            worker_epoch=None,
            worker_epoch_file=epoch_file,
            role="denoiser",
            endpoint="ws://10.0.0.7:30000/v1/realtime_video/generate",
            reservation_endpoint="http://10.0.0.7:30000/v1/realtime_worker",
            az="us-east-2a",
            capacity=4,
            model_revision="model-sha",
            vae_fingerprint="taew2_2",
        )

        assert await reporter.heartbeat_once() is True
        assert client.posts[-1][1]["worker_epoch"] == "epoch-a"

        epoch_file.write_text("epoch-b\n")
        client.state_payload["worker_epoch"] = "epoch-b"
        assert await reporter.heartbeat_once() is True
        assert client.posts[-1][1]["worker_epoch"] == "epoch-b"

    asyncio.run(run())
