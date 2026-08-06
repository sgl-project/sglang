# SPDX-License-Identifier: Apache-2.0

import logging
import sys

from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server import (
    _parse_args,
    create_app,
)
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
)


def test_coordinator_http_session_lifecycle_and_structured_rejection(caplog):
    coordinator = RealtimeCoordinator(
        InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30),
        wait_timeout_s=0,
    )
    client = TestClient(create_app(coordinator))

    for role in ("denoiser", "vae"):
        response = client.post(
            "/v1/workers/heartbeat",
            json={
                "worker_id": f"{role}-a",
                "worker_epoch": f"{role}-epoch",
                "role": role,
                "endpoint": f"ws://{role}-a.cluster.local/generate",
                "reservation_endpoint": (
                    f"http://{role}-a.cluster.local/v1/realtime_worker"
                ),
                "az": "us-east-2a",
                "capacity": 1,
                "model_revision": "minwm-r1",
                "vae_fingerprint": "taew2_2",
                "lifecycle": "ready",
                "active_sessions": 0,
                "runnable_sessions": 0,
                "blocked_sessions": 0,
                "queue_depth": 0,
                "service_time_ms": 0,
            },
        )
        assert response.status_code == 204

    capacity = client.get("/v1/capacity")
    assert capacity.status_code == 200
    assert capacity.json()["roles"]["denoiser"]["free_slots"] == 1
    assert capacity.json()["roles"]["vae"]["free_slots"] == 1

    request = {
        "user_id": "user-a",
        "session_id": "session-a",
        "generation_id": "generation-a",
        "model_revision": "minwm-r1",
        "vae_fingerprint": "taew2_2",
        "trace_id": "trace-a",
    }
    with caplog.at_level(logging.INFO):
        response = client.post("/v1/sessions/admit", json=request)
    assert response.status_code == 200
    assignment = response.json()
    assert assignment["denoiser"]["worker_id"] == "denoiser-a"
    assert assignment["denoiser"]["worker_epoch"] == "denoiser-epoch"
    assert assignment["vae"]["worker_id"] == "vae-a"
    assert '"event":"coordinator.admit_complete"' in caplog.text
    assert '"trace_id":"trace-a"' in caplog.text

    rejected = client.post(
        "/v1/sessions/admit",
        json={**request, "session_id": "session-b"},
    )
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["reason"] == "USER_SESSION_LIMIT"

    renewed = client.post("/v1/sessions/renew", json=assignment)
    assert renewed.status_code == 200
    released = client.request("DELETE", "/v1/sessions/release", json=renewed.json())
    assert released.status_code == 204
    assert client.get("/healthz").json() == {"status": "ok"}


def test_coordinator_cli_bounds_worker_reservation_calls(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["realtime_coordinator_server"])

    assert _parse_args().worker_reservation_timeout_s == 2.0
