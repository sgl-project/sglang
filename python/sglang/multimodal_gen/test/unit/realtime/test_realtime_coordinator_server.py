# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server import (
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
                "role": role,
                "endpoint": f"ws://{role}-a.cluster.local/generate",
                "az": "us-east-2a",
                "capacity": 1,
                "model_revision": "minwm-r1",
                "vae_fingerprint": "taew2_2",
            },
        )
        assert response.status_code == 204

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
