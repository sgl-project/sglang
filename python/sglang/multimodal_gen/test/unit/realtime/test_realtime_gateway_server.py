# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import sys

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints import realtime_gateway_server
from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
    HTTPCoordinatorClient,
    _parse_args,
    _parse_ui_config,
    create_app,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import decode_message
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    SessionAssignment,
    WorkerSlot,
)


class _Coordinator:
    def __init__(self, *, ready=True):
        self.ready = ready

    async def health(self):
        if not self.ready:
            raise RuntimeError("coordinator unavailable")
        return {"status": "ready"}


class _TraceQuery:
    def __init__(self):
        self.calls = []

    async def query(self, trace_id, **kwargs):
        self.calls.append((trace_id, kwargs))
        return {
            "trace_id": trace_id,
            "events": [{"event": "server.chunk_complete", "trace_seq": 9}],
            "next_cursor": 9,
        }


def test_gateway_parses_ui_config_for_the_served_webui():
    assert _parse_ui_config(
        '{"generationModes":["i2v","t2v"],"defaultGenerationMode":"t2v"}'
    ) == {
        "generationModes": ["i2v", "t2v"],
        "defaultGenerationMode": "t2v",
    }


def test_gateway_trace_events_use_the_independent_otlp_log_plane():
    source = inspect.getsource(realtime_gateway_server)

    assert "emit_realtime_trace_payload" in source
    assert 'logger.info(\n        "realtime_trace %s"' not in source


def test_gateway_coordinator_release_treats_lost_lease_as_idempotent():
    class Response:
        status_code = 409

        @staticmethod
        def json():
            return {"detail": {"reason": "LEASE_LOST"}}

    class Client:
        async def request(self, method, path, *, json):
            assert method == "DELETE"
            assert path == "/v1/sessions/release"
            assert json["token"] == "token-a"
            return Response()

    async def run():
        client = HTTPCoordinatorClient.__new__(HTTPCoordinatorClient)
        client._client = Client()
        slot = WorkerSlot(
            worker_id="worker-a",
            role="denoiser",
            endpoint="ws://worker-a/generate",
            az="us-east-2a",
            slot_index=0,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        await client.release(
            SessionAssignment(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                token="token-a",
                expires_at=1,
                denoiser=slot,
                vae=slot,
            )
        )

    asyncio.run(run())


def test_gateway_serves_trace_query_and_accepts_sanitized_client_metrics():
    query = _TraceQuery()
    app = create_app(
        _Coordinator(),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
        trace_query=query,
    )
    client = TestClient(app)

    response = client.get("/v1/realtime_video/traces/trace-a?after=7&limit=220")
    assert response.status_code == 200
    assert response.json()["next_cursor"] == 9
    assert query.calls == [("trace-a", {"after": 7, "limit": 220})]

    response = client.post(
        "/v1/realtime_video/traces/trace-a/client-events",
        json={
            "events": [
                {
                    "name": "client.chunk_first_rendered",
                    "seq": 2,
                    "display_lag_ms": 84,
                    "prompt": "must not be accepted",
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json() == {"accepted": 1}


def test_gateway_maps_first_cloudwatch_transport_failure_to_503():
    class FailingTraceQuery:
        async def query(self, *_args, **_kwargs):
            raise OSError("CloudWatch connection reset")

    app = create_app(
        _Coordinator(),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
        trace_query=FailingTraceQuery(),
    )

    response = TestClient(app).get("/v1/realtime_video/traces/trace-a")

    assert response.status_code == 503
    assert response.json()["detail"] == "trace query unavailable"


def test_gateway_readiness_depends_on_coordinator():
    ready_client = TestClient(
        create_app(
            _Coordinator(),
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url="ws://gateway/v1/internal/realtime_output",
        )
    )
    unavailable_client = TestClient(
        create_app(
            _Coordinator(ready=False),
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url="ws://gateway/v1/internal/realtime_output",
        )
    )

    assert ready_client.get("/readyz").status_code == 200
    response = unavailable_client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["detail"] == "coordinator unavailable"


def test_gateway_cli_defaults_to_a_bounded_64_waiter_queue(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "realtime_gateway_server",
            "--coordinator-url=http://coordinator:18081",
            "--model-revision=minwm-r1",
        ],
    )

    assert _parse_args().max_admission_waiters == 64


@pytest.mark.parametrize(
    ("reason", "retry_after_s", "expected_code"),
    (
        ("CAPACITY_EXHAUSTED", 0.25, 1013),
        ("USER_SESSION_LIMIT", None, 1008),
    ),
)
def test_gateway_uses_retryable_close_semantics_for_capacity_only(
    reason, retry_after_s, expected_code
):
    class RejectingCoordinator(_Coordinator):
        async def admit(self, **_request):
            raise CoordinatorRejected(reason, retry_after_s=retry_after_s)

    app = create_app(
        RejectingCoordinator(),
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        internal_output_url="ws://gateway/v1/internal/realtime_output",
        release_grace_s=0,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_video/generate") as websocket:
            message = decode_message(websocket.receive_bytes())
            assert message["type"] == "error"
            assert message["reason"] == reason
            assert message.get("retry_after_s") == retry_after_s
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_bytes()
            assert closed.value.code == expected_code
