# SPDX-License-Identifier: Apache-2.0

import inspect

from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints import realtime_gateway_server
from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
    _parse_ui_config,
    create_app,
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
