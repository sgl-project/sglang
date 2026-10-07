"""Exercise feature routing through FastAPI without starting an engine."""

import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from sglang.srt.managers.io_struct import (
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.utils.auth import AuthLevel, add_api_key_middleware
from sglang.srt.weight_sync.gpu_delta.http import register_gpu_delta_routes
from sglang.srt.weight_sync.gpu_delta.io import (
    AbortGpuDeltaReqInput,
    ApplyGpuDeltaReqInput,
    ClearGpuDeltaStateReqInput,
    GetGpuDeltaInfoReqInput,
    GetGpuDeltaStatusReqInput,
    GpuDeltaReqOutput,
    PrepareGpuDeltaReqInput,
    ResumeGpuDeltaReqInput,
    UpdateWeightsFromGpuDeltaReqInput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def http_delta():
    class CustomRoute(APIRoute):
        pass

    app = FastAPI()
    app.router.route_class = CustomRoute
    state = SimpleNamespace(tokenizer_manager=None)
    register_gpu_delta_routes(app, lambda: state)
    add_api_key_middleware(app, api_key="ordinary-key", admin_api_key="admin-key")
    calls = []
    reply = {"success": True, "message": "", "participants": []}

    async def dispatch(obj):
        calls.append(obj)
        return reply

    # Global state is populated at startup, after the routes are registered.
    state.tokenizer_manager = SimpleNamespace(
        gpu_delta=SimpleNamespace(request=dispatch)
    )
    with TestClient(app, headers={"Authorization": "Bearer admin-key"}) as client:
        yield client, app, calls, reply, CustomRoute


def test_delta_routes_preserve_typed_requests_auth_and_app_route_class(http_delta):
    client, app, calls, reply, route_class = http_delta
    session = {"session_id": "publication-1"}
    receipts = [
        {
            "identity": {"engine_id": "e0", "rank_id": "original"},
            "state": "APPLIED",
            "session_id": "publication-1",
            "target_version": 1,
        }
    ]
    cases = [
        (
            "update_weights_from_gpu_delta",
            UpdateWeightsFromGpuDeltaReqInput,
            {"manifest_path": "/immutable/manifest.json"},
        ),
        (
            "update_weights_from_gpu_delta",
            UpdateWeightsFromGpuDeltaReqInput,
            {
                "manifest_path": "/immutable/manifest.json",
                "release_state": False,
                "flush_cache": False,
                "abort_all_requests": True,
            },
        ),
        ("clear_gpu_delta_state", ClearGpuDeltaStateReqInput, {}),
        ("get_gpu_delta_info", GetGpuDeltaInfoReqInput, {"engine_id": "e0"}),
        (
            "prepare_gpu_delta",
            PrepareGpuDeltaReqInput,
            session
            | {
                "manifest_path": "/immutable/manifest.json",
                "manifest_sha256": "a" * 64,
                "stream_id": "run-1",
                "base_version": 0,
                "target_version": 1,
                "plan_digest": "b" * 64,
                "participants": [],
            },
        ),
        ("get_gpu_delta_status", GetGpuDeltaStatusReqInput, session),
        (
            "apply_gpu_delta",
            ApplyGpuDeltaReqInput,
            session | {"flush_cache": False, "abort_all_requests": True},
        ),
        (
            "resume_gpu_delta",
            ResumeGpuDeltaReqInput,
            session | {"keep_pause": True},
        ),
        ("abort_gpu_delta", AbortGpuDeltaReqInput, session),
    ]
    for name, request_type, payload in cases:
        route = next(route for route in app.routes if route.path == f"/{name}")
        assert isinstance(route, route_class)
        assert route.methods == {"POST"}
        assert route.endpoint._auth_level is AuthLevel.ADMIN_OPTIONAL
        rejected = client.post(
            route.path,
            json=payload,
            headers={"Authorization": "Bearer ordinary-key"},
        )
        assert rejected.status_code == 401
        result = client.post(route.path, json=payload)
        assert result.status_code == 200 and result.json() == reply
        assert isinstance(calls[-1], request_type)
        received = msgpack_decode(msgpack_encode(calls[-1]))
        assert type(received) is request_type
        assert received == calls[-1]
    response = GpuDeltaReqOutput(
        success=True, message="", participant=receipts[0], rid="control-reply"
    )
    received = msgpack_decode(msgpack_encode(response))
    assert type(received) is GpuDeltaReqOutput and received == response
    assert len(calls) == len(cases)

    # A real scheduler failure retains the same typed HTTP route and response.
    reply.update(success=False, message="delta payload checksum failed")
    result = client.post("/get_gpu_delta_status", json={"session_id": "p"})
    assert result.status_code == 409 and result.json() == reply
    assert len(calls) == len(cases) + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
