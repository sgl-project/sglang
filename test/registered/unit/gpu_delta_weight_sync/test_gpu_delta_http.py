"""Exercise feature routing through FastAPI without starting an engine."""

import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sglang.srt.managers.io_struct import (
    AbortWeightsFromDeltaReqInput,
    GetWeightsDeltaInfoReqInput,
    GetWeightsDeltaStatusReqInput,
    PrepareWeightsFromDeltaReqInput,
    ResumeWeightsFromDeltaReqInput,
    UpdateWeightsFromDeltaReqInput,
)
from sglang.srt.utils.auth import AuthLevel, add_api_key_middleware
from sglang.srt.weight_sync.gpu_delta_http import register_gpu_delta_routes
from sglang.srt.weight_sync.gpu_delta_session import GpuDeltaConflict
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

    async def dispatch(obj, request):
        assert request.app is app
        calls.append(obj)
        if isinstance(reply.get("error"), Exception):
            raise reply["error"]
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
    cases = [
        ("get_weights_delta_info", GetWeightsDeltaInfoReqInput, {"engine_id": "e0"}),
        (
            "prepare_weights_from_delta",
            PrepareWeightsFromDeltaReqInput,
            session
            | {
                "engine_id": "e0",
                "manifest_path": "/immutable/manifest.json",
                "manifest_sha256": "a" * 64,
                "stream_id": "run-1",
                "base_version": 0,
                "target_version": 1,
                "plan_digest": "b" * 64,
                "participants": [],
                "cohort": [],
                "host_tensor_names": {"host": []},
            },
        ),
        ("get_weights_delta_status", GetWeightsDeltaStatusReqInput, session),
        (
            "update_weights_from_delta",
            UpdateWeightsFromDeltaReqInput,
            session,
        ),
        (
            "resume_weights_from_delta",
            ResumeWeightsFromDeltaReqInput,
            session | {"receipts": []},
        ),
        ("abort_weights_from_delta", AbortWeightsFromDeltaReqInput, session),
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
    assert len(calls) == len(cases)


def test_delta_failure_is_conflict_and_invalid_body_never_dispatches(http_delta):
    client, _, calls, reply, _ = http_delta
    reply.update(success=False, message="original participant changed")
    result = client.post("/get_weights_delta_info", json={"engine_id": "e0"})
    assert result.status_code == 409 and result.json() == reply
    result = client.post("/prepare_weights_from_delta", json={"session_id": "p"})
    assert result.status_code == 422
    assert len(calls) == 1
    reply["error"] = GpuDeltaConflict("another delta session is active")
    result = client.post("/get_weights_delta_info", json={"engine_id": "e0"})
    assert result.status_code == 409
    assert result.json() == {
        "success": False,
        "message": "another delta session is active",
        "participants": [],
    }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
