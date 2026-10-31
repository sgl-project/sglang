import socket
import threading
import time
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import uvicorn
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import FlexibleArgumentParser
from sglang.srt.utils.auth import (
    AuthLevel,
    _iter_effective_routes,
    add_api_key_middleware,
    app_has_admin_force_endpoints,
    auth_level,
)


def _server_args(api_key=None, admin_api_key=None):
    pipeline_config = SimpleNamespace(
        supports_action_endpoint=lambda: False,
        supports_openpi_endpoint=lambda: False,
    )
    return SimpleNamespace(
        api_key=api_key,
        admin_api_key=admin_api_key,
        enable_metrics=False,
        pipeline_config=pipeline_config,
    )


def _route_auth_level(app, path):
    route = next(
        route
        for route in _iter_effective_routes(app)
        if getattr(route, "path", None) == path
    )
    return getattr(route.endpoint, "_auth_level", AuthLevel.NORMAL)


@pytest.mark.parametrize("transport", ["http", "websocket"])
@pytest.mark.parametrize(
    "level,api_key,admin_api_key,statuses",
    [
        (AuthLevel.NORMAL, None, None, [200, 200, 200, 200]),
        (AuthLevel.NORMAL, "user-secret", None, [401, 401, 200, 401]),
        (AuthLevel.NORMAL, None, "admin-secret", [200, 200, 200, 200]),
        (AuthLevel.NORMAL, "user-secret", "admin-secret", [401, 401, 200, 401]),
        (AuthLevel.ADMIN_OPTIONAL, None, None, [200, 200, 200, 200]),
        (AuthLevel.ADMIN_OPTIONAL, "user-secret", None, [401, 401, 200, 401]),
        (AuthLevel.ADMIN_OPTIONAL, None, "admin-secret", [401, 401, 401, 200]),
        (
            AuthLevel.ADMIN_OPTIONAL,
            "user-secret",
            "admin-secret",
            [401, 401, 401, 200],
        ),
        (AuthLevel.ADMIN_FORCE, None, None, [403, 403, 403, 403]),
        (AuthLevel.ADMIN_FORCE, "user-secret", None, [403, 403, 403, 403]),
        (AuthLevel.ADMIN_FORCE, None, "admin-secret", [401, 401, 401, 200]),
        (AuthLevel.ADMIN_FORCE, "user-secret", "admin-secret", [401, 401, 401, 200]),
    ],
)
def test_nested_routes_preserve_management_auth_level(
    level, api_key, admin_api_key, statuses, transport
):
    router = APIRouter()

    @auth_level(level)
    async def management():
        return {"success": True}

    @auth_level(level)
    async def management_ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"success": True})

    if transport == "http":
        router.add_api_route("/management", management, methods=["POST"])
    else:
        router.add_api_websocket_route("/management", management_ws)

    outer = APIRouter()
    outer.include_router(router, prefix="/v1")
    app = FastAPI()
    app.include_router(outer, prefix="/api")

    assert _route_auth_level(app, "/api/v1/management") == level
    assert app_has_admin_force_endpoints(app) == (level == AuthLevel.ADMIN_FORCE)
    add_api_key_middleware(app, api_key=api_key, admin_api_key=admin_api_key)
    client = TestClient(app)
    for token, status in zip(
        (None, "wrong-secret", "user-secret", "admin-secret"), statuses
    ):
        headers = {} if token is None else {"Authorization": f"Bearer {token}"}
        if transport == "http":
            assert (
                client.post("/api/v1/management", headers=headers).status_code == status
            )
        elif status == 200:
            with client.websocket_connect("/api/v1/management", headers=headers) as ws:
                assert ws.receive_json() == {"success": True}
        else:
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect("/api/v1/management", headers=headers):
                    pass
            # TestClient exposes the ASGI close code before acceptance.
            assert exc.value.code == 1008


@pytest.mark.parametrize("token", [None, "wrong-secret"])
def test_websocket_denial_returns_http_403_on_the_wire(token):
    """A rejected upgrade returns HTTP 403 through a real Uvicorn server."""
    app = FastAPI()
    entered = False

    @app.websocket("/stream")
    async def stream(websocket: WebSocket):
        nonlocal entered
        entered = True
        await websocket.accept()
        await websocket.close()

    add_api_key_middleware(app, api_key="user-secret", admin_api_key=None)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                loop="asyncio",
                lifespan="off",
                log_level="critical",
                timeout_graceful_shutdown=1,
            )
        )
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started, "Uvicorn failed to start"
            headers = {
                "Connection": "Upgrade",
                "Upgrade": "websocket",
                "Sec-WebSocket-Version": "13",
                "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
            }
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"
            response = httpx.get(
                f"http://127.0.0.1:{port}/stream",
                headers=headers,
                trust_env=False,
                timeout=5,
            )
            assert response.status_code == 403
            assert not entered
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive(), "Uvicorn failed to stop"


@pytest.mark.parametrize(
    "path",
    ["/v1/realtime_video/generate", "/v1/actions/realtime", "/openpi/policy"],
)
@pytest.mark.parametrize("token", [None, "wrong-secret", "admin-secret"])
def test_diffusion_websockets_reject_unauthorized_handshakes(monkeypatch, path, token):
    server_args = _server_args(api_key="user-secret", admin_api_key="admin-secret")
    server_args.pipeline_config.supports_action_endpoint = lambda: True
    server_args.pipeline_config.supports_openpi_endpoint = lambda: True
    forward = AsyncMock()
    monkeypatch.setattr(async_scheduler_client, "forward", forward)
    app = create_app(server_args)
    assert _route_auth_level(app, path) == AuthLevel.NORMAL
    client = TestClient(app)

    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(path, headers=headers):
            pass
    assert exc.value.code == 1008
    forward.assert_not_awaited()


def test_cli_parser_accepts_api_keys_without_exposing_them_in_repr():
    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)

    args, remaining = parser.parse_known_args(
        [
            "--model-path",
            "/fake",
            "--api-key",
            "user-secret",
            "--admin-api-key",
            "admin-secret",
        ]
    )

    assert remaining == []
    assert args.api_key == "user-secret"
    assert args.admin_api_key == "admin-secret"

    server_arg_fields = {field.name: field for field in fields(ServerArgs)}
    assert not server_arg_fields["api_key"].repr
    assert not server_arg_fields["admin_api_key"].repr


def test_normal_routes_remain_open_when_no_key_is_configured():
    client = TestClient(create_app(_server_args()))

    assert client.get("/openapi.json").status_code == 200
    assert client.post("/v1/set_lora").status_code == 422


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/v1/set_lora", {"lora_nickname": "default"}),
        ("/v1/merge_lora_weights", {}),
        ("/v1/unmerge_lora_weights", {}),
        ("/update_weights_from_disk", {"model_path": "/fake"}),
        ("/update_weights_from_tensor", {"serialized_named_tensors": ["fake"]}),
        (
            "/update_weights_from_tensor_checker",
            {
                "target_module": "transformer",
                "expected_named_tensors_sha256": {"fake.weight": "0" * 64},
            },
        ),
        ("/get_weights_checksum", {"module_names": ["transformer"]}),
        ("/release_memory_occupation", {}),
        ("/resume_memory_occupation", {}),
    ],
)
@pytest.mark.parametrize(
    "api_key,admin_api_key,token,status",
    [
        (None, None, None, 200),
        ("user-secret", None, None, 401),
        ("user-secret", None, "user-secret", 200),
        (None, "admin-secret", None, 401),
        (None, "admin-secret", "admin-secret", 200),
        ("user-secret", "admin-secret", "user-secret", 401),
        ("user-secret", "admin-secret", "admin-secret", 200),
    ],
)
def test_management_routes_follow_srt_auth_policy(
    monkeypatch, path, payload, api_key, admin_api_key, token, status
):
    forward = AsyncMock(
        return_value=SimpleNamespace(error=None, output={"success": True})
    )
    monkeypatch.setattr(async_scheduler_client, "forward", forward)
    client = TestClient(create_app(_server_args(api_key, admin_api_key)))

    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    assert client.post(path, json=payload, headers=headers).status_code == status
    if status == 200:
        forward.assert_awaited_once()
    else:
        forward.assert_not_awaited()


def test_admin_key_only_protects_management_and_keeps_normal_routes_open():
    client = TestClient(create_app(_server_args(admin_api_key="admin-secret")))

    assert client.get("/openapi.json").status_code == 200
    assert client.post("/v1/set_lora").status_code == 401
    assert (
        client.post(
            "/v1/set_lora", headers={"Authorization": "Bearer admin-secret"}
        ).status_code
        == 422
    )


def test_api_key_protects_normal_routes_and_keeps_probes_public():
    client = TestClient(create_app(_server_args(api_key="user-secret")))

    assert client.get("/openapi.json").status_code == 401
    assert (
        client.get(
            "/openapi.json", headers={"Authorization": "Bearer wrong-secret"}
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/openapi.json", headers={"Authorization": "Bearer user-secret"}
        ).status_code
        == 200
    )
    assert client.get("/liveness").status_code == 200
    assert client.get("/liveness-admin").status_code == 401
    assert (
        client.post(
            "/v1/set_lora", headers={"Authorization": "Bearer user-secret"}
        ).status_code
        == 422
    )


def test_cors_preflight_does_not_require_authentication():
    client = TestClient(create_app(_server_args(api_key="user-secret")))

    response = client.options(
        "/openapi.json",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://example.com"


@pytest.mark.parametrize(
    "path,token,status",
    [
        ("/openapi.json", None, 401),
        ("/openapi.json", "wrong-secret", 401),
        ("/openapi.json", "user-secret", 200),
        ("/forced", "user-secret", 403),
    ],
)
def test_cors_headers_on_authentication_responses(path, token, status):
    app = create_app(_server_args(api_key="user-secret"))

    @app.get("/forced")
    @auth_level(AuthLevel.ADMIN_FORCE)
    async def forced():
        return {"success": True}

    headers = {"Origin": "https://example.com"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    response = TestClient(app).get(path, headers=headers)

    assert response.status_code == status
    assert response.headers.get("access-control-allow-origin") in (
        "*",
        "https://example.com",
    )
    assert response.headers["access-control-allow-credentials"] == "true"
    if status in (401, 403):
        assert response.json() == {
            "error": "Unauthorized" if status == 401 else "Forbidden"
        }
