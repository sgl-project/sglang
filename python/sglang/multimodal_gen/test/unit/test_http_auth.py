from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import FlexibleArgumentParser
from sglang.srt.utils.auth import AuthLevel, _iter_effective_routes


def _server_args(api_key=None, admin_api_key=None):
    pipeline_config = SimpleNamespace(
        supports_action_endpoint=lambda: False,
        supports_openpi_endpoint=lambda: False,
    )
    return SimpleNamespace(
        api_key=api_key,
        admin_api_key=admin_api_key,
        pipeline_config=pipeline_config,
    )


def _route_auth_level(app, path):
    route = next(
        route
        for route in _iter_effective_routes(app)
        if getattr(route, "path", None) == path
    )
    return getattr(route.endpoint, "_auth_level", AuthLevel.NORMAL)


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


def test_sensitive_management_routes_require_admin_auth():
    app = create_app(_server_args(api_key="user-secret", admin_api_key="admin-secret"))
    client = TestClient(app)

    assert (
        client.post(
            "/v1/set_lora", headers={"Authorization": "Bearer user-secret"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/v1/set_lora", headers={"Authorization": "Bearer admin-secret"}
        ).status_code
        == 422
    )

    for path in (
        "/v1/set_lora",
        "/v1/merge_lora_weights",
        "/v1/unmerge_lora_weights",
        "/update_weights_from_disk",
        "/update_weights_from_tensor",
        "/release_memory_occupation",
        "/resume_memory_occupation",
        "/update_weights_from_tensor_checker",
        "/get_weights_checksum",
    ):
        assert _route_auth_level(app, path) == AuthLevel.ADMIN_OPTIONAL
