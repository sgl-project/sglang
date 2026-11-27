# SPDX-License-Identifier: Apache-2.0
"""The diffusion app holds its routes to the levels in srt/utils/auth_levels.json."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
from sglang.srt.utils.auth import undeclared_routes


def _app(api_key=None, admin_api_key=None):
    # Both optional routers mounted, so every route the server can serve is here.
    args = SimpleNamespace(
        enable_metrics=False,
        media_gateway_key_file=None,
        api_key=api_key,
        admin_api_key=admin_api_key,
        pipeline_config=SimpleNamespace(
            supports_action_endpoint=lambda: True,
            supports_openpi_endpoint=lambda: True,
        ),
    )
    return create_app(args)


def test_every_route_declares_its_level():
    assert undeclared_routes(_app()) == []


def test_admin_route_needs_the_admin_key():
    # 403 without an admin key, even with the API key sent; 401 for a wrong key.
    path = "/update_weights_from_disk"
    client = TestClient(_app(api_key="user"))
    assert (
        client.post(path, headers={"Authorization": "Bearer user"}).status_code == 403
    )
    client = TestClient(
        _app(api_key="user", admin_api_key="admin"), raise_server_exceptions=False
    )
    assert (
        client.post(path, headers={"Authorization": "Bearer user"}).status_code == 401
    )
    # The admin key gets past auth; the handler's own answer does not matter here.
    response = client.post(path, headers={"Authorization": "Bearer admin"})
    assert response.status_code not in (401, 403)
