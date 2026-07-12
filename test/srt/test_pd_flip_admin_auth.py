import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTH_PATH = ROOT / "python" / "sglang" / "srt" / "utils" / "auth.py"
HTTP_SERVER_PATH = (
    ROOT / "python" / "sglang" / "srt" / "entrypoints" / "http_server.py"
)


def _load_auth_module():
    spec = importlib.util.spec_from_file_location("pd_flip_auth_under_test", AUTH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _decide(auth, *, header=None, admin_api_key="admin-secret"):
    return auth.decide_request_auth(
        method="POST",
        path="/pd_flip/runtime_role/set",
        authorization_header=header,
        api_key="user-secret",
        admin_api_key=admin_api_key,
        auth_level=auth.AuthLevel.ADMIN_REQUIRED,
    )


def test_admin_required_returns_503_when_admin_key_is_not_configured():
    auth = _load_auth_module()

    decision = _decide(
        auth, header="Bearer user-secret", admin_api_key=None
    )

    assert not decision.allowed
    assert decision.error_status_code == 503


def test_admin_required_returns_401_for_missing_or_wrong_bearer():
    auth = _load_auth_module()

    for header in (None, "Bearer wrong", "Bearer user-secret"):
        decision = _decide(auth, header=header)
        assert not decision.allowed
        assert decision.error_status_code == 401


def test_admin_required_accepts_the_configured_admin_bearer():
    auth = _load_auth_module()

    assert _decide(auth, header="Bearer admin-secret").allowed


def test_existing_admin_levels_keep_their_absent_key_behavior():
    auth = _load_auth_module()
    common = dict(
        method="POST",
        path="/admin/control",
        authorization_header=None,
        api_key=None,
        admin_api_key=None,
    )

    optional = auth.decide_request_auth(
        **common, auth_level=auth.AuthLevel.ADMIN_OPTIONAL
    )
    forced = auth.decide_request_auth(**common, auth_level=auth.AuthLevel.ADMIN_FORCE)

    assert optional.allowed
    assert not forced.allowed
    assert forced.error_status_code == 403


def test_middleware_install_helper_detects_admin_required_endpoint():
    auth = _load_auth_module()

    @auth.auth_level(auth.AuthLevel.ADMIN_REQUIRED)
    def required_endpoint():
        pass

    app = SimpleNamespace(
        router=SimpleNamespace(routes=[SimpleNamespace(endpoint=required_endpoint)])
    )

    assert auth.app_has_admin_middleware_endpoints(app)
    assert auth.app_has_admin_force_endpoints(app)  # Legacy server import remains valid.


def _make_fastapi_app(auth, *, admin_api_key):
    fastapi = pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")

    app = fastapi.FastAPI()

    @app.get("/legacy-optional")
    @auth.auth_level(auth.AuthLevel.ADMIN_OPTIONAL)
    async def legacy_optional():
        return {"ok": True}

    @app.post("/admin-required")
    @auth.auth_level(auth.AuthLevel.ADMIN_REQUIRED)
    async def admin_required():
        return {"ok": True}

    # Mirror the HTTP server's middleware-install condition. In particular, an
    # ADMIN_REQUIRED route must install middleware even when neither key exists.
    if admin_api_key or auth.app_has_admin_force_endpoints(app):
        auth.add_api_key_middleware(
            app,
            api_key=None,
            admin_api_key=admin_api_key,
        )

    return testclient.TestClient(app)


def test_middleware_is_installed_for_admin_required_without_keys():
    auth = _load_auth_module()

    with _make_fastapi_app(auth, admin_api_key=None) as client:
        assert client.get("/legacy-optional").status_code == 200
        response = client.post("/admin-required")

    assert response.status_code == 503


def test_admin_required_middleware_enforces_configured_admin_key():
    auth = _load_auth_module()

    with _make_fastapi_app(auth, admin_api_key="admin-secret") as client:
        assert client.post("/admin-required").status_code == 401
        assert client.post(
            "/admin-required", headers={"Authorization": "Bearer wrong"}
        ).status_code == 401
        assert client.post(
            "/admin-required",
            headers={"Authorization": "Bearer admin-secret"},
        ).status_code == 200


def test_every_pd_flip_http_route_requires_admin_configuration():
    tree = ast.parse(HTTP_SERVER_PATH.read_text(encoding="utf-8"))
    markers = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        paths = []
        auth_marker = None
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if (
                isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "app"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                paths.append(decorator.args[0].value)
            if (
                isinstance(decorator.func, ast.Name)
                and decorator.func.id == "auth_level"
                and decorator.args
            ):
                auth_marker = ast.unparse(decorator.args[0])
        for path in paths:
            if path.startswith("/pd_flip/"):
                markers[path] = auth_marker

    assert markers, "expected at least one /pd_flip/* worker route"
    assert set(markers.values()) == {"AuthLevel.ADMIN_REQUIRED"}, markers
