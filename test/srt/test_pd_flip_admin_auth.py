import ast
import importlib.util
import sys
from pathlib import Path


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
