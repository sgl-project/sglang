import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


HTTP_SERVER = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "entrypoints"
    / "http_server.py"
)


@dataclass
class StatusReq:
    session_id: str | None = None


def load_helper():
    tree = ast.parse(HTTP_SERVER.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_pd_flip_status_req_from_request"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"PDFlipMigrationStatusReq": StatusReq, "Request": object}
    exec(compile(module, str(HTTP_SERVER), "exec"), namespace)
    return namespace[helper.name]


def test_status_request_preserves_optional_session_query():
    helper = load_helper()
    assert helper(SimpleNamespace(query_params={"session_id": "session-7"})) == StatusReq(
        "session-7"
    )
    assert helper(SimpleNamespace(query_params={})) == StatusReq(None)


def test_status_route_uses_query_helper():
    source = HTTP_SERVER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    route = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "get_pd_flip_migration_status"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_pd_flip_status_req_from_request"
        for node in ast.walk(route)
    )
