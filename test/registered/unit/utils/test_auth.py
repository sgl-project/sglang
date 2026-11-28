"""Unit tests for srt/utils/auth.py — no server, no model loading."""

import json
import unittest
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sglang.srt.environ import envs
from sglang.srt.utils.auth import (
    AuthDecision,
    AuthLevel,
    _unique_keys,
    add_api_key_middleware,
    admin_auth_headers,
    decide_request_auth,
    undeclared_routes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestAuthDecision(CustomTestCase):
    def test_not_allowed_with_custom_status(self):
        decision = AuthDecision(allowed=False, error_status_code=403)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.error_status_code, 403)

    def test_frozen(self):
        decision = AuthDecision(allowed=True)
        with self.assertRaises(AttributeError):
            decision.allowed = False


class TestAuthLevel(CustomTestCase):
    def test_is_string_enum(self):
        self.assertIsInstance(AuthLevel.NORMAL, str)
        # str mixin allows direct comparison with string values
        self.assertEqual(AuthLevel.NORMAL, "normal")

    def test_a_route_declared_twice_is_refused(self):
        # serde_json keeps the last of two keys silently; this load is the guard.
        with self.assertRaisesRegex(ValueError, "/x"):
            _unique_keys([("/x", "normal"), ("/x", "admin_force")])


class TestDecideRequestAuth(CustomTestCase):
    """Tests for the pure decide_request_auth function."""

    # ==================== Always-Allowed Paths ====================

    def test_options_method_always_allowed(self):
        decision = decide_request_auth(
            method="OPTIONS",
            path="/v1/chat/completions",
            authorization_header=None,
            api_key="secret",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertTrue(decision.allowed)

    def test_health_path_always_allowed(self):
        decision = decide_request_auth(
            method="GET",
            path="/health",
            authorization_header=None,
            api_key="secret",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_health_subpath_always_allowed(self):
        decision = decide_request_auth(
            method="GET",
            path="/health_generate",
            authorization_header=None,
            api_key="secret",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_ready_path_always_allowed(self):
        decision = decide_request_auth(
            method="GET",
            path="/ready",
            authorization_header=None,
            api_key="secret",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_metrics_path_always_allowed(self):
        decision = decide_request_auth(
            method="GET",
            path="/metrics",
            authorization_header=None,
            api_key="secret",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    # ==================== NORMAL Auth Level ====================

    def test_normal_no_keys_configured(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header=None,
            api_key=None,
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_normal_with_api_key_correct(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_normal_with_api_key_wrong(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="Bearer wrong-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(decision.allowed)

    def test_normal_with_api_key_missing_header(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header=None,
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(decision.allowed)

    def test_normal_only_admin_key_configured(self):
        """When only admin_api_key is configured, normal endpoints allow all."""
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header=None,
            api_key=None,
            admin_api_key="admin-secret",
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    # ==================== ADMIN_FORCE Auth Level ====================

    def test_admin_force_no_admin_key_configured(self):
        """ADMIN_FORCE without admin_api_key configured returns 403."""
        decision = decide_request_auth(
            method="POST",
            path="/admin/endpoint",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.error_status_code, 403)

    def test_admin_force_correct_admin_key(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/endpoint",
            authorization_header="Bearer admin-secret",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertTrue(decision.allowed)

    def test_admin_force_wrong_admin_key(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/endpoint",
            authorization_header="Bearer wrong-key",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.error_status_code, 401)

    def test_admin_force_api_key_not_accepted(self):
        """ADMIN_FORCE rejects api_key, only accepts admin_api_key."""
        decision = decide_request_auth(
            method="POST",
            path="/admin/endpoint",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(decision.allowed)

    # ==================== ADMIN_OPTIONAL Auth Level ====================

    def test_admin_optional_no_keys_configured(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header=None,
            api_key=None,
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(decision.allowed)

    def test_admin_optional_only_api_key_correct(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(decision.allowed)

    def test_admin_optional_only_api_key_wrong(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer wrong-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertFalse(decision.allowed)

    def test_admin_optional_only_admin_key_correct(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer admin-secret",
            api_key=None,
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(decision.allowed)

    def test_admin_optional_both_keys_requires_admin(self):
        """When both keys configured, ADMIN_OPTIONAL requires admin_api_key."""
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertFalse(decision.allowed)

    def test_admin_optional_both_keys_admin_accepted(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer admin-secret",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(decision.allowed)

    # ==================== Bearer Token Edge Cases ====================

    def test_malformed_authorization_header(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="NotBearer my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(decision.allowed)

    def test_empty_authorization_header(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(decision.allowed)

    def test_bearer_case_insensitive(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="BEARER my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)


AUTH_DECISIONS_FIXTURE = (
    Path(__file__).parents[4] / "rust/sglang-server/testdata/auth_decisions.json"
)
_API, _ADMIN = "sk-api", "sk-admin"


def build_auth_decisions_fixture() -> dict:
    """`decide_request_auth`'s status for each key setup, level and header; the
    Rust server's `auth.rs` tests answer the same cases.

    Regenerate the file with `render_auth_decisions_fixture()`.
    """
    authorizations = [
        None,
        "",
        "Bearer nope",
        f"Bearer {_API}",
        f"Bearer {_ADMIN}",
        f"bEaReR {_ADMIN}",
        f"Bearer  {_ADMIN}",
        f"Bearer {_ADMIN} ",
        f"Basic {_ADMIN}",
        "Bearer",
        f"Bearer {_ADMIN[:-1]}",
    ]
    requests = [("POST", "/generate", level) for level in AuthLevel]
    # Probes and preflights skip the check at any level.
    requests += [
        ("GET", path, AuthLevel.ADMIN_FORCE)
        for path in ["/health", "/health_generate", "/metrics", "/metrics/x", "/ready"]
    ]
    requests.append(("OPTIONS", "/generate", AuthLevel.ADMIN_FORCE))
    cases = []
    for api_key, admin_api_key in [
        (None, None),
        (_API, None),
        (None, _ADMIN),
        (_API, _ADMIN),
        ("", ""),
    ]:
        for method, path, level in requests:
            statuses = []
            for authorization in authorizations:
                decision = decide_request_auth(
                    method=method,
                    path=path,
                    authorization_header=authorization,
                    api_key=api_key,
                    admin_api_key=admin_api_key,
                    auth_level=level,
                )
                statuses.append(200 if decision.allowed else decision.error_status_code)
            cases.append(
                {
                    "api_key": api_key,
                    "admin_api_key": admin_api_key,
                    "method": method,
                    "path": path,
                    "level": level.value,
                    "statuses": statuses,
                }
            )
    return {"authorizations": authorizations, "cases": cases}


def render_auth_decisions_fixture() -> str:
    """The fixture as written: one case per line."""
    fixture = build_auth_decisions_fixture()
    cases = ",\n".join(f"    {json.dumps(case)}" for case in fixture["cases"])
    authorizations = json.dumps(fixture["authorizations"])
    return (
        f'{{\n  "authorizations": {authorizations},\n  "cases": [\n{cases}\n  ]\n}}\n'
    )


class TestRustAuthDecisionsFixture(CustomTestCase):
    def test_fixture_matches_python(self):
        self.assertEqual(
            AUTH_DECISIONS_FIXTURE.read_text(),
            render_auth_decisions_fixture(),
            "regenerate auth_decisions.json with render_auth_decisions_fixture()",
        )


class TestApiKeyMiddleware(CustomTestCase):
    """Routes take their levels from `auth_levels.json`; `/undeclared` is NORMAL."""

    def setUp(self):
        app = FastAPI()
        self.keys = (None, None)

        @app.get("/generate")
        async def normal():
            return {}

        @app.post("/start_profile")
        async def force():
            return {}

        @app.post("/dumper/{method}")
        async def force_by_method(method: str):
            return {}

        @app.post("/undeclared")
        async def undeclared():
            return {}

        @app.websocket("/v1/realtime")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            await websocket.close()

        add_api_key_middleware(app, get_keys=lambda: self.keys)
        self.app = app
        self.client = TestClient(app)

    def test_keys_are_read_per_request(self):
        self.assertEqual(self.client.get("/generate").status_code, 200)
        self.assertEqual(self.client.post("/undeclared").status_code, 200)
        for path in ("/start_profile", "/dumper/x"):
            self.assertEqual(self.client.post(path).status_code, 403, path)
        self.keys = ("user", "admin")
        admin = {"Authorization": "Bearer admin"}
        user = {"Authorization": "Bearer user"}
        for normal in (self.client.get("/generate"), self.client.post("/undeclared")):
            self.assertEqual(normal.status_code, 401)
        self.assertEqual(self.client.get("/generate", headers=user).status_code, 200)
        self.assertEqual(self.client.post("/undeclared", headers=user).status_code, 200)
        for path in ("/start_profile", "/dumper/x"):
            self.assertEqual(self.client.post(path, headers=user).status_code, 401)
            self.assertEqual(self.client.post(path, headers=admin).status_code, 200)

    def test_a_wrong_method_is_checked_at_the_routes_level(self):
        self.assertEqual(self.client.get("/start_profile").status_code, 403)
        self.keys = ("user", "admin")
        user = {"Authorization": "Bearer user"}
        admin = {"Authorization": "Bearer admin"}
        self.assertEqual(
            self.client.get("/start_profile", headers=user).status_code, 401
        )
        self.assertEqual(
            self.client.get("/start_profile", headers=admin).status_code, 405
        )

    def test_websocket_requires_the_api_key(self):
        self.keys = ("user", None)
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/v1/realtime"):
                pass
        with self.client.websocket_connect(
            "/v1/realtime", headers={"Authorization": "Bearer user"}
        ):
            pass

    def test_undeclared_routes_lists_the_extra_route(self):
        self.assertEqual(undeclared_routes(self.app), ["/undeclared"])


class TestAdminAuthHeaders(CustomTestCase):
    def test_given_key_wins_over_env(self):
        with envs.SGLANG_ADMIN_API_KEY.override("sk-env"):
            self.assertEqual(
                admin_auth_headers("sk-arg"), {"Authorization": "Bearer sk-arg"}
            )
            self.assertEqual(
                admin_auth_headers(None), {"Authorization": "Bearer sk-env"}
            )

    def test_no_key_sends_no_header(self):
        with envs.SGLANG_ADMIN_API_KEY.override(None):
            self.assertEqual(admin_auth_headers(None), {})


if __name__ == "__main__":
    unittest.main()
