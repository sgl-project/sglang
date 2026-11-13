"""Unit tests for srt/utils/auth.py — no server, no model loading."""

import unittest

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sglang.srt.utils.auth import (
    AuthDecision,
    AuthLevel,
    add_api_key_middleware,
    auth_level,
    decide_request_auth,
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


class TestAuthLevelDecorator(CustomTestCase):
    def test_decorator_sets_auth_level(self):
        @auth_level(AuthLevel.ADMIN_FORCE)
        def my_endpoint():
            pass

        self.assertEqual(my_endpoint._auth_level, AuthLevel.ADMIN_FORCE)


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


class TestApiKeyMiddleware(CustomTestCase):
    def setUp(self):
        app = FastAPI()
        self.keys = (None, None)

        @app.get("/normal")
        async def normal():
            return {}

        @app.post("/force")
        @auth_level(AuthLevel.ADMIN_FORCE)
        async def force():
            return {}

        @app.post("/force/{name}")
        @auth_level(AuthLevel.ADMIN_FORCE)
        async def force_by_name(name: str):
            return {}

        @app.post("/force/shadowed")
        async def shadowed():
            return {}

        @app.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            await websocket.close()

        add_api_key_middleware(app, get_keys=lambda: self.keys)
        self.client = TestClient(app)

    def test_keys_are_read_per_request(self):
        self.assertEqual(self.client.get("/normal").status_code, 200)
        self.assertEqual(self.client.post("/force").status_code, 403)
        self.assertEqual(self.client.post("/force/shadowed").status_code, 403)
        self.keys = ("user", "admin")
        admin = {"Authorization": "Bearer admin"}
        user = {"Authorization": "Bearer user"}
        self.assertEqual(self.client.get("/normal").status_code, 401)
        self.assertEqual(self.client.get("/normal", headers=user).status_code, 200)
        for path in ("/force", "/force/x"):
            self.assertEqual(self.client.post(path, headers=user).status_code, 401)
            self.assertEqual(self.client.post(path, headers=admin).status_code, 200)

    def test_websocket_requires_the_api_key(self):
        self.keys = ("user", None)
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/ws"):
                pass
        with self.client.websocket_connect(
            "/ws", headers={"Authorization": "Bearer user"}
        ):
            pass


if __name__ == "__main__":
    unittest.main()
