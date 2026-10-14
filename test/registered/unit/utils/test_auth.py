"""Unit tests for srt/utils/auth.py — no server, no model loading."""

import unittest
from pathlib import Path

from sglang.srt.utils.auth import (
    ADMIN_API_KEY_HELP,
    API_KEY_HELP,
    AuthDecision,
    AuthLevel,
    auth_level,
    decide_request_auth,
    single_key_authorizes_admin_optional,
    warn_single_key_admin_optional,
)

# Paths named by #43706. Literals, so dropping one from the warning or help fails here.
_ADMIN_OPTIONAL_EXAMPLES = (
    "/update_weights_from_disk",
    "/load_lora_adapter",
    "/start_profile",
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SERVER_ARGS_DOC = (
    _REPO_ROOT / "docs" / "docs" / "advanced_features" / "server_arguments.mdx"
)
_SERVING_FIELDS = (
    _REPO_ROOT / "python" / "sglang" / "srt" / "arg_groups" / "fields" / "serving.py"
)
_HTTP_SERVER = (
    _REPO_ROOT / "python" / "sglang" / "srt" / "entrypoints" / "http_server.py"
)

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

    def test_admin_optional_legacy_single_key_compat(self):
        """Ordinary keys still authorize ADMIN_OPTIONAL management routes.

        The single-key fallback is the #15908 compatibility path. Issue #43706
        warns about it; flipping this branch would reject existing deployments
        that call management endpoints with only --api-key.
        """
        path = "/update_weights_from_disk"
        cases = (
            ("no_credential", None, "user-secret", None, False),
            ("ordinary_api_key", "Bearer user-secret", "user-secret", None, True),
            ("wrong_ordinary_key", "Bearer wrong", "user-secret", None, False),
            (
                "both_keys_ordinary_rejected",
                "Bearer user-secret",
                "user-secret",
                "admin-secret",
                False,
            ),
            (
                "both_keys_admin_accepted",
                "Bearer admin-secret",
                "user-secret",
                "admin-secret",
                True,
            ),
            ("neither_key_open", None, None, None, True),
            ("empty_strings_count_as_unset", None, "", "", True),
        )
        for name, header, api_key, admin_api_key, allowed in cases:
            with self.subTest(name=name):
                decision = decide_request_auth(
                    method="POST",
                    path=path,
                    authorization_header=header,
                    api_key=api_key,
                    admin_api_key=admin_api_key,
                    auth_level=AuthLevel.ADMIN_OPTIONAL,
                )
                self.assertIs(decision.allowed, allowed)


class TestSingleKeyAdminOptionalWarning(CustomTestCase):
    """Startup warning for the legacy single-key management credential."""

    def test_warns_only_for_the_single_key_shape(self):
        shapes = (
            ("ordinary_only", "user-secret", None, True),
            ("both_keys", "user-secret", "admin-secret", False),
            ("neither_key", None, None, False),
            ("admin_only", None, "admin-secret", False),
            ("empty_ordinary_key", "", None, False),
            ("empty_admin_key_counts_as_unset", "user-secret", "", True),
        )
        for name, api_key, admin_api_key, expect_warning in shapes:
            with self.subTest(name=name):
                self.assertIs(
                    single_key_authorizes_admin_optional(api_key, admin_api_key),
                    expect_warning,
                )
                if expect_warning:
                    with self.assertLogs(
                        "sglang.srt.utils.auth", level="WARNING"
                    ) as logs:
                        warn_single_key_admin_optional(api_key, admin_api_key)
                    text = "\n".join(logs.output)
                    self.assertEqual(len(logs.records), 1)
                    self.assertEqual(logs.records[0].levelname, "WARNING")
                    for path in _ADMIN_OPTIONAL_EXAMPLES:
                        self.assertIn(path, text)
                    self.assertIn("--admin-api-key", text)
                    self.assertNotIn(str(api_key), text)
                else:
                    with self.assertNoLogs("sglang.srt.utils.auth", level="WARNING"):
                        warn_single_key_admin_optional(api_key, admin_api_key)

    def test_help_and_docs_name_the_exposure(self):
        """Operators can see the single-key exposure without reading the issue."""
        for help_text in (API_KEY_HELP, ADMIN_API_KEY_HELP):
            for path in _ADMIN_OPTIONAL_EXAMPLES:
                self.assertIn(path, help_text)
            self.assertIn("--admin-api-key", help_text)
        self.assertIn("do not accept --api-key", ADMIN_API_KEY_HELP)

        serving_source = _SERVING_FIELDS.read_text(encoding="utf-8")
        self.assertIn("API_KEY_HELP", serving_source)
        self.assertIn("ADMIN_API_KEY_HELP", serving_source)

        http_server = _HTTP_SERVER.read_text(encoding="utf-8")
        self.assertIn("warn_single_key_admin_optional(", http_server)

        doc = _SERVER_ARGS_DOC.read_text(encoding="utf-8")
        for path in _ADMIN_OPTIONAL_EXAMPLES:
            self.assertGreaterEqual(doc.count(path), 2)
        self.assertIn("ordinary API key also authorizes", doc)
        self.assertIn("ordinary key also authorizes", doc)


if __name__ == "__main__":
    unittest.main()
