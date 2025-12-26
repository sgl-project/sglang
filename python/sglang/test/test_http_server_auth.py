"""
Unit tests for HTTP server admin auth.

Usage:
    python3 -m pytest test/test_http_server_admin_auth.py -v
"""

import importlib.util
import os
import sys
import unittest


def _load_auth_module():
    """Load auth.py directly, avoiding importing the full sglang package.

    This keeps the test importable even if optional runtime deps (e.g. orjson/httpx)
    are not installed in the unit test environment.
    """
    this_dir = os.path.dirname(__file__)
    python_dir = os.path.abspath(os.path.join(this_dir, "..", ".."))
    auth_path = os.path.join(python_dir, "sglang", "srt", "utils", "auth.py")

    module_name = "_sglang_srt_utils_auth_for_test"
    spec = importlib.util.spec_from_file_location(module_name, auth_path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    # dataclasses (py3.12) may consult sys.modules during class processing
    sys.modules[module_name] = m
    spec.loader.exec_module(m)
    return m


_auth = _load_auth_module()
decide_request_auth = _auth.decide_request_auth


class TestHttpServerAdminAuth(unittest.TestCase):
    def test_no_admin_key_backward_compatible(self):
        optional_paths = {"/clear_hicache_storage_backend"}
        force_paths = {"/force_auth_demo"}

        # No api_key configured -> everything open (legacy)
        self.assertTrue(
            decide_request_auth(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key=None,
                admin_api_key=None,
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertTrue(
            decide_request_auth(
                method="POST",
                path="/clear_hicache_storage_backend",
                authorization_header=None,
                api_key=None,
                admin_api_key=None,
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )

        # Force-auth endpoints must be rejected when admin_api_key is not configured
        d = decide_request_auth(
            method="POST",
            path="/force_auth_demo",
            authorization_header=None,
            api_key=None,
            admin_api_key=None,
            admin_optional_auth_paths=optional_paths,
            admin_force_auth_paths=force_paths,
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.status_code, 403)

        # api_key configured -> everything gated by that key (including admin-ish routes) (legacy)
        self.assertFalse(
            decide_request_auth(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key="user",
                admin_api_key=None,
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertTrue(
            decide_request_auth(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key=None,
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertFalse(
            decide_request_auth(
                method="POST",
                path="/clear_hicache_storage_backend",
                authorization_header="Bearer wrong",
                api_key="user",
                admin_api_key=None,
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertTrue(
            decide_request_auth(
                method="POST",
                path="/clear_hicache_storage_backend",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key=None,
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )

        # Force-auth endpoints must still be rejected even if api_key is configured
        d = decide_request_auth(
            method="POST",
            path="/force_auth_demo",
            authorization_header="Bearer user",
            api_key="user",
            admin_api_key=None,
            admin_optional_auth_paths=optional_paths,
            admin_force_auth_paths=force_paths,
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.status_code, 403)

    def test_with_admin_key_admin_routes_require_admin(self):
        optional_paths = {"/clear_hicache_storage_backend"}
        force_paths = {"/force_auth_demo"}

        # admin_api_key only:
        # - normal endpoints open
        # - optional/force endpoints require admin_api_key
        self.assertTrue(
            decide_request_auth(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key=None,
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertTrue(
            decide_request_auth(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key=None,
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )

        # Optional endpoints require admin_api_key when admin_api_key is configured.
        self.assertTrue(
            decide_request_auth(
                method="POST",
                path="/clear_hicache_storage_backend",
                authorization_header="Bearer admin",
                api_key=None,
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertFalse(
            decide_request_auth(
                method="POST",
                path="/clear_hicache_storage_backend",
                authorization_header="Bearer user",
                api_key=None,
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )

        d = decide_request_auth(
            method="POST",
            path="/force_auth_demo",
            authorization_header="Bearer admin",
            api_key=None,
            admin_api_key="admin",
            admin_optional_auth_paths=optional_paths,
            admin_force_auth_paths=force_paths,
        )
        self.assertTrue(d.allowed)

    def test_with_admin_key_but_no_user_api_key(self):
        optional_paths = {"/clear_hicache_storage_backend"}
        force_paths = {"/force_auth_demo"}

        # both api_key and admin_api_key configured:
        # - normal endpoints require api_key
        # - optional endpoints require admin_api_key (api_key is NOT accepted)
        # - force endpoints require admin_api_key
        self.assertTrue(
            decide_request_auth(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertFalse(
            decide_request_auth(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        # Optional endpoints must require admin_api_key when both keys are configured.
        self.assertFalse(
            decide_request_auth(
                method="POST",
                path="/clear_hicache_storage_backend",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertTrue(
            decide_request_auth(
                method="POST",
                path="/clear_hicache_storage_backend",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertFalse(
            decide_request_auth(
                method="POST",
                path="/force_auth_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )
        self.assertTrue(
            decide_request_auth(
                method="POST",
                path="/force_auth_demo",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                admin_optional_auth_paths=optional_paths,
                admin_force_auth_paths=force_paths,
            ).allowed
        )


if __name__ == "__main__":
    unittest.main()
