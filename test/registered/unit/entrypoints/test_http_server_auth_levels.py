import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import app
from sglang.srt.utils.auth import (
    _get_auth_level_from_app_and_scope,
    decide_request_auth,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

ADMIN_PATHS = [
    "/load_lora_adapter",
    "/load_lora_adapter_from_tensors",
    "/unload_lora_adapter",
    "/start_profile",
    "/stop_profile",
    "/set_trace_level",
    "/freeze_gc",
]


class TestAdminEndpointAuthLevels(CustomTestCase):
    def _decide(self, path: str, authorization_header: str | None):
        scope = {"type": "http", "method": "POST", "path": path, "root_path": ""}
        return decide_request_auth(
            method="POST",
            path=path,
            authorization_header=authorization_header,
            api_key=None,
            admin_api_key="admin-secret",
            auth_level=_get_auth_level_from_app_and_scope(app, scope),
        )

    def test_state_mutating_endpoints_require_admin_api_key(self):
        """A route missing @auth_level resolves to NORMAL, which a server started
        with only --admin-api-key leaves open to unauthenticated requests."""
        for path in ADMIN_PATHS:
            with self.subTest(path=path):
                self.assertFalse(self._decide(path, None).allowed)
                self.assertTrue(self._decide(path, "Bearer admin-secret").allowed)


class TestHiCacheClearAdminGuard(CustomTestCase):
    """POST /hicache/storage-backend/clear must refuse when the server was
    started without --admin-api-key (issue #44191: an ordinary API key could
    otherwise erase the persistent hierarchical cache backend)."""

    def _call_clear_endpoint(self, admin_api_key):
        serving = MagicMock()
        serving.admin_api_key = admin_api_key
        global_state = MagicMock()
        global_state.tokenizer_manager.clear_hicache_storage = AsyncMock(
            return_value=SimpleNamespace(success=True, message="")
        )
        with (
            patch.object(http_server, "get_serving", return_value=serving),
            patch.object(http_server, "_global_state", global_state),
        ):
            return asyncio.run(http_server.clear_hicache_storage_backend()), (
                global_state.tokenizer_manager.clear_hicache_storage
            )

    def test_clear_refuses_without_admin_api_key(self):
        resp, clear_mock = self._call_clear_endpoint(admin_api_key=None)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("admin API key", resp.body.decode())
        clear_mock.assert_not_called()

    def test_clear_delegates_with_admin_api_key_configured(self):
        resp, clear_mock = self._call_clear_endpoint(admin_api_key="admin-secret")
        clear_mock.assert_called_once_with()
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
