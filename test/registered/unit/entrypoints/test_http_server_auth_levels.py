import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import app
from sglang.srt.utils.auth import AUTH_LEVELS, _iter_leaf_routes, undeclared_routes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# Declared routes neither imported app has: the dumper's (registered under
# DUMPER_SERVER_PORT=reuse), the metrics mount (added at launch) and the Rust
# server's routes under it, and the Rust server's relay.
SERVED_ELSEWHERE = {
    "/dumper/{method}",
    "/metrics",
    "/metrics/",
    "/metrics/{path:path}",
    "/internal/peer",
}


class TestAuthLevels(CustomTestCase):
    """An undeclared route is NORMAL at runtime, so sglang's own must be declared."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.disaggregation.encoder.http_server import app as encoder_app

        cls.apps = (app, encoder_app)

    def test_every_route_declares_its_level(self):
        for served in self.apps:
            self.assertEqual(undeclared_routes(served), [])

    def test_every_declared_route_is_served(self):
        served = {
            prefix + route.path
            for served_app in self.apps
            for prefix, route in _iter_leaf_routes(served_app.routes)
        }
        self.assertLessEqual(set(AUTH_LEVELS) - served, SERVED_ELSEWHERE)

    def test_api_key_middleware_is_outermost_on_the_module_level_app(self):
        self.assertEqual(app.user_middleware[0].cls.__name__, "_ApiKeyASGIMiddleware")


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
