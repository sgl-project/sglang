import unittest

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


if __name__ == "__main__":
    unittest.main()
