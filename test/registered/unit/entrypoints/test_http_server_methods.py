"""HTTP method regression tests without a running model or scheduler."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

STATE_CHANGING_PATHS = [
    "/flush_cache",
    "/start_profile",
    "/stop_profile",
    "/set_trace_level",
    "/freeze_gc",
    "/start_expert_distribution_record",
    "/stop_expert_distribution_record",
    "/dump_expert_distribution_record",
    "/release_memory_occupation",
    "/resume_memory_occupation",
    "/slow_down",
    "/open_session",
    "/close_session",
    "/configure_logging",
]


class TestStateChangingHttpMethods(CustomTestCase):
    def setUp(self):
        # Exercise the production routes without starting a model or installing
        # authentication middleware: method rejection must not depend on auth.
        app = FastAPI()
        app.router.routes.extend(http_server.app.router.routes)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_get_and_head_are_rejected_before_handler_execution(self):
        manager = AsyncMock()
        with (
            patch.object(
                http_server,
                "_global_state",
                SimpleNamespace(tokenizer_manager=manager),
            ),
            patch.object(http_server, "set_global_trace_level") as set_trace,
        ):
            for path in STATE_CHANGING_PATHS:
                for method in ("GET", "HEAD"):
                    with self.subTest(path=path, method=method):
                        response = self.client.request(
                            method, path, params={"level": 1}, json={}
                        )
                        self.assertEqual(response.status_code, 405)
                        self.assertEqual(response.headers["allow"], "POST")
            self.assertEqual(manager.mock_calls, [])
            set_trace.assert_not_called()

    def test_openapi_advertises_post_only(self):
        paths = self.client.app.openapi()["paths"]
        for path in STATE_CHANGING_PATHS:
            with self.subTest(path=path):
                self.assertIn("post", paths[path])
                self.assertNotIn("get", paths[path])
        for path in ("/ready", "/health", "/get_model_info", "/get_weights_by_name"):
            with self.subTest(path=path):
                self.assertIn("get", paths[path])

    def test_post_flush_cache_preserves_timeout_and_response(self):
        flush = AsyncMock(return_value=SimpleNamespace(success=True))
        state = SimpleNamespace(tokenizer_manager=SimpleNamespace(flush_cache=flush))
        with patch.object(http_server, "_global_state", state):
            response = self.client.post("/flush_cache", params={"timeout": 2.5})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Cache flushed.", response.text)
        flush.assert_awaited_once_with(timeout_s=2.5)


if __name__ == "__main__":
    unittest.main()
