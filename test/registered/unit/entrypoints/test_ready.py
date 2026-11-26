import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _call_endpoint(
    handler,
    *,
    is_pause: bool = False,
    gracefully_exit: bool = False,
    server_status: ServerStatus = ServerStatus.Up,
):
    tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
    tokenizer_manager.is_pause = is_pause
    tokenizer_manager.gracefully_exit = gracefully_exit
    tokenizer_manager.server_status = server_status

    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer_manager))
    try:
        return asyncio.run(handler())
    finally:
        http_server._global_state = prior_state


class TestReadyEndpoint(CustomTestCase):
    def call_ready(self, **state):
        return _call_endpoint(http_server.ready, **state)

    def test_ready_while_accepting_requests(self):
        self.assertEqual(self.call_ready().status_code, 200)

    def test_not_ready_while_paused(self):
        self.assertEqual(self.call_ready(is_pause=True).status_code, 503)

    def test_not_ready_while_starting(self):
        self.assertEqual(
            self.call_ready(server_status=ServerStatus.Starting).status_code, 503
        )

    def test_not_ready_while_unhealthy(self):
        self.assertEqual(
            self.call_ready(server_status=ServerStatus.UnHealthy).status_code, 503
        )

    def test_not_ready_while_gracefully_exiting(self):
        self.assertEqual(self.call_ready(gracefully_exit=True).status_code, 503)


class TestSageMakerPingEndpoint(CustomTestCase):
    def call_ping(self, **state):
        return _call_endpoint(http_server.sagemaker_health, **state)

    def test_ping_ok_with_empty_body_when_up(self):
        # SageMaker's container contract: HTTP 200 with an empty body.
        response = self.call_ping()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, b"")

    def test_ping_ok_while_paused(self):
        # Unlike /ready, pausing must not fail the health check and get the instance replaced.
        self.assertEqual(self.call_ping(is_pause=True).status_code, 200)

    def test_ping_unavailable_while_starting(self):
        """/ping must not report healthy before warmup finishes."""
        self.assertEqual(
            self.call_ping(server_status=ServerStatus.Starting).status_code, 503
        )

    def test_ping_unavailable_while_unhealthy(self):
        self.assertEqual(
            self.call_ping(server_status=ServerStatus.UnHealthy).status_code, 503
        )

    def test_ping_unavailable_while_gracefully_exiting(self):
        self.assertEqual(self.call_ping(gracefully_exit=True).status_code, 503)


if __name__ == "__main__":
    unittest.main(verbosity=2)
