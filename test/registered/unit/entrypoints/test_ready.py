import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestReadyEndpoint(CustomTestCase):
    def call_ready(
        self,
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
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=tokenizer_manager)
        )
        try:
            return asyncio.run(http_server.ready())
        finally:
            http_server._global_state = prior_state

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
