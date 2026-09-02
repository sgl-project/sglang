import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestReadyEndpoint(CustomTestCase):
    def call_ready(self, *, is_pause: bool):
        prior_state = http_server.get_global_state()
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=SimpleNamespace(is_pause=is_pause))
        )
        try:
            return asyncio.run(http_server.ready())
        finally:
            http_server._global_state = prior_state

    def test_ready_while_accepting_requests(self):
        self.assertEqual(self.call_ready(is_pause=False).status_code, 200)

    def test_not_ready_while_paused(self):
        self.assertEqual(self.call_ready(is_pause=True).status_code, 503)


if __name__ == "__main__":
    unittest.main(verbosity=2)
