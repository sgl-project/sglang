"""/health_generate after a scheduler outside the process tree (a Ray actor) exited.

Requests are rejected with 503 then, so the health endpoint must answer 503 too:
output that arrives after the exit (sent before it, or by surviving data-parallel
ranks) must not mark the server Up again.
"""

import asyncio
import time
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StubTokenizerManager:
    """Answers each probe with output, like a scheduler that keeps producing it."""

    is_generation = True
    gracefully_exit = False

    def __init__(self, exit_error_after_probe=None):
        self.server_status = ServerStatus.Up
        self.scheduler_exit_error = None
        self.last_receive_tstamp = 0.0
        self.rid_to_state = {}
        self.probes = 0
        self._exit_error_after_probe = exit_error_after_probe

    async def generate_request(self, obj, request=None):
        self.probes += 1
        if self._exit_error_after_probe is not None:
            self.scheduler_exit_error = self._exit_error_after_probe
        self.last_receive_tstamp = time.time() + 1
        await asyncio.Event().wait()
        yield {}


class TestHealthGenerateSchedulerExit(CustomTestCase):
    def setUp(self):
        # The probe reads the disaggregation config.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _call(self, tokenizer_manager):
        prior_state = http_server.get_global_state()
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=tokenizer_manager)
        )
        request = SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
        try:
            return asyncio.run(http_server.health_generate(request))
        finally:
            http_server._global_state = prior_state

    def test_answering_scheduler_is_healthy(self):
        tm = _StubTokenizerManager()
        self.assertEqual(self._call(tm).status_code, 200)
        self.assertEqual(tm.server_status, ServerStatus.Up)

    def test_exit_before_probe_returns_503_without_probing(self):
        tm = _StubTokenizerManager()
        tm.scheduler_exit_error = "SGLang scheduler actor died"
        self.assertEqual(self._call(tm).status_code, 503)
        self.assertEqual(tm.probes, 0)

    def test_exit_during_probe_returns_503_despite_output(self):
        tm = _StubTokenizerManager(exit_error_after_probe="SGLang scheduler actor died")
        tm.server_status = ServerStatus.UnHealthy
        self.assertEqual(self._call(tm).status_code, 503)
        self.assertEqual(tm.probes, 1)
        self.assertEqual(tm.server_status, ServerStatus.UnHealthy)
        self.assertEqual(tm.rid_to_state, {})


if __name__ == "__main__":
    unittest.main()
