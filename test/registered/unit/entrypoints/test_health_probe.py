"""The one-token generation probe shared by /health and the frontend stall
watchdog: it reports whether the scheduler answered, and always cleans up."""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import health_probe
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(self, *, answers: bool, is_generation: bool = True):
        self.is_generation = is_generation
        self.rid_to_state = {}
        self.last_receive_tstamp = 0.0
        self.answers = answers
        self.probes = []
        self.cancelled = False

    async def generate_request(self, obj, request=None):
        self.probes.append(obj)
        self.rid_to_state[obj.rid] = object()
        if self.answers:
            self.last_receive_tstamp = time.time() + 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        yield {}


def _null_disagg():
    return patch.object(
        health_probe,
        "get_disagg",
        return_value=SimpleNamespace(disaggregation_mode="null"),
    )


class TestRunGenerationProbe(CustomTestCase):
    def _run(self, tm):
        async def main():
            ok = await health_probe.run_generation_probe(
                tm, timeout=0.2, poll_interval=0.01
            )
            # Let the cancelled task unwind.
            await asyncio.sleep(0)
            return ok

        with _null_disagg():
            return asyncio.run(main())

    def test_answered_probe_reports_healthy_and_cleans_up(self):
        tm = _FakeTokenizerManager(answers=True)
        self.assertTrue(self._run(tm))
        self.assertEqual(tm.rid_to_state, {})
        self.assertTrue(tm.cancelled)
        self.assertTrue(tm.probes[0].rid.startswith(HEALTH_CHECK_RID_PREFIX))

    def test_unanswered_probe_times_out_and_cleans_up(self):
        """A probe left in rid_to_state would count as an in-flight request and
        keep the frontend watchdog probing forever."""
        tm = _FakeTokenizerManager(answers=False)
        self.assertFalse(self._run(tm))
        self.assertEqual(tm.rid_to_state, {})
        self.assertTrue(tm.cancelled)


class TestProbeRequest(CustomTestCase):
    def test_pd_generation_probe_carries_the_fake_bootstrap(self):
        with patch.object(
            health_probe,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="prefill"),
        ):
            probe = health_probe._build_probe_request(is_generation=True, rid="r")
        self.assertIsInstance(probe, GenerateReqInput)
        self.assertEqual(probe.bootstrap_host, FAKE_BOOTSTRAP_HOST)
        self.assertEqual(probe.bootstrap_room, 0)
        self.assertEqual(probe.input_ids, [0])
        self.assertEqual(probe.sampling_params["max_new_tokens"], 1)

    def test_embedding_server_gets_an_embedding_probe(self):
        with _null_disagg():
            probe = health_probe._build_probe_request(is_generation=False, rid="r")
        self.assertIsInstance(probe, EmbeddingReqInput)


if __name__ == "__main__":
    unittest.main()
