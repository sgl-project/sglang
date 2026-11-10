"""The /health failure py-spy dump must never break the 503 path: disabled
by default, shared between concurrent probes, and a broken dump still leaves
the server marked unhealthy."""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=16, suite="base-a-test-cpu")

_PATH = "sglang.srt.entrypoints.http_server.pyspy_dump_all_sglang_processes"


class TestHealthFailPyspyDump(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        http_server._health_fail_pyspy_dump = None

    async def test_disabled_by_default_never_dumps(self):
        """At the default 0 nothing is scheduled and the state flip is immediate."""
        ran = []
        # Pin to 0 explicitly: the test environment may export the knob (e.g. an
        # e2e pod box env), and "default" then never exercises this branch.
        with envs.SGLANG_DEBUG_HEALTH_FAIL_PYSPY_DUMP_INTERVAL_SECS.override(0):
            with patch(_PATH) as dump:
                await http_server._maybe_pyspy_dump_on_health_fail()
                http_server._run_after_health_fail_dump(lambda: ran.append(1))
        dump.assert_not_called()
        self.assertEqual(ran, [1])

    async def test_broken_dump_still_flips_state_after_it(self):
        """A raising dump surfaces as one log line, not as a 500 or a skipped flip."""
        ran = []

        def broken(**kwargs):
            raise RuntimeError("py-spy exploded")

        with envs.SGLANG_DEBUG_HEALTH_FAIL_PYSPY_DUMP_INTERVAL_SECS.override(60):
            with patch(_PATH, side_effect=broken):
                with self.assertLogs(http_server.logger, level="ERROR") as logs:
                    await http_server._maybe_pyspy_dump_on_health_fail()
                    http_server._run_after_health_fail_dump(lambda: ran.append(1))
        self.assertEqual(ran, [1])
        self.assertEqual(sum("py-spy exploded" in line for line in logs.output), 1)

    async def test_concurrent_probes_share_one_dump_and_flip_after_it(self):
        """Concurrent probes share one dump; a cancelled probe waits for it to flip."""
        release = threading.Event()
        calls = []
        ran = []

        def slow(**kwargs):
            calls.append(kwargs)
            release.wait(timeout=10)
            return True

        with envs.SGLANG_DEBUG_HEALTH_FAIL_PYSPY_DUMP_INTERVAL_SECS.override(60):
            with patch(_PATH, side_effect=slow):
                first = asyncio.create_task(
                    http_server._maybe_pyspy_dump_on_health_fail()
                )
                await asyncio.sleep(0)  # first reaches its await on the shared future
                second = asyncio.create_task(
                    http_server._maybe_pyspy_dump_on_health_fail()
                )
                await asyncio.sleep(0)  # second joins the same future
                second.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await second
                # The cancelled probe's state flip is deferred until the dump ends.
                http_server._run_after_health_fail_dump(lambda: ran.append("second"))
                self.assertEqual(ran, [])
                release.set()
                await first  # done callbacks (clear, deferred flip) ran before this
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], {"min_interval_secs": 60})
        self.assertEqual(ran, ["second"])
        self.assertIsNone(http_server._health_fail_pyspy_dump)

    async def test_unhealthy_flip_guards_against_recovery_during_dump(self):
        """Without a newer probe the state flips to UnHealthy after the dump;
        a probe that succeeded while the dump ran keeps the server Up."""
        release = threading.Event()

        def slow(**kwargs):
            release.wait(timeout=10)
            return True

        async def generate_request(gri, request):
            await asyncio.Event().wait()
            yield  # never reached: the probe times out first

        def new_manager():
            return SimpleNamespace(
                gracefully_exit=False,
                server_status=http_server.ServerStatus.Up,
                is_generation=True,
                last_receive_tstamp=time.time() - 10.0,
                rid_to_state={},
                generate_request=generate_request,
            )

        request = SimpleNamespace(url=SimpleNamespace(path="/health_generate"))

        async def run_probe(manager):
            manager.last_receive_tstamp = time.time() - 10.0
            probe = asyncio.create_task(http_server.health_generate(request))
            await asyncio.sleep(1.2)  # let the patched 1 s health loop time out
            for _ in range(10_000):
                if http_server._health_fail_pyspy_dump is not None:
                    return probe
                if probe.done():
                    self.fail(f"probe died before the dump: {probe.exception()!r}")
                await asyncio.sleep(0)
            self.fail("dump future never appeared")

        tm = new_manager()
        with (
            patch.object(http_server, "HEALTH_CHECK_TIMEOUT", 1),
            envs.SGLANG_DEBUG_HEALTH_FAIL_PYSPY_DUMP_INTERVAL_SECS.override(60),
            envs.SGLANG_DIAG_BYPASS_HEALTH_GENERATE.override(False),
            patch(_PATH, side_effect=slow),
            patch.object(
                http_server,
                "get_disagg",
                lambda: SimpleNamespace(
                    disaggregation_mode=http_server.DisaggregationMode.NULL.value
                ),
            ),
            patch.object(
                http_server, "_global_state", SimpleNamespace(tokenizer_manager=tm)
            ),
        ):
            # A newer probe succeeded while the dump ran: its Up wins.
            probe = await run_probe(tm)
            tm.last_receive_tstamp = time.time()
            release.set()
            response = await probe
            self.assertEqual(response.status_code, 503)
            self.assertEqual(tm.server_status, http_server.ServerStatus.Up)

            # No newer probe: the flip to UnHealthy happens after the dump.
            http_server._health_fail_pyspy_dump = None
            release = threading.Event()
            tm.server_status = http_server.ServerStatus.Up
            probe = await run_probe(tm)
            release.set()
            response = await probe
            self.assertEqual(response.status_code, 503)
            self.assertEqual(tm.server_status, http_server.ServerStatus.UnHealthy)


if __name__ == "__main__":
    unittest.main()
