import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
        from sglang.srt.runtime_context import get_context

        # The warmup fan-out width comes from the published topology.
        override = get_context().override_server_args(dp_size=4)
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(dp_size=4)
        all_started = asyncio.Event()
        calls = []
        sessions = []

        class Response:
            status = 200

            async def __aenter__(self):
                if len(calls) == server_args.dp_size:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5)
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self):
                return b""

        class Session:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, *args, **kwargs):
                calls.append((args, kwargs))
                return Response()

        with patch("sglang.srt.entrypoints.http_server.aiohttp.ClientSession", Session):
            status_codes = await _send_disaggregation_warmup_requests(
                url="http://localhost:30000",
                headers={"Authorization": "Bearer token"},
                ssl_verify=False,
                timeout=123,
            )

        self.assertEqual(status_codes, [200] * server_args.dp_size)
        self.assertEqual(len(calls), server_args.dp_size)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(
            sessions[0].kwargs["headers"], {"Authorization": "Bearer token"}
        )
        self.assertEqual(sessions[0].kwargs["timeout"].total, 123)

        calls_by_rank = {
            kwargs["json"]["routed_dp_rank"]: (args, kwargs) for args, kwargs in calls
        }
        self.assertEqual(set(calls_by_rank), set(range(server_args.dp_size)))

        for dp_rank, (args, kwargs) in calls_by_rank.items():
            self.assertEqual(args, ("http://localhost:30000/generate",))
            self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12, 13])
            self.assertEqual(kwargs["json"]["bootstrap_host"], FAKE_BOOTSTRAP_HOST)
            self.assertEqual(kwargs["json"]["bootstrap_room"], dp_rank)
            self.assertFalse(kwargs["ssl"])


class TestSkippedWarmupGcFreeze(unittest.TestCase):
    def test_gc_freeze_waits_for_the_http_server(self):
        """With server warmup skipped, the startup GC freeze must reach the HTTP
        server, and the server reports Up only after it."""
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(skip_server_warmup=True)
        override.install()
        self.addCleanup(override.restore)

        seconds = [0]  # simulated time; the server listens from second 2
        freezes = []

        def sleep(duration):
            seconds[0] += duration

        def get(url, **kwargs):
            if seconds[0] < 2:
                raise requests.exceptions.ConnectionError("refused")
            return SimpleNamespace(status_code=200, text="", json=lambda: {})

        def post(url, **kwargs):
            if seconds[0] < 2:
                raise requests.exceptions.ConnectionError("refused")
            freezes.append((url, state.tokenizer_manager.server_status))
            return SimpleNamespace(raise_for_status=lambda: None)

        server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:30000")
        state = SimpleNamespace(tokenizer_manager=SimpleNamespace(server_status=None))
        with (
            patch.object(http_server, "_global_state", state),
            patch.object(http_server, "ssl_verify_of", lambda _: True),
            patch.object(http_server.time, "sleep", sleep),
            patch.object(http_server.requests, "get", get),
            patch.object(http_server.requests, "post", post),
        ):
            http_server._wait_and_warmup(server_args)

        self.assertEqual(freezes, [("http://127.0.0.1:30000/freeze_gc", None)])
        self.assertEqual(state.tokenizer_manager.server_status, ServerStatus.Up)

    def test_gc_freeze_gives_up_on_a_server_that_never_answers(self):
        """With server warmup skipped, a server that cannot reach its own URL must
        still report Up within the wait cap; it used to poll for up to 12 minutes."""
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(skip_server_warmup=True)
        override.install()
        self.addCleanup(override.restore)

        seconds = [0.0]  # simulated time; every request hangs until its timeout
        freezes = []

        def sleep(duration):
            seconds[0] += duration

        def get(url, timeout, **kwargs):
            seconds[0] += timeout
            raise requests.exceptions.ReadTimeout("no answer")

        def post(url, **kwargs):
            freezes.append(url)
            return SimpleNamespace(raise_for_status=lambda: None)

        server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:30000")
        state = SimpleNamespace(tokenizer_manager=SimpleNamespace(server_status=None))
        with (
            patch.object(http_server, "_global_state", state),
            patch.object(http_server, "ssl_verify_of", lambda _: True),
            patch.object(http_server.time, "sleep", sleep),
            patch.object(http_server.time, "monotonic", lambda: seconds[0]),
            patch.object(http_server.requests, "get", get),
            patch.object(http_server.requests, "post", post),
        ):
            http_server._wait_and_warmup(server_args)

        self.assertLessEqual(seconds[0], http_server._SKIPPED_WARMUP_WAIT_S + 1)
        self.assertEqual(freezes, [])
        self.assertEqual(state.tokenizer_manager.server_status, ServerStatus.Up)


if __name__ == "__main__":
    unittest.main()
