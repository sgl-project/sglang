import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import (
    _freeze_gc_after_server_warmup,
    _send_disaggregation_warmup_requests,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()


class TestFreezeGcAfterWarmup(unittest.TestCase):
    """The freeze must survive a listener that is not accepting yet.

    Warmup polls /model_info until the server answers, but it is skipped under
    --skip-server-warmup, and uvicorn logs "startup complete" before the socket
    accepts, so the POST lands in that gap and the freeze is silently lost.
    """

    def setUp(self):
        # _freeze_gc_after_server_warmup reads get_serving(), which fails closed
        # until a context is published.
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        # ssl_verify_of wants a full ServerArgs and is irrelevant here.
        p = patch("sglang.srt.entrypoints.http_server.ssl_verify_of", lambda _a: False)
        p.start()
        self.addCleanup(p.stop)
        self.server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:30000")

    def _run(self, post):
        with (
            patch("sglang.srt.entrypoints.http_server.requests.post", post),
            patch("sglang.srt.entrypoints.http_server.logger") as log,
        ):
            _freeze_gc_after_server_warmup(self.server_args)
        return log

    def test_a_refused_connection_is_retried_until_the_listener_accepts(self):
        attempts = []

        def post(*args, **kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise requests.exceptions.ConnectionError("connection refused")
            return SimpleNamespace(raise_for_status=lambda: None)

        log = self._run(post)
        self.assertEqual(len(attempts), 3)
        log.warning.assert_not_called()

    def test_a_non_connection_error_is_not_retried(self):
        attempts = []

        def post(*args, **kwargs):
            attempts.append(1)
            raise requests.exceptions.HTTPError("500")

        log = self._run(post)
        self.assertEqual(len(attempts), 1)
        log.warning.assert_called_once()

    def test_the_retry_is_bounded(self):
        def post(*args, **kwargs):
            raise requests.exceptions.ConnectionError("connection refused")

        with patch(
            "sglang.srt.entrypoints.http_server._FREEZE_GC_CONNECT_TIMEOUT_SECS", 0
        ):
            log = self._run(post)
        log.warning.assert_called_once()
