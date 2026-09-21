import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
    _serving_url,
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


class TestWarmupProbeAddress(unittest.TestCase):
    """The warmup has to reach the server this process actually runs.

    The listener binds the resolved serving config, which a restore that
    redirects the listen address changes without touching the startup record.
    """

    def serving(self, host):
        return patch(
            "sglang.srt.entrypoints.http_server.get_serving",
            return_value=SimpleNamespace(host=host, port=30184),
        )

    def test_probes_follow_the_resolved_address(self):
        cases = (
            ("127.0.0.1", "http://127.0.0.1:30184"),
            ("0.0.0.0", "http://127.0.0.1:30184"),
            ("::", "http://[::1]:30184"),
            ("::1", "http://[::1]:30184"),
            ("my-server", "http://my-server:30184"),
        )
        for host, expected in cases:
            with self.subTest(host=host):
                with self.serving(host):
                    self.assertEqual(
                        _serving_url(SimpleNamespace(ssl_certfile=None)), expected
                    )


if __name__ == "__main__":
    unittest.main()
