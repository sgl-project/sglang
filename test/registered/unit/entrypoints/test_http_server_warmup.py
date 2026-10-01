import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
    launch_server,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
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


class TestRustServerSidecarLifecycle(unittest.TestCase):
    def test_one_sidecar_wraps_the_top_level_rust_launch(self):
        events = []
        sidecar = MagicMock()
        sidecar.stop.side_effect = lambda: events.append("stop")
        scheduler_init_result = MagicMock()
        scheduler_init_result.block_until_scheduler_exits.side_effect = lambda: (
            events.append("block")
        )

        with (
            envs.SGLANG_RUST_SERVER.override(True),
            get_context().override_server_args(
                grpc_port=30001,
                sidecar="dynamo.sglang.sidecar",
                skip_server_warmup=True,
            ),
            patch(
                "sglang.srt.entrypoints.http_server.Engine._launch_subprocesses",
                return_value=(
                    None,
                    None,
                    None,
                    scheduler_init_result,
                    None,
                    None,
                ),
            ),
            patch(
                "sglang.srt.entrypoints.sidecar.start_sidecar",
                side_effect=lambda: (events.append("start"), sidecar)[1],
            ) as start_sidecar,
        ):
            launch_server(SimpleNamespace())

        start_sidecar.assert_called_once_with()
        sidecar.stop.assert_called_once_with()
        self.assertEqual(events, ["start", "block", "stop"])


if __name__ == "__main__":
    unittest.main()
