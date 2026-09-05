import asyncio
import unittest
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


class TestRustStartup(unittest.TestCase):
    def test_all_ranks_warm_up_before_ready(self):
        args = SimpleNamespace(
            url=lambda port: f"http://localhost:{port}",
            api_key=None,
            skip_tokenizer_init=False,
            debug_tensor_dump_input_file=None,
        )
        for skip, fail, mode in [
            (False, False, "null"),
            (False, False, "prefill"),
            (False, True, "prefill"),
            (True, False, "null"),
        ]:
            override = get_context().override_server_args(
                dp_size=2,
                port=30000,
                skip_server_warmup=skip,
                disaggregation_mode=mode,
            )
            override.install()
            self.addCleanup(override.restore)
            barrier, warmed, ready = Barrier(2, timeout=5), set(), []

            def post(url, **kwargs):
                self.assertEqual(kwargs["headers"]["X-SGLang-Startup-Token"], "token")
                if url.endswith("/generate"):
                    barrier.wait()
                    warmed.add(url)
                    return Mock(status_code=503 if fail and ":30001/" in url else 200)
                self.assertEqual(len(warmed), 0 if skip else 2)
                ready.append(url)
                return Mock(status_code=200)

            with (
                self.subTest(skip=skip, fail=fail, mode=mode),
                patch.object(http_server.requests, "post", side_effect=post),
                patch.object(http_server, "ssl_verify_of", return_value=False),
                patch.object(http_server, "kill_process_tree") as kill,
            ):
                result = http_server._warmup_and_mark_rust_server_ready(
                    args, SimpleNamespace(instance_id="token")
                )
            self.assertEqual(result, not fail)
            self.assertEqual(kill.call_count, int(fail))
            self.assertEqual(
                ready,
                []
                if fail
                else [args.url(p) + "/startup_ready" for p in (30000, 30001)],
            )


if __name__ == "__main__":
    unittest.main()
