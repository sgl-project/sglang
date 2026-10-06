import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import (
    _execute_server_warmup,
    _send_disaggregation_warmup_requests,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

URLS = ["http://localhost:30000", "http://localhost:30001"]


class TestRustServerWarmup(unittest.TestCase):
    def test_warms_each_local_listener_concurrently(self):
        server_args = SimpleNamespace(url=lambda port: f"http://localhost:{port}")
        model_info = SimpleNamespace(
            status_code=200, json=lambda: {"is_generation": True}
        )
        # Fails unless both listeners' warmups are in flight together.
        both_posted = threading.Barrier(len(URLS), timeout=5)

        def post(url, **kwargs):
            both_posted.wait()
            return SimpleNamespace(status_code=200)

        with (
            get_context().override_server_args(),
            envs.SGLANG_RUST_SERVER.override(True),
            patch.object(
                http_server, "rust_listener_ports_on_node", return_value=[30000, 30001]
            ),
            patch.object(http_server, "ssl_verify_of", return_value=True),
            patch.object(http_server, "kill_process_tree"),
            patch.object(http_server.time, "sleep"),
            patch.object(http_server.requests, "get", return_value=model_info) as get,
            patch.object(http_server.requests, "post", side_effect=post) as posted,
        ):
            self.assertTrue(_execute_server_warmup(server_args))

        self.assertEqual(
            [c.args[0] for c in get.call_args_list],
            [url + "/model_info" for url in URLS],
        )
        self.assertEqual(
            sorted(c.args[0] for c in posted.call_args_list),
            [url + "/generate" for url in URLS],
        )
        for c in posted.call_args_list:
            self.assertEqual(c.kwargs["json"]["text"], "The capital city of France is")
            self.assertEqual(c.kwargs["headers"], {"x-sglang-startup-warmup": "1"})


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
        dp_ranks_per_url = 2
        num_requests = len(URLS) * dp_ranks_per_url
        all_started = asyncio.Event()
        calls = []
        sessions = []

        class Response:
            status = 200

            async def __aenter__(self):
                if len(calls) == num_requests:
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
                urls=URLS,
                dp_ranks_per_url=dp_ranks_per_url,
                headers={"Authorization": "Bearer token"},
                ssl_verify=False,
                timeout=123,
            )

        self.assertEqual(status_codes, [200] * num_requests)
        self.assertEqual(len(calls), num_requests)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(
            sessions[0].kwargs["headers"], {"Authorization": "Bearer token"}
        )
        self.assertEqual(sessions[0].kwargs["timeout"].total, 123)

        calls_by_target = {
            (args[0], kwargs["json"]["routed_dp_rank"]): kwargs
            for args, kwargs in calls
        }
        self.assertEqual(
            set(calls_by_target),
            {
                (url + "/generate", dp_rank)
                for url in URLS
                for dp_rank in range(dp_ranks_per_url)
            },
        )

        for (_, dp_rank), kwargs in calls_by_target.items():
            self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12, 13])
            self.assertEqual(kwargs["json"]["bootstrap_host"], FAKE_BOOTSTRAP_HOST)
            self.assertEqual(kwargs["json"]["bootstrap_room"], dp_rank)
            self.assertFalse(kwargs["ssl"])


if __name__ == "__main__":
    unittest.main()
