import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
        for rust, payload in [
            (False, None),
            (True, None),
            (True, {"input_ids": [10, 11, 12]}),
        ]:
            with (
                self.subTest(rust=rust, payload=payload),
                envs.SGLANG_RUST_SERVER.override(rust),
            ):
                await self.check_requests(rust, payload)

    async def check_requests(self, rust, payload):
        override = get_context().override_server_args(dp_size=4, port=30000)
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(
            dp_size=4, url=lambda port=30000: f"http://[::1]:{port}"
        )
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
            urls = http_server._server_warmup_urls(server_args)
            self.assertEqual(
                urls,
                [f"http://[::1]:{30000 + (rank if rust else 0)}" for rank in range(4)],
            )
            status_codes = await http_server._send_warmup_requests(
                urls=urls,
                json_data=payload,
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

        for dp_rank, (args, kwargs) in enumerate(calls):
            self.assertEqual(args, (urls[dp_rank] + "/generate",))
            self.assertFalse(kwargs["ssl"])
            if payload is not None:
                self.assertEqual(kwargs["json"], payload)
            else:
                self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12, 13])
                self.assertEqual(kwargs["json"]["bootstrap_host"], FAKE_BOOTSTRAP_HOST)
                self.assertEqual(kwargs["json"]["bootstrap_room"], dp_rank)
                self.assertEqual(kwargs["json"]["routed_dp_rank"], dp_rank)


class TestRustServerStartup(unittest.TestCase):
    def test_warmup_before_ready(self):
        args = SimpleNamespace(
            url=lambda port=30000: f"http://localhost:{port}",
            api_key=None,
            admin_api_key=None,
            language_only=False,
            language_model_only=False,
            skip_tokenizer_init=True,
            debug_tensor_dump_input_file=None,
        )
        ports = SimpleNamespace(instance_id="token")
        urls = [args.url(port=p) for p in (30000, 30001)]
        for skip, mode, codes, fail_ready in [
            (False, "null", [200, 200], False),
            (False, "null", [200, 200], True),
            (False, "null", [200, 503], False),
            (False, "prefill", [200, 200], False),
            (False, "prefill", [200, 503], False),
            (True, "null", [], False),
        ]:
            override = get_context().override_server_args(
                dp_size=2,
                port=30000,
                skip_server_warmup=skip,
                disaggregation_mode=mode,
            )
            override.install()
            self.addCleanup(override.restore)
            scheduler, callback, kill = Mock(), Mock(), Mock()
            warmup = AsyncMock(return_value=codes)

            def ready(url, **kw):
                self.assertEqual(warmup.await_count, int(not skip))
                self.assertEqual(kw["headers"]["X-SGLang-Startup-Token"], "token")
                if fail_ready and url == urls[-1] + "/startup_ready":
                    raise http_server.requests.exceptions.HTTPError("not ready")
                return Mock()

            with (
                self.subTest(skip=skip, mode=mode, codes=codes, fail_ready=fail_ready),
                envs.SGLANG_RUST_SERVER.override(True),
                patch.object(
                    http_server.Engine,
                    "_launch_subprocesses",
                    return_value=(None, None, ports, scheduler, None, None),
                ),
                patch.object(
                    http_server.requests,
                    "get",
                    return_value=Mock(
                        status_code=200, json=lambda: {"is_generation": True}
                    ),
                ),
                patch.object(http_server.requests, "post", side_effect=ready) as post,
                patch.multiple(
                    http_server,
                    time=Mock(),
                    ssl_verify_of=Mock(return_value=False),
                    kill_process_tree=kill,
                    _send_warmup_requests=warmup,
                ),
            ):
                http_server.launch_server(args, launch_callback=callback)
            warmed = skip or all(code == 200 for code in codes)
            success = warmed and not fail_ready
            self.assertEqual(kill.call_count, int(not success))
            self.assertEqual(callback.call_count, int(success))
            self.assertEqual(
                scheduler.block_until_scheduler_exits.call_count, int(success)
            )
            self.assertEqual(
                [c.args[0] for c in post.call_args_list],
                [url + "/startup_ready" for url in urls] if warmed else [],
            )
            if not skip:
                self.assertEqual(warmup.call_args.kwargs["urls"], urls)
                payload = warmup.call_args.kwargs["json_data"]
                if mode == "null":
                    self.assertEqual(payload["input_ids"], [10, 11, 12])
                else:
                    self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
