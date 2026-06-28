"""Unit tests for the public HTTP server tuning CLI flags.

Usage:
    python3 -m pytest test/registered/unit/server_args/test_http_server_tuning.py -v
"""

import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

try:
    import granian  # noqa: F401

    _HAS_GRANIAN = True
except ImportError:
    _HAS_GRANIAN = False

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Mock get_device() so all tests run on CPU-only CI runners.
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestKeepAliveDefault(unittest.TestCase):
    def test_env_default_is_65s(self):
        # 65s aligns with Go net/http (90s), reqwest (90s), Node (60s)
        # client pool defaults. Old 5s default tripped pool-reuse races.
        self.assertEqual(envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(), 65)


class TestHttpTuningDefaults(unittest.TestCase):
    def test_defaults_are_unset(self):
        # None == "use server default" for everything except backlog (2048
        # matches uvicorn/Granian's own default and is explicit to make the
        # value discoverable without reading uvicorn source).
        args = ServerArgs(model_path="dummy")
        self.assertIsNone(args.timeout_keep_alive)
        self.assertEqual(args.http_backlog, 2048)
        self.assertIsNone(args.http_limit_concurrency)
        self.assertIsNone(args.http_timeout_graceful_shutdown)
        self.assertIsNone(args.http2_max_concurrent_streams)
        self.assertIsNone(args.http2_max_frame_size)


class TestHttpTuningCliFlags(unittest.TestCase):
    def test_all_flags_parse(self):
        args = prepare_server_args(
            [
                "--model-path",
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                "--timeout-keep-alive",
                "120",
                "--http-backlog",
                "4096",
                "--http-limit-concurrency",
                "500",
                "--http-timeout-graceful-shutdown",
                "30",
                "--http2-max-concurrent-streams",
                "256",
                "--http2-max-frame-size",
                "65536",
            ]
        )
        self.assertEqual(args.timeout_keep_alive, 120)
        self.assertEqual(args.http_backlog, 4096)
        self.assertEqual(args.http_limit_concurrency, 500)
        self.assertEqual(args.http_timeout_graceful_shutdown, 30)
        self.assertEqual(args.http2_max_concurrent_streams, 256)
        self.assertEqual(args.http2_max_frame_size, 65536)


class TestKeepAliveResolution(unittest.TestCase):
    # The CLI flag overrides the env var; the env var is the fallback when
    # the CLI flag isn't set. The resolution logic lives in
    # _setup_and_run_http_server; we exercise it directly here so changes
    # don't silently regress.

    @staticmethod
    def _resolve(server_args):
        if server_args.timeout_keep_alive is not None:
            return server_args.timeout_keep_alive
        return envs.SGLANG_TIMEOUT_KEEP_ALIVE.get()

    def test_cli_flag_wins(self):
        args = ServerArgs(model_path="dummy", timeout_keep_alive=999)
        self.assertEqual(self._resolve(args), 999)

    def test_env_var_used_when_cli_unset(self):
        args = ServerArgs(model_path="dummy")
        self.assertEqual(self._resolve(args), envs.SGLANG_TIMEOUT_KEEP_ALIVE.get())


class TestUvicornWiring(unittest.TestCase):
    # Verify the uvicorn-side tuning kwargs flow through to uvicorn.run.
    # Granian-side wiring is covered by TestGranianHttp2Settings below.
    # _run_setup mocks uvicorn.run + skips _execute_server_warmup so we
    # can inspect the kwargs without actually starting a server.

    @staticmethod
    def _run_setup(server_args, mock_uvicorn_run):
        from sglang.srt.entrypoints.http_server import _setup_and_run_http_server

        # _setup_and_run_http_server tries to set up logging + middleware
        # before calling uvicorn. Stub out the pieces we don't care about.
        with patch("uvicorn.run", mock_uvicorn_run), patch(
            "sglang.srt.entrypoints.http_server.set_global_state"
        ), patch(
            "sglang.srt.entrypoints.http_server.set_uvicorn_logging_configs"
        ), patch(
            "sglang.srt.entrypoints.http_server.add_prometheus_track_response_middleware"
        ):
            try:
                _setup_and_run_http_server(
                    server_args,
                    tokenizer_manager=None,
                    template_manager=None,
                    port_args=None,
                    scheduler_infos=[{}],
                    subprocess_watchdog=None,
                )
            except Exception:
                # The test only cares about the kwargs at the uvicorn.run
                # boundary; any post-call exception is irrelevant.
                pass

    def test_default_no_flags_set(self):
        from unittest.mock import MagicMock

        args = ServerArgs(model_path="dummy")
        mock = MagicMock()
        self._run_setup(args, mock)
        self.assertTrue(mock.called)
        kwargs = mock.call_args.kwargs
        # Default behavior: keep-alive from env (65 with new default),
        # backlog=2048, no limit_concurrency or timeout_graceful_shutdown.
        self.assertEqual(kwargs["timeout_keep_alive"], 65)
        self.assertEqual(kwargs["backlog"], 2048)
        self.assertNotIn("limit_concurrency", kwargs)
        self.assertNotIn("timeout_graceful_shutdown", kwargs)

    def test_all_uvicorn_flags_threaded_through(self):
        from unittest.mock import MagicMock

        args = ServerArgs(
            model_path="dummy",
            timeout_keep_alive=120,
            http_backlog=4096,
            http_limit_concurrency=500,
            http_timeout_graceful_shutdown=30,
        )
        mock = MagicMock()
        self._run_setup(args, mock)
        self.assertTrue(mock.called)
        kwargs = mock.call_args.kwargs
        self.assertEqual(kwargs["timeout_keep_alive"], 120)
        self.assertEqual(kwargs["backlog"], 4096)
        self.assertEqual(kwargs["limit_concurrency"], 500)
        self.assertEqual(kwargs["timeout_graceful_shutdown"], 30)


@unittest.skipUnless(_HAS_GRANIAN, "granian not installed (pip install sglang[http2])")
class TestGranianHttp2Settings(unittest.TestCase):
    # Regression guard: Granian's HTTP/2 tunables go through
    # http2_settings=HTTP2Settings(...), NOT as flat top-level kwargs.
    # Force the multi-worker path (tokenizer_worker_num=2) so we hit
    # granian.Granian (mockable) rather than the embedded server.

    def test_passes_http2_settings_when_tunables_set(self):
        from granian.http import HTTP2Settings

        from sglang.srt.entrypoints.http_server import _run_granian_server

        with patch("granian.Granian") as MockGranian:
            _run_granian_server(
                host="127.0.0.1",
                port=30000,
                log_level="info",
                tokenizer_worker_num=2,
                http2_max_concurrent_streams=256,
                http2_max_frame_size=65536,
            )

        kwargs = MockGranian.call_args.kwargs
        self.assertIn("http2_settings", kwargs)
        self.assertIsInstance(kwargs["http2_settings"], HTTP2Settings)
        self.assertEqual(kwargs["http2_settings"].max_concurrent_streams, 256)
        self.assertEqual(kwargs["http2_settings"].max_frame_size, 65536)
        # Don't smuggle the same values as flat kwargs (that'd be a TypeError
        # from Granian and is the bug this test guards against).
        self.assertNotIn("http2_max_concurrent_streams", kwargs)
        self.assertNotIn("http2_max_frame_size", kwargs)

    def test_no_http2_settings_when_tunables_unset(self):
        from sglang.srt.entrypoints.http_server import _run_granian_server

        with patch("granian.Granian") as MockGranian:
            _run_granian_server(
                host="127.0.0.1",
                port=30000,
                log_level="info",
                tokenizer_worker_num=2,
            )

        kwargs = MockGranian.call_args.kwargs
        # When the operator sets nothing, we let Granian use its defaults
        # rather than pinning them from our side.
        self.assertNotIn("http2_settings", kwargs)


if __name__ == "__main__":
    unittest.main()
