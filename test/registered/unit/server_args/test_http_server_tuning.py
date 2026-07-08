"""Unit tests for the public HTTP server tuning CLI flags.

Usage:
    python3 -m pytest test/registered/unit/server_args/test_http_server_tuning.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

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


class TestHttpTuningConfig(unittest.TestCase):
    # Defaults, CLI parsing, and CLI-vs-env resolution for the new tunables.

    def test_env_keep_alive_default_is_5s(self):
        # 5s is conservative for backward compat. Operators with longer
        # client pools override via SGLANG_TIMEOUT_KEEP_ALIVE,
        # --timeout-keep-alive, or the Engine constructor.
        self.assertEqual(envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(), 5)

    def test_dataclass_defaults_are_unset(self):
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
        self.assertIsNone(args.http2_keep_alive_interval)
        self.assertIsNone(args.http2_keep_alive_timeout)
        self.assertIsNone(args.http1_header_read_timeout)
        self.assertIsNone(args.http1_max_buffer_size)

    def test_all_cli_flags_parse(self):
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
                "--http2-keep-alive-interval",
                "30",
                "--http2-keep-alive-timeout",
                "10",
                "--http1-header-read-timeout",
                "15000",
                "--http1-max-buffer-size",
                "1048576",
            ]
        )
        self.assertEqual(args.timeout_keep_alive, 120)
        self.assertEqual(args.http_backlog, 4096)
        self.assertEqual(args.http_limit_concurrency, 500)
        self.assertEqual(args.http_timeout_graceful_shutdown, 30)
        self.assertEqual(args.http2_max_concurrent_streams, 256)
        self.assertEqual(args.http2_max_frame_size, 65536)
        self.assertEqual(args.http2_keep_alive_interval, 30)
        self.assertEqual(args.http2_keep_alive_timeout, 10)
        self.assertEqual(args.http1_header_read_timeout, 15000)
        self.assertEqual(args.http1_max_buffer_size, 1048576)

    def test_keep_alive_cli_overrides_env(self):
        from sglang.srt.utils.http_server_tuning import resolved_keep_alive_timeout

        # CLI value wins regardless of env var.
        args = ServerArgs(model_path="dummy", timeout_keep_alive=999)
        self.assertEqual(resolved_keep_alive_timeout(args), 999)

        # CLI unset → env var (which has its own default).
        args = ServerArgs(model_path="dummy")
        self.assertEqual(
            resolved_keep_alive_timeout(args), envs.SGLANG_TIMEOUT_KEEP_ALIVE.get()
        )


class TestUvicornWiring(unittest.TestCase):
    # Verify the tuning kwargs actually flow into uvicorn.run. Without
    # this, the resolution helpers could be subtly wrong (e.g. assembling
    # kwargs that uvicorn doesn't accept) and no test would catch it
    # until a real server startup.

    @staticmethod
    def _run_setup(server_args, mock_uvicorn_run):
        from sglang.srt.entrypoints.http_server import _setup_and_run_http_server

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
        args = ServerArgs(model_path="dummy")
        mock = MagicMock()
        self._run_setup(args, mock)
        self.assertTrue(mock.called)
        kwargs = mock.call_args.kwargs
        # keep-alive comes from env (5 by default), backlog from the
        # dataclass default (2048), no limit_concurrency or
        # timeout_graceful_shutdown unless the operator set them.
        self.assertEqual(kwargs["timeout_keep_alive"], 5)
        self.assertEqual(kwargs["backlog"], 2048)
        self.assertNotIn("limit_concurrency", kwargs)
        self.assertNotIn("timeout_graceful_shutdown", kwargs)

    def test_all_uvicorn_flags_threaded_through(self):
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
    # Granian's HTTP/2 tunables go through http2_settings=HTTP2Settings(...),
    # NOT as flat top-level kwargs. Force the multi-worker path so we hit
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
                http2_settings_kwargs={
                    "max_concurrent_streams": 256,
                    "max_frame_size": 65536,
                },
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

    def test_passes_http1_settings_when_tunables_set(self):
        # Symmetric with the HTTP/2 case above: HTTP/1.1-specific knobs
        # go through granian.http.HTTP1Settings, not as top-level kwargs.
        from granian.http import HTTP1Settings

        from sglang.srt.entrypoints.http_server import _run_granian_server

        with patch("granian.Granian") as MockGranian:
            _run_granian_server(
                host="127.0.0.1",
                port=30000,
                log_level="info",
                tokenizer_worker_num=2,
                http1_settings_kwargs={
                    "header_read_timeout": 15000,
                    "max_buffer_size": 1048576,
                },
            )

        kwargs = MockGranian.call_args.kwargs
        self.assertIn("http1_settings", kwargs)
        self.assertIsInstance(kwargs["http1_settings"], HTTP1Settings)
        self.assertEqual(kwargs["http1_settings"].header_read_timeout, 15000)
        self.assertEqual(kwargs["http1_settings"].max_buffer_size, 1048576)

    def test_no_http1_settings_when_tunables_unset(self):
        from sglang.srt.entrypoints.http_server import _run_granian_server

        with patch("granian.Granian") as MockGranian:
            _run_granian_server(
                host="127.0.0.1", port=30000, log_level="info", tokenizer_worker_num=2
            )

        kwargs = MockGranian.call_args.kwargs
        self.assertNotIn("http1_settings", kwargs)


class TestHttp2FlagValidation(unittest.TestCase):
    # __post_init__ -> _handle_ssl_validation cross-checks between
    # --enable-http2 and the uvicorn-only / Granian-only tunables.

    def test_http2_only_flags_without_enable_http2_warns(self):
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
            ServerArgs(model_path="dummy", http2_max_concurrent_streams=256)
        self.assertTrue(
            any("--http2-max-concurrent-streams" in msg for msg in cm.output)
        )

    @unittest.skipUnless(
        _HAS_GRANIAN, "granian not installed (pip install sglang[http2])"
    )
    def test_uvicorn_only_flags_with_enable_http2_warns(self):
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
            ServerArgs(
                model_path="dummy", enable_http2=True, http_limit_concurrency=500
            )
        self.assertTrue(any("--http-limit-concurrency" in msg for msg in cm.output))

    @unittest.skipUnless(
        _HAS_GRANIAN, "granian not installed (pip install sglang[http2])"
    )
    def test_max_frame_size_out_of_range_raises(self):
        # Below the RFC 7540 SETTINGS_MAX_FRAME_SIZE minimum (2^14).
        with self.assertRaises(ValueError):
            ServerArgs(model_path="dummy", enable_http2=True, http2_max_frame_size=100)
        # Above the RFC 7540 SETTINGS_MAX_FRAME_SIZE maximum (2^24-1).
        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                enable_http2=True,
                http2_max_frame_size=16777216,
            )

    @unittest.skipUnless(
        _HAS_GRANIAN, "granian not installed (pip install sglang[http2])"
    )
    def test_max_concurrent_streams_non_positive_raises(self):
        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy", enable_http2=True, http2_max_concurrent_streams=0
            )

    @unittest.skipUnless(
        _HAS_GRANIAN, "granian not installed (pip install sglang[http2])"
    )
    def test_valid_http2_flags_with_enable_http2_do_not_raise(self):
        # Should not raise -- values are within the valid HTTP/2 range.
        ServerArgs(
            model_path="dummy",
            enable_http2=True,
            http2_max_concurrent_streams=256,
            http2_max_frame_size=65536,
        )


if __name__ == "__main__":
    unittest.main()
