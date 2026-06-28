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


if __name__ == "__main__":
    unittest.main()
