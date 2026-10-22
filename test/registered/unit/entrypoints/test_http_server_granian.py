"""SGLANG_TIMEOUT_KEEP_ALIVE reaches Granian's HTTP/1.1 idle timeout when set."""

import os
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

try:
    from granian.http import HTTP1Settings

    _HAS_GRANIAN = True
except ImportError:
    _HAS_GRANIAN = False

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@unittest.skipUnless(_HAS_GRANIAN, "granian not installed (pip install sglang[http2])")
class TestGranianHTTP1Settings(CustomTestCase):
    def setUp(self):
        super().setUp()
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_unset_keeps_granian_default(self):
        envs.SGLANG_TIMEOUT_KEEP_ALIVE.clear()
        self.assertEqual(http_server._granian_http1_settings(), HTTP1Settings())

    def test_explicit_value_sets_header_read_timeout(self):
        with envs.SGLANG_TIMEOUT_KEEP_ALIVE.override(300):
            settings = http_server._granian_http1_settings()
        self.assertEqual(settings, HTTP1Settings(header_read_timeout=300_000))

    def test_non_positive_value_is_rejected(self):
        with envs.SGLANG_TIMEOUT_KEEP_ALIVE.override(0):
            with self.assertRaisesRegex(ValueError, "SGLANG_TIMEOUT_KEEP_ALIVE"):
                http_server._granian_http1_settings()

    def test_settings_reach_both_granian_servers(self):
        cases = ((1, "granian.server.embed.Server"), (2, "granian.Granian"))
        for tokenizer_worker_num, server_cls in cases:
            with self.subTest(tokenizer_worker_num=tokenizer_worker_num):
                with (
                    envs.SGLANG_TIMEOUT_KEEP_ALIVE.override(300),
                    patch.object(
                        http_server.uvloop, "run", side_effect=lambda coro: coro.close()
                    ),
                    patch(server_cls) as mock_server,
                ):
                    http_server._run_granian_server(
                        host="127.0.0.1",
                        port=30000,
                        log_level="info",
                        http2_max_concurrent_streams=64,
                        http2_initial_connection_window_size=65536,
                        tokenizer_worker_num=tokenizer_worker_num,
                    )
                kwargs = mock_server.call_args.kwargs
                self.assertEqual(kwargs["http1_settings"].header_read_timeout, 300_000)
                self.assertEqual(kwargs["http2_settings"].max_concurrent_streams, 64)


if __name__ == "__main__":
    unittest.main()
