"""With --enable-http2, Granian closes an idle HTTP/1.1 keep-alive connection
when its header read timeout fires, so SGLANG_TIMEOUT_KEEP_ALIVE must reach
that timeout without ever lowering Granian's default.
"""

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
        # Restore the environment even when an assertion fails inside override().
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_unset_keeps_granian_default(self):
        envs.SGLANG_TIMEOUT_KEEP_ALIVE.clear()
        settings = http_server._granian_http1_settings()
        self.assertEqual(settings, HTTP1Settings())

    def test_long_keep_alive_raises_header_read_timeout(self):
        with envs.SGLANG_TIMEOUT_KEEP_ALIVE.override(1800):
            settings = http_server._granian_http1_settings()
        # Seconds become milliseconds; the other HTTP/1.1 settings keep their defaults.
        self.assertEqual(settings, HTTP1Settings(header_read_timeout=1_800_000))

    def test_short_keep_alive_never_lowers_granian_default(self):
        for keep_alive in (-1, 0, 5, 30):
            with self.subTest(keep_alive=keep_alive):
                with envs.SGLANG_TIMEOUT_KEEP_ALIVE.override(keep_alive):
                    settings = http_server._granian_http1_settings()
                self.assertEqual(settings, HTTP1Settings())

    def test_invalid_value_keeps_granian_default(self):
        os.environ["SGLANG_TIMEOUT_KEEP_ALIVE"] = "abc"
        with self.assertWarns(UserWarning):
            settings = http_server._granian_http1_settings()
        self.assertEqual(settings, HTTP1Settings())

    def _granian_kwargs(self, server_cls, tokenizer_worker_num):
        # The embedded server is driven by uvloop.run; close the coroutine instead.
        skip_serve = patch.object(
            http_server.uvloop, "run", side_effect=lambda coro: coro.close()
        )
        with skip_serve, patch(server_cls) as mock_server:
            http_server._run_granian_server(
                host="127.0.0.1",
                port=30000,
                log_level="info",
                http2_max_concurrent_streams=64,
                http2_initial_connection_window_size=65536,
                tokenizer_worker_num=tokenizer_worker_num,
            )
        mock_server.assert_called_once()
        return mock_server.call_args.kwargs

    def test_settings_reach_embedded_and_multiprocess_servers(self):
        cases = ((1, "granian.server.embed.Server"), (2, "granian.Granian"))
        with envs.SGLANG_TIMEOUT_KEEP_ALIVE.override(300):
            for tokenizer_worker_num, server_cls in cases:
                with self.subTest(tokenizer_worker_num=tokenizer_worker_num):
                    kwargs = self._granian_kwargs(server_cls, tokenizer_worker_num)
                    http1, http2 = kwargs["http1_settings"], kwargs["http2_settings"]
                    self.assertEqual(http1.header_read_timeout, 300_000)
                    self.assertEqual(http2.max_concurrent_streams, 64)
                    self.assertEqual(http2.initial_connection_window_size, 65536)


if __name__ == "__main__":
    unittest.main()
