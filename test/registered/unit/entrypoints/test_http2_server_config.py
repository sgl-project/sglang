import importlib.util
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.http_server import _run_granian_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@unittest.skipUnless(
    importlib.util.find_spec("granian"), "granian is required for HTTP/2"
)
class TestGranianHTTP2Config(unittest.TestCase):
    def test_passes_explicit_max_concurrent_streams(self):
        configured = {}

        class FakeEmbeddedServer:
            def __init__(self, **kwargs):
                configured.update(kwargs)

            async def serve(self):
                return None

            def stop(self):
                return None

        with patch("granian.server.embed.Server", FakeEmbeddedServer):
            _run_granian_server(
                host="127.0.0.1",
                port=30000,
                log_level="info",
                http2_max_concurrent_streams=37,
            )

        self.assertEqual(configured["http2_settings"].max_concurrent_streams, 37)


if __name__ == "__main__":
    unittest.main()
