"""SIGTERM drains the embedded Rust server (``SGLANG_RUST_SERVER=1``) as it
drains the Python server: ``/health`` answers 503 while an in-flight stream
runs to its ``[DONE]``, a request arriving meanwhile is still served, and the
server then exits 0 and releases its port."""

import json
import signal
import socket
import threading
import time
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

# Long enough that the stream outlives the SIGTERM and the health checks.
MAX_NEW_TOKENS = 2000


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestRustServerGracefulShutdown(CustomTestCase):
    def test_sigterm_drains_in_flight_stream_then_exits(self):
        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={"SGLANG_RUST_SERVER": "1"},
        )
        try:
            self._check_drain(process)
        finally:
            kill_process_tree(process.pid)

    def _check_drain(self, process):
        lines = []
        started = threading.Event()

        def stream():
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": "Count to a million:",
                    "stream": True,
                    "sampling_params": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "ignore_eos": True,
                        "temperature": 0,
                    },
                },
                stream=True,
                timeout=300,
            )
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    lines.append(line)
                    started.set()

        streamer = threading.Thread(target=stream)
        streamer.start()
        self.assertTrue(started.wait(60), "the stream never started")

        process.send_signal(signal.SIGTERM)
        # The drain begins from a Rust thread, without waiting for the
        # scheduler loop.
        time.sleep(0.5)
        self.assertEqual(
            requests.get(DEFAULT_URL_FOR_TEST + "/health").status_code, 503
        )
        self.assertEqual(
            requests.get(DEFAULT_URL_FOR_TEST + "/health_generate").status_code, 503
        )
        # Admitted during the drain, as Python admits it.
        late = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={"text": "Hello", "sampling_params": {"max_new_tokens": 4}},
            timeout=60,
        )
        self.assertEqual(late.status_code, 200, late.text)
        self.assertTrue(streamer.is_alive(), "the stream ended before the checks")

        streamer.join(300)
        self.assertEqual(lines[-1], "data: [DONE]")
        final = json.loads(lines[-2][len("data: ") :])
        self.assertEqual(final["meta_info"]["completion_tokens"], MAX_NEW_TOKENS)

        self.assertEqual(process.wait(timeout=120), 0)
        url = urlparse(DEFAULT_URL_FOR_TEST)
        with self.assertRaises(ConnectionRefusedError):
            socket.create_connection((url.hostname, url.port), timeout=5).close()


if __name__ == "__main__":
    unittest.main()
