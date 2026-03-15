"""
Test that non-streaming requests are correctly aborted on client disconnect
when --enable-metrics is enabled.

Background: @app.middleware("http") uses Starlette's BaseHTTPMiddleware whose
call_next() replaces the ASGI `receive` callable.  This breaks
request.is_disconnected() in downstream handlers, preventing abort.
http_middleware_patch.py fixes this by providing a pure ASGI call_next.

This test verifies the fix works end-to-end.
"""

import multiprocessing
import os
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
    read_output,
)

STDOUT_FILE = "/tmp/test_abort_metrics_stdout.txt"
STDERR_FILE = "/tmp/test_abort_metrics_stderr.txt"


class TestAbortWithMetrics(CustomTestCase):
    """Verify non-streaming abort works when --enable-metrics is on."""

    def _launch_server(self, base_url, extra_args=None):
        """Launch a server with --enable-metrics and captured output."""
        stdout = open(STDOUT_FILE, "w")
        stderr = open(STDERR_FILE, "w")

        other_args = [
            "--enable-metrics",
            "--log-level",
            "debug",
            "--chunked-prefill-size",
            "8192",
        ]
        if extra_args:
            other_args.extend(extra_args)

        process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(stdout, stderr),
        )
        return process, stdout, stderr

    def _collect_output(self):
        """Start a thread to collect server stderr output lines."""
        output_lines = []
        t = threading.Thread(target=read_output, args=(output_lines, STDERR_FILE))
        t.start()
        return output_lines, t

    def _cleanup(self, process, stdout, stderr, thread):
        kill_process_tree(process.pid)
        stdout.close()
        stderr.close()
        for f in (STDOUT_FILE, STDERR_FILE):
            if os.path.exists(f):
                os.remove(f)
        thread.join()

    def test_abort_non_streaming_with_metrics(self):
        """Client disconnect on non-streaming request should trigger abort
        even with --enable-metrics enabled."""

        port = random.randint(4000, 5000)
        base_url = f"http://127.0.0.1:{port}"

        process, stdout, stderr = self._launch_server(base_url)
        output_lines, t = self._collect_output()

        try:
            # Send many non-streaming requests in a subprocess, then kill it
            # to simulate client disconnect (same pattern as TestAbort).
            def client_process_func():
                def send_one(_):
                    requests.post(
                        f"{base_url}/v1/chat/completions",
                        json={
                            "model": DEFAULT_MODEL_NAME_FOR_TEST,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Write a long essay about AI.",
                                }
                            ],
                            "max_tokens": 2048,
                            "temperature": 0,
                            "stream": False,
                        },
                    )

                with ThreadPoolExecutor(16) as executor:
                    list(executor.map(send_one, range(16)))

            p = multiprocessing.Process(target=client_process_func)
            p.start()
            time.sleep(0.5)
            p.terminate()
            time.sleep(10)
        finally:
            self._cleanup(process, stdout, stderr, t)

        has_abort = any("Abort" in line for line in output_lines)
        self.assertTrue(
            has_abort,
            "Server should abort requests when client disconnects "
            "with --enable-metrics enabled, but no 'Abort' found in server logs.\n"
            f"Captured {len(output_lines)} log lines.",
        )

    def test_metrics_still_work(self):
        """Verify prometheus metrics are still collected after the patch."""

        port = random.randint(4000, 5000)
        base_url = f"http://127.0.0.1:{port}"

        process, stdout, stderr = self._launch_server(base_url)
        _, t = self._collect_output()

        try:
            # Send a normal request to generate metrics
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": DEFAULT_MODEL_NAME_FOR_TEST,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 8,
                    "temperature": 0,
                    "stream": False,
                },
            )
            self.assertEqual(response.status_code, 200)

            # Check prometheus metrics endpoint
            metrics_response = requests.get(f"{base_url}/metrics")
            self.assertEqual(metrics_response.status_code, 200)

            metrics_text = metrics_response.text
            self.assertIn(
                "sglang:http_requests_total",
                metrics_text,
                "Prometheus http_requests_total counter should be present",
            )
            self.assertIn(
                "sglang:http_responses_total",
                metrics_text,
                "Prometheus http_responses_total counter should be present",
            )
        finally:
            self._cleanup(process, stdout, stderr, t)


if __name__ == "__main__":
    unittest.main()
