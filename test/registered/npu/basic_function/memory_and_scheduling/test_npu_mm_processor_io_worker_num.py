import os
import tempfile
import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=900, suite="nightly-acc-8-npu-a5", nightly=True)

# Minimal valid 8x8 white JPEG base64 for testing (JPEG minimum block size is 8x8)
_BASE64_WHITE_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAIAAgBAREA/8QAHwAAAQUBAQEB"
    "AQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1Fh"
    "ByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZ"
    "WmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG"
    "x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APv6/9k="
)


class TestMmProcessorIoWorkerNum(CustomTestCase):
    """Testcase: Verify --mm-processor-worker-num and --mm-io-worker-num parameters.

    [Test Category] Parameter
    [Test Target] --mm-processor-worker-num, --mm-io-worker-num
    """

    model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.stderr_lines = []

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def setUp(self):
        self._stderr_file = None
        self._stderr_fh = None
        self.reader_thread = None

    def tearDown(self):
        self._cleanup_stderr()

    def _cleanup_stderr(self):
        """Close stderr file handle, join reader thread, unlink temp file."""
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=5)
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except OSError:
                pass
        if self._stderr_file is not None:
            try:
                os.unlink(self._stderr_file)
            except OSError:
                pass

    def _read_stderr(self):
        """Background thread: read stderr lines into stderr_lines."""
        with open(self._stderr_file, "r") as f:
            for line in f:
                self.stderr_lines.append(line)

    def _launch_server(self, other_args, env=None):
        """Launch server with stderr capture. Stores process in self.process."""
        # Clean up any previous server resources
        self._cleanup_stderr()

        fd, self._stderr_file = tempfile.mkstemp(suffix=".txt", prefix="sglang_stderr_")
        os.close(fd)
        self._stderr_fh = open(self._stderr_file, "w")

        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(None, self._stderr_fh),
        )

        self.stderr_lines.clear()
        self.reader_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.reader_thread.start()

    def _stop_server(self):
        """Kill server process and collect remaining stderr."""
        if self.process:
            time.sleep(2)  # Let logs flush
            kill_process_tree(self.process.pid)
            self.process = None

    def _send_image_request(self):
        """Send a base64 image chat request and return the response."""
        return requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{_BASE64_WHITE_JPEG}"
                                },
                            },
                            {
                                "type": "text",
                                "text": "What color is this image? Answer in one word.",
                            },
                        ],
                    }
                ],
                "max_tokens": 32,
                "temperature": 0,
            },
            timeout=120,
        )

    # ========================= Test Cases =========================

    def test_explicit_params_concurrency_logs(self):
        """Case 1: Explicit params produce concurrency log lines + functional correctness."""
        other_args = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--trust-remote-code",
            "--enable-multimodal",
            "--mm-processor-worker-num",
            "4",
            "--mm-attention-backend",
            "ascend_attn",
            "--mm-io-worker-num",
            "8",
        ]
        self._launch_server(other_args)

        # Verify server is healthy
        resp = requests.get(f"{self.base_url}/health_generate", timeout=10)
        self.assertEqual(resp.status_code, 200)

        # Send base64 image request
        chat_resp = self._send_image_request()
        self.assertEqual(chat_resp.status_code, 200)
        content = chat_resp.json()["choices"][0]["message"]["content"]
        self.assertIn(
            "white",
            content.lower(),
            f"Expected response about a white image, got: '{content}'",
        )

        self._stop_server()

        # Check log messages
        stderr_text = "".join(self.stderr_lines)

        # Processor concurrency log
        self.assertRegex(
            stderr_text,
            r"Multimodal processor concurrency enabled with \d+ isolated worker threads \(explicit\)",
            f"Expected processor concurrency log not found. Logs:\n{stderr_text[-2000:]}",
        )

        # IO worker log (only printed when > 4)
        self.assertRegex(
            stderr_text,
            r"Multimodal data loading enabled with \d+ worker threads \(explicit\)",
            f"Expected IO worker log not found. Logs:\n{stderr_text[-2000:]}",
        )

    def test_npu_auto_locks_processor_to_one(self):
        """Case 2: NPU auto locks processor to 1, no concurrency log appears."""
        other_args = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--trust-remote-code",
            "--enable-multimodal",
        ]
        self._launch_server(other_args)

        # Verify server is healthy
        resp = requests.get(f"{self.base_url}/health_generate", timeout=10)
        self.assertEqual(resp.status_code, 200)

        self._stop_server()

        stderr_text = "".join(self.stderr_lines)

        # Processor concurrency line should NOT appear (locked to 1)
        self.assertNotRegex(
            stderr_text,
            r"Multimodal processor concurrency enabled",
            f"NPU should lock processor to 1, but concurrency log found.\n{stderr_text[-2000:]}",
        )

        # IO worker log for qwen-vl default (16, >4) should appear
        self.assertRegex(
            stderr_text,
            r"Multimodal data loading enabled with \d+ worker threads",
            f"Expected IO worker log for qwen-vl default. Logs:\n{stderr_text[-2000:]}",
        )

    def test_disable_fast_image_processor_unlocks_concurrency(self):
        """Case 3: --disable-fast-image-processor unlocks processor concurrency on NPU."""
        other_args = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--trust-remote-code",
            "--enable-multimodal",
            "--disable-fast-image-processor",
        ]
        self._launch_server(other_args)

        # Verify server is healthy
        resp = requests.get(f"{self.base_url}/health_generate", timeout=10)
        self.assertEqual(resp.status_code, 200)

        self._stop_server()

        stderr_text = "".join(self.stderr_lines)

        # Processor concurrency should now appear with "(auto)"
        self.assertRegex(
            stderr_text,
            r"Multimodal processor concurrency enabled with \d+ isolated worker threads \(auto\)",
            f"Expected processor concurrency (auto) after disable-fast-image-processor. Logs:\n{stderr_text[-2000:]}",
        )

    def test_functional_correctness_with_multiple_workers(self):
        """Case 5: worker=1 and worker=4 produce identical greedy output."""
        # Launch with worker=1 first
        other_args_1 = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--trust-remote-code",
            "--enable-multimodal",
            "--disable-fast-image-processor",
            "--mm-processor-worker-num",
            "1",
            "--mm-io-worker-num",
            "8",
        ]
        self._launch_server(other_args_1)

        # Verify healthy and send request
        requests.get(f"{self.base_url}/health_generate", timeout=10)
        resp_1 = self._send_image_request()
        self.assertEqual(resp_1.status_code, 200)
        output_1 = resp_1.json()["choices"][0]["message"]["content"]
        self._stop_server()

        # Launch with worker=4
        other_args_4 = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--trust-remote-code",
            "--enable-multimodal",
            "--disable-fast-image-processor",
            "--mm-processor-worker-num",
            "4",
            "--mm-io-worker-num",
            "8",
        ]
        self._launch_server(other_args_4)

        requests.get(f"{self.base_url}/health_generate", timeout=10)
        resp_4 = self._send_image_request()
        self.assertEqual(resp_4.status_code, 200)
        output_4 = resp_4.json()["choices"][0]["message"]["content"]
        self._stop_server()

        # Assert identical greedy output
        self.assertEqual(
            output_1,
            output_4,
            f"Worker=1 output '{output_1}' != Worker=4 output '{output_4}'",
        )

        # Verify content is correct (white image)
        self.assertIn(
            "white",
            output_1.lower(),
            f"Expected response about a white image, got: '{output_1}'",
        )

    def test_env_io_workers_suppressed_by_explicit_param(self):
        """Case 4: SGLANG_IO_WORKERS env var is suppressed when --mm-io-worker-num is explicit."""
        env = os.environ.copy()
        env["SGLANG_IO_WORKERS"] = "8"

        other_args = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--trust-remote-code",
            "--enable-multimodal",
            "--mm-io-worker-num",
            "16",
        ]
        self._launch_server(other_args, env=env)

        # Verify server is healthy
        resp = requests.get(f"{self.base_url}/health_generate", timeout=10)
        self.assertEqual(resp.status_code, 200)

        self._stop_server()

        stderr_text = "".join(self.stderr_lines)

        # IO log should show "(explicit)" not "(environment)"
        self.assertRegex(
            stderr_text,
            r"Multimodal data loading enabled with 16 worker threads \(explicit\)",
            f"Expected IO worker log with '(explicit)' despite SGLANG_IO_WORKERS=8. Logs:\n{stderr_text[-2000:]}",
        )


if __name__ == "__main__":
    unittest.main()
