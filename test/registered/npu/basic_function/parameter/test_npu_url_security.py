import io
import os
import random
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from PIL import Image

from sglang.test.ascend.test_ascend_utils import QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=3600, suite="nightly-1-npu-a3", nightly=True)


_IMAGE_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image in one short sentence.<|im_end|>\n<|im_start|>assistant\n"
)
_SAMPLING_PARAMS = {"max_new_tokens": 16, "min_new_tokens": 4, "temperature": 0}
_COMMON_OTHER_ARGS = [
    "--trust-remote-code",
    "--enable-multimodal",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static",
    0.7,
    "--log-level",
    "info",
]


def _small_png_bytes():
    """A deterministic ~1 KB PNG that is a valid image for the VLM."""
    image = Image.new("RGB", (64, 64), (220, 40, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _big_body_bytes():
    """Deterministic incompressible bytes larger than the 1 MiB test limit."""
    return random.Random(20240601).randbytes(2 * 1024 * 1024 + 777)


class _MediaOriginHandler(BaseHTTPRequestHandler):
    """Serves the fixed media bodies registered on the server instance."""

    def do_GET(self):
        body = self.server.media_bodies.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The server may stop reading mid-body (download size cap).
            pass

    def log_message(self, format, *args):
        pass


def media_http_server():
    """A local origin exposing /small (valid PNG) and /big (oversized body)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MediaOriginHandler)
    server.media_bodies = {
        "/small": _small_png_bytes(),
        "/big": _big_body_bytes(),
    }
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


class _UrlSecurityTestBase(CustomTestCase):
    """Shared helpers for the media URL security parameter tests."""

    def _assert_generate_ok(self, response):
        """The request completed a full download-decode-prefill-decode round.

        The contracts under test are about request handling, not generation
        quality: min_new_tokens keeps the output non-empty, and any completed
        request decodes at least one token.
        """
        self.assertEqual(response.status_code, 200, response.text)
        out = response.json()
        self.assertGreaterEqual(out["meta_info"]["completion_tokens"], 1)
        self.assertTrue(out["text"])


class TestNpuAllowedMediaDomains(_UrlSecurityTestBase):
    """Testcase: Verify --allowed-media-domains restricts the hosts the server downloads client-supplied media URLs from, while whitelisted hosts keep working.

    [Test Category] Parameter
    [Test Target] --allowed-media-domains
    """

    PORT = 31410

    @classmethod
    def setUpClass(cls):
        cls.origin, cls.origin_port = media_http_server()
        cls.base_url = f"http://127.0.0.1:{cls.PORT}"
        cls.process = popen_launch_server(
            QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_OTHER_ARGS + ["--allowed-media-domains", "127.0.0.1"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        if hasattr(cls, "origin") and cls.origin:
            cls.origin.shutdown()
            cls.origin.server_close()

    def _post(self, image_data):
        payload = {
            "text": _IMAGE_PROMPT,
            "image_data": image_data,
            "sampling_params": _SAMPLING_PARAMS,
        }
        return requests.post(self.base_url + "/generate", json=payload, timeout=120)

    def test_server_info_reports_allowed_domains(self):
        # The startup configuration echoes the allowlist on /server_info.
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["allowed_media_domains"], ["127.0.0.1"])

    def test_whitelisted_image_url_accepted(self):
        # An image URL on the whitelisted host is downloaded and decoded.
        response = self._post([f"http://127.0.0.1:{self.origin_port}/small"])
        self._assert_generate_ok(response)

    def test_non_whitelisted_host_rejected(self):
        # localhost is a different hostname from 127.0.0.1 (exact-match after
        # normalization, so a DNS alias cannot smuggle traffic past the allowlist).
        response = self._post([f"http://localhost:{self.origin_port}/small"])
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Media URL domain is not allowed", response.text)

    def test_file_scheme_url_still_accepted(self):
        # Community behavior: --allowed-media-domains constrains HTTP(S) URLs
        # only; a client-supplied local file:// URL keeps working and is not
        # subject to the allowlist.
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            handle.write(_small_png_bytes())
            handle.close()
            response = self._post([f"file://{handle.name}"])
            self._assert_generate_ok(response)
        finally:
            if os.path.exists(handle.name):
                os.remove(handle.name)


class TestNpuMediaUrlMaxFileSizeMb(_UrlSecurityTestBase):
    """Testcase: Verify --media-url-max-file-size-mb caps the size of one client-supplied remote media download, rejecting oversized bodies while the stream is read.

    [Test Category] Parameter
    [Test Target] --media-url-max-file-size-mb
    """

    PORT = 31420

    @classmethod
    def setUpClass(cls):
        cls.origin, cls.origin_port = media_http_server()
        cls.base_url = f"http://127.0.0.1:{cls.PORT}"
        cls.process = popen_launch_server(
            QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_OTHER_ARGS + ["--media-url-max-file-size-mb", "1"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        if hasattr(cls, "origin") and cls.origin:
            cls.origin.shutdown()
            cls.origin.server_close()

    def _post(self, image_data):
        payload = {
            "text": _IMAGE_PROMPT,
            "image_data": image_data,
            "sampling_params": _SAMPLING_PARAMS,
        }
        return requests.post(self.base_url + "/generate", json=payload, timeout=120)

    def test_server_info_reports_download_limit(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["media_url_max_file_size_mb"], 1)

    def test_download_size_cap_enforced(self):
        # Under the cap: the small PNG downloads and decodes fine.
        response = self._post([f"http://127.0.0.1:{self.origin_port}/small"])
        self._assert_generate_ok(response)

        # Over the cap: the ~2 MiB body is rejected against the 1 MiB limit.
        response = self._post([f"http://127.0.0.1:{self.origin_port}/big"])
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("exceeds the", response.text)
        self.assertIn("byte download limit", response.text)


if __name__ == "__main__":
    unittest.main()
