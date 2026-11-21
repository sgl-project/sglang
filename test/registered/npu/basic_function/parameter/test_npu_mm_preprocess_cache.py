import hashlib
import io
import os
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

register_npu_ci(est_time=4500, suite="nightly-1-npu-a3", nightly=True)


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
    image = Image.new("RGB", (64, 64), (40, 90, 220))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _small_png_sha256():
    return "sha256:" + hashlib.sha256(_small_png_bytes()).hexdigest()


class _MediaOriginHandler(BaseHTTPRequestHandler):
    """Serves the fixed media body registered on the server instance."""

    def do_GET(self):
        body = self.server.media_bodies.get(self.path)
        if body is None:
            # Unknown paths answer 404, so a request that falls back to
            # actually downloading from this origin can never succeed.
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


def media_http_server():
    """A local origin exposing /small (one valid PNG) and 404 for other paths."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MediaOriginHandler)
    server.media_bodies = {"/small": _small_png_bytes()}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


class _MmCacheTestBase(CustomTestCase):
    """Shared helpers for the mm preprocess cache parameter tests."""

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


class TestNpuMmPreprocessCacheSizeMb(_MmCacheTestBase):
    """Testcase: Verify --mm-preprocess-cache-size-mb sets the CPU budget of the content-addressed multimodal preprocess cache, enables it, and keeps hash verification on while the caller hashes are not trusted.

    [Test Category] Parameter
    [Test Target] --mm-preprocess-cache-size-mb
    """

    PORT = 31430
    OUT_LOG_PATH = "./out_log_31430.txt"
    ERR_LOG_PATH = "./err_log_31430.txt"

    @classmethod
    def setUpClass(cls):
        cls.origin, cls.origin_port = media_http_server()
        cls.base_url = f"http://127.0.0.1:{cls.PORT}"
        cls.out_log_file = open(cls.OUT_LOG_PATH, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.ERR_LOG_PATH, "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_OTHER_ARGS + ["--mm-preprocess-cache-size-mb", "256"],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        if hasattr(cls, "origin") and cls.origin:
            cls.origin.shutdown()
            cls.origin.server_close()
        for path in (cls.OUT_LOG_PATH, cls.ERR_LOG_PATH):
            if os.path.exists(path):
                os.remove(path)

    def _post(self, image_data, mm_content_hashes=None):
        payload = {
            "text": _IMAGE_PROMPT,
            "image_data": image_data,
            "sampling_params": _SAMPLING_PARAMS,
        }
        if mm_content_hashes is not None:
            payload["mm_content_hashes"] = mm_content_hashes
        return requests.post(self.base_url + "/generate", json=payload, timeout=120)

    def _err_log(self):
        self.err_log_file.seek(0)
        return self.err_log_file.read()

    def test_server_info_reports_cache_budget(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mm_preprocess_cache_size_mb"], 256)

    def test_cache_enabled_and_repeated_request(self):
        # The cache announces itself with its total budget at startup.
        response = self._post([f"http://127.0.0.1:{self.origin_port}/small"])
        self._assert_generate_ok(response)

        # A second identical request is served from the preprocess cache.
        response = self._post([f"http://127.0.0.1:{self.origin_port}/small"])
        self.assertEqual(response.status_code, 200, response.text)

        log = self._err_log()
        self.assertIn("Multimodal preprocess cache enabled", log)
        self.assertIn("256 MiB total", log)
        # Hashes are verified, not trusted, unless --trust-mm-content-hashes
        # is also set.
        self.assertIn("caller content hashes are verified", log)

    def test_untrusted_hash_still_reads_media(self):
        # Without --trust-mm-content-hashes, a caller hash never skips the
        # media read: an unreadable URL must fail even with the correct hash.
        response = self._post(
            [f"http://127.0.0.1:{self.origin_port}/missing.png"],
            mm_content_hashes=[_small_png_sha256()],
        )
        self.assertNotEqual(response.status_code, 200, response.text)


class TestNpuMmPreprocessCacheDisabled(_MmCacheTestBase):
    """Testcase: Verify --mm-preprocess-cache-size-mb=0 disables the multimodal preprocess cache while image inference keeps working.

    [Test Category] Parameter
    [Test Target] --mm-preprocess-cache-size-mb
    """

    PORT = 31440
    OUT_LOG_PATH = "./out_log_31440.txt"
    ERR_LOG_PATH = "./err_log_31440.txt"

    @classmethod
    def setUpClass(cls):
        cls.origin, cls.origin_port = media_http_server()
        cls.base_url = f"http://127.0.0.1:{cls.PORT}"
        cls.out_log_file = open(cls.OUT_LOG_PATH, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.ERR_LOG_PATH, "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_OTHER_ARGS + ["--mm-preprocess-cache-size-mb", "0"],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        if hasattr(cls, "origin") and cls.origin:
            cls.origin.shutdown()
            cls.origin.server_close()
        for path in (cls.OUT_LOG_PATH, cls.ERR_LOG_PATH):
            if os.path.exists(path):
                os.remove(path)

    def test_server_info_reports_zero_budget(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mm_preprocess_cache_size_mb"], 0)

    def test_cache_disabled_and_inference_still_works(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": _IMAGE_PROMPT,
                "image_data": [f"http://127.0.0.1:{self.origin_port}/small"],
                "sampling_params": _SAMPLING_PARAMS,
            },
            timeout=120,
        )
        self._assert_generate_ok(response)

        self.err_log_file.seek(0)
        self.assertNotIn(
            "Multimodal preprocess cache enabled", self.err_log_file.read()
        )


class TestNpuTrustMmContentHashes(_MmCacheTestBase):
    """Testcase: Verify --trust-mm-content-hashes lets caller-provided SHA-256 content hashes resolve cached preprocessing without reading the media, while wrong hashes are still rejected.

    [Test Category] Parameter
    [Test Target] --trust-mm-content-hashes
    """

    PORT = 31450
    OUT_LOG_PATH = "./out_log_31450.txt"
    ERR_LOG_PATH = "./err_log_31450.txt"

    @classmethod
    def setUpClass(cls):
        cls.origin, cls.origin_port = media_http_server()
        cls.base_url = f"http://127.0.0.1:{cls.PORT}"
        cls.out_log_file = open(cls.OUT_LOG_PATH, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.ERR_LOG_PATH, "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_OTHER_ARGS
            + [
                "--trust-mm-content-hashes",
                "--mm-preprocess-cache-size-mb",
                "256",
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        if hasattr(cls, "origin") and cls.origin:
            cls.origin.shutdown()
            cls.origin.server_close()
        for path in (cls.OUT_LOG_PATH, cls.ERR_LOG_PATH):
            if os.path.exists(path):
                os.remove(path)

    def _post(self, path, content_hash):
        # On the native /generate API the caller hash travels in the
        # mm_content_hashes field; the {"url": ..., "content_hash": ...} dict
        # form only exists on the OpenAI chat path.
        payload = {
            "text": _IMAGE_PROMPT,
            "image_data": [f"http://127.0.0.1:{self.origin_port}{path}"],
            "mm_content_hashes": [content_hash],
            "sampling_params": _SAMPLING_PARAMS,
        }
        return requests.post(self.base_url + "/generate", json=payload, timeout=120)

    def test_server_info_reports_trusted_hashes(self):
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["trust_mm_content_hashes"])
        # The trusted fast path resolves artifacts from the preprocess cache,
        # so the cache must be enabled alongside the flag.
        self.assertEqual(response.json()["mm_preprocess_cache_size_mb"], 256)

    def test_correct_hash_with_live_url_accepted(self):
        response = self._post("/small", _small_png_sha256())
        self._assert_generate_ok(response)

        self.err_log_file.seek(0)
        self.assertIn("caller content hashes are trusted", self.err_log_file.read())

    def test_trusted_hash_skips_download(self):
        # Populate the preprocess cache with the image's artifact.
        response = self._post("/small", _small_png_sha256())
        self._assert_generate_ok(response)

        # The same hash against an unreadable URL still succeeds: the trusted
        # hash resolves the cached artifact without downloading the media.
        # Without the flag (see TestNpuMmPreprocessCacheSizeMb) this fails.
        response = self._post("/missing.png", _small_png_sha256())
        self._assert_generate_ok(response)

    def test_wrong_hash_rejected(self):
        wrong_hash = "sha256:" + "ab" * 32
        response = self._post("/small", wrong_hash)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("content hash mismatch", response.text)


if __name__ == "__main__":
    unittest.main()
