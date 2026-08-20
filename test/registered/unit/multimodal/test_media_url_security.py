"""Security tests for client-supplied remote multimodal media URLs."""

import http.server
import threading
import tempfile
import unittest
from unittest.mock import patch

import requests

from sglang.srt.utils.common import (
    _normalize_video_input,
    configure_media_url_security,
    download_remote_media,
    get_image_bytes,
    load_audio,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _MediaHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/media":
            payload = b"remote-media"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/same-host-redirect":
            self.send_response(302)
            self.send_header("Location", "/media")
            self.end_headers()
            return

        if self.path == "/other-host-redirect":
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://localhost:{self.server.server_port}/redirect-target",
            )
            self.end_headers()
            return

        if self.path == "/redirect-target":
            self.server.redirect_target_reached = True
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"must-not-be-fetched")
            return

        if self.path == "/oversized":
            self.send_response(200)
            self.send_header("Content-Length", str(2 * 1024 * 1024))
            self.end_headers()
            return

        if self.path == "/chunked-oversized":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * (1024 * 1024 + 1))
            return

        if self.path == "/redirect-loop":
            self.send_response(302)
            self.send_header("Location", "/redirect-loop")
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, *_):
        pass


class TestMediaURLSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MediaHandler)
        cls.server.redirect_target_reached = False
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.redirect_target_reached = False
        configure_media_url_security([], max_file_size_mb=64)

    def tearDown(self):
        configure_media_url_security([], max_file_size_mb=64)

    def _url(self, path, host="127.0.0.1"):
        return f"http://{host}:{self.port}{path}"

    def test_unrestricted_mode_preserves_remote_media_compatibility(self):
        self.assertEqual(
            download_remote_media(self._url("/media", host="localhost"), timeout=5),
            b"remote-media",
        )

    def test_exact_domain_allowlist(self):
        configure_media_url_security(["127.0.0.1"], max_file_size_mb=64)
        self.assertEqual(
            download_remote_media(self._url("/media"), timeout=5), b"remote-media"
        )
        with self.assertRaisesRegex(ValueError, "not allowed"):
            download_remote_media(self._url("/media", host="localhost"), timeout=5)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            download_remote_media("http://169.254.169.254/latest/meta-data", timeout=5)

    def test_redirect_destination_is_checked_before_fetch(self):
        configure_media_url_security(["127.0.0.1"], max_file_size_mb=64)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            download_remote_media(self._url("/other-host-redirect"), timeout=5)
        self.assertFalse(self.server.redirect_target_reached)

    def test_same_domain_redirect_is_allowed(self):
        configure_media_url_security(["127.0.0.1"], max_file_size_mb=64)
        self.assertEqual(
            download_remote_media(self._url("/same-host-redirect"), timeout=5),
            b"remote-media",
        )

    def test_redirect_count_is_bounded(self):
        configure_media_url_security(["127.0.0.1"], max_file_size_mb=64)
        with self.assertRaises(requests.exceptions.TooManyRedirects):
            download_remote_media(self._url("/redirect-loop"), timeout=5)

    def test_declared_oversized_response_is_rejected(self):
        configure_media_url_security(["127.0.0.1"], max_file_size_mb=1)
        with self.assertRaisesRegex(ValueError, "download limit"):
            download_remote_media(self._url("/oversized"), timeout=5)

    def test_streamed_oversized_response_is_rejected(self):
        configure_media_url_security(["127.0.0.1"], max_file_size_mb=1)
        with self.assertRaisesRegex(ValueError, "download limit"):
            download_remote_media(self._url("/chunked-oversized"), timeout=5)

    def test_invalid_allowlist_entries_are_rejected(self):
        for domain in (
            "https://media.example.com",
            "media.example.com/path",
            "media.example.com:443",
            "",
        ):
            with self.subTest(domain=domain):
                with self.assertRaises(ValueError):
                    configure_media_url_security([domain], max_file_size_mb=64)

    def test_backslash_userinfo_parser_confusion_cannot_bypass_allowlist(self):
        configure_media_url_security(["safe.example.org"], max_file_size_mb=64)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            download_remote_media(
                r"https://evil.example\@safe.example.org/media", timeout=5
            )

    def test_non_public_destinations_require_explicit_opt_in(self):
        configure_media_url_security(
            [], max_file_size_mb=64, allow_private_networks=False
        )
        for url in (
            self._url("/media"),
            self._url("/media", host="localhost"),
            "http://169.254.169.254/latest/meta-data",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, "non-public IP"):
                    download_remote_media(url, timeout=5)

        configure_media_url_security(
            [], max_file_size_mb=64, allow_private_networks=True
        )
        self.assertEqual(
            download_remote_media(self._url("/media"), timeout=5), b"remote-media"
        )

    def test_all_common_loaders_share_the_policy(self):
        blocked = ValueError("media URL domain is not allowed")
        with patch(
            "sglang.srt.utils.common.download_remote_media", side_effect=blocked
        ) as download:
            for loader in (
                get_image_bytes,
                _normalize_video_input,
                load_audio,
            ):
                with self.subTest(loader=loader.__name__):
                    with self.assertRaisesRegex(ValueError, "not allowed"):
                        loader("https://blocked.example/media")
            self.assertEqual(download.call_count, 3)

    def test_local_media_paths_require_explicit_opt_in(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            image_file.write(b"test-image")
            image_file.flush()
            local_path = image_file.name
            file_uri = f"file://{local_path}"

            configure_media_url_security(
                [], max_file_size_mb=64, allow_local_file_paths=False
            )
            for loader, value in (
                (get_image_bytes, local_path),
                (get_image_bytes, file_uri),
                (_normalize_video_input, local_path),
                (_normalize_video_input, file_uri),
                (load_audio, local_path),
                (load_audio, file_uri),
            ):
                with self.subTest(loader=loader.__name__, value=value):
                    with self.assertRaisesRegex(
                        ValueError, "Local media paths are disabled"
                    ):
                        loader(value)

            configure_media_url_security(
                [], max_file_size_mb=64, allow_local_file_paths=True
            )
            self.assertEqual(get_image_bytes(local_path), b"test-image")
            self.assertEqual(get_image_bytes(file_uri), b"test-image")


if __name__ == "__main__":
    unittest.main()
