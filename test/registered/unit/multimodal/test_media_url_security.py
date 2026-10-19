"""Security tests for client-supplied remote multimodal media URLs."""

import http.server
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from sglang.srt.utils.common import (
    _normalize_video_input,
    configure_media_url_security,
    download_remote_media,
    get_image_bytes,
    get_video_bytes,
    load_audio,
    load_image,
    load_video,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


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

    def test_local_regular_file_over_the_cap_is_rejected(self):
        """Local image, audio, and video inputs over the media byte cap are refused."""
        configure_media_url_security([], max_file_size_mb=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.bin"
            path.write_bytes(b"")
            os.truncate(path, 1024 * 1024 + 1)
            uri = path.as_uri()
            with self.assertRaisesRegex(ValueError, "size limit"):
                get_image_bytes(str(path))
            with self.assertRaisesRegex(ValueError, "size limit"):
                load_image(uri)
            with self.assertRaisesRegex(ValueError, "size limit"):
                load_audio(uri)
            with self.assertRaisesRegex(ValueError, "size limit"):
                _normalize_video_input(uri)
            with self.assertRaisesRegex(ValueError, "size limit"):
                get_video_bytes(str(path))

    @unittest.skipUnless(os.path.exists("/dev/zero"), "requires /dev/zero")
    def test_local_device_node_is_rejected(self):
        """A device path is refused instead of being read until EOF."""
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            _normalize_video_input("file:///dev/zero")
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            get_image_bytes("/dev/zero")
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            load_image("file:///dev/zero")
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            load_audio("file:///dev/zero")
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            get_video_bytes("file:///dev/zero")

    def test_local_file_within_the_cap_is_read(self):
        """A small local file is still read, including when the byte cap is disabled."""
        payload = b"small-local-media"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.bin"
            path.write_bytes(payload)
            configure_media_url_security([], max_file_size_mb=1)
            self.assertEqual(get_image_bytes(str(path)), payload)
            self.assertEqual(get_video_bytes(str(path)), payload)
            self.assertEqual(_normalize_video_input(path.as_uri()), payload)
            configure_media_url_security([], max_file_size_mb=0)
            self.assertEqual(get_image_bytes(str(path)), payload)
            self.assertEqual(_normalize_video_input(path.as_uri()), payload)

    def test_percent_encoded_video_path_is_the_file_that_was_checked(self):
        """A %20 in the path must not select a different file from the size check."""
        configure_media_url_security([], max_file_size_mb=1)
        with tempfile.TemporaryDirectory() as directory:
            checked = Path(directory) / "a b.mp4"
            checked.write_bytes(b"tiny")
            literal = Path(directory) / "a%20b.mp4"
            literal.write_bytes(b"")
            os.truncate(literal, 1024 * 1024 + 1)
            self.assertEqual(_normalize_video_input(str(literal)), b"tiny")
            with patch(
                "sglang.srt.utils.common.VideoDecoderWrapper",
                return_value="decoder",
            ) as wrapper:
                self.assertEqual(load_video(str(literal), use_gpu=False), "decoder")
            wrapper.assert_called_once_with(b"tiny", device="cpu")

    def test_preprocess_snapshot_rejects_an_oversized_local_image(self):
        """The preprocess-cache snapshot uses the same local byte cap."""
        from sglang.srt.multimodal.cache.identity import _read_media_bytes

        configure_media_url_security([], max_file_size_mb=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            path.write_bytes(b"")
            os.truncate(path, 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "size limit"):
                _read_media_bytes(str(path))


if __name__ == "__main__":
    unittest.main()
