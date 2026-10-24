"""Security tests for client-supplied remote multimodal media URLs and local paths."""

import http.server
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import requests

from sglang.srt.utils.common import (
    _assert_local_media_path_allowed,
    _normalize_video_input,
    configure_media_url_security,
    download_remote_media,
    get_image_bytes,
    load_audio,
    read_local_media,
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

    def test_local_media_disabled_when_domain_allowlist_configured(self):
        configure_media_url_security(["127.0.0.1"], max_file_size_mb=64)
        for input_target in ("/etc/hosts", "file:///etc/hosts"):
            with self.subTest(target=input_target):
                with self.assertRaisesRegex(ValueError, "Local media access is disabled"):
                    get_image_bytes(input_target)
                with self.assertRaisesRegex(ValueError, "Local media access is disabled"):
                    _normalize_video_input(input_target)
                with self.assertRaisesRegex(ValueError, "Local media access is disabled"):
                    _assert_local_media_path_allowed(input_target)

    def test_allowed_local_media_paths_permits_whitelisted_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "sample.png")
            with open(file_path, "wb") as f:
                f.write(b"sample-image-content")

            configure_media_url_security(
                allowed_local_media_paths=[tmpdir], max_file_size_mb=64
            )
            # Direct path
            self.assertEqual(get_image_bytes(file_path), b"sample-image-content")
            # file:// URI
            self.assertEqual(
                get_image_bytes(f"file://{file_path}"), b"sample-image-content"
            )
            # Video normalization
            self.assertEqual(
                _normalize_video_input(f"file://{file_path}"),
                os.path.realpath(file_path),
            )

    def test_allowed_local_media_paths_rejects_unwhitelisted_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            safe_file = os.path.join(tmpdir1, "allowed.png")
            with open(safe_file, "wb") as f:
                f.write(b"allowed")

            secret_file = os.path.join(tmpdir2, "secret.png")
            with open(secret_file, "wb") as f:
                f.write(b"secret")

            # Symlink pointing outside the allowed directory
            symlink_file = os.path.join(tmpdir1, "symlink_to_secret.png")
            try:
                os.symlink(secret_file, symlink_file)
            except OSError:
                symlink_file = None

            configure_media_url_security(
                allowed_local_media_paths=[tmpdir1], max_file_size_mb=64
            )

            # Unwhitelisted absolute path
            with self.assertRaisesRegex(ValueError, "not within allowed paths"):
                get_image_bytes(secret_file)

            # Path traversal attempt
            traversal_path = os.path.join(tmpdir1, "..", os.path.basename(tmpdir2), "secret.png")
            with self.assertRaisesRegex(ValueError, "not within allowed paths"):
                get_image_bytes(traversal_path)

            # Symlink escape attempt
            if symlink_file is not None:
                with self.assertRaisesRegex(ValueError, "not within allowed paths"):
                    get_image_bytes(symlink_file)

    def test_local_media_rejects_special_device_files_and_directories(self):
        configure_media_url_security([], max_file_size_mb=64)
        for special_target in ("/dev/zero", "/dev/null", "file:///dev/zero", "/tmp"):
            if not os.path.exists(special_target):
                continue
            with self.subTest(target=special_target):
                with self.assertRaisesRegex(ValueError, "not a regular file"):
                    get_image_bytes(special_target)

    def test_local_media_file_size_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            oversized_file = os.path.join(tmpdir, "large.bin")
            with open(oversized_file, "wb") as f:
                f.write(b"x" * (1024 * 1024 + 1))  # 1MB + 1 byte

            configure_media_url_security(
                allowed_local_media_paths=[tmpdir], max_file_size_mb=1
            )
            with self.assertRaisesRegex(ValueError, "download limit"):
                get_image_bytes(oversized_file)

    def test_empty_allowed_local_media_paths_blocks_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.png")
            with open(test_file, "wb") as f:
                f.write(b"test")

            configure_media_url_security(allowed_local_media_paths=[])
            with self.assertRaisesRegex(ValueError, "Local media access is not allowed"):
                get_image_bytes(test_file)

    def test_load_image_and_audio_enforce_local_security(self):
        from PIL import Image
        from sglang.srt.utils.common import load_image

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "valid.png")
            img = Image.new("RGB", (10, 10), color="blue")
            img.save(img_path)

            # Case 1: Configured allowed local paths allows the image
            configure_media_url_security(allowed_local_media_paths=[tmpdir])
            loaded_img, _ = load_image(img_path)
            self.assertEqual((loaded_img.width, loaded_img.height), (10, 10))

            loaded_img2, _ = load_image(f"file://{img_path}")
            self.assertEqual((loaded_img2.width, loaded_img2.height), (10, 10))

            # Case 2: Configured domain allowlist blocks local image and audio access
            configure_media_url_security(allowed_media_domains=["example.com"])
            with self.assertRaisesRegex(ValueError, "Local media access is disabled"):
                load_image(img_path)
            with self.assertRaisesRegex(ValueError, "Local media access is disabled"):
                load_image(f"file://{img_path}")
            with self.assertRaisesRegex(ValueError, "Local media access is disabled"):
                load_audio("/etc/hosts")


if __name__ == "__main__":
    unittest.main()
