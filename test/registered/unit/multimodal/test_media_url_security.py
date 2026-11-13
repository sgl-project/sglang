"""Security tests for client-supplied remote multimodal media URLs."""

import http.server
import os
import socket
import threading
import unittest
from unittest.mock import MagicMock, patch

import requests

from sglang.srt.multimodal.media_io import (
    _normalize_video_input,
    configure_media_url_security,
    download_remote_media,
    get_image_bytes,
    load_audio,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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
        self.private_media_opt_in = patch.dict(
            os.environ, {"SGLANG_ALLOW_PRIVATE_MEDIA_URLS": "1"}
        )
        self.private_media_opt_in.start()
        self.addCleanup(self.private_media_opt_in.stop)
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
            "sglang.srt.multimodal.media_io.download_remote_media", side_effect=blocked
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


class TestMediaURLAddresses(unittest.TestCase):
    def setUp(self):
        configure_media_url_security([])
        self.env = patch.dict(os.environ, {"SGLANG_ALLOW_PRIVATE_MEDIA_URLS": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.session = MagicMock()
        self.session_patch = patch(
            "sglang.srt.multimodal.media_io.get_mm_http_session",
            return_value=self.session,
        )
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.response = self.session.get.return_value.__enter__.return_value
        self.response.status_code = 200
        self.response.headers = {}
        self.response.iter_content.return_value = [b"media"]

    @staticmethod
    def _addresses(*addresses):
        return [
            (
                socket.AF_INET6 if ":" in ip else socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (ip, 80),
            )
            for ip in addresses
        ]

    def test_non_global_destinations_are_rejected_before_fetch(self):
        for host, address in [
            ("127.0.0.1", "127.0.0.1"),
            ("localhost", "127.0.0.1"),
            ("media.example", "127.0.0.1"),
            ("2130706433", "127.0.0.1"),
            ("metadata.example", "169.254.169.254"),
            ("private.example", "10.0.0.1"),
            ("private.example", "172.16.0.1"),
            ("private.example", "192.168.0.1"),
            ("cgnat.example", "100.64.0.1"),
            ("unspecified.example", "0.0.0.0"),
            ("[::1]", "::1"),
            ("[::]", "::"),
            ("[fc00::1]", "fc00::1"),
            ("[fe80::1]", "fe80::1"),
            ("[::ffff:127.0.0.1]", "::ffff:127.0.0.1"),
            ("[::127.0.0.1]", "::127.0.0.1"),
            ("nat64.example", "64:ff9b::7f00:1"),
            ("nat64.example", "64:ff9b::a00:1"),
            ("multicast.example", "239.1.1.1"),
            ("multicast.example", "ff0e::1"),
        ]:
            with (
                self.subTest(host=host, address=address),
                patch(
                    "sglang.srt.multimodal.media_io.socket.getaddrinfo",
                    return_value=self._addresses(address),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "local or private"):
                    download_remote_media(f"http://{host}/media", timeout=5)
                self.session.get.assert_not_called()

    def test_public_destinations_are_allowed(self):
        with patch(
            "sglang.srt.multimodal.media_io.socket.getaddrinfo",
            return_value=self._addresses(
                "8.8.8.8", "2001:4860:4860::8888", "64:ff9b::808:808"
            ),
        ):
            self.assertEqual(
                download_remote_media("https://media.example/media", timeout=5),
                b"media",
            )
        self.session.get.assert_called_once()

    def test_any_private_dns_answer_is_rejected(self):
        with patch(
            "sglang.srt.multimodal.media_io.socket.getaddrinfo",
            return_value=self._addresses("8.8.8.8", "127.0.0.1"),
        ):
            with self.assertRaisesRegex(ValueError, "local or private"):
                download_remote_media("https://media.example/media", timeout=5)
        self.session.get.assert_not_called()

    def test_redirect_to_private_destination_is_rejected_before_fetch(self):
        self.response.status_code = 302
        self.response.headers = {"Location": "http://127.0.0.1/media"}
        self.response.url = "https://media.example/media"
        with patch(
            "sglang.srt.multimodal.media_io.socket.getaddrinfo",
            side_effect=[self._addresses("8.8.8.8"), self._addresses("127.0.0.1")],
        ):
            with self.assertRaisesRegex(ValueError, "local or private"):
                download_remote_media(self.response.url, timeout=5)
        self.session.get.assert_called_once()

    def test_failed_or_empty_dns_answer_is_rejected(self):
        for kwargs in [{"return_value": []}, {"side_effect": socket.gaierror("DNS")}]:
            with (
                self.subTest(kwargs=kwargs),
                patch("sglang.srt.multimodal.media_io.socket.getaddrinfo", **kwargs),
            ):
                with self.assertRaisesRegex(ValueError, "could not be resolved"):
                    download_remote_media("https://media.example/media", timeout=5)
                self.session.get.assert_not_called()

    def test_direct_decoders_share_the_policy(self):
        from sglang.srt.multimodal.audio_from_video import decode_audio_container
        from sglang.srt.utils.video_decoder import VideoDecoderWrapper

        with patch(
            "sglang.srt.multimodal.media_io.socket.getaddrinfo",
            return_value=self._addresses("127.0.0.1"),
        ):
            with self.assertRaisesRegex(ValueError, "local or private"):
                VideoDecoderWrapper("http://localhost/media")
            with self.assertRaisesRegex(ValueError, "local or private"):
                decode_audio_container(
                    "http://localhost/media", target_sr=16000, mono=True
                )
        self.session.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
