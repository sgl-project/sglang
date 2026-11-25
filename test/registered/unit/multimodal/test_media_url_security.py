"""Security tests for client-supplied remote multimodal media URLs."""

import http.server
import os
import socket
import threading
import unittest
from unittest.mock import patch

import requests

from sglang.srt.multimodal.media_io import (
    _assert_media_url_allowed,
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
            self.server.media_hits += 1
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
        cls.server.media_hits = 0
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
        self.server.media_hits = 0
        configure_media_url_security([], max_file_size_mb=64)

    def tearDown(self):
        configure_media_url_security([], max_file_size_mb=64)

    def _url(self, path, host="127.0.0.1"):
        return f"http://{host}:{self.port}{path}"

    def test_private_ip_urls_are_blocked_by_default(self):
        for url in (
            self._url("/media"),
            self._url("/media", host="localhost"),
            "http://169.254.169.254/latest/meta-data",
            "http://10.0.0.5/media",
            "http://192.168.1.1/media",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, "non-public IP"):
                    download_remote_media(url, timeout=5)
        self.assertEqual(self.server.media_hits, 0)

    def test_allow_private_media_hosts_opt_in_restores_local_downloads(self):
        configure_media_url_security(
            [], max_file_size_mb=64, allow_private_media_hosts=True
        )
        self.assertEqual(
            download_remote_media(self._url("/media", host="localhost"), timeout=5),
            b"remote-media",
        )

    def test_ipv6_loopback_and_mapped_addresses_are_blocked(self):
        for url in (
            "http://[::1]/media",
            "http://[::ffff:127.0.0.1]/media",
            "http://[fe80::1]/media",
            "http://[::]/media",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, "non-public IP"):
                    _assert_media_url_allowed(url)

    def test_noncanonical_private_ip_spellings_are_blocked(self):
        for host in ("0x7f.0.0.1", "2130706433", "0177.0.0.1"):
            with self.subTest(host=host):
                with self.assertRaisesRegex(ValueError, "non-public IP"):
                    download_remote_media(f"http://{host}:{self.port}/media", timeout=5)
        self.assertEqual(self.server.media_hits, 0)

    def test_connection_goes_to_the_checked_address(self):
        # A rebinding resolver: public when first asked, loopback afterwards.
        answers = iter(["93.184.216.34", "127.0.0.1", "127.0.0.1"])
        looked_up = []
        real_getaddrinfo = socket.getaddrinfo

        def getaddrinfo(host, port, *args, **kwargs):
            looked_up.append(host)
            if host == "rebind.test":
                address = (next(answers), port or 0)
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", address)]
            if host == "93.184.216.34":
                # Stands in for an unreachable public host, without network.
                raise socket.gaierror(socket.EAI_NONAME, "unreachable in this test")
            return real_getaddrinfo(host, port, *args, **kwargs)

        with (
            patch.dict(os.environ, {"no_proxy": "*"}),
            patch("socket.getaddrinfo", getaddrinfo),
        ):
            with self.assertRaises(requests.exceptions.ConnectionError):
                download_remote_media(
                    f"http://rebind.test:{self.port}/media", timeout=5
                )
        self.assertEqual(self.server.media_hits, 0)
        self.assertEqual(looked_up, ["rebind.test", "93.184.216.34"])

    def test_proxied_download_still_checks_the_host(self):
        real_getaddrinfo = socket.getaddrinfo

        def getaddrinfo(host, port, *args, **kwargs):
            if host == "internal.test":
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))]
            return real_getaddrinfo(host, port, *args, **kwargs)

        # The test server stands in for the proxy; it must not be asked.
        proxy = self._url("")
        with (
            patch.dict(os.environ, {"http_proxy": proxy, "no_proxy": ""}),
            patch("socket.getaddrinfo", getaddrinfo),
        ):
            with self.assertRaisesRegex(ValueError, "non-public IP"):
                download_remote_media("http://internal.test/media", timeout=5)

    def test_exact_domain_allowlist(self):
        configure_media_url_security(["127.0.0.1"], max_file_size_mb=64)
        self.assertEqual(
            download_remote_media(self._url("/media"), timeout=5), b"remote-media"
        )
        with self.assertRaisesRegex(ValueError, "not allowed") as blocked:
            download_remote_media(self._url("/media", host="localhost"), timeout=5)
        self.assertNotIn("127.0.0.1", str(blocked.exception))
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


if __name__ == "__main__":
    unittest.main()
