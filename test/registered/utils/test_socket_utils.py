import os
import socket
import unittest
from unittest.mock import patch

from sglang.srt.utils.network import (
    _get_addrinfos_for_bind,
    bind_port,
    get_free_port,
    get_open_port,
    is_port_available,
    try_bind_socket,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.utils import normalize_base_url, release_port, reserve_port

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestTryBindSocket(CustomTestCase):
    def test_bind_ephemeral_port(self):
        """try_bind_socket() with port=0 should bind to an OS-assigned port."""
        sock = try_bind_socket()
        try:
            port = sock.getsockname()[1]
            self.assertGreater(port, 0)
            self.assertLessEqual(port, 65535)
        finally:
            sock.close()

    def test_bind_specific_port(self):
        """try_bind_socket(port=N) should bind to that exact port."""
        port = get_free_port()
        sock = try_bind_socket(port=port)
        try:
            self.assertEqual(sock.getsockname()[1], port)
        finally:
            sock.close()

    def test_bind_with_listen(self):
        """try_bind_socket(listen=True) should return a listening socket."""
        sock = try_bind_socket(listen=True)
        try:
            # A listening socket has a valid bound address
            port = sock.getsockname()[1]
            self.assertGreater(port, 0)
        finally:
            sock.close()

    def test_bind_with_host(self):
        """try_bind_socket(host='127.0.0.1') should bind to localhost."""
        sock = try_bind_socket(host="127.0.0.1")
        try:
            addr = sock.getsockname()
            self.assertEqual(addr[0], "127.0.0.1")
        finally:
            sock.close()

    def test_bind_occupied_port_raises(self):
        """try_bind_socket should raise OSError if port is occupied."""
        sock1 = try_bind_socket(host="127.0.0.1", reuse_addr=False)
        try:
            port = sock1.getsockname()[1]
            with self.assertRaises(OSError):
                try_bind_socket(host="127.0.0.1", port=port, reuse_addr=False)
        finally:
            sock1.close()

    def test_returns_correct_family(self):
        """Returned socket should be AF_INET or AF_INET6."""
        sock = try_bind_socket()
        try:
            self.assertIn(sock.family, (socket.AF_INET, socket.AF_INET6))
        finally:
            sock.close()

    def test_gaierror_fallback(self):
        """_get_addrinfos_for_bind should fall back to AF_INET on gaierror."""
        with patch(
            "sglang.srt.utils.network.socket.getaddrinfo",
            side_effect=socket.gaierror("mocked"),
        ):
            infos = _get_addrinfos_for_bind()
            self.assertEqual(len(infos), 1)
            family, socktype, _, _, sockaddr = infos[0]
            self.assertEqual(family, socket.AF_INET)
            self.assertEqual(sockaddr[0], "0.0.0.0")

    def test_gaierror_fallback_preserves_host(self):
        """Fallback should use the provided host, not default to 0.0.0.0."""
        with patch(
            "sglang.srt.utils.network.socket.getaddrinfo",
            side_effect=socket.gaierror("mocked"),
        ):
            infos = _get_addrinfos_for_bind(host="10.0.0.1", port=8080)
            self.assertEqual(infos[0][4], ("10.0.0.1", 8080))


class TestSocketUtilities(CustomTestCase):
    def test_is_port_available(self):
        """is_port_available should return True for a free port."""
        port = get_free_port()
        self.assertTrue(is_port_available(port))

    def test_is_port_available_occupied(self):
        """is_port_available should return False for an occupied port."""
        sock = try_bind_socket(port=0, reuse_addr=False, listen=True)
        try:
            port = sock.getsockname()[1]
            self.assertFalse(is_port_available(port))
        finally:
            sock.close()

    def test_get_free_port(self):
        """get_free_port should return a valid port number."""
        port = get_free_port()
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    def test_bind_port(self):
        """bind_port should return a listening socket."""
        port = get_free_port()
        sock = bind_port(port)
        try:
            self.assertEqual(sock.getsockname()[1], port)
        finally:
            sock.close()

    def test_get_open_port(self):
        """get_open_port should return a valid port number."""
        port = get_open_port()
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    def test_get_open_port_with_env_var(self):
        """get_open_port should respect SGLANG_PORT env var."""
        free_port = get_free_port()
        with patch.dict(os.environ, {"SGLANG_PORT": str(free_port)}):
            port = get_open_port()
            self.assertEqual(port, free_port)

    def test_get_open_port_env_var_occupied_increments(self):
        """get_open_port should increment if SGLANG_PORT is occupied."""
        sock = try_bind_socket(port=0, reuse_addr=False, listen=True)
        try:
            occupied_port = sock.getsockname()[1]
            with patch.dict(os.environ, {"SGLANG_PORT": str(occupied_port)}):
                port = get_open_port()
                # Should skip the occupied port and return a higher one
                self.assertGreater(port, occupied_port)
        finally:
            sock.close()


class TestReservePort(CustomTestCase):
    def test_reserve_port_returns_port_and_socket(self):
        """reserve_port should return a (port, socket) tuple."""
        port, sock = reserve_port("127.0.0.1")
        try:
            self.assertGreaterEqual(port, 30000)
            self.assertLess(port, 40000)
            self.assertEqual(sock.getsockname()[1], port)
        finally:
            release_port(sock)

    def test_reserve_port_custom_range(self):
        """reserve_port should respect custom start/end range."""
        port, sock = reserve_port("127.0.0.1", start=40000, end=41000)
        try:
            self.assertGreaterEqual(port, 40000)
            self.assertLess(port, 41000)
        finally:
            release_port(sock)

    def test_reserve_port_holds_port(self):
        """The reserved port should not be available until released."""
        port, sock = reserve_port("127.0.0.1")
        try:
            # Verify port is held by trying to bind the same family explicitly
            with self.assertRaises(OSError):
                s = try_bind_socket(host="127.0.0.1", port=port, reuse_addr=False)
                s.close()
        finally:
            release_port(sock)

    def test_reserve_port_no_free_port_raises(self):
        """reserve_port should raise RuntimeError if no port is available."""
        with patch(
            "sglang.srt.utils.network.try_bind_socket",
            side_effect=OSError("mocked"),
        ):
            with self.assertRaises(RuntimeError):
                reserve_port("127.0.0.1", start=50000, end=50002)


class TestNormalizeBaseUrl(CustomTestCase):
    def test_ipv4_host(self):
        """normalize_base_url should produce http://host:port for IPv4."""
        url = normalize_base_url("127.0.0.1", 8080)
        self.assertEqual(url, "http://127.0.0.1:8080")

    def test_ipv6_host(self):
        """normalize_base_url should bracket IPv6 addresses."""
        url = normalize_base_url("::1", 8080)
        self.assertEqual(url, "http://[::1]:8080")

    def test_hostname(self):
        """normalize_base_url should work with hostnames."""
        url = normalize_base_url("localhost", 3000)
        self.assertEqual(url, "http://localhost:3000")

    def test_deprecated_scheme_passthrough(self):
        """normalize_base_url should pass through host with scheme (deprecated)."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            url = normalize_base_url("http://myhost", 9000)
        self.assertEqual(url, "http://myhost:9000")


if __name__ == "__main__":
    unittest.main()
