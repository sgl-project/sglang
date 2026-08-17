"""Unit tests for sglang.srt.utils.network — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import os
import socket
import unittest
from unittest.mock import patch

from sglang.srt.utils.network import (
    NetworkAddress,
    _is_ipv6,
    _parse_port,
    _wrap,
    get_free_port,
    get_local_ip_auto,
    get_open_port,
    is_port_available,
    is_valid_ipv6_address,
    try_bind_socket,
)
from sglang.test.test_utils import CustomTestCase


class TestIsValidIpv6Address(CustomTestCase):
    def test_valid_ipv6(self):
        self.assertTrue(is_valid_ipv6_address("::1"))
        self.assertTrue(is_valid_ipv6_address("2001:db8::1"))
        self.assertTrue(is_valid_ipv6_address("fe80::1"))
        self.assertTrue(is_valid_ipv6_address("::"))

    def test_invalid_ipv6(self):
        self.assertFalse(is_valid_ipv6_address("127.0.0.1"))
        self.assertFalse(is_valid_ipv6_address("not-an-ip"))
        self.assertFalse(is_valid_ipv6_address(""))
        self.assertFalse(is_valid_ipv6_address("[::1]"))


class TestIsIpv6(CustomTestCase):
    def test_valid(self):
        self.assertTrue(_is_ipv6("::1"))
        self.assertTrue(_is_ipv6("2001:4860:4860::8888"))

    def test_invalid(self):
        self.assertFalse(_is_ipv6("192.168.1.1"))
        self.assertFalse(_is_ipv6("hostname"))


class TestWrap(CustomTestCase):
    def test_wraps_ipv6(self):
        self.assertEqual(_wrap("::1"), "[::1]")
        self.assertEqual(_wrap("2001:db8::1"), "[2001:db8::1]")

    def test_passes_through_ipv4_and_hostname(self):
        self.assertEqual(_wrap("127.0.0.1"), "127.0.0.1")
        self.assertEqual(_wrap("my-host"), "my-host")


class TestParsePort(CustomTestCase):
    def test_valid_ports(self):
        self.assertEqual(_parse_port("0"), 0)
        self.assertEqual(_parse_port("8000"), 8000)
        self.assertEqual(_parse_port("65535"), 65535)

    def test_invalid_non_numeric(self):
        with self.assertRaises(ValueError, msg="Invalid port number"):
            _parse_port("abc")

    def test_out_of_range(self):
        with self.assertRaises(ValueError, msg="Port out of range"):
            _parse_port("65536")
        with self.assertRaises(ValueError, msg="Port out of range"):
            _parse_port("-1")


class TestNetworkAddress(CustomTestCase):
    def test_basic_creation(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        self.assertEqual(addr.host, "127.0.0.1")
        self.assertEqual(addr.port, 8000)

    def test_strips_ipv6_brackets(self):
        addr = NetworkAddress("[::1]", 8000)
        self.assertEqual(addr.host, "::1")

    def test_is_ipv6_property(self):
        self.assertTrue(NetworkAddress("::1", 80).is_ipv6)
        self.assertFalse(NetworkAddress("127.0.0.1", 80).is_ipv6)

    def test_family_property(self):
        self.assertEqual(NetworkAddress("::1", 80).family, socket.AF_INET6)
        self.assertEqual(NetworkAddress("127.0.0.1", 80).family, socket.AF_INET)

    def test_to_url(self):
        self.assertEqual(
            NetworkAddress("127.0.0.1", 8000).to_url(), "http://127.0.0.1:8000"
        )
        self.assertEqual(NetworkAddress("::1", 8000).to_url(), "http://[::1]:8000")
        self.assertEqual(
            NetworkAddress("127.0.0.1", 443).to_url("https"), "https://127.0.0.1:443"
        )

    def test_to_tcp(self):
        self.assertEqual(
            NetworkAddress("127.0.0.1", 5000).to_tcp(), "tcp://127.0.0.1:5000"
        )
        self.assertEqual(NetworkAddress("::1", 5000).to_tcp(), "tcp://[::1]:5000")

    def test_to_host_port_str(self):
        self.assertEqual(
            NetworkAddress("127.0.0.1", 8000).to_host_port_str(), "127.0.0.1:8000"
        )
        self.assertEqual(NetworkAddress("::1", 8000).to_host_port_str(), "[::1]:8000")

    def test_to_bind_tuple(self):
        self.assertEqual(
            NetworkAddress("127.0.0.1", 8000).to_bind_tuple(), ("127.0.0.1", 8000)
        )

    def test_str_and_repr(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        self.assertEqual(str(addr), "127.0.0.1:8000")
        self.assertEqual(repr(addr), "NetworkAddress('127.0.0.1', 8000)")
        self.assertEqual(str(NetworkAddress("::1", 8000)), "[::1]:8000")

    def test_frozen(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        with self.assertRaises(AttributeError):
            addr.host = "0.0.0.0"


class TestNetworkAddressParse(CustomTestCase):
    def test_parse_ipv4(self):
        addr = NetworkAddress.parse("127.0.0.1:8000")
        self.assertEqual(addr.host, "127.0.0.1")
        self.assertEqual(addr.port, 8000)

    def test_parse_hostname(self):
        addr = NetworkAddress.parse("my-host:9000")
        self.assertEqual(addr.host, "my-host")
        self.assertEqual(addr.port, 9000)

    def test_parse_bracketed_ipv6(self):
        addr = NetworkAddress.parse("[::1]:8000")
        self.assertEqual(addr.host, "::1")
        self.assertEqual(addr.port, 8000)

    def test_parse_empty_raises(self):
        with self.assertRaises(ValueError, msg="Empty address"):
            NetworkAddress.parse("")

    def test_parse_missing_port_raises(self):
        with self.assertRaises(ValueError, msg="Missing port"):
            NetworkAddress.parse("127.0.0.1")

    def test_parse_empty_host_raises(self):
        with self.assertRaises(ValueError, msg="Empty host"):
            NetworkAddress.parse(":8000")

    def test_parse_missing_bracket_raises(self):
        with self.assertRaises(ValueError, msg="Missing closing bracket"):
            NetworkAddress.parse("[::1:8000")

    def test_parse_invalid_ipv6_in_brackets_raises(self):
        with self.assertRaises(ValueError, msg="Invalid IPv6"):
            NetworkAddress.parse("[not-ipv6]:8000")

    def test_parse_bare_ipv6_raises(self):
        with self.assertRaises(ValueError, msg="ambiguous"):
            NetworkAddress.parse("::1:8000")

    def test_parse_bracket_missing_port_raises(self):
        with self.assertRaises(ValueError, msg="Expected ':port'"):
            NetworkAddress.parse("[::1]")


class TestNetworkAddressResolve(CustomTestCase):
    def test_resolve_host_ip_passthrough(self):
        self.assertEqual(NetworkAddress.resolve_host("127.0.0.1"), "127.0.0.1")
        self.assertEqual(NetworkAddress.resolve_host("::1"), "::1")

    @patch("sglang.srt.utils.network.socket.getaddrinfo")
    def test_resolve_host_dns(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))
        ]
        self.assertEqual(NetworkAddress.resolve_host("example.com"), "93.184.216.34")

    @patch("sglang.srt.utils.network.socket.getaddrinfo")
    def test_resolve_host_failure(self, mock_getaddrinfo):
        mock_getaddrinfo.side_effect = socket.gaierror("DNS failure")
        with self.assertRaises(ValueError, msg="Cannot resolve"):
            NetworkAddress.resolve_host("nonexistent.invalid")

    def test_resolved_ip_returns_self(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        resolved = addr.resolved()
        self.assertIs(resolved, addr)


class TestPortOperations(CustomTestCase):
    def test_is_port_available_on_free_port(self):
        sock = try_bind_socket(port=0)
        port = sock.getsockname()[1]
        sock.close()
        self.assertTrue(is_port_available(port))

    def test_is_port_available_on_occupied_port(self):
        sock = try_bind_socket(port=0, listen=True)
        port = sock.getsockname()[1]
        try:
            self.assertFalse(is_port_available(port))
        finally:
            sock.close()

    def test_try_bind_socket_returns_bound_socket(self):
        sock = try_bind_socket(port=0)
        self.assertIsInstance(sock, socket.socket)
        port = sock.getsockname()[1]
        self.assertGreater(port, 0)
        sock.close()

    def test_try_bind_socket_with_listen(self):
        sock = try_bind_socket(port=0, listen=True)
        port = sock.getsockname()[1]
        self.assertGreater(port, 0)
        sock.close()

    def test_get_free_port_returns_valid_port(self):
        port = get_free_port()
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    @patch.dict("os.environ", {}, clear=False)
    def test_get_open_port_without_env(self):
        # Ensure SGLANG_PORT is not set so get_open_port uses OS-assigned port
        os.environ.pop("SGLANG_PORT", None)
        port = get_open_port()
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    @patch("sglang.srt.utils.network.is_port_available", return_value=True)
    @patch.dict("os.environ", {"SGLANG_PORT": "12345"}, clear=False)
    def test_get_open_port_with_env(self, mock_available):
        port = get_open_port()
        self.assertEqual(port, 12345)


class TestGetLocalIpAuto(CustomTestCase):
    @patch.dict(
        "os.environ", {"SGLANG_HOST_IP": "10.0.0.1", "HOST_IP": ""}, clear=False
    )
    def test_returns_env_sglang_host_ip(self):
        self.assertEqual(get_local_ip_auto(), "10.0.0.1")

    @patch.dict(
        "os.environ", {"SGLANG_HOST_IP": "", "HOST_IP": "10.0.0.2"}, clear=False
    )
    def test_returns_env_host_ip(self):
        self.assertEqual(get_local_ip_auto(), "10.0.0.2")

    @patch.dict("os.environ", {"SGLANG_HOST_IP": "", "HOST_IP": ""}, clear=False)
    @patch("sglang.srt.utils.network.get_local_ip_by_nic", return_value=None)
    @patch("sglang.srt.utils.network.get_local_ip_by_remote", return_value=None)
    def test_returns_fallback(self, _mock_remote, _mock_nic):
        self.assertEqual(get_local_ip_auto(fallback="0.0.0.0"), "0.0.0.0")

    @patch.dict("os.environ", {"SGLANG_HOST_IP": "", "HOST_IP": ""}, clear=False)
    @patch("sglang.srt.utils.network.get_local_ip_by_nic", return_value=None)
    @patch("sglang.srt.utils.network.get_local_ip_by_remote", return_value=None)
    def test_raises_without_fallback(self, _mock_remote, _mock_nic):
        with self.assertRaises(ValueError, msg="Can not get local ip"):
            get_local_ip_auto()


if __name__ == "__main__":
    unittest.main()
