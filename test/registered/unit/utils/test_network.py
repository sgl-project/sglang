"""Unit tests for srt/utils/network.py — no server, no model loading."""

import socket
import unittest

from sglang.srt.utils.network import (
    NetworkAddress,
    _is_ipv6,
    _parse_port,
    _wrap,
    is_valid_ipv6_address,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class TestIsValidIPv6Address(CustomTestCase):
    def test_valid_full(self):
        self.assertTrue(is_valid_ipv6_address("2001:db8::1"))

    def test_valid_loopback(self):
        self.assertTrue(is_valid_ipv6_address("::1"))

    def test_valid_all_zeros(self):
        self.assertTrue(is_valid_ipv6_address("::"))

    def test_valid_full_expanded(self):
        self.assertTrue(
            is_valid_ipv6_address("2001:0db8:0000:0000:0000:0000:0000:0001")
        )

    def test_invalid_ipv4(self):
        self.assertFalse(is_valid_ipv6_address("192.168.1.1"))

    def test_invalid_hostname(self):
        self.assertFalse(is_valid_ipv6_address("example.com"))

    def test_invalid_empty(self):
        self.assertFalse(is_valid_ipv6_address(""))

    def test_invalid_garbage(self):
        self.assertFalse(is_valid_ipv6_address("not-an-ip"))

    def test_invalid_bracketed(self):
        # brackets are not part of the address itself
        self.assertFalse(is_valid_ipv6_address("[::1]"))


class TestIsIPv6(CustomTestCase):
    def test_loopback(self):
        self.assertTrue(_is_ipv6("::1"))

    def test_full_address(self):
        self.assertTrue(_is_ipv6("2001:db8::1"))

    def test_ipv4(self):
        self.assertFalse(_is_ipv6("127.0.0.1"))

    def test_hostname(self):
        self.assertFalse(_is_ipv6("localhost"))

    def test_empty(self):
        self.assertFalse(_is_ipv6(""))


class TestWrap(CustomTestCase):
    def test_ipv6_gets_wrapped(self):
        self.assertEqual(_wrap("::1"), "[::1]")

    def test_ipv6_full_gets_wrapped(self):
        self.assertEqual(_wrap("2001:db8::1"), "[2001:db8::1]")

    def test_ipv4_unchanged(self):
        self.assertEqual(_wrap("127.0.0.1"), "127.0.0.1")

    def test_hostname_unchanged(self):
        self.assertEqual(_wrap("example.com"), "example.com")


class TestParsePort(CustomTestCase):
    def test_valid_port(self):
        self.assertEqual(_parse_port("8080"), 8080)

    def test_zero(self):
        self.assertEqual(_parse_port("0"), 0)

    def test_max_port(self):
        self.assertEqual(_parse_port("65535"), 65535)

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("abc")

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("-1")

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("65536")


class TestNetworkAddressInit(CustomTestCase):
    def test_basic_ipv4(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        self.assertEqual(addr.host, "127.0.0.1")
        self.assertEqual(addr.port, 8080)

    def test_ipv6_brackets_stripped(self):
        addr = NetworkAddress("[::1]", 8080)
        self.assertEqual(addr.host, "::1")

    def test_ipv6_no_brackets(self):
        addr = NetworkAddress("::1", 8080)
        self.assertEqual(addr.host, "::1")

    def test_frozen_immutable(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        with self.assertRaises((AttributeError, TypeError)):
            addr.host = "0.0.0.0"


class TestNetworkAddressProperties(CustomTestCase):
    def test_is_ipv6_true(self):
        addr = NetworkAddress("::1", 8080)
        self.assertTrue(addr.is_ipv6)

    def test_is_ipv6_false(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        self.assertFalse(addr.is_ipv6)

    def test_family_ipv4(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        self.assertEqual(addr.family, socket.AF_INET)

    def test_family_ipv6(self):
        addr = NetworkAddress("::1", 8080)
        self.assertEqual(addr.family, socket.AF_INET6)


class TestNetworkAddressURLGeneration(CustomTestCase):
    def test_to_url_ipv4(self):
        addr = NetworkAddress("127.0.0.1", 30000)
        self.assertEqual(addr.to_url(), "http://127.0.0.1:30000")

    def test_to_url_ipv6(self):
        addr = NetworkAddress("::1", 30000)
        self.assertEqual(addr.to_url(), "http://[::1]:30000")

    def test_to_url_custom_scheme(self):
        addr = NetworkAddress("127.0.0.1", 30000)
        self.assertEqual(addr.to_url("https"), "https://127.0.0.1:30000")

    def test_to_tcp_ipv4(self):
        addr = NetworkAddress("127.0.0.1", 5555)
        self.assertEqual(addr.to_tcp(), "tcp://127.0.0.1:5555")

    def test_to_tcp_ipv6(self):
        addr = NetworkAddress("2001:db8::1", 5555)
        self.assertEqual(addr.to_tcp(), "tcp://[2001:db8::1]:5555")

    def test_to_host_port_str_ipv4(self):
        addr = NetworkAddress("10.0.0.1", 9000)
        self.assertEqual(addr.to_host_port_str(), "10.0.0.1:9000")

    def test_to_host_port_str_ipv6(self):
        addr = NetworkAddress("::1", 9000)
        self.assertEqual(addr.to_host_port_str(), "[::1]:9000")

    def test_str(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        self.assertEqual(str(addr), "127.0.0.1:8080")

    def test_repr(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        self.assertIn("NetworkAddress", repr(addr))

    def test_to_bind_tuple(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        self.assertEqual(addr.to_bind_tuple(), ("127.0.0.1", 8080))

    def test_to_bind_tuple_ipv6(self):
        # brackets are stripped, raw host returned
        addr = NetworkAddress("[::1]", 8080)
        self.assertEqual(addr.to_bind_tuple(), ("::1", 8080))


class TestNetworkAddressParse(CustomTestCase):
    def test_parse_ipv4(self):
        addr = NetworkAddress.parse("127.0.0.1:8080")
        self.assertEqual(addr.host, "127.0.0.1")
        self.assertEqual(addr.port, 8080)

    def test_parse_ipv6_bracketed(self):
        addr = NetworkAddress.parse("[::1]:8080")
        self.assertEqual(addr.host, "::1")
        self.assertEqual(addr.port, 8080)

    def test_parse_ipv6_full(self):
        addr = NetworkAddress.parse("[2001:db8::1]:9000")
        self.assertEqual(addr.host, "2001:db8::1")
        self.assertEqual(addr.port, 9000)

    def test_parse_hostname(self):
        addr = NetworkAddress.parse("example.com:443")
        self.assertEqual(addr.host, "example.com")
        self.assertEqual(addr.port, 443)

    def test_parse_empty_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("")

    def test_parse_no_port_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1")

    def test_parse_missing_closing_bracket_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[::1:8080")

    def test_parse_bare_ipv6_raises(self):
        # bare IPv6 without brackets is ambiguous
        with self.assertRaises(ValueError):
            NetworkAddress.parse("::1:8080")

    def test_parse_invalid_port_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1:abc")

    def test_parse_port_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1:99999")

    def test_parse_invalid_ipv6_in_brackets_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[notanip]:8080")

    def test_parse_empty_host_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse(":8080")

    def test_parse_roundtrip_ipv4(self):
        original = "192.168.0.1:5000"
        addr = NetworkAddress.parse(original)
        self.assertEqual(str(addr), original)

    def test_parse_roundtrip_ipv6(self):
        addr = NetworkAddress.parse("[::1]:5000")
        self.assertEqual(str(addr), "[::1]:5000")


class TestNetworkAddressResolveHost(CustomTestCase):
    def test_resolve_ipv4_passthrough(self):
        ip = NetworkAddress.resolve_host("127.0.0.1")
        self.assertEqual(ip, "127.0.0.1")

    def test_resolve_ipv6_passthrough(self):
        ip = NetworkAddress.resolve_host("::1")
        self.assertEqual(ip, "::1")

    def test_resolve_invalid_hostname_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.resolve_host("this-hostname-definitely-does-not-exist.invalid")

    def test_resolved_same_when_already_ip(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        resolved = addr.resolved()
        self.assertIs(resolved, addr)


class TestNetworkAddressEquality(CustomTestCase):
    def test_equal_same_values(self):
        a = NetworkAddress("127.0.0.1", 8080)
        b = NetworkAddress("127.0.0.1", 8080)
        self.assertEqual(a, b)

    def test_not_equal_different_port(self):
        a = NetworkAddress("127.0.0.1", 8080)
        b = NetworkAddress("127.0.0.1", 9090)
        self.assertNotEqual(a, b)

    def test_not_equal_different_host(self):
        a = NetworkAddress("127.0.0.1", 8080)
        b = NetworkAddress("0.0.0.0", 8080)
        self.assertNotEqual(a, b)

    def test_hashable(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        s = {addr}
        self.assertIn(addr, s)


if __name__ == "__main__":
    unittest.main()
