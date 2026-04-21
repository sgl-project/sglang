"""Unit tests for srt/utils/network.py — no server, no model loading."""

import socket
import unittest

from sglang.srt.utils.network import (
    NetworkAddress,
    _is_ipv6,
    _parse_port,
    _wrap,
    is_port_available,
    is_valid_ipv6_address,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


# ---------------------------------------------------------------------------
# is_valid_ipv6_address
# ---------------------------------------------------------------------------


class TestIsValidIpv6Address(CustomTestCase):
    def test_loopback(self):
        self.assertTrue(is_valid_ipv6_address("::1"))

    def test_full_address(self):
        self.assertTrue(is_valid_ipv6_address("2001:db8::1"))

    def test_all_zeros(self):
        self.assertTrue(is_valid_ipv6_address("::"))

    def test_ipv4_is_not_ipv6(self):
        self.assertFalse(is_valid_ipv6_address("127.0.0.1"))

    def test_hostname_is_not_ipv6(self):
        self.assertFalse(is_valid_ipv6_address("localhost"))

    def test_empty_string(self):
        self.assertFalse(is_valid_ipv6_address(""))

    def test_bracketed_ipv6_is_not_valid(self):
        # brackets are not part of the address itself
        self.assertFalse(is_valid_ipv6_address("[::1]"))


# ---------------------------------------------------------------------------
# _is_ipv6  (internal helper)
# ---------------------------------------------------------------------------


class TestIsIpv6(CustomTestCase):
    def test_loopback(self):
        self.assertTrue(_is_ipv6("::1"))

    def test_ipv4(self):
        self.assertFalse(_is_ipv6("192.168.1.1"))

    def test_hostname(self):
        self.assertFalse(_is_ipv6("example.com"))

    def test_empty(self):
        self.assertFalse(_is_ipv6(""))


# ---------------------------------------------------------------------------
# _wrap
# ---------------------------------------------------------------------------


class TestWrap(CustomTestCase):
    def test_ipv6_gets_brackets(self):
        self.assertEqual(_wrap("::1"), "[::1]")

    def test_ipv4_passthrough(self):
        self.assertEqual(_wrap("127.0.0.1"), "127.0.0.1")

    def test_hostname_passthrough(self):
        self.assertEqual(_wrap("example.com"), "example.com")

    def test_full_ipv6_gets_brackets(self):
        self.assertEqual(_wrap("2001:db8::1"), "[2001:db8::1]")


# ---------------------------------------------------------------------------
# _parse_port
# ---------------------------------------------------------------------------


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

    def test_over_max_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("65536")

    def test_float_string_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("80.5")


# ---------------------------------------------------------------------------
# NetworkAddress — construction and auto-strip
# ---------------------------------------------------------------------------


class TestNetworkAddressConstruction(CustomTestCase):
    def test_ipv4(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        self.assertEqual(addr.host, "127.0.0.1")
        self.assertEqual(addr.port, 8080)

    def test_ipv6_plain(self):
        addr = NetworkAddress("::1", 8080)
        self.assertEqual(addr.host, "::1")

    def test_ipv6_bracketed_auto_strip(self):
        addr = NetworkAddress("[::1]", 8080)
        self.assertEqual(addr.host, "::1")

    def test_hostname(self):
        addr = NetworkAddress("my-host.local", 9000)
        self.assertEqual(addr.host, "my-host.local")


# ---------------------------------------------------------------------------
# NetworkAddress.is_ipv6 / family
# ---------------------------------------------------------------------------


class TestNetworkAddressProperties(CustomTestCase):
    def test_is_ipv6_true(self):
        self.assertTrue(NetworkAddress("::1", 80).is_ipv6)

    def test_is_ipv6_false_ipv4(self):
        self.assertFalse(NetworkAddress("127.0.0.1", 80).is_ipv6)

    def test_is_ipv6_false_hostname(self):
        self.assertFalse(NetworkAddress("example.com", 80).is_ipv6)

    def test_family_ipv6(self):
        self.assertEqual(NetworkAddress("::1", 80).family, socket.AF_INET6)

    def test_family_ipv4(self):
        self.assertEqual(NetworkAddress("127.0.0.1", 80).family, socket.AF_INET)


# ---------------------------------------------------------------------------
# NetworkAddress.to_url / to_tcp / to_host_port_str
# ---------------------------------------------------------------------------


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
        addr = NetworkAddress("::1", 5555)
        self.assertEqual(addr.to_tcp(), "tcp://[::1]:5555")

    def test_to_host_port_str_ipv4(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        self.assertEqual(addr.to_host_port_str(), "127.0.0.1:8000")

    def test_to_host_port_str_ipv6(self):
        addr = NetworkAddress("::1", 8000)
        self.assertEqual(addr.to_host_port_str(), "[::1]:8000")

    def test_str_matches_to_host_port_str(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        self.assertEqual(str(addr), addr.to_host_port_str())

    def test_repr(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        self.assertEqual(repr(addr), "NetworkAddress('127.0.0.1', 8000)")


# ---------------------------------------------------------------------------
# NetworkAddress.to_bind_tuple
# ---------------------------------------------------------------------------


class TestNetworkAddressBindTuple(CustomTestCase):
    def test_ipv4(self):
        addr = NetworkAddress("0.0.0.0", 8080)
        self.assertEqual(addr.to_bind_tuple(), ("0.0.0.0", 8080))

    def test_ipv6_returns_unwrapped(self):
        # Brackets must NOT appear in the raw socket tuple
        addr = NetworkAddress("::1", 8080)
        host, port = addr.to_bind_tuple()
        self.assertFalse(host.startswith("["))
        self.assertEqual(host, "::1")
        self.assertEqual(port, 8080)


# ---------------------------------------------------------------------------
# NetworkAddress.parse
# ---------------------------------------------------------------------------


class TestNetworkAddressParse(CustomTestCase):
    # --- valid IPv4 ---

    def test_parse_ipv4_loopback(self):
        addr = NetworkAddress.parse("127.0.0.1:8080")
        self.assertEqual(addr.host, "127.0.0.1")
        self.assertEqual(addr.port, 8080)

    def test_parse_ipv4_all_interfaces(self):
        addr = NetworkAddress.parse("0.0.0.0:30000")
        self.assertEqual(addr.host, "0.0.0.0")
        self.assertEqual(addr.port, 30000)

    # --- valid IPv6 (bracketed) ---

    def test_parse_ipv6_loopback(self):
        addr = NetworkAddress.parse("[::1]:8000")
        self.assertEqual(addr.host, "::1")
        self.assertEqual(addr.port, 8000)

    def test_parse_ipv6_full(self):
        addr = NetworkAddress.parse("[2001:db8::1]:443")
        self.assertEqual(addr.host, "2001:db8::1")
        self.assertEqual(addr.port, 443)

    # --- valid hostname ---

    def test_parse_hostname(self):
        addr = NetworkAddress.parse("my-host.local:9000")
        self.assertEqual(addr.host, "my-host.local")
        self.assertEqual(addr.port, 9000)

    # --- edge cases ---

    def test_parse_port_zero(self):
        addr = NetworkAddress.parse("127.0.0.1:0")
        self.assertEqual(addr.port, 0)

    def test_parse_port_max(self):
        addr = NetworkAddress.parse("127.0.0.1:65535")
        self.assertEqual(addr.port, 65535)

    # --- error cases ---

    def test_parse_empty_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("")

    def test_parse_missing_port_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1")

    def test_parse_bare_ipv6_raises(self):
        # Bare IPv6 without brackets is ambiguous
        with self.assertRaises(ValueError):
            NetworkAddress.parse("::1:8000")

    def test_parse_invalid_bracket_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[::1")  # missing closing bracket

    def test_parse_bracket_non_ipv6_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[127.0.0.1]:8080")  # not a valid IPv6

    def test_parse_invalid_port_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1:notaport")

    def test_parse_port_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1:99999")

    def test_parse_empty_host_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse(":8080")


# ---------------------------------------------------------------------------
# NetworkAddress.resolve_host — only test IP passthrough (no real DNS)
# ---------------------------------------------------------------------------


class TestNetworkAddressResolveHost(CustomTestCase):
    def test_ipv4_passthrough(self):
        result = NetworkAddress.resolve_host("127.0.0.1")
        self.assertEqual(result, "127.0.0.1")

    def test_ipv6_passthrough(self):
        result = NetworkAddress.resolve_host("::1")
        self.assertEqual(result, "::1")

    def test_unresolvable_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.resolve_host("this-host-definitely-does-not-exist.invalid")


# ---------------------------------------------------------------------------
# NetworkAddress.resolved — for already-IP addresses
# ---------------------------------------------------------------------------


class TestNetworkAddressResolved(CustomTestCase):
    def test_ipv4_returns_self(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        resolved = addr.resolved()
        self.assertIs(resolved, addr)

    def test_ipv6_returns_self(self):
        addr = NetworkAddress("::1", 8080)
        resolved = addr.resolved()
        self.assertIs(resolved, addr)


# ---------------------------------------------------------------------------
# NetworkAddress — equality and hash (frozen dataclass)
# ---------------------------------------------------------------------------


class TestNetworkAddressEquality(CustomTestCase):
    def test_equal(self):
        a = NetworkAddress("127.0.0.1", 8080)
        b = NetworkAddress("127.0.0.1", 8080)
        self.assertEqual(a, b)

    def test_not_equal_port(self):
        a = NetworkAddress("127.0.0.1", 8080)
        b = NetworkAddress("127.0.0.1", 9090)
        self.assertNotEqual(a, b)

    def test_not_equal_host(self):
        a = NetworkAddress("127.0.0.1", 8080)
        b = NetworkAddress("0.0.0.0", 8080)
        self.assertNotEqual(a, b)

    def test_hashable_and_usable_in_set(self):
        a = NetworkAddress("127.0.0.1", 8080)
        b = NetworkAddress("127.0.0.1", 8080)
        s = {a, b}
        self.assertEqual(len(s), 1)

    def test_frozen(self):
        addr = NetworkAddress("127.0.0.1", 8080)
        with self.assertRaises((AttributeError, TypeError)):
            addr.port = 9090


# ---------------------------------------------------------------------------
# is_port_available — basic sanity (uses a real socket, but no server)
# ---------------------------------------------------------------------------


class TestIsPortAvailable(CustomTestCase):
    def test_occupied_port_not_available(self):
        # Bind and listen — a listening socket cannot be re-bound even with
        # SO_REUSEADDR, so is_port_available must return False.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            self.assertFalse(is_port_available(port))

    def test_free_port_available(self):
        # Get a free port by letting the OS assign one, then release it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        # Socket is now closed — port should be free
        self.assertTrue(is_port_available(port))


if __name__ == "__main__":
    unittest.main()
