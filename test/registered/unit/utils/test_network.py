import unittest

from sglang.srt.utils.network import (
    NetworkAddress,
    _is_ipv6,
    _parse_port,
    _wrap,
    get_free_port,
    is_port_available,
    is_valid_ipv6_address,
    is_zmq_endpoint_ipv6,
    resolve_base_url,
    resolve_host_port,
    wait_port_available,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIsValidIpv6Address(CustomTestCase):
    """is_valid_ipv6_address validates IPv6 address strings."""

    def test_loopback(self):
        self.assertTrue(is_valid_ipv6_address("::1"))

    def test_full_address(self):
        self.assertTrue(
            is_valid_ipv6_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        )

    def test_compressed(self):
        self.assertTrue(is_valid_ipv6_address("2001:db8::1"))

    def test_all_zeros(self):
        self.assertTrue(is_valid_ipv6_address("::"))

    def test_ipv4_rejected(self):
        self.assertFalse(is_valid_ipv6_address("192.168.1.1"))

    def test_empty_string(self):
        self.assertFalse(is_valid_ipv6_address(""))

    def test_garbage(self):
        self.assertFalse(is_valid_ipv6_address("not_an_address"))

    def test_ipv4_mapped_ipv6(self):
        self.assertTrue(is_valid_ipv6_address("::ffff:192.0.2.1"))

    def test_bracketed_rejected(self):
        # Brackets are not part of a valid IPv6 address literal
        self.assertFalse(is_valid_ipv6_address("[::1]"))


class TestIsIpv6(CustomTestCase):
    """_is_ipv6 is an internal helper with same semantics as is_valid_ipv6_address."""

    def test_valid(self):
        self.assertTrue(_is_ipv6("::1"))
        self.assertTrue(_is_ipv6("fe80::1"))

    def test_invalid(self):
        self.assertFalse(_is_ipv6("127.0.0.1"))
        self.assertFalse(_is_ipv6("hostname"))


class TestWrap(CustomTestCase):
    """_wrap brackets IPv6 addresses, passes IPv4/hostnames through."""

    def test_ipv6_wrapped(self):
        self.assertEqual(_wrap("::1"), "[::1]")
        self.assertEqual(_wrap("2001:db8::1"), "[2001:db8::1]")

    def test_ipv4_passthrough(self):
        self.assertEqual(_wrap("127.0.0.1"), "127.0.0.1")
        self.assertEqual(_wrap("10.0.0.1"), "10.0.0.1")

    def test_hostname_passthrough(self):
        self.assertEqual(_wrap("localhost"), "localhost")
        self.assertEqual(_wrap("my-server.example.com"), "my-server.example.com")


class TestParsePort(CustomTestCase):
    """_parse_port parses and validates TCP port numbers."""

    def test_valid_ports(self):
        self.assertEqual(_parse_port("0"), 0)
        self.assertEqual(_parse_port("80"), 80)
        self.assertEqual(_parse_port("8080"), 8080)
        self.assertEqual(_parse_port("65535"), 65535)

    def test_boundary_values(self):
        self.assertEqual(_parse_port("0"), 0)
        self.assertEqual(_parse_port("65535"), 65535)

    def test_negative_port_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("-1")

    def test_port_too_large_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("65536")

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("abc")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("")

    def test_float_string_raises(self):
        with self.assertRaises(ValueError):
            _parse_port("80.5")


class TestIsZmqEndpointIpv6(CustomTestCase):
    """is_zmq_endpoint_ipv6 detects bracketed IPv6 in ZMQ TCP endpoints."""

    def test_ipv6_endpoint(self):
        self.assertTrue(is_zmq_endpoint_ipv6("tcp://[::1]:5555"))
        self.assertTrue(is_zmq_endpoint_ipv6("tcp://[2001:db8::1]:8080"))

    def test_ipv4_endpoint(self):
        self.assertFalse(is_zmq_endpoint_ipv6("tcp://127.0.0.1:5555"))

    def test_wildcard_endpoint(self):
        self.assertFalse(is_zmq_endpoint_ipv6("tcp://*:5555"))

    def test_no_tcp_prefix(self):
        self.assertFalse(is_zmq_endpoint_ipv6("ipc:///tmp/test"))

    def test_missing_closing_bracket(self):
        self.assertFalse(is_zmq_endpoint_ipv6("tcp://[::1:5555"))

    def test_invalid_ipv6_in_brackets(self):
        self.assertFalse(is_zmq_endpoint_ipv6("tcp://[not_ipv6]:5555"))


class TestNetworkAddressInit(CustomTestCase):
    """NetworkAddress __init__ and __post_init__ bracket stripping."""

    def test_basic_construction(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        self.assertEqual(addr.host, "127.0.0.1")
        self.assertEqual(addr.port, 8000)

    def test_ipv6_bracket_stripping(self):
        addr = NetworkAddress("[::1]", 8000)
        self.assertEqual(addr.host, "::1")

    def test_ipv6_without_brackets(self):
        addr = NetworkAddress("::1", 8000)
        self.assertEqual(addr.host, "::1")

    def test_frozen_dataclass(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        with self.assertRaises(AttributeError):
            addr.host = "0.0.0.0"


class TestNetworkAddressProperties(CustomTestCase):
    """NetworkAddress is_ipv6 and family properties."""

    def test_ipv4_not_ipv6(self):
        addr = NetworkAddress("127.0.0.1", 80)
        self.assertFalse(addr.is_ipv6)

    def test_ipv6_detected(self):
        addr = NetworkAddress("::1", 80)
        self.assertTrue(addr.is_ipv6)

    def test_hostname_not_ipv6(self):
        addr = NetworkAddress("localhost", 80)
        self.assertFalse(addr.is_ipv6)

    def test_family_ipv4(self):
        import socket

        addr = NetworkAddress("127.0.0.1", 80)
        self.assertEqual(addr.family, socket.AF_INET)

    def test_family_ipv6(self):
        import socket

        addr = NetworkAddress("::1", 80)
        self.assertEqual(addr.family, socket.AF_INET6)


class TestNetworkAddressUrls(CustomTestCase):
    """NetworkAddress to_url, to_tcp, to_host_port_str."""

    def test_to_url_ipv4(self):
        addr = NetworkAddress("127.0.0.1", 30000)
        self.assertEqual(addr.to_url(), "http://127.0.0.1:30000")

    def test_to_url_ipv6(self):
        addr = NetworkAddress("::1", 30000)
        self.assertEqual(addr.to_url(), "http://[::1]:30000")

    def test_to_url_custom_scheme(self):
        addr = NetworkAddress("127.0.0.1", 443)
        self.assertEqual(addr.to_url("https"), "https://127.0.0.1:443")

    def test_to_tcp_ipv4(self):
        addr = NetworkAddress("10.0.0.1", 5555)
        self.assertEqual(addr.to_tcp(), "tcp://10.0.0.1:5555")

    def test_to_tcp_ipv6(self):
        addr = NetworkAddress("2001:db8::1", 5555)
        self.assertEqual(addr.to_tcp(), "tcp://[2001:db8::1]:5555")

    def test_to_host_port_str_ipv4(self):
        addr = NetworkAddress("192.168.1.1", 8080)
        self.assertEqual(addr.to_host_port_str(), "192.168.1.1:8080")

    def test_to_host_port_str_ipv6(self):
        addr = NetworkAddress("::1", 8080)
        self.assertEqual(addr.to_host_port_str(), "[::1]:8080")

    def test_to_bind_tuple(self):
        addr = NetworkAddress("0.0.0.0", 9000)
        self.assertEqual(addr.to_bind_tuple(), ("0.0.0.0", 9000))


class TestNetworkAddressParse(CustomTestCase):
    """NetworkAddress.parse handles IPv4, IPv6 (bracketed), and hostnames."""

    def test_parse_ipv4(self):
        addr = NetworkAddress.parse("127.0.0.1:8000")
        self.assertEqual(addr.host, "127.0.0.1")
        self.assertEqual(addr.port, 8000)

    def test_parse_ipv6_bracketed(self):
        addr = NetworkAddress.parse("[::1]:8000")
        self.assertEqual(addr.host, "::1")
        self.assertEqual(addr.port, 8000)

    def test_parse_ipv6_full(self):
        addr = NetworkAddress.parse("[2001:db8::1]:443")
        self.assertEqual(addr.host, "2001:db8::1")
        self.assertEqual(addr.port, 443)

    def test_parse_hostname(self):
        addr = NetworkAddress.parse("my-hostname:8000")
        self.assertEqual(addr.host, "my-hostname")
        self.assertEqual(addr.port, 8000)

    def test_parse_empty_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("")

    def test_parse_no_port_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1")

    def test_parse_empty_host_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse(":8000")

    def test_parse_bare_ipv6_raises(self):
        # Bare IPv6 without brackets is ambiguous
        with self.assertRaises(ValueError):
            NetworkAddress.parse("::1:8000")

    def test_parse_missing_bracket_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[::1:8000")

    def test_parse_invalid_ipv6_in_brackets_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[not_ipv6]:8000")

    def test_parse_no_port_after_bracket_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[::1]")

    def test_parse_invalid_port_raises(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1:99999")


class TestNetworkAddressStringRepresentation(CustomTestCase):
    """NetworkAddress __str__ and __repr__."""

    def test_str_ipv4(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        self.assertEqual(str(addr), "127.0.0.1:8000")

    def test_str_ipv6(self):
        addr = NetworkAddress("::1", 8000)
        self.assertEqual(str(addr), "[::1]:8000")

    def test_repr(self):
        addr = NetworkAddress("127.0.0.1", 8000)
        self.assertEqual(repr(addr), "NetworkAddress('127.0.0.1', 8000)")


class TestResolveBaseUrl(CustomTestCase):
    """resolve_base_url returns base_url if set, else constructs from host/port."""

    def test_base_url_provided(self):
        result = resolve_base_url("http://custom:9000", "127.0.0.1", 8000)
        self.assertEqual(result, "http://custom:9000")

    def test_base_url_empty_constructs_ipv4(self):
        result = resolve_base_url("", "127.0.0.1", 8000)
        self.assertEqual(result, "http://127.0.0.1:8000")

    def test_base_url_empty_constructs_ipv6(self):
        result = resolve_base_url("", "::1", 8000)
        self.assertEqual(result, "http://[::1]:8000")


class TestResolveHostPort(CustomTestCase):
    """resolve_host_port returns base_url if set, else host:port string."""

    def test_base_url_provided(self):
        result = resolve_host_port("custom:9000", "127.0.0.1", 8000)
        self.assertEqual(result, "custom:9000")

    def test_base_url_empty_constructs_ipv4(self):
        result = resolve_host_port("", "127.0.0.1", 8000)
        self.assertEqual(result, "127.0.0.1:8000")

    def test_base_url_empty_constructs_ipv6(self):
        result = resolve_host_port("", "::1", 8000)
        self.assertEqual(result, "[::1]:8000")


class TestWaitPortAvailable(CustomTestCase):
    """wait_port_available rejects invalid port numbers."""

    def test_negative_port_raises(self):
        with self.assertRaises(ValueError, msg="invalid port number"):
            wait_port_available(-1, "test_port")

    def test_port_too_large_raises(self):
        with self.assertRaises(ValueError, msg="invalid port number"):
            wait_port_available(65536, "test_port")


class TestGetFreePortAndIsPortAvailable(CustomTestCase):
    """get_free_port returns available ports; is_port_available validates them."""

    def test_get_free_port_returns_valid_range(self):
        port = get_free_port()
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    def test_free_port_is_available(self):
        port = get_free_port()
        self.assertTrue(is_port_available(port))


if __name__ == "__main__":
    unittest.main()
