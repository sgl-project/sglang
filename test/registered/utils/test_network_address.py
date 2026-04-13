import socket
import unittest
from unittest.mock import patch

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils.network import NetworkAddress
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")

# Mock get_device() so ServerArgs tests run on CPU-only CI runners
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestNetworkAddressIPv4(unittest.TestCase):
    def test_basic_properties(self):
        na = NetworkAddress("127.0.0.1", 30000)
        self.assertEqual(na.host, "127.0.0.1")
        self.assertEqual(na.port, 30000)
        self.assertFalse(na.is_ipv6)
        self.assertEqual(na.family, socket.AF_INET)

    def test_to_url(self):
        na = NetworkAddress("10.0.0.1", 8080)
        self.assertEqual(na.to_url(), "http://10.0.0.1:8080")
        self.assertEqual(na.to_url("https"), "https://10.0.0.1:8080")

    def test_to_tcp(self):
        self.assertEqual(
            NetworkAddress("10.0.0.1", 25000).to_tcp(), "tcp://10.0.0.1:25000"
        )

    def test_to_host_port_str(self):
        self.assertEqual(
            NetworkAddress("192.168.1.1", 443).to_host_port_str(), "192.168.1.1:443"
        )

    def test_to_bind_tuple(self):
        self.assertEqual(
            NetworkAddress("0.0.0.0", 30000).to_bind_tuple(), ("0.0.0.0", 30000)
        )

    def test_str(self):
        self.assertEqual(str(NetworkAddress("127.0.0.1", 30000)), "127.0.0.1:30000")


class TestNetworkAddressIPv6(unittest.TestCase):
    def test_basic_properties(self):
        na = NetworkAddress("::1", 30000)
        self.assertEqual(na.host, "::1")
        self.assertEqual(na.port, 30000)
        self.assertTrue(na.is_ipv6)
        self.assertEqual(na.family, socket.AF_INET6)

    def test_to_url(self):
        self.assertEqual(NetworkAddress("::1", 8080).to_url(), "http://[::1]:8080")

    def test_to_url_custom_scheme(self):
        na = NetworkAddress("2001:db8::1", 443)
        self.assertEqual(na.to_url("https"), "https://[2001:db8::1]:443")
        self.assertEqual(na.to_url("instance"), "instance://[2001:db8::1]:443")

    def test_to_tcp(self):
        self.assertEqual(NetworkAddress("::1", 25000).to_tcp(), "tcp://[::1]:25000")

    def test_to_host_port_str(self):
        self.assertEqual(NetworkAddress("::1", 443).to_host_port_str(), "[::1]:443")

    def test_to_bind_tuple_raw(self):
        self.assertEqual(NetworkAddress("::1", 30000).to_bind_tuple(), ("::1", 30000))

    def test_full_ipv6_address(self):
        na = NetworkAddress("2001:0db8:85a3:0000:0000:8a2e:0370:7334", 80)
        self.assertTrue(na.is_ipv6)
        self.assertEqual(
            na.to_url(), "http://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:80"
        )

    def test_str(self):
        self.assertEqual(str(NetworkAddress("::1", 30000)), "[::1]:30000")


class TestNetworkAddressHostname(unittest.TestCase):
    def test_hostname(self):
        na = NetworkAddress("my-server", 8080)
        self.assertFalse(na.is_ipv6)
        self.assertEqual(na.family, socket.AF_INET)
        self.assertEqual(na.to_url(), "http://my-server:8080")
        self.assertEqual(na.to_tcp(), "tcp://my-server:8080")

    def test_localhost(self):
        na = NetworkAddress("localhost", 30000)
        self.assertFalse(na.is_ipv6)
        self.assertEqual(na.to_url(), "http://localhost:30000")


class TestNetworkAddressParse(unittest.TestCase):
    def test_parse_ipv4(self):
        na = NetworkAddress.parse("127.0.0.1:8000")
        self.assertEqual(na, NetworkAddress("127.0.0.1", 8000))

    def test_parse_ipv4_high_port(self):
        self.assertEqual(NetworkAddress.parse("10.0.0.1:65535").port, 65535)

    def test_parse_ipv6_loopback(self):
        na = NetworkAddress.parse("[::1]:8000")
        self.assertEqual(na, NetworkAddress("::1", 8000))
        self.assertTrue(na.is_ipv6)

    def test_parse_ipv6_full(self):
        na = NetworkAddress.parse("[2001:db8::1]:30000")
        self.assertEqual(na, NetworkAddress("2001:db8::1", 30000))

    def test_parse_ipv6_all_interfaces(self):
        na = NetworkAddress.parse("[::]:8080")
        self.assertEqual(na, NetworkAddress("::", 8080))

    def test_parse_hostname(self):
        na = NetworkAddress.parse("my-server:9000")
        self.assertEqual(na, NetworkAddress("my-server", 9000))

    def test_parse_fqdn(self):
        na = NetworkAddress.parse("node1.cluster.local:25000")
        self.assertEqual(na, NetworkAddress("node1.cluster.local", 25000))

    def test_roundtrip_ipv4(self):
        na = NetworkAddress("10.0.0.1", 8080)
        self.assertEqual(NetworkAddress.parse(na.to_host_port_str()), na)

    def test_roundtrip_ipv6(self):
        na = NetworkAddress("::1", 30000)
        self.assertEqual(NetworkAddress.parse(na.to_host_port_str()), na)

    def test_roundtrip_hostname(self):
        na = NetworkAddress("my-host", 443)
        self.assertEqual(NetworkAddress.parse(na.to_host_port_str()), na)


class TestNetworkAddressParseErrors(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("")

    def test_no_port(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1")

    def test_bare_ipv6_ambiguous(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("::1:8000")

    def test_missing_closing_bracket(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[::1:8000")

    def test_invalid_ipv6_in_brackets(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[not-ipv6]:8000")

    def test_bracket_no_port(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("[::1]")

    def test_invalid_port_string(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1:abc")

    def test_port_out_of_range(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1:70000")

    def test_negative_port(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse("127.0.0.1:-1")

    def test_empty_host(self):
        with self.assertRaises(ValueError):
            NetworkAddress.parse(":8000")


class TestNetworkAddressBracketStripping(unittest.TestCase):
    def test_strip_brackets(self):
        na = NetworkAddress("[::1]", 8000)
        self.assertEqual(na.host, "::1")
        self.assertTrue(na.is_ipv6)

    def test_no_brackets(self):
        na = NetworkAddress("::1", 8000)
        self.assertEqual(na.host, "::1")

    def test_ipv4_passthrough(self):
        na = NetworkAddress("127.0.0.1", 30000)
        self.assertEqual(na.host, "127.0.0.1")
        self.assertFalse(na.is_ipv6)

    def test_hostname_passthrough(self):
        na = NetworkAddress("myhost", 30000)
        self.assertEqual(na.host, "myhost")


class TestNetworkAddressImmutability(unittest.TestCase):
    def test_frozen(self):
        na = NetworkAddress("127.0.0.1", 30000)
        with self.assertRaises(AttributeError):
            na.host = "0.0.0.0"
        with self.assertRaises(AttributeError):
            na.port = 8080

    def test_hashable(self):
        a = NetworkAddress("::1", 8000)
        b = NetworkAddress("::1", 8000)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)

    def test_inequality(self):
        a = NetworkAddress("127.0.0.1", 8000)
        b = NetworkAddress("127.0.0.1", 8001)
        self.assertNotEqual(a, b)


class TestPortArgsIPv6(unittest.TestCase):
    """PortArgs.init_new() IPv6 address parsing via NetworkAddress.parse()."""

    def test_init_new_with_ipv6_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://[2001:db8::1]:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertTrue(
            port_args.detokenizer_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_invalid_ipv6_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[invalid-ipv6]:25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)
        self.assertIn("Invalid IPv6 address inside brackets", str(context.exception))

    def test_init_new_with_malformed_ipv6_address_missing_bracket(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1:25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)
        self.assertIn("Missing closing bracket", str(context.exception))

    def test_init_new_with_malformed_ipv6_address_missing_port(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)
        self.assertIn("Expected ':port' after closing bracket", str(context.exception))

    def test_init_new_with_malformed_ipv6_address_invalid_port(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:abcde"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)
        self.assertIn("Invalid port number", str(context.exception))

    def test_init_new_with_malformed_ipv6_address_wrong_separator(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]#25000"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)
        self.assertIn("Expected ':port' after closing bracket", str(context.exception))


class TestServerArgsIPv6Url(unittest.TestCase):
    """ServerArgs.url() IPv6 formatting (moved from test_server_args.py)."""

    def test_url_rewrites_ipv6_all_interfaces_to_loopback(self):
        server_args = ServerArgs(model_path="dummy", host="::")
        self.assertEqual(server_args.url(), "http://[::1]:30000")

    @patch("os.path.isfile", return_value=True)
    def test_url_returns_https_with_ssl_and_ipv6(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy",
            host="::1",
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
        )
        self.assertEqual(server_args.url(), "https://[::1]:30000")


if __name__ == "__main__":
    unittest.main()
