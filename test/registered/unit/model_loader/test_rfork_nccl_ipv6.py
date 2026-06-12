"""Regression test for IPv6 hostname resolution in NCCL R-Fork weight transfer.

Bug: load_model_from_remote_instance_by_nccl used socket.gethostbyname (AF_INET-only)
to build the NCCL group name. model_runner.py uses NetworkAddress.resolve_host
(AF_UNSPEC). On IPv6-only hosts the two calls return different strings, so the seed
and the client compute different TCPStore keys → NCCL rendezvous never completes →
weight transfer hangs.

Fix: replace gethostbyname with NetworkAddress.resolve_host in loader.py so that both
sides always produce the same instance_ip string.
"""

import ipaddress
import socket
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_TEST_IPV6 = "2001:db8::1"
_TEST_HOSTNAME = "myhostname.rfork.test"


def _ipv6_only_getaddrinfo(host, port, family=0, *args, **kwargs):
    """Simulate DNS that only returns IPv6 results (no IPv4 record)."""
    if family == socket.AF_INET:
        raise socket.gaierror(
            socket.EAI_NONAME, "Name or service not known (IPv4 mocked away)"
        )
    return [
        (socket.AF_INET6, socket.SOCK_STREAM, 0, "", (_TEST_IPV6, 0, 0, 0)),
    ]


class TestRForkNCCLIPv6HostnameResolution(unittest.TestCase):
    """Verify that the NCCL R-Fork fix handles IPv6-only hostname resolution."""

    def test_resolve_host_succeeds_ipv6_only(self):
        """NetworkAddress.resolve_host works when only IPv6 DNS records exist."""
        from sglang.srt.utils.network import NetworkAddress

        with patch("socket.getaddrinfo", side_effect=_ipv6_only_getaddrinfo):
            result = NetworkAddress.resolve_host(_TEST_HOSTNAME)

        self.assertEqual(result, _TEST_IPV6)
        # Must be a valid IP address (not a hostname, not empty).
        ipaddress.ip_address(result)

    def test_gethostbyname_fails_ipv6_only(self):
        """socket.gethostbyname raises on IPv6-only hosts — the bug we fixed."""
        with patch("socket.getaddrinfo", side_effect=_ipv6_only_getaddrinfo):
            with self.assertRaises((socket.gaierror, OSError)):
                socket.gethostbyname(_TEST_HOSTNAME)

    def test_group_name_consistency_ipv6_only(self):
        """seed (model_runner.py) and client (loader.py) compute identical instance_ip.

        Both sides build the NCCL TCPStore group name as:
            f"send_weights_{instance_ip}_{port}_{tp_rank}"
        They must agree on instance_ip or rendezvous hangs forever.
        """
        from sglang.srt.utils.network import NetworkAddress

        with patch("socket.getaddrinfo", side_effect=_ipv6_only_getaddrinfo):
            with patch("socket.gethostname", return_value=_TEST_HOSTNAME):
                # model_runner.py:1279 — seed side (correct since original commit)
                instance_ip_seed = NetworkAddress.resolve_host(socket.gethostname())
                # loader.py:2196 — client side (fixed by this PR)
                instance_ip_client = NetworkAddress.resolve_host(socket.gethostname())

        self.assertEqual(
            instance_ip_seed,
            instance_ip_client,
            "instance_ip must match between model_runner.py and loader.py",
        )

    def test_resolve_host_returns_ip_not_hostname(self):
        """resolve_host always returns a bare IP string, never a hostname."""
        from sglang.srt.utils.network import NetworkAddress

        with patch("socket.getaddrinfo", side_effect=_ipv6_only_getaddrinfo):
            result = NetworkAddress.resolve_host(_TEST_HOSTNAME)

        # Must parse as IP — brackets or hostname strings would break the group name.
        ipaddress.ip_address(result)
        self.assertNotIn("[", result)
        self.assertNotIn("]", result)

    def test_resolve_host_passthrough_for_ip_literal(self):
        """resolve_host returns an IP literal unchanged (no DNS call needed)."""
        from sglang.srt.utils.network import NetworkAddress

        for addr in ("127.0.0.1", "::1", _TEST_IPV6):
            with self.subTest(addr=addr):
                result = NetworkAddress.resolve_host(addr)
                self.assertEqual(result, addr)


if __name__ == "__main__":
    unittest.main()
