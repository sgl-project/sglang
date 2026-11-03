"""Unit tests for srt/utils/network.py: ZMQ peer allowlist and source pinning."""

import unittest

import zmq

from sglang.srt.environ import envs
from sglang.srt.utils import network
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestZmqSourceIpEndpoint(CustomTestCase):
    def test_rewrites_only_matching_tcp_ip_endpoints(self):
        """`SGLANG_ZMQ_SOURCE_IP` is spliced in with libzmq's `src:0;dst` form;
        hostnames, ipc endpoints and the other address family pass unchanged."""
        cases = [
            ("10.0.0.5", "tcp://10.0.0.1:5555", "tcp://10.0.0.5:0;10.0.0.1:5555"),
            (
                "2001:db8::2",
                "tcp://[2001:db8::1]:5555",
                "tcp://[2001:db8::2]:0;[2001:db8::1]:5555",
            ),
            ("2001:db8::2", "tcp://127.0.0.1:5555", "tcp://127.0.0.1:5555"),
            ("10.0.0.5", "tcp://somehost:5555", "tcp://somehost:5555"),
            ("10.0.0.5", "ipc:///tmp/sock", "ipc:///tmp/sock"),
            (
                "10.0.0.5",
                "tcp://10.0.0.9:0;10.0.0.1:5555",
                "tcp://10.0.0.9:0;10.0.0.1:5555",
            ),
            (None, "tcp://10.0.0.1:5555", "tcp://10.0.0.1:5555"),
        ]
        for source, endpoint, expected in cases:
            with self.subTest(source=source, endpoint=endpoint):
                with envs.SGLANG_ZMQ_SOURCE_IP.override(source):
                    if source is None:
                        envs.SGLANG_ZMQ_SOURCE_IP.clear()
                    self.assertEqual(network._with_source_ip(endpoint), expected)


class TestZmqAllowlistAdmitsPinnedSource(CustomTestCase):
    """A multi-homed client whose kernel picks a source address outside the
    peer allowlist is rejected and its PUSH blocks; pinning the source to the
    allowlisted address must get the message through."""

    def _delivered(self, source: str | None) -> bool:
        server_ctx, client_ctx = zmq.Context(), zmq.Context()
        try:
            with envs.SGLANG_ZMQ_ALLOWED_PEERS.override("127.0.0.2"):
                port, server = network.get_zmq_socket(server_ctx, zmq.PULL)
                with envs.SGLANG_ZMQ_SOURCE_IP.override(source):
                    if source is None:
                        envs.SGLANG_ZMQ_SOURCE_IP.clear()
                    client = network.get_zmq_socket(
                        client_ctx, zmq.PUSH, f"tcp://127.0.0.1:{port}", bind=False
                    )
            client.send(b"ping")
            return server.poll(2000) != 0
        finally:
            with network._zmq_authenticators_lock:
                auth = network._zmq_authenticators.pop(server_ctx.underlying, None)
            if auth is not None:
                auth.stop()
            server_ctx.destroy(linger=0)
            client_ctx.destroy(linger=0)

    def test_unpinned_connect_is_rejected(self):
        self.assertFalse(self._delivered(None))

    def test_pinned_connect_is_admitted(self):
        self.assertTrue(self._delivered("127.0.0.2"))


if __name__ == "__main__":
    unittest.main()
