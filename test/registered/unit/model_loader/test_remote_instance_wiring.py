"""Unit tests for the remote-instance (R-Fork) rendezvous wiring.

Regression guards for three defects found while running R-Fork between two hosts:
an IPv6 seed address that could not be expressed as a URL at all, group ports
accepted inside the kernel's ephemeral range, and a seed refusal that nothing
looked at -- which is what turned each of the others into a silent hang.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.model_path_hook import validate_remote_instance_group_ports
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    _raise_for_seed_refusal,
)
from sglang.srt.utils.network import NetworkAddress, local_ephemeral_port_range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

IPV6 = "fdbd:dccd:cdc2:12d1:0:314::"
IPV4 = "10.124.162.216"


class TestSeedUrlIsIpv6Safe(CustomTestCase):
    """A bare IPv6 seed address used to be interpolated straight into
    f"http://{ip}:{port}", producing an unparsable URL, so R-Fork could not be
    used at all on an IPv6-only host."""

    def test_ipv6_host_is_bracketed(self):
        self.assertEqual(NetworkAddress(IPV6, 30000).to_url(), f"http://[{IPV6}]:30000")
        # The instance:// connector URL goes through urlparse on the other side,
        # which needs the brackets to find the port.
        self.assertEqual(
            NetworkAddress(IPV6, 20000).to_url("instance"), f"instance://[{IPV6}]:20000"
        )

    def test_ipv4_host_is_untouched(self):
        self.assertEqual(NetworkAddress(IPV4, 30000).to_url(), f"http://{IPV4}:30000")


class TestSeedRefusalIsSurfaced(CustomTestCase):
    """The group-creation request is fire-and-forget on a worker thread, so nothing
    downstream inspects the response. A seed that refuses used to leave the client
    sitting in the group build until the process-group timeout fired tens of
    minutes later, with no log line naming a cause."""

    def _response(self, status, body=None, text=""):
        class _Resp:
            status_code = status

            def __init__(self):
                self.text = text

            def json(self):
                if body is None:
                    raise ValueError("not json")
                return body

        return _Resp()

    def test_success_is_silent(self):
        _raise_for_seed_refusal(self._response(200), "x", "http://seed:1")

    def test_seed_message_reaches_the_exception(self):
        # This is the whole point: the operator needs the seed's own reason, e.g.
        # "Failed to init group: address already in use".
        resp = self._response(
            400, body={"success": False, "message": "Failed to init group: EADDRINUSE."}
        )
        with self.assertRaises(RuntimeError) as cm:
            _raise_for_seed_refusal(resp, "group creation", "http://seed:1")
        self.assertIn("EADDRINUSE", str(cm.exception))
        self.assertIn("group creation", str(cm.exception))
        self.assertIn("http://seed:1", str(cm.exception))

    def test_non_json_body_still_raises(self):
        # A proxy or a crashed worker can answer with plain text; losing the raise
        # here would put the silent hang straight back.
        resp = self._response(502, body=None, text="upstream gone")
        with self.assertRaises(RuntimeError) as cm:
            _raise_for_seed_refusal(resp, "group creation", "http://seed:1")
        self.assertIn("upstream gone", str(cm.exception))


class TestGroupPortValidation(CustomTestCase):
    """Send-weights group ports are listened on by the seed, but the kernel also
    hands out ports in the ephemeral range to outbound connections, so an
    overlapping choice races and fails with EADDRINUSE some of the time."""

    def _validate(self, ports):
        validate_remote_instance_group_ports(
            SimpleNamespace(
                remote_instance_weight_loader_send_weights_group_ports=ports,
            )
        )

    def test_ephemeral_ports_are_rejected(self):
        port_range = local_ephemeral_port_range()
        if port_range is None:
            self.skipTest("kernel ephemeral range not readable on this platform")
        low, high = port_range
        with self.assertRaises(ValueError) as cm:
            self._validate([low, low + 1])
        # The message has to name the range, or the operator cannot act on it.
        self.assertIn(str(low), str(cm.exception))
        self.assertIn(str(high), str(cm.exception))

    def test_ports_below_the_range_are_accepted(self):
        port_range = local_ephemeral_port_range()
        if port_range is None:
            self.skipTest("kernel ephemeral range not readable on this platform")
        self._validate([port_range[0] - 2, port_range[0] - 1])

    def test_unset_ports_are_accepted(self):
        # transfer_engine / modelexpress do not use group ports at all.
        self._validate(None)


if __name__ == "__main__":
    unittest.main()
