"""Unit tests for the ``/server_info`` ``kv_events`` block surfaced for
router-side introspection (sgl-router and equivalents).

These tests exercise only the pure projection logic — they do not stand up
an HTTP server. The full e2e contract is covered separately by the
manual ``test_kv_events.py`` flow.
"""

import json
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.kv_events import KVEventsConfig
from sglang.srt.entrypoints.http_server import _resolve_kv_events_block
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKvEventsResolvedBlock(unittest.TestCase):
    """``KVEventsConfig.resolved_block`` shape and wildcard preservation."""

    def test_default_endpoint_splits_into_host_and_port(self):
        cfg = KVEventsConfig(publisher="zmq", endpoint="tcp://*:5557", topic="kv")
        block = cfg.resolved_block(block_size=64, dp_size=1)
        # Wildcard host is preserved literally — consumer substitutes its
        # own routable address.
        self.assertEqual(block["endpoint_host"], "*")
        self.assertEqual(block["endpoint_port_base"], 5557)
        self.assertEqual(block["publisher"], "zmq")
        self.assertEqual(block["topic"], "kv")
        self.assertEqual(block["block_size"], 64)
        self.assertEqual(block["dp_size"], 1)

    def test_explicit_host_is_preserved(self):
        cfg = KVEventsConfig(
            publisher="zmq", endpoint="tcp://10.1.2.3:6000", topic="kv"
        )
        block = cfg.resolved_block(block_size=64, dp_size=2)
        self.assertEqual(block["endpoint_host"], "10.1.2.3")
        self.assertEqual(block["endpoint_port_base"], 6000)
        self.assertEqual(block["dp_size"], 2)

    def test_ipv6_host_round_trips(self):
        # IPv6 literals carry their own colons inside `[]`; we use the
        # LAST colon as the port separator which keeps the bracketed
        # host intact.
        cfg = KVEventsConfig(publisher="zmq", endpoint="tcp://[::1]:5557", topic="")
        block = cfg.resolved_block(block_size=32, dp_size=1)
        self.assertEqual(block["endpoint_host"], "[::1]")
        self.assertEqual(block["endpoint_port_base"], 5557)

    def test_malformed_endpoint_falls_back_to_empty(self):
        # If the operator misconfigures the endpoint, we still emit a
        # block (so the consumer can see the publisher kind) but with an
        # unusable host/port pair that will fail loudly at connect time
        # — better than silently dropping the block.
        cfg = KVEventsConfig(publisher="zmq", endpoint="garbage", topic="")
        block = cfg.resolved_block(block_size=64, dp_size=1)
        self.assertEqual(block["endpoint_host"], "")
        self.assertEqual(block["endpoint_port_base"], 0)


class TestResolveKvEventsBlockFromServerArgs(unittest.TestCase):
    """``_resolve_kv_events_block`` (the ``/server_info`` shim).

    Exercised via ``SimpleNamespace`` rather than a real ``ServerArgs``
    instance so the test stays a pure unit test with no model-config
    plumbing.
    """

    def test_returns_none_when_kv_events_config_unset(self):
        args = SimpleNamespace(kv_events_config=None, page_size=64, dp_size=1)
        self.assertIsNone(_resolve_kv_events_block(args))

    def test_returns_none_when_kv_events_config_empty_string(self):
        args = SimpleNamespace(kv_events_config="", page_size=64, dp_size=1)
        self.assertIsNone(_resolve_kv_events_block(args))

    def test_parses_zmq_block_with_wildcard_host(self):
        raw = json.dumps(
            {"publisher": "zmq", "endpoint": "tcp://*:5557", "topic": "kv"}
        )
        args = SimpleNamespace(kv_events_config=raw, page_size=64, dp_size=1)
        block = _resolve_kv_events_block(args)
        self.assertIsNotNone(block)
        self.assertEqual(block["publisher"], "zmq")
        self.assertEqual(block["endpoint_host"], "*")
        self.assertEqual(block["endpoint_port_base"], 5557)
        self.assertEqual(block["topic"], "kv")
        self.assertEqual(block["block_size"], 64)
        self.assertEqual(block["dp_size"], 1)

    def test_picks_up_dp_size_from_server_args(self):
        raw = json.dumps({"publisher": "zmq", "endpoint": "tcp://*:5557", "topic": ""})
        args = SimpleNamespace(kv_events_config=raw, page_size=128, dp_size=4)
        block = _resolve_kv_events_block(args)
        self.assertEqual(block["block_size"], 128)
        self.assertEqual(block["dp_size"], 4)

    def test_malformed_json_returns_none_without_raising(self):
        args = SimpleNamespace(
            kv_events_config="{not valid json", page_size=64, dp_size=1
        )
        # Defensive path: we don't want a malformed flag to crash
        # /server_info — return None so the key is simply omitted.
        self.assertIsNone(_resolve_kv_events_block(args))


if __name__ == "__main__":
    unittest.main()
