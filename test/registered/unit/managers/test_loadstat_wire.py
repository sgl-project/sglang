"""Wire contract and port/rank gating for the LoadStat load snapshot.

Locks the msgpack array shape the router's `cache_aware_zmq` policy decodes:

    ["LoadStat", num_running_reqs, num_waiting_reqs, num_tokens,
     max_total_num_tokens, attn_dp_rank]

The consumer decodes these positionally, so a field reorder or rename is a
silent cross-language break; this pins the encoding. TestLoadPublisherGating
additionally pins which schedulers publish and on which port.
CPU-only.
"""

import unittest
from unittest.mock import MagicMock, patch

import msgspec.msgpack

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.scheduler_components.load_publisher import (
    LoadStat,
    SchedulerLoadPublisher,
)
from sglang.srt.utils.event_publisher import ZmqEventPublisher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLoadStatWire(CustomTestCase):
    def test_loadstat_msgpack_array_shape(self):
        # `attn_dp_rank` is stamped by ZmqEventPublisher.publish in production;
        # set it here to assert the full on-the-wire shape.
        stat = LoadStat(
            num_running_reqs=7,
            num_waiting_reqs=3,
            num_tokens=1024,
            max_total_num_tokens=8192,
            attn_dp_rank=2,
        )
        # Same encoder the publisher thread uses (msgspec.msgpack.Encoder).
        raw = msgspec.msgpack.Encoder().encode(stat)
        decoded = msgspec.msgpack.Decoder().decode(raw)

        # tag=True + array_like → [tag, *fields] in declaration order. The Rust
        # decoder reads the tag + first four counts and ignores any trailing
        # fields (attn_dp_rank).
        self.assertEqual(
            decoded,
            ["LoadStat", 7, 3, 1024, 8192, 2],
            "LoadStat wire shape must match the Rust decoder's expectation",
        )

    def test_loadstat_tag_is_class_name(self):
        # The Rust decoder matches the literal tag string "LoadStat"; guard
        # against an accidental msgspec `tag=` override or class rename.
        raw = msgspec.msgpack.Encoder().encode(
            LoadStat(
                num_running_reqs=0,
                num_waiting_reqs=0,
                num_tokens=0,
                max_total_num_tokens=0,
            )
        )
        decoded = msgspec.msgpack.Decoder().decode(raw)
        # omit_defaults does not trim trailing fields of an array_like struct, so
        # attn_dp_rank is always emitted -- as null when unset. The decoder must
        # tolerate a null there, not an absent element.
        self.assertEqual(decoded, ["LoadStat", 0, 0, 0, 0, None])


ZMQ_ENDPOINT = '{"publisher": "zmq", "endpoint": "tcp://*:5557"}'


class TestLoadPublisherGating(CustomTestCase):
    """One load publisher per independent KV cache, on a derivable port.

    Getting either half wrong makes several schedulers bind the same port, which
    is an uncaught ZMQError at startup for a bind-style endpoint and — worse —
    silently merges every worker's load onto one rank for a connect-style one.
    """

    def _build(self, *, config=ZMQ_ENDPOINT, dp_size=1, **ps_overrides):
        """Construct a publisher with ZmqEventPublisher stubbed out (its __init__
        binds a real socket), returning (publisher, captured_ctor_call)."""
        with patch(
            "sglang.srt.managers.scheduler_components.load_publisher.ZmqEventPublisher"
        ) as zmq_pub:
            zmq_pub.offset_endpoint_port = ZmqEventPublisher.offset_endpoint_port
            pub = SchedulerLoadPublisher(
                kv_events_config=config,
                ps=ParallelState.trivial(**ps_overrides),
                dp_size=dp_size,
            )
        return pub, zmq_pub

    def test_enabled_on_rank_zero(self):
        pub, zmq_pub = self._build()
        self.assertTrue(pub.enable)
        zmq_pub.assert_called_once()

    def test_disabled_off_pp_rank_zero(self):
        # Every PP stage shares attn_tp_rank/attn_cp_rank 0, so without the
        # pp_rank gate they all bind the same load port.
        pub, zmq_pub = self._build(pp_rank=1, pp_size=2)
        self.assertFalse(pub.enable)
        zmq_pub.assert_not_called()

    def test_disabled_off_attn_tp_and_cp_rank_zero(self):
        for override in ({"attn_tp_rank": 1}, {"attn_cp_rank": 1}):
            with self.subTest(**override):
                pub, zmq_pub = self._build(**override)
                self.assertFalse(pub.enable)
                zmq_pub.assert_not_called()

    def test_pure_dp_keys_the_load_port_by_dp_rank(self):
        # Pure DP: attn_dp_size == 1 and every worker has attn_dp_rank == 0, so
        # the publisher must key off dp_rank or all replicas collide on one port.
        _, zmq_pub = self._build(attn_dp_size=1, attn_dp_rank=0, dp_rank=2, dp_size=4)
        self.assertEqual(zmq_pub.call_args.args[0], 2)

    def test_dp_attention_keys_the_load_port_by_attn_dp_rank(self):
        _, zmq_pub = self._build(attn_dp_size=4, attn_dp_rank=3, dp_rank=0, dp_size=4)
        self.assertEqual(zmq_pub.call_args.args[0], 3)

    def test_load_port_is_packed_after_the_kv_range(self):
        _, zmq_pub = self._build(dp_size=2)
        self.assertEqual(zmq_pub.call_args.kwargs["endpoint"], "tcp://*:5559")

    def test_non_tcp_endpoint_declines_instead_of_raising(self):
        # ipc:// and inproc:// are valid KV-event endpoints; offset_endpoint_port
        # raises on ipc://, which must not take down scheduler startup.
        for endpoint in ("ipc:///tmp/kv.sock", "inproc://kv"):
            with self.subTest(endpoint=endpoint):
                pub, zmq_pub = self._build(
                    config='{"publisher": "zmq", "endpoint": "%s"}' % endpoint
                )
                self.assertFalse(pub.enable)
                zmq_pub.assert_not_called()

    def test_disabled_paths_clear_enable(self):
        # enable must be False on every bail-out, or publish_load_stat keeps
        # computing the (non-trivial) load snapshot for a null sink.
        for label, config in (
            ("no config", None),
            ("null publisher", '{"publisher": "null"}'),
            ("malformed", "{not json"),
        ):
            with self.subTest(label):
                pub, _ = self._build(config=config)
                self.assertFalse(pub.enable)

    def test_publish_skips_snapshot_when_disabled(self):
        pub, _ = self._build(config='{"publisher": "null"}')
        provider = MagicMock()
        pub.publish_load_stat(provider, force=True)
        provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
