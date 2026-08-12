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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec.msgpack

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.scheduler_components.load_publisher import (
    LoadStat,
    SchedulerLoadPublisher,
)
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

    def test_underivable_endpoint_declines_instead_of_raising(self):
        # ipc:// and inproc:// are valid KV-event endpoints but carry no port
        # to offset; tcp:// endpoints without an integer port (a port-less
        # host, or ZMQ's ephemeral-port wildcard "tcp://*:*", on which the KV
        # publisher itself starts fine) are underivable too. None of them may
        # take down scheduler startup over a load socket.
        for endpoint in (
            "ipc:///tmp/kv.sock",
            "inproc://kv",
            "tcp://somehost",
            "tcp://*:*",
            "tcp://somehost:-100",
        ):
            with self.subTest(endpoint=endpoint):
                pub, zmq_pub = self._build(
                    config='{"publisher": "zmq", "endpoint": "%s"}' % endpoint
                )
                self.assertFalse(pub.enable)
                zmq_pub.assert_not_called()

    def test_disabled_paths_leave_the_publisher_unbound(self):
        # Every bail-out must leave `publisher` None (surfaced as
        # enable == False) so publish_load_stat returns before computing
        # the (non-trivial) load snapshot.
        for label, config in (
            ("no config", None),
            ("null publisher", '{"publisher": "null"}'),
            ("malformed", "{not json"),
        ):
            with self.subTest(label):
                pub, _ = self._build(config=config)
                self.assertFalse(pub.enable)

    def test_replay_port_collision_skips_past_the_replay_range(self):
        # Conventional config inherited from upstream: KV on 5557, replay on
        # 5558. With dp_size=1 the load socket would land exactly on the
        # replay ROUTER's port; instead of declining (which would silently
        # turn the feature off on exactly this common config) the load range
        # packs after the replay range: 5558 + dp_size = 5559.
        pub, zmq_pub = self._build(
            config='{"publisher": "zmq", "endpoint": "tcp://*:5557", '
            '"replay_endpoint": "tcp://*:5558"}'
        )
        self.assertTrue(pub.enable)
        self.assertEqual(zmq_pub.call_args.kwargs["endpoint"], "tcp://*:5559")

    def test_replay_skip_covers_the_whole_per_rank_range(self):
        # dp_size=4: KV range 5557..5560, replay ROUTER range 5558..5561; the
        # first candidate (5561) still collides with the replay range's tail,
        # so the load range packs after it: 5558 + 4 = 5562.
        pub, zmq_pub = self._build(
            config='{"publisher": "zmq", "endpoint": "tcp://*:5557", '
            '"replay_endpoint": "tcp://*:5558"}',
            dp_size=4,
        )
        self.assertTrue(pub.enable)
        self.assertEqual(zmq_pub.call_args.kwargs["endpoint"], "tcp://*:5562")

    def test_replay_far_away_keeps_the_packed_port(self):
        # No overlap with the replay range => the load range stays right
        # after the KV range (no needless jump past a distant replay port).
        pub, zmq_pub = self._build(
            config='{"publisher": "zmq", "endpoint": "tcp://*:5557", '
            '"replay_endpoint": "tcp://*:6000"}'
        )
        self.assertTrue(pub.enable)
        self.assertEqual(zmq_pub.call_args.kwargs["endpoint"], "tcp://*:5558")

    def test_port_overflow_declines_instead_of_crashing(self):
        # kv base 65535 + dp_size pushes the load range past u16;
        # /server_info omits the key for the same reason.
        pub, zmq_pub = self._build(
            config='{"publisher": "zmq", "endpoint": "tcp://*:65535"}'
        )
        self.assertFalse(pub.enable)
        zmq_pub.assert_not_called()

    def test_publish_skips_snapshot_when_disabled(self):
        pub, _ = self._build(config='{"publisher": "null"}')
        provider = MagicMock()
        pub.publish_load_stat(provider, force=True)
        provider.assert_not_called()

    @staticmethod
    def _provider(running):
        return MagicMock(
            return_value=SimpleNamespace(
                num_running_reqs=running,
                num_waiting_reqs=2,
                num_used_tokens=3,
                max_total_num_tokens=4,
            )
        )

    def test_unchanged_stat_is_deduped_to_the_heartbeat(self):
        # force=True fires per idle-loop iteration (which busy-spins without
        # --sleep-on-idle); an unchanged gauge must go out once per heartbeat,
        # not per iteration. time is patched so the test cannot race the
        # wall clock.
        pub, _ = self._build()
        provider = self._provider(running=1)
        with patch(
            "sglang.srt.managers.scheduler_components.load_publisher.time"
        ) as fake_time:
            fake_time.monotonic.return_value = 100.0
            pub.publish_load_stat(provider, force=True)  # first: publishes
            pub.publish_load_stat(provider, force=True)  # unchanged: deduped
            self.assertEqual(pub.publisher.publish.call_count, 1)
            fake_time.monotonic.return_value = 101.5  # heartbeat elapsed
            pub.publish_load_stat(provider, force=True)
            self.assertEqual(pub.publisher.publish.call_count, 2)

    def test_changed_stat_publishes_immediately(self):
        # The busy->idle (and idle->busy) transition must never be delayed:
        # a changed gauge bypasses the heartbeat dedup even when the last
        # send was a moment ago.
        pub, _ = self._build()
        with patch(
            "sglang.srt.managers.scheduler_components.load_publisher.time"
        ) as fake_time:
            fake_time.monotonic.return_value = 100.0
            pub.publish_load_stat(self._provider(running=7), force=True)
            pub.publish_load_stat(self._provider(running=0), force=True)
            self.assertEqual(pub.publisher.publish.call_count, 2)


if __name__ == "__main__":
    unittest.main()
