"""Wire contract and port/rank gating for the LoadStat load snapshot.

Locks the msgpack array shape the sgl-router `cache_aware_zmq` policy will
decode positionally (that consumer lands with the router PR; it is not yet
in this tree, so this pins only the Python side):

    ["LoadStat", num_running_reqs, num_waiting_reqs, num_tokens,
     max_total_num_tokens, attn_dp_rank]

carried as the payload of a three-frame message ``[b"load", BE-i64 seq,
payload]``. A field reorder or rename is a silent cross-language break, so
`test_loadstat_golden_bytes` pins the exact encoding — assert the same hex
on the Rust side when that PR lands to actually close the loop.
TestLoadPublisherGating pins which schedulers publish and on which port.
CPU-only: the socket bind is stubbed at the `_open_pub_socket` seam.
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
    def test_loadstat_golden_bytes(self):
        # Exact on-the-wire encoding. Assert the identical hex from the Rust
        # decoder's test when the router PR lands — that is what actually pins
        # a cross-language format; the decode-round-trip below only pins Python.
        raw = msgspec.msgpack.Encoder().encode(
            LoadStat(
                num_running_reqs=7,
                num_waiting_reqs=3,
                num_tokens=1024,
                max_total_num_tokens=8192,
                attn_dp_rank=2,
            )
        )
        self.assertEqual(raw.hex(), "96a84c6f6164537461740703cd0400cd200002")

    def test_loadstat_msgpack_array_shape(self):
        raw = msgspec.msgpack.Encoder().encode(
            LoadStat(
                num_running_reqs=7,
                num_waiting_reqs=3,
                num_tokens=1024,
                max_total_num_tokens=8192,
                attn_dp_rank=2,
            )
        )
        # tag=True + array_like → [tag, *fields] in declaration order; the
        # router reads the tag + four counts and ignores the trailing field.
        self.assertEqual(
            msgspec.msgpack.Decoder().decode(raw),
            ["LoadStat", 7, 3, 1024, 8192, 2],
        )

    def test_loadstat_tag_is_class_name(self):
        # The tag is the literal "LoadStat"; guard against an accidental
        # msgspec `tag=` override or a class rename.
        raw = msgspec.msgpack.Encoder().encode(
            LoadStat(
                num_running_reqs=0,
                num_waiting_reqs=0,
                num_tokens=0,
                max_total_num_tokens=0,
            )
        )
        decoded = msgspec.msgpack.Decoder().decode(raw)
        # LoadStat sets no omit_defaults, so the trailing field is always
        # emitted (null when unset); a decoder must tolerate it.
        self.assertEqual(decoded, ["LoadStat", 0, 0, 0, 0, None])


ZMQ_ENDPOINT = '{"publisher": "zmq", "endpoint": "tcp://*:5557"}'


class TestLoadPublisherGating(CustomTestCase):
    """One load publisher per independent KV cache, on a resolvable port.

    Getting either half wrong makes several schedulers bind the same port,
    which is an uncaught ZMQError at startup for a bind-style endpoint and —
    worse — silently merges every worker's load onto one rank for a
    connect-style one.
    """

    def _build(
        self, *, config=ZMQ_ENDPOINT, dp_size=1, explicit="auto", **ps_overrides
    ):
        """Construct a publisher with the socket bind stubbed out, returning
        (publisher, captured _open_pub_socket mock). Opts in via explicit="auto"
        by default (the feature is off without it). dp_size lives on the ps,
        which the publisher reads (no separate param to disagree with it)."""
        with patch(
            "sglang.srt.managers.scheduler_components.load_publisher."
            "_open_pub_socket"
        ) as open_sock:
            pub = SchedulerLoadPublisher(
                kv_events_config=config,
                ps=ParallelState.trivial(dp_size=dp_size, **ps_overrides),
                load_publish_endpoint=explicit,
            )
        return pub, open_sock

    def test_disabled_by_default(self):
        # Off unless opted in: a bare --kv-events-config user (no
        # --load-publish-endpoint) reserves no load port, so an upgrade can't
        # collide with a co-hosted neighbor's KV bind.
        pub, open_sock = self._build(explicit=None)
        self.assertFalse(pub.enable)
        open_sock.assert_not_called()

    def test_enabled_on_rank_zero(self):
        pub, open_sock = self._build()  # explicit="auto"
        self.assertTrue(pub.enable)
        open_sock.assert_called_once_with("tcp://*:5558")

    def test_disabled_off_pp_rank_zero(self):
        # Every PP stage shares attn_tp_rank/attn_cp_rank 0, so without the
        # pp_rank gate they all bind the same load port.
        pub, open_sock = self._build(pp_rank=1, pp_size=2)
        self.assertFalse(pub.enable)
        open_sock.assert_not_called()

    def test_disabled_off_attn_tp_and_cp_rank_zero(self):
        for override in ({"attn_tp_rank": 1}, {"attn_cp_rank": 1}):
            with self.subTest(**override):
                pub, open_sock = self._build(**override)
                self.assertFalse(pub.enable)
                open_sock.assert_not_called()

    def test_pure_dp_keys_the_load_port_by_dp_rank(self):
        # Pure DP: attn_dp_size == 1 and every worker has attn_dp_rank == 0, so
        # the publisher must key off dp_rank or all replicas collide on one
        # port. kv 5557 + dp_size 4 => base 5561; rank 2 binds 5563.
        _, open_sock = self._build(attn_dp_size=1, attn_dp_rank=0, dp_rank=2, dp_size=4)
        open_sock.assert_called_once_with("tcp://*:5563")

    def test_dp_attention_keys_the_load_port_by_attn_dp_rank(self):
        _, open_sock = self._build(attn_dp_size=4, attn_dp_rank=3, dp_rank=0, dp_size=4)
        open_sock.assert_called_once_with("tcp://*:5564")

    def test_load_port_is_packed_after_the_kv_range(self):
        _, open_sock = self._build(dp_size=2)
        open_sock.assert_called_once_with("tcp://*:5559")

    def test_accepts_every_bind_style_host(self):
        for host, expected in (
            ("*", "tcp://*:5558"),
            ("0.0.0.0", "tcp://0.0.0.0:5558"),
            ("[::]", "tcp://[::]:5558"),
        ):
            with self.subTest(host=host):
                pub, open_sock = self._build(
                    config='{"publisher": "zmq", "endpoint": "tcp://%s:5557"}' % host
                )
                self.assertTrue(pub.enable)
                open_sock.assert_called_once_with(expected)

    def test_unresolvable_endpoint_declines_instead_of_raising(self):
        # ipc:// and inproc:// are valid KV-event endpoints but carry no port
        # to pack after; port-less/malformed tcp shapes are underivable; and a
        # concrete host (IPv4 or IPv6 — "::" appears inside every IPv6
        # address, so this must not be a substring test) would be *connected
        # to* rather than bound, publishing into a void. None of them may
        # take down scheduler startup over a load socket.
        for endpoint in (
            "ipc:///tmp/kv.sock",
            "inproc://kv",
            "tcp://somehost",
            "tcp://*:*",
            "tcp://somehost:-100",
            "tcp://10.0.0.5:5557",
            "tcp://[2001:db8::5]:5557",
            "tcp://::1:5557",
        ):
            with self.subTest(endpoint=endpoint):
                pub, open_sock = self._build(
                    config='{"publisher": "zmq", "endpoint": "%s"}' % endpoint
                )
                self.assertFalse(pub.enable)
                open_sock.assert_not_called()

    def test_disabled_paths_leave_the_publisher_unbound(self):
        # Every bail-out must leave the socket unbound (surfaced as
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
        pub, open_sock = self._build(
            config='{"publisher": "zmq", "endpoint": "tcp://*:5557", '
            '"replay_endpoint": "tcp://*:5558"}'
        )
        self.assertTrue(pub.enable)
        open_sock.assert_called_once_with("tcp://*:5559")

    def test_replay_skip_covers_the_whole_per_rank_range(self):
        # dp_size=4: KV range 5557..5560, replay ROUTER range 5558..5561; the
        # first candidate (5561) still collides with the replay range's tail,
        # so the load range packs after it: 5558 + 4 = 5562.
        pub, open_sock = self._build(
            config='{"publisher": "zmq", "endpoint": "tcp://*:5557", '
            '"replay_endpoint": "tcp://*:5558"}',
            dp_size=4,
        )
        self.assertTrue(pub.enable)
        open_sock.assert_called_once_with("tcp://*:5562")

    def test_replay_far_away_keeps_the_packed_port(self):
        # No overlap with the replay range => the load range stays right
        # after the KV range (no needless jump past a distant replay port).
        pub, open_sock = self._build(
            config='{"publisher": "zmq", "endpoint": "tcp://*:5557", '
            '"replay_endpoint": "tcp://*:6000"}'
        )
        self.assertTrue(pub.enable)
        open_sock.assert_called_once_with("tcp://*:5558")

    def test_port_overflow_declines_instead_of_crashing(self):
        # kv base 65535 + dp_size pushes the load range past u16;
        # /server_info omits the key for the same reason.
        pub, open_sock = self._build(
            config='{"publisher": "zmq", "endpoint": "tcp://*:65535"}'
        )
        self.assertFalse(pub.enable)
        open_sock.assert_not_called()

    def test_explicit_endpoint_moves_the_range(self):
        # --load-publish-endpoint sets the range outright; rank r still binds
        # base + r (pure DP keys by dp_rank).
        pub, open_sock = self._build(explicit="tcp://*:7000")
        self.assertTrue(pub.enable)
        open_sock.assert_called_once_with("tcp://*:7000")

        _, open_sock = self._build(
            explicit="tcp://*:7000",
            attn_dp_size=1,
            attn_dp_rank=0,
            dp_rank=2,
            dp_size=4,
        )
        open_sock.assert_called_once_with("tcp://*:7002")

    def test_explicit_endpoint_must_be_bindable(self):
        # A concrete host would be connected to rather than bound.
        pub, open_sock = self._build(explicit="tcp://10.0.0.5:7000")
        self.assertFalse(pub.enable)
        open_sock.assert_not_called()

    def test_explicit_off_disables_load_publishing(self):
        # The operator's off switch: KV events without the extra port range.
        # /server_info omits the load keys through the same resolver.
        pub, open_sock = self._build(explicit="off")
        self.assertFalse(pub.enable)
        open_sock.assert_not_called()

    def test_bind_failure_disables_without_raising(self):
        # An occupied port must not take down scheduler startup over a routing
        # hint; the publisher logs and stays a no-op. Opted in (auto) so the
        # bind is actually reached — otherwise the feature is just off.
        import zmq

        with patch(
            "sglang.srt.managers.scheduler_components.load_publisher."
            "_open_pub_socket",
            side_effect=zmq.ZMQError,
        ) as open_sock:
            pub = SchedulerLoadPublisher(
                kv_events_config=ZMQ_ENDPOINT,
                ps=ParallelState.trivial(),
                load_publish_endpoint="auto",
            )
        open_sock.assert_called_once()  # the bind was attempted and failed
        self.assertFalse(pub.enable)
        pub.publish_load_stat(MagicMock(), force=True)  # still a no-op

    def test_close_is_idempotent_and_disables(self):
        pub, _ = self._build()
        socket = pub._socket
        pub.close()
        pub.close()
        socket.close.assert_called_once()
        self.assertFalse(pub.enable)
        provider = MagicMock()
        pub.publish_load_stat(provider, force=True)
        provider.assert_not_called()

    def test_explicit_endpoint_needs_an_advertisable_kv_endpoint(self):
        # Discovery rides on /server_info's kv_events block, which is absent
        # for non-tcp (or port-less) KV endpoints — binding the explicit
        # range anyway would claim a port no router can ever find.
        for kv_endpoint in ("ipc:///tmp/kv.sock", "inproc://kv", "tcp://0.0.0.0"):
            with self.subTest(kv_endpoint=kv_endpoint):
                pub, open_sock = self._build(
                    config='{"publisher": "zmq", "endpoint": "%s"}' % kv_endpoint,
                    explicit="tcp://*:7000",
                )
                self.assertFalse(pub.enable)
                open_sock.assert_not_called()

    def test_explicit_endpoint_inside_the_kv_range_declines(self):
        # The KV publisher binds its own range later and unguarded, so taking
        # one of its ports would kill startup blaming the KV publisher.
        pub, open_sock = self._build(dp_size=4, explicit="tcp://*:5558")
        self.assertFalse(pub.enable)
        open_sock.assert_not_called()

    # ----- publish path -------------------------------------------------

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

    def test_publish_skips_snapshot_when_disabled(self):
        pub, _ = self._build(config='{"publisher": "null"}')
        provider = MagicMock()
        pub.publish_load_stat(provider, force=True)
        provider.assert_not_called()

    def test_caller_supplied_snapshot_bypasses_the_provider(self):
        # The scheduler hands in the snapshot it already computed for the
        # DP-balancing sink; the provider is the fallback for cycles where
        # that sink was throttled — it must not run when a snapshot is given.
        pub, _ = self._build()
        provider = MagicMock()
        snap = SimpleNamespace(
            num_running_reqs=1,
            num_waiting_reqs=2,
            num_used_tokens=3,
            max_total_num_tokens=4,
        )
        pub.publish_load_stat(provider, force=True, snapshot=snap)
        provider.assert_not_called()
        self.assertEqual(pub._socket.send_multipart.call_count, 1)

    def test_publish_frames_are_topic_seq_payload(self):
        # Three frames, matching the KV-event socket's layout so one
        # subscriber loop handles both.
        pub, _ = self._build()
        pub.publish_load_stat(self._provider(running=1), force=True)
        (frames,), _ = pub._socket.send_multipart.call_args
        topic, seq, payload = frames
        self.assertEqual(topic, b"load")
        self.assertEqual(seq, (0).to_bytes(8, "big"))
        self.assertEqual(
            msgspec.msgpack.Decoder().decode(payload),
            ["LoadStat", 1, 2, 3, 4, 0],
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
            self.assertEqual(pub._socket.send_multipart.call_count, 1)
            fake_time.monotonic.return_value = 101.5  # heartbeat elapsed
            pub.publish_load_stat(provider, force=True)
            self.assertEqual(pub._socket.send_multipart.call_count, 2)

    def test_call_throttle_stays_engaged_across_dedup_hits(self):
        # Regression: the counter must reset when the throttle PASSES, not
        # when a send happens. Resetting only on the send path let one dedup
        # hit saturate the counter, running the O(queue) provider every step.
        # A working counter fires the provider at counts 5 and 10.
        pub, _ = self._build()
        provider = self._provider(running=1)
        with patch(
            "sglang.srt.managers.scheduler_components.load_publisher.time"
        ) as fake_time:
            fake_time.monotonic.return_value = 100.0
            for _ in range(10):
                pub.publish_load_stat(provider)
            self.assertEqual(provider.call_count, 2)

    def test_provider_failure_never_raises(self):
        # get_loads raising must not crash the scheduler loop; it warns and
        # leaves the counter reset (not saturated).
        def boom():
            raise RuntimeError("get_loads exploded")

        pub, _ = self._build()
        with self.assertLogs(
            "sglang.srt.managers.scheduler_components.load_publisher",
            level="WARNING",
        ):
            pub.publish_load_stat(boom, force=True)
        self.assertEqual(pub._publish_counter, 0)

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
            self.assertEqual(pub._socket.send_multipart.call_count, 2)


class TestLoadStatIntegration(CustomTestCase):
    """The one path every gating test stubs: a real socket bind + SUB
    round-trip. Covers _open_pub_socket (bind, HWM/LINGER/IPV6 order) and the
    three-frame wire end to end."""

    def test_binds_and_delivers_three_decodable_frames(self):
        import socket as _socket
        import time as _time

        import zmq

        # Probe on "" (all interfaces) to match ZMQ's wildcard bind, and retry:
        # probe-then-bind is a TOCTOU race and the publisher swallows bind
        # errors, so a lost race shows up only as a disabled publisher.
        pub = None
        for _ in range(3):
            with _socket.socket() as probe:
                probe.bind(("", 0))
                port = probe.getsockname()[1]
            pub = SchedulerLoadPublisher(
                kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
                ps=ParallelState.trivial(),
                load_publish_endpoint=f"tcp://*:{port}",
            )
            if pub.enable:
                break
        self.assertTrue(pub.enable, "load socket never bound a free port")
        self.addCleanup(pub.close)

        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.connect(f"tcp://127.0.0.1:{port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "load")  # exact advertised topic
        self.addCleanup(sub.close)

        snap = SimpleNamespace(
            num_running_reqs=7,
            num_waiting_reqs=3,
            num_used_tokens=1024,
            max_total_num_tokens=8192,
        )
        # PUB/SUB drops messages sent before the subscription propagates, so
        # re-publish until one lands (heartbeat reset each pass).
        frames = None
        deadline = _time.time() + 5
        while frames is None and _time.time() < deadline:
            pub._last_publish_ts = 0.0
            pub.publish_load_stat(lambda: snap, force=True)
            if sub.poll(100):
                frames = sub.recv_multipart()
        self.assertIsNotNone(frames, "no load frame received within 5s")

        topic, seq, payload = frames
        self.assertEqual(topic, b"load")
        self.assertEqual(len(seq), 8)
        self.assertEqual(
            msgspec.msgpack.Decoder().decode(payload),
            ["LoadStat", 7, 3, 1024, 8192, 0],
        )


if __name__ == "__main__":
    unittest.main()
