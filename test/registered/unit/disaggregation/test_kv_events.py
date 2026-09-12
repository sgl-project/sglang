"""Unit tests for srt/disaggregation/kv_events KV-event publisher rank selection.

Covers the data-parallel rank used to offset each scheduler's KV-event
publisher port, across pure DP, DP-attention, and single-replica modes. The
port offset must make every independent KV cache publish on a distinct port so
the router can subscribe per replica (the `dp_size` it reads from
`/server_info`).
"""

import time
import unittest
from collections import OrderedDict, deque
from queue import Queue
from unittest.mock import patch

import msgspec
import zmq

from sglang.srt.disaggregation.kv_events import (
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithMetadata,
    KVEventBatch,
    StorageMedium,
    ZmqEventPublisher,
    resolve_load_pub_range,
    select_kv_publisher_dp_rank,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestResolveLoadPubRange(CustomTestCase):
    """The single source of truth both the bind and /server_info route through."""

    @staticmethod
    def _base(kv, replay=None, dp_size=1, explicit="auto"):
        resolved, _ = resolve_load_pub_range(
            kv_endpoint=kv,
            replay_endpoint=replay,
            dp_size=dp_size,
            load_publish_endpoint=explicit,
        )
        return None if resolved is None else resolved[1]

    def test_off_by_default(self):
        # Opt-in: unset or "off" reserves nothing, even with a valid config.
        self.assertIsNone(self._base("tcp://*:5557", explicit=None))
        self.assertIsNone(self._base("tcp://*:5557", explicit="off"))

    def test_auto_packs_after_kv_range(self):
        self.assertEqual(self._base("tcp://*:5557"), 5558)
        self.assertEqual(self._base("tcp://*:5557", dp_size=2), 5559)

    def test_auto_skips_an_overlapping_replay_range(self):
        # Conventional replay = kv + 1 always overlaps the packed candidate.
        self.assertEqual(self._base("tcp://*:5557", "tcp://*:5558"), 5559)
        self.assertEqual(self._base("tcp://*:5557", "tcp://*:5558", dp_size=4), 5562)

    def test_non_adjacent_replay_leaves_packing_unchanged(self):
        self.assertEqual(self._base("tcp://*:5557", "tcp://*:6000"), 5558)

    def test_auto_declines_connect_style_and_underivable_endpoints(self):
        for kv in (
            "tcp://10.0.0.5:5557",  # concrete host: connect-style
            "tcp://[2001:db8::5]:5557",  # concrete IPv6 ("::" is not a wildcard)
            "tcp://::1:5557",  # bare IPv6: ambiguous
            "tcp://host",  # no port
            "ipc:///tmp/kv",
            None,
        ):
            with self.subTest(kv=kv):
                self.assertIsNone(self._base(kv))

    def test_auto_declines_on_u16_overflow(self):
        self.assertIsNone(self._base("tcp://*:65535"))

    def test_explicit_endpoint_moves_and_validates_the_range(self):
        self.assertEqual(self._base("tcp://*:5557", explicit="tcp://*:7000"), 7000)
        # A concrete explicit host, or one overlapping the kv range, declines.
        self.assertIsNone(self._base("tcp://*:5557", explicit="tcp://10.0.0.5:7000"))
        self.assertIsNone(
            self._base("tcp://*:5557", dp_size=4, explicit="tcp://*:5558")
        )

    def test_reason_is_set_only_for_actionable_declines(self):
        # Off by default is unremarkable (no reason); an opt-in the operator
        # asked for that can't resolve is worth surfacing.
        _, quiet = resolve_load_pub_range(
            kv_endpoint="tcp://10.0.0.5:5557", replay_endpoint=None, dp_size=1
        )
        self.assertIsNone(quiet)
        _, auto_loud = resolve_load_pub_range(
            kv_endpoint="tcp://10.0.0.5:5557",  # connect-style: can't derive
            replay_endpoint=None,
            dp_size=1,
            load_publish_endpoint="auto",
        )
        self.assertIsNotNone(auto_loud)
        # A missing config surfaces at startup — no message may render a bare
        # "None". Both the likely mistakes (auto and an explicit address
        # without --kv-events-config) go through this.
        for endpoint in ("auto", "tcp://*:7000"):
            with self.subTest(endpoint=endpoint):
                _, no_cfg = resolve_load_pub_range(
                    kv_endpoint=None,
                    replay_endpoint=None,
                    dp_size=1,
                    load_publish_endpoint=endpoint,
                )
                self.assertIsNotNone(no_cfg)
                self.assertNotIn("None", no_cfg)
                self.assertIn("--kv-events-config", no_cfg)


class TestSelectKvPublisherDpRank(CustomTestCase):
    def test_select_rank_across_modes(self):
        # (label, attn_dp_size, attn_dp_rank, dp_rank, expected)
        cases = [
            # Pure DP (no dp-attention): attn_dp_rank is 0 for every worker,
            # so the replica is distinguished by dp_rank.
            ("pure_dp_worker0", 1, 0, 0, 0),
            ("pure_dp_worker1", 1, 0, 1, 1),
            ("pure_dp_worker3", 1, 0, 3, 3),
            # DP-attention: each attn-dp rank owns a KV shard; distinguish by
            # attn_dp_rank. dp_rank is ignored entirely in this mode.
            ("dp_attention_rank0", 2, 0, None, 0),
            ("dp_attention_rank1", 2, 1, None, 1),
            ("dp_attention_ignores_dp_rank", 2, 1, 99, 1),
            # Single replica / no DP.
            ("single_dp_rank_none", 1, 0, None, 0),
            ("single_dp_rank_zero", 1, 0, 0, 0),
        ]
        for label, attn_dp_size, attn_dp_rank, dp_rank, expected in cases:
            with self.subTest(label):
                self.assertEqual(
                    select_kv_publisher_dp_rank(attn_dp_size, attn_dp_rank, dp_rank),
                    expected,
                )

    def test_workers_bind_sequential_ports_per_replica(self):
        # Each replica r must publish on port_base + r, since the router opens
        # one SUB socket per rank at port_base + r. Regression: pre-fix every
        # pure-DP worker offset by attn_dp_rank == 0, so all collapsed onto the
        # single port tcp://*:5557 -> the 2nd worker crashed binding an
        # already-bound port.
        endpoint = "tcp://*:5557"
        expected = [f"tcp://*:{5557 + r}" for r in range(4)]

        # Pure DP: replica index is dp_rank (attn_dp_rank is 0 for all).
        pure_dp = [
            ZmqEventPublisher.offset_endpoint_port(
                endpoint, select_kv_publisher_dp_rank(1, 0, r)
            )
            for r in range(4)
        ]
        self.assertEqual(pure_dp, expected)

        # DP-attention: replica index is attn_dp_rank.
        dp_attention = [
            ZmqEventPublisher.offset_endpoint_port(
                endpoint, select_kv_publisher_dp_rank(4, a, None)
            )
            for a in range(4)
        ]
        self.assertEqual(dp_attention, expected)

    def test_publisher_rank_count_matches_advertised_dp_size(self):
        # The router subscribes to `dp_size` per-rank ports (from /server_info).
        # The engine must produce exactly `dp_size` distinct publisher ranks in
        # both modes, otherwise some subscribed ports get no data.
        for dp_size in (1, 2, 4):
            with self.subTest(f"pure_dp_{dp_size}"):
                ranks = {
                    select_kv_publisher_dp_rank(
                        attn_dp_size=1, attn_dp_rank=0, dp_rank=r
                    )
                    for r in range(dp_size)
                }
                self.assertEqual(len(ranks), dp_size)
            with self.subTest(f"dp_attention_{dp_size}"):
                ranks = {
                    select_kv_publisher_dp_rank(
                        attn_dp_size=dp_size, attn_dp_rank=a, dp_rank=None
                    )
                    for a in range(dp_size)
                }
                self.assertEqual(len(ranks), dp_size)


class TestBlockStoredWireFormat(CustomTestCase):
    def _event(self, metadata=None):
        event_type = BlockStored if metadata is None else BlockStoredWithMetadata
        kwargs = dict(
            block_hashes=[123],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
        )
        if metadata is not None:
            kwargs["metadata"] = metadata
        return event_type(**kwargs)

    def test_unsalted_event_keeps_legacy_array_shape(self):
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(self._event()))
        self.assertEqual(len(decoded), 7)

    def test_salted_event_appends_typed_metadata(self):
        event = self._event(BlockStoredMetadata(cache_salt="tenant-a"))
        encoded = msgspec.msgpack.encode(event)
        decoded = msgspec.msgpack.decode(encoded)
        round_tripped = msgspec.msgpack.decode(encoded, type=BlockStoredWithMetadata)
        self.assertEqual(len(decoded), 8)
        self.assertEqual(decoded[7], {"cache_salt": "tenant-a"})
        self.assertEqual(round_tripped.metadata.cache_salt, "tenant-a")

    def test_salted_event_remains_compatible_with_typed_batch_consumers(self):
        batch = KVEventBatch(
            ts=1.0,
            events=[self._event(BlockStoredMetadata(cache_salt="tenant-a"))],
        )
        round_tripped = msgspec.msgpack.decode(
            msgspec.msgpack.encode(batch), type=KVEventBatch
        )
        self.assertEqual(round_tripped.events[0].block_hashes, [123])


class TestZmqReplayBackpressure(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.ctx = zmq.Context()
        self.addCleanup(self.ctx.destroy, linger=0)
        # Exercise the real socket setup and replay loop on the test thread.
        self.publisher = ZmqEventPublisher.__new__(ZmqEventPublisher)
        self.publisher._ctx = self.ctx
        self.publisher._pub = None
        self.publisher._replay = None
        self.publisher._endpoint = "inproc://live"
        self.publisher._replay_endpoint = "inproc://replay"
        self.publisher._hwm = 100
        self.publisher._buffer = deque(maxlen=1024)
        self.publisher._pending_replays = OrderedDict()
        socket_factory = zmq.Context.socket

        def low_hwm_socket(context, socket_type):
            socket = socket_factory(context, socket_type)
            if socket_type == zmq.ROUTER:
                socket.setsockopt(zmq.SNDHWM, 1)
            return socket

        with patch.object(zmq.Context, "socket", low_hwm_socket):
            self.publisher._socket_setup()

    def _client(self, identity=b"reader"):
        client = self.ctx.socket(zmq.DEALER)
        client.setsockopt(zmq.IDENTITY, identity)
        client.setsockopt(zmq.RCVHWM, 1)
        client.connect("inproc://replay")
        return client

    def _request(self, client, start=0):
        client.send_multipart((b"", start.to_bytes(8, "big")))
        self.assertTrue(self.publisher._replay.poll(1000, zmq.POLLIN))
        self.publisher._service_replay()

    def _collect(self, client):
        frames = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            while client.poll(0, zmq.POLLIN):
                message = client.recv_multipart()
                frames.append(message)
                if message == [b"", self.publisher.END_SEQ, b""]:
                    return frames
            self.publisher._service_replay()
            client.poll(1, zmq.POLLIN)
        self.fail("Replay did not deliver its terminal marker")

    def test_full_queue_retries_tail_and_pins_requested_history(self):
        batches = [(seq, f"batch-{seq}".encode()) for seq in range(1024)]
        self.publisher._buffer.extend(batches)
        client = self._client()
        self._request(client)
        self.assertIn(b"reader", self.publisher._pending_replays)

        # New live events can evict every original batch during a slow replay.
        self.publisher._buffer.extend((seq, b"new") for seq in range(1024, 2048))
        frames = self._collect(client)
        self.assertEqual(
            frames,
            [[b"", seq.to_bytes(8, "big"), payload] for seq, payload in batches]
            + [[b"", self.publisher.END_SEQ, b""]],
        )
        self.assertFalse(self.publisher._pending_replays)

    def test_terminal_marker_is_retried_when_queue_is_full(self):
        # Inproc capacity is the sender HWM + receiver HWM, both set to one.
        self.publisher._buffer.extend([(0, b"first"), (1, b"second")])
        client = self._client()
        self._request(client)
        self.assertIn(b"reader", self.publisher._pending_replays)
        self.assertFalse(self.publisher._pending_replays[b"reader"].batches)
        self.assertEqual(
            self._collect(client),
            [
                [b"", (0).to_bytes(8, "big"), b"first"],
                [b"", (1).to_bytes(8, "big"), b"second"],
                [b"", self.publisher.END_SEQ, b""],
            ],
        )

    def test_stalled_reader_does_not_block_another_reader(self):
        self.publisher._buffer.extend((seq, b"data") for seq in range(16))
        stalled = self._client(b"stalled")
        self._request(stalled)
        healthy = self._client(b"healthy")
        self._request(healthy)
        frames = self._collect(healthy)
        self.assertEqual(len(frames), 17)
        self.assertEqual(
            [int.from_bytes(frame[1], "big") for frame in frames[:-1]],
            list(range(16)),
        )
        self.assertIn(b"stalled", self.publisher._pending_replays)

    def test_stalled_replay_does_not_block_live_publication(self):
        self.publisher._buffer.extend((seq, b"data") for seq in range(16))
        client = self._client()
        self._request(client)
        subscriber = self.ctx.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.connect("inproc://live")
        publisher = self.publisher
        publisher._running = True
        publisher._seq_gen = iter([16])
        publisher._topic_bytes = b""
        publisher._event_queue = Queue()
        publisher._event_queue.put(KVEventBatch(ts=1.0, events=[]))
        publisher._event_queue.put(None)
        publisher._publisher_thread()
        self.assertTrue(subscriber.poll(1000, zmq.POLLIN))
        frames = subscriber.recv_multipart()
        self.assertEqual(frames[:2], [b"", (16).to_bytes(8, "big")])
        self.assertEqual(msgspec.msgpack.decode(frames[2], type=KVEventBatch).ts, 1.0)
        self.assertIn(b"reader", publisher._pending_replays)

    def test_send_budget_yields_before_finishing_replay(self):
        self.publisher.REPLAY_SEND_BUDGET = 1
        self.publisher._buffer.extend([(0, b"old"), (1, b"first"), (2, b"second")])
        client = self._client()
        self._request(client, start=1)
        self.assertEqual(
            list(self.publisher._pending_replays[b"reader"].batches), [(2, b"second")]
        )
        self.assertEqual(
            self._collect(client),
            [
                [b"", (1).to_bytes(8, "big"), b"first"],
                [b"", (2).to_bytes(8, "big"), b"second"],
                [b"", self.publisher.END_SEQ, b""],
            ],
        )

    def test_idle_replay_expires_and_releases_admission_slot(self):
        self.publisher.MAX_PENDING_REPLAYS = 1
        self.publisher._buffer.extend((seq, b"data") for seq in range(16))
        stalled = self._client(b"stalled")
        self._request(stalled)
        healthy = self._client(b"healthy")
        self._request(healthy)
        self.assertEqual(list(self.publisher._pending_replays), [b"stalled"])
        state = self.publisher._pending_replays[b"stalled"]
        with patch(
            "sglang.srt.disaggregation.kv_events.time.monotonic",
            return_value=state.last_progress + self.publisher.REPLAY_IDLE_TIMEOUT,
        ):
            self.publisher._service_replay()
        self.assertNotIn(b"stalled", self.publisher._pending_replays)
        self.assertEqual(len(self._collect(healthy)), 17)

    def test_disconnected_reader_does_not_retain_replay(self):
        self.publisher._buffer.extend((seq, b"data") for seq in range(16))
        client = self._client()
        self._request(client)
        client.close(linger=0)
        deadline = time.monotonic() + 1
        while self.publisher._pending_replays and time.monotonic() < deadline:
            self.publisher._service_replay()
            time.sleep(0.001)
        self.assertFalse(self.publisher._pending_replays)


if __name__ == "__main__":
    unittest.main()
