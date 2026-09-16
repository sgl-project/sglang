"""Unit tests for srt/disaggregation/kv_events KV-event publisher rank selection.

Covers the data-parallel rank used to offset each scheduler's KV-event
publisher port, across pure DP, DP-attention, and single-replica modes. The
port offset must make every independent KV cache publish on a distinct port so
the router can subscribe per replica (the `dp_size` it reads from
`/server_info`).
"""

import atexit
import tempfile
import time
import unittest
import uuid

import msgspec
import zmq

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    NullEventPublisher,
    StorageMedium,
    ZmqEventPublisher,
    resolve_load_pub_range,
    select_kv_publisher_dp_rank,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLocalKvEventSource(CustomTestCase):
    def _publisher(self, **kwargs):
        publisher = ZmqEventPublisher(**kwargs)
        atexit.unregister(publisher.shutdown)
        self.addCleanup(publisher.shutdown)
        return publisher

    def test_bound_source_uses_actual_port_and_global_rank(self):
        # The source describes a real rank-4 publisher, not local rank zero.
        with zmq.Context.instance().socket(zmq.PUB) as probe:
            port = probe.bind_to_random_port("tcp://127.0.0.1")
        publisher = self._publisher(
            attn_dp_rank=4, endpoint=f"tcp://*:{port - 4}", topic="kv"
        )
        source = publisher.describe_local_source(64)
        self.assertEqual(
            source,
            {
                "dp_rank": 4,
                "endpoint": f"tcp://127.0.0.1:{port}",
                "topic": "kv",
                "block_size": 64,
            },
        )
        with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
            subscriber.setsockopt_string(zmq.SUBSCRIBE, source["topic"])
            subscriber.connect(source["endpoint"])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                publisher.publish(KVEventBatch(ts=1.0, events=[AllBlocksCleared()]))
                if subscriber.poll(100):
                    topic, _, payload = subscriber.recv_multipart()
                    self.assertEqual(topic, b"kv")
                    batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
                    self.assertEqual(batch.attn_dp_rank, 4)
                    self.assertEqual(batch.events, [AllBlocksCleared()])
                    break
            else:
                self.fail("No event received from the advertised local source")

    def test_ephemeral_bind_and_replay_report_resolved_ports(self):
        publisher = self._publisher(
            attn_dp_rank=0,
            endpoint="tcp://0.0.0.0:0",
            replay_endpoint="tcp://*:0",
        )
        source = publisher.describe_local_source(64)
        self.assertEqual(
            source["endpoint"],
            publisher._pub.getsockopt_string(zmq.LAST_ENDPOINT).replace(
                "0.0.0.0", "127.0.0.1"
            ),
        )
        self.assertEqual(
            source["replay_endpoint"],
            publisher._replay.getsockopt_string(zmq.LAST_ENDPOINT).replace(
                "0.0.0.0", "127.0.0.1"
            ),
        )
        self.assertNotEqual(source["endpoint"], source["replay_endpoint"])
        self.assertFalse(source["endpoint"].endswith(":0"))

    def test_ipc_source_preserves_bound_path(self):
        directory = tempfile.TemporaryDirectory(prefix="kv-", dir="/tmp")
        self.addCleanup(directory.cleanup)
        publisher = self._publisher(
            attn_dp_rank=0, endpoint=f"ipc://{directory.name}/events"
        )
        self.assertEqual(
            publisher.describe_local_source(64)["endpoint"],
            f"ipc://{directory.name}/events",
        )

    def test_non_subscribable_publishers_are_not_advertised(self):
        for endpoint in (
            "tcp://127.0.0.1:5557",  # Connect-style PUB, not a listening source.
            f"inproc://kv-source-{uuid.uuid4().hex}",  # Same-process only.
        ):
            with self.subTest(endpoint=endpoint):
                publisher = self._publisher(attn_dp_rank=0, endpoint=endpoint)
                self.assertIsNone(publisher.describe_local_source(64))
        self.assertIsNone(NullEventPublisher().describe_local_source(64))


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
    def _event(self, **extra):
        return BlockStored(
            block_hashes=[123],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
            **extra,
        )

    def test_event_is_a_tagged_map(self):
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(self._event()))
        self.assertIsInstance(decoded, dict)
        self.assertEqual(decoded["type"], "BlockStored")
        self.assertEqual(
            set(decoded),
            {
                "type",
                "block_hashes",
                "parent_block_hash",
                "token_ids",
                "block_size",
                "lora_id",
                "medium",
            },
        )

    def test_salt_and_session_are_named_fields(self):
        event = self._event(cache_salt="tenant-a", session_id="session-a")
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(event))
        self.assertEqual(decoded["cache_salt"], "tenant-a")
        self.assertEqual(decoded["session_id"], "session-a")

    def test_one_decoder_reads_a_mixed_batch(self):
        batch = KVEventBatch(
            ts=1.0,
            events=[
                self._event(),
                self._event(cache_salt="tenant-a"),
                self._event(session_id="session-a"),
                BlockRemoved(block_hashes=[123], medium=StorageMedium.GPU),
                AllBlocksCleared(),
            ],
        )
        round_tripped = msgspec.msgpack.decode(
            msgspec.msgpack.encode(batch), type=KVEventBatch
        )
        stored = round_tripped.events[:3]
        self.assertEqual([e.cache_salt for e in stored], [None, "tenant-a", None])
        self.assertEqual([e.session_id for e in stored], [None, None, "session-a"])
        self.assertIsInstance(round_tripped.events[3], BlockRemoved)
        self.assertIsInstance(round_tripped.events[4], AllBlocksCleared)

    def test_batch_stays_a_positional_array_of_maps(self):
        batch = KVEventBatch(ts=1.0, events=[self._event()], attn_dp_rank=0)
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(batch))
        self.assertEqual(decoded[0], 1.0)
        self.assertEqual(decoded[2], 0)
        self.assertIsInstance(decoded[1][0], dict)
        self.assertEqual(len(decoded), 3)


if __name__ == "__main__":
    unittest.main()
