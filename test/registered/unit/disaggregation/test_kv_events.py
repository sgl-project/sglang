"""Unit tests for srt/disaggregation/kv_events KV-event publisher rank selection.

Covers the data-parallel rank used to offset each scheduler's KV-event
publisher port, across pure DP, DP-attention, and single-replica modes. The
port offset must make every independent KV cache publish on a distinct port so
the router can subscribe per replica (the `dp_size` it reads from
`/server_info`).
"""

import unittest

import msgspec
import zmq

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    StorageMedium,
    ZmqEventPublisher,
    kv_event_namespace_seed,
    namespaced_block_hash,
    resolve_load_pub_range,
    select_kv_publisher_dp_rank,
)
from sglang.srt.runtime_context import describe_kv_events_publisher
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import get_free_port
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


class TestNamespacedBlockHash(CustomTestCase):
    def test_values_are_pinned(self):
        # Rust has its own copy of this function, and its test uses the same
        # values. Thus the two copies must give the same hashes, and the hashes
        # must not change.
        seed = kv_event_namespace_seed(extra_key="lora-a", cache_salt=None)
        self.assertEqual(
            namespaced_block_hash(123, namespace_seed=seed), 9220659119954863560
        )
        self.assertEqual(
            namespaced_block_hash(-5, namespace_seed=seed), -1810974732006142875
        )


class TestReplay(CustomTestCase):
    def test_descriptor_advertises_replay_port(self):
        def descriptor(**cfg):
            cfg = msgspec.json.encode({"publisher": "zmq", **cfg}).decode()
            args = ServerArgs(model_path="dummy", page_size=16, kv_events_config=cfg)
            return describe_kv_events_publisher(args)

        self.assertNotIn("replay_endpoint_port_base", descriptor())
        self.assertEqual(
            descriptor(replay_endpoint="tcp://*:6000")["replay_endpoint_port_base"],
            6000,
        )

    def test_router_serves_buffered_batches_then_end_seq(self):
        # The router's DEALER client relies on this exact framing.
        replay = f"tcp://127.0.0.1:{get_free_port()}"
        publisher = ZmqEventPublisher(
            attn_dp_rank=0, endpoint="inproc://kv-replay-test", replay_endpoint=replay
        )
        dealer = zmq.Context.instance().socket(zmq.DEALER)
        dealer.setsockopt(zmq.RCVTIMEO, 5000)
        dealer.connect(replay)
        try:
            for _ in range(3):
                publisher.publish(KVEventBatch(ts=1.0, events=[AllBlocksCleared()]))
            publisher._event_queue.join()
            dealer.send_multipart([b"", (1).to_bytes(8, "big")])
            frames = [dealer.recv_multipart() for _ in range(3)]
        finally:
            dealer.close(linger=0)
            publisher.shutdown()
        seqs = [int.from_bytes(f[1], "big", signed=True) for f in frames]
        self.assertEqual(seqs, [1, 2, -1])
        self.assertEqual([f[0] for f in frames], [b""] * 3)
        self.assertEqual(frames[2][2], b"")


if __name__ == "__main__":
    unittest.main()
