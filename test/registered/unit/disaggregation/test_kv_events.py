"""Unit tests for srt/disaggregation/kv_events KV-event publisher rank selection.

Covers the data-parallel rank used to offset each scheduler's KV-event
publisher port, across pure DP, DP-attention, and single-replica modes. The
port offset must make every independent KV cache publish on a distinct port so
the router can subscribe per replica (the `dp_size` it reads from
`/server_info`).
"""

import unittest

import msgspec

from sglang.srt.disaggregation.kv_events import (
    CACHE_SALT_EXTRA_KEY_PREFIX,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithMetadata,
    KVEventBatch,
    StorageMedium,
    ZmqEventPublisher,
    resolve_load_pub_range,
    cache_salt_extra_keys,
    select_kv_publisher_dp_rank,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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
    def _event(self, cache_salt=None):
        # Mirrors how KVCacheEventRecorder.record_store builds an event: a salt
        # goes into both the positional extra_keys slot and the typed struct.
        kwargs = dict(
            block_hashes=[123],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
            extra_keys=cache_salt_extra_keys(cache_salt),
        )
        if cache_salt is None:
            return BlockStored(**kwargs)
        return BlockStoredWithMetadata(
            **kwargs, metadata=BlockStoredMetadata(cache_salt=cache_salt)
        )

    def test_event_matches_the_positional_layout_routers_parse(self):
        # These fields are read by position by KV-aware routers, which parse one
        # layout for every framework. The order below is that layout, not ours to
        # choose: a field landing on the wrong slot is misread as its neighbour
        # rather than rejected. Reordering or inserting anywhere but the end
        # breaks every consumer, so pin the whole tuple.
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(self._event()))
        self.assertEqual(
            decoded,
            [
                "BlockStored",
                [123],  # block_hashes
                None,  # parent_block_hash
                [1, 2],  # token_ids
                2,  # block_size
                None,  # lora_id
                StorageMedium.GPU,  # medium
                None,  # lora_name
                None,  # extra_keys
                None,  # group_idx
                None,  # kv_cache_spec_kind
                None,  # kv_cache_spec_sliding_window
                None,  # locality
                None,  # ownership
            ],
        )

    def test_salt_rides_in_extra_keys_and_past_the_positional_layout(self):
        # Regression: the salt used to be appended straight after `medium`, which
        # is the router's `lora_name: Option<String>` slot -- a map there fails to
        # decode and takes the whole published batch down with it. It now goes in
        # the positional `extra_keys` slot the router actually reads a cache
        # namespace from, while the typed struct stays for name-decoding
        # consumers, parked past every position the router interprets.
        event = self._event("tenant-a")
        encoded = msgspec.msgpack.encode(event)
        decoded = msgspec.msgpack.decode(encoded)
        round_tripped = msgspec.msgpack.decode(encoded, type=BlockStoredWithMetadata)

        self.assertEqual(decoded[7], None, "lora_name slot must stay a string-or-null")
        self.assertEqual(decoded[8], [[f"{CACHE_SALT_EXTRA_KEY_PREFIX}tenant-a"]])
        self.assertEqual(decoded[14], {"cache_salt": "tenant-a"})
        self.assertEqual(round_tripped.metadata.cache_salt, "tenant-a")

    def test_extra_keys_helper_leaves_the_slot_null_when_unsalted(self):
        # The overwhelmingly common event carries no salt; it must not start
        # emitting an empty list, which a consumer would index into.
        self.assertIsNone(cache_salt_extra_keys(None))
        self.assertEqual(
            cache_salt_extra_keys("tenant-a"),
            [[f"{CACHE_SALT_EXTRA_KEY_PREFIX}tenant-a"]],
        )

    def test_cache_salt_prefix_matches_the_router_constant(self):
        # Copied from dynamo's DYNAMO_CACHE_SALT_PREFIX
        # (lib/kv-router/src/zmq_wire/extra_keys.rs). The router strips exactly
        # this prefix to recover the namespace; drift silently yields no match.
        self.assertEqual(CACHE_SALT_EXTRA_KEY_PREFIX, "dynamo-cache-salt:")

    def test_removed_and_cleared_reserve_the_same_trailing_slots(self):
        # BlockRemoved shares BlockStored's trailing tail, and the router only
        # engages that parse when all five slots are present.
        removed = msgspec.msgpack.decode(
            msgspec.msgpack.encode(BlockRemoved(block_hashes=[123]))
        )
        self.assertEqual(
            removed,
            [
                "BlockRemoved",
                [123],  # block_hashes
                None,  # medium
                None,  # group_idx
                None,  # kv_cache_spec_kind
                None,  # kv_cache_spec_sliding_window
                None,  # locality
                None,  # ownership
            ],
        )
        cleared = msgspec.msgpack.decode(msgspec.msgpack.encode(AllBlocksCleared()))
        self.assertEqual(cleared, ["AllBlocksCleared", None])  # ownership

    def test_salted_event_remains_compatible_with_typed_batch_consumers(self):
        batch = KVEventBatch(
            ts=1.0,
            events=[self._event("tenant-a")],
        )
        round_tripped = msgspec.msgpack.decode(
            msgspec.msgpack.encode(batch), type=KVEventBatch
        )
        self.assertEqual(round_tripped.events[0].block_hashes, [123])


if __name__ == "__main__":
    unittest.main()
