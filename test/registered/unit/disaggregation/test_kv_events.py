"""Unit tests for srt/disaggregation/kv_events publisher rank selection and
the KV event wire contract.

Covers the data-parallel rank used to offset each scheduler's KV-event
publisher port, across pure DP, DP-attention, and single-replica modes. The
port offset must make every independent KV cache publish on a distinct port so
the router can subscribe per replica (the `dp_size` it reads from
`/server_info`).

Also covers the two array shapes a `BlockStored` event takes on the wire:
salted and unsalted. The consumer side type that decodes both is included also.
"""

import hashlib
import time
import unittest
import uuid
from array import array
from typing import Union
from unittest.mock import Mock, patch

import msgspec
import torch
import zmq

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredView,
    BlockStoredWithMetadata,
    KVEventBatch,
    KVEventBatchView,
    StorageMedium,
    ZmqEventPublisher,
    resolve_load_pub_range,
    select_kv_publisher_dp_rank,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _block_stored(cache_salt=None, *, block_hashes=(123,), token_ids=(1, 2)):
    """A BlockStored in the shape its producer would emit it."""
    kwargs = dict(
        block_hashes=list(block_hashes),
        parent_block_hash=None,
        token_ids=list(token_ids),
        block_size=len(token_ids),
        lora_id=None,
        medium=StorageMedium.GPU,
    )
    if cache_salt is None:
        return BlockStored(**kwargs)
    return BlockStoredWithMetadata(
        **kwargs, metadata=BlockStoredMetadata(cache_salt=cache_salt)
    )


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
        return _block_stored(None if metadata is None else metadata.cache_salt)

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


class _WithMetadataBatch(KVEventBatch):
    """The batch a consumer reaches for first. The producer's metadata struct
    which rejects every unsalted event on the same stream."""

    events: list[BlockStoredWithMetadata]


class _AmbiguousBatch(KVEventBatch):
    """Both producer structs in one union. Rejected for duplicate tags."""

    events: list[Union[BlockStored, BlockStoredWithMetadata]]


class TestBlockStoredViewDecoding(CustomTestCase):
    """The consumer side shape for a stream that mixes salted and unsalted stores.

    A salted request appends a metadata element under the same "BlockStored"
    tag, so one stream carries two array lengths. Each producer struct reads
    exactly one of them and the two cannot share a union which left the salt
    undecodable by any exported type. `BlockStoredView` is what a consumer
    decodes instead.
    """

    @staticmethod
    def _payload(*events):
        return msgspec.msgpack.encode(KVEventBatch(ts=1.0, events=list(events)))

    def test_producer_structs_cannot_decode_a_mixed_stream(self):
        # The metadata struct rejects the unsalted events
        # sharing its stream and msgspec rejects the union that would have
        # covered both shapes together.
        with self.assertRaises(msgspec.ValidationError):
            msgspec.msgpack.decode(
                self._payload(_block_stored()), type=_WithMetadataBatch
            )
        with self.assertRaises(TypeError):
            msgspec.msgpack.Decoder(_AmbiguousBatch)

    def test_view_decodes_both_shapes_from_one_stream(self):
        payload = self._payload(
            _block_stored(block_hashes=(11,)),
            _block_stored("tenant-a", block_hashes=(22,)),
            BlockRemoved(block_hashes=[11], medium=StorageMedium.GPU),
            AllBlocksCleared(),
        )
        batch = msgspec.msgpack.Decoder(KVEventBatchView).decode(payload)

        self.assertEqual(
            [type(event) for event in batch.events],
            [BlockStoredView, BlockStoredView, BlockRemoved, AllBlocksCleared],
        )
        self.assertEqual(
            [event.cache_salt for event in batch.events[:2]], [None, "tenant-a"]
        )
        self.assertEqual(
            [event.block_hashes[0] for event in batch.events[:2]], [11, 22]
        )

    def test_view_keeps_every_legacy_field_positioned(self):
        # The optional element is appended, so both array lengths must still
        # land each legacy field on the same attribute.
        for salt in (None, "tenant-a"):
            with self.subTest(cache_salt=salt):
                payload = self._payload(
                    _block_stored(salt, block_hashes=(11,), token_ids=(7, 8, 9))
                )
                event = (
                    msgspec.msgpack.Decoder(KVEventBatchView).decode(payload).events[0]
                )

                self.assertEqual(event.block_hashes, [11])
                self.assertIsNone(event.parent_block_hash)
                self.assertEqual(event.token_ids, [7, 8, 9])
                self.assertEqual(event.block_size, 3)
                self.assertIsNone(event.lora_id)
                self.assertEqual(event.medium, StorageMedium.GPU)
                self.assertEqual(event.cache_salt, salt)

    def test_view_stays_a_block_stored_subclass(self):
        # Consumers branch on isinstance(event, BlockStored). Redeclaring the
        # view as a standalone struct would silently drop them into the
        # unhandled event branch.
        self.assertTrue(issubclass(BlockStoredView, BlockStored))

    def test_view_is_not_wire_safe_to_publish(self):
        # Pins why the producer keeps two structs. omit_defaults does not trim
        # a trailing default in an array_like struct, so publishing the view
        # would append a null that legacy 7 element consumers do not expect.
        view = BlockStoredView(
            block_hashes=[1],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
        )
        self.assertEqual(len(msgspec.msgpack.decode(msgspec.msgpack.encode(view))), 8)


class TestKVEventStreamRoundTrip(CustomTestCase):
    """Real cache -> publisher -> ZMQ -> consumer, over a mixed salt stream.

    Guards the path the issue reports as broken end to end. A subscriber to a
    live stream could read the block hashes but never the salt that namespaces
    them because no exported type decoded both event shapes. One capture
    serves every case below and the frame is the fixture.
    """

    TOPIC = "kv-events"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.events = cls._record_real_events()
        cls.payload = cls._round_trip(cls.events)

    @staticmethod
    def _page_hashes(token_ids, prior_hash=None, page_size=None):
        """Pure python stand-in for ``get_hash_str``.

        The native page hash is a JIT compiled, Linux only C++ extension. The
        wire contract under test does not depend on which digest it produces,
        only on the recorder chaining and namespacing them.
        """

        def digest(chunk, prior):
            hasher = hashlib.sha256()
            if prior:
                hasher.update(bytes.fromhex(prior))
            for token in chunk:
                for element in token if isinstance(token, tuple) else (token,):
                    hasher.update(int(element).to_bytes(4, "little", signed=False))
            return hasher.hexdigest()

        if page_size is None:
            return digest(token_ids, prior_hash)
        hashes = []
        running = prior_hash
        for start in range(0, len(token_ids), page_size):
            running = digest(token_ids[start : start + page_size], running)
            hashes.append(running)
        return hashes

    @classmethod
    def _record_real_events(cls):
        """Drive a real RadixCache: one unsalted request, two tenants, evict."""
        allocator = Mock()
        allocator.device = torch.device("cpu")
        with patch("sglang.srt.mem_cache.utils.get_hash_str", cls._page_hashes):
            cache = RadixCache.create_simulated(
                mock_allocator=allocator, page_size=4, enable_kv_cache_events=True
            )
            cache.take_events()  # drop the AllBlocksCleared from construction

            for tokens, salt in (
                ([1, 2, 3, 4, 5, 6, 7, 8], None),
                ([1, 2, 3, 4, 9, 10, 11, 12], "tenant-a"),
                ([21, 22, 23, 24], "tenant-b"),
            ):
                cache.insert(
                    InsertParams(
                        key=RadixKey(array("q", tokens), cache_salt=salt),
                        value=torch.tensor(tokens, dtype=torch.int64),
                    )
                )
            cache.evict(EvictParams(num_tokens=4))
            return cache.take_events()

    @classmethod
    def _round_trip(cls, events):
        """Publish through a real ZmqEventPublisher and return the raw frame."""
        endpoint = f"inproc://kv-events-{uuid.uuid4().hex}"
        publisher = ZmqEventPublisher(
            attn_dp_rank=0, endpoint=endpoint, topic=cls.TOPIC
        )
        cls.addClassCleanup(publisher.shutdown)
        subscriber = zmq.Context.instance().socket(zmq.SUB)
        cls.addClassCleanup(subscriber.close, 0)
        subscriber.connect(endpoint)
        subscriber.setsockopt_string(zmq.SUBSCRIBE, cls.TOPIC)

        # PUB drops whatever is sent before the subscription propagates, so
        # republish until a frame arrives.
        payload = None
        deadline = time.monotonic() + 30
        while payload is None and time.monotonic() < deadline:
            publisher.publish(KVEventBatch(ts=time.time(), events=events))
            if subscriber.poll(200):
                _topic, _seq, payload = subscriber.recv_multipart()
        if payload is None:
            raise AssertionError("publisher delivered no frame")
        while subscriber.poll(100):  # drain the republished duplicates
            subscriber.recv_multipart()
        return payload

    def _decode(self, batch_type):
        return msgspec.msgpack.Decoder(batch_type).decode(self.payload)

    def test_salted_stream_round_trips_to_a_typed_consumer(self):
        produced = [event for event in self.events if isinstance(event, BlockStored)]
        self.assertEqual(
            [type(event) for event in produced],
            [BlockStored, BlockStoredWithMetadata, BlockStoredWithMetadata],
        )

        batch = self._decode(KVEventBatchView)
        stored = [event for event in batch.events if isinstance(event, BlockStoredView)]
        self.assertEqual(
            [event.cache_salt for event in stored], [None, "tenant-a", "tenant-b"]
        )
        self.assertEqual(
            [event.block_hashes for event in stored],
            [event.block_hashes for event in produced],
        )
        self.assertTrue(
            any(isinstance(event, BlockRemoved) for event in batch.events),
            "eviction should reach the consumer as BlockRemoved",
        )

    def test_salt_namespaces_the_hashes_a_consumer_indexes(self):
        # The salted request shares a 4 token prefix with the unsalted one yet
        # emits its own chain. Consumers must index emitted hashes per salt,
        # not treat an equal prefix as an equal block.
        stored = [
            event
            for event in self._decode(KVEventBatchView).events
            if isinstance(event, BlockStoredView)
        ]
        unsalted, salted = stored[0], stored[1]

        self.assertEqual(unsalted.token_ids[:4], salted.token_ids[:4])
        self.assertNotEqual(unsalted.block_hashes[0], salted.block_hashes[0])

    def test_legacy_consumers_still_decode_the_same_frame(self):
        # The fix is decoder side only. The bytes a legacy KVEventBatch
        # consumer sees must be unchanged, salt or no salt.
        legacy = self._decode(KVEventBatch)
        view = self._decode(KVEventBatchView)

        self.assertEqual(
            [event.block_hashes for event in legacy.events],
            [event.block_hashes for event in view.events],
        )
        self.assertFalse(
            any(isinstance(event, BlockStoredView) for event in legacy.events),
            "moving the view into KVEventBatch would change what legacy "
            "consumers decode",
        )


if __name__ == "__main__":
    unittest.main()
