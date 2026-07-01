"""Unit tests for srt/disaggregation/kv_events."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import msgspec

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventPublisherFactory,
    KVEventBatch,
    KVEventsConfig,
    NullEventPublisher,
    ZmqEventPublisher,
)
from sglang.test.test_utils import CustomTestCase


class TestOffsetEndpointPort(CustomTestCase):
    """Tests for ZmqEventPublisher.offset_endpoint_port (pure static method)."""

    def test_none_endpoint_returns_none_for_any_rank(self):
        self.assertIsNone(ZmqEventPublisher.offset_endpoint_port(None, 0))
        self.assertIsNone(ZmqEventPublisher.offset_endpoint_port(None, 5))

    def test_rank_zero_returns_endpoint_unchanged(self):
        self.assertEqual(
            ZmqEventPublisher.offset_endpoint_port("tcp://*:5557", 0),
            "tcp://*:5557",
        )
        self.assertEqual(
            ZmqEventPublisher.offset_endpoint_port("inproc://cache", 0),
            "inproc://cache",
        )

    def test_inproc_endpoint_gets_dp_suffix(self):
        self.assertEqual(
            ZmqEventPublisher.offset_endpoint_port("inproc://cache", 2),
            "inproc://cache_dp2",
        )

    def test_tcp_wildcard_port_offset_by_rank(self):
        self.assertEqual(
            ZmqEventPublisher.offset_endpoint_port("tcp://*:5557", 1),
            "tcp://*:5558",
        )

    def test_tcp_host_port_offset_by_rank(self):
        self.assertEqual(
            ZmqEventPublisher.offset_endpoint_port("tcp://host:5557", 3),
            "tcp://host:5560",
        )

    def test_tcp_without_colon_returns_unchanged(self):
        # Contains "tcp" but has no ":" to locate a port, so it is returned as-is.
        self.assertEqual(
            ZmqEventPublisher.offset_endpoint_port("tcpsocket", 1),
            "tcpsocket",
        )

    def test_unknown_scheme_raises_value_error(self):
        with self.assertRaises(ValueError):
            ZmqEventPublisher.offset_endpoint_port("udp://host:5557", 1)


class TestKVEventsConfig(CustomTestCase):
    """Tests for KVEventsConfig.from_cli."""

    def test_from_cli_defaults(self):
        config = KVEventsConfig.from_cli("{}")
        self.assertEqual(config.publisher, "null")
        self.assertEqual(config.endpoint, "tcp://*:5557")
        self.assertIsNone(config.replay_endpoint)
        self.assertEqual(config.buffer_steps, 10_000)
        self.assertEqual(config.hwm, 100_000)
        self.assertEqual(config.max_queue_size, 100_000)
        self.assertEqual(config.topic, "")

    def test_from_cli_overrides(self):
        config = KVEventsConfig.from_cli(
            '{"publisher": "zmq", "endpoint": "tcp://localhost:6000"}'
        )
        self.assertEqual(config.publisher, "zmq")
        self.assertEqual(config.endpoint, "tcp://localhost:6000")
        # Untouched fields keep their defaults.
        self.assertIsNone(config.replay_endpoint)

    def test_from_cli_invalid_json_raises(self):
        with self.assertRaises(Exception):
            KVEventsConfig.from_cli("{not valid json")


class TestMsgspecRoundTrips(CustomTestCase):
    """msgspec.msgpack encode/decode round-trips for KV cache event structs."""

    def setUp(self):
        self.encoder = msgspec.msgpack.Encoder()

    def _round_trip(self, value, typ):
        decoder = msgspec.msgpack.Decoder(typ)
        return decoder.decode(self.encoder.encode(value))

    def test_block_stored_round_trip(self):
        event = BlockStored(
            block_hashes=[1, 2, 3],
            parent_block_hash=10,
            token_ids=[4, 5, 6],
            block_size=16,
            lora_id=7,
            medium="GPU",
        )
        self.assertEqual(self._round_trip(event, BlockStored), event)

    def test_block_removed_round_trip(self):
        event = BlockRemoved(block_hashes=[9, 8], medium="CPU_PINNED")
        self.assertEqual(self._round_trip(event, BlockRemoved), event)

    def test_all_blocks_cleared_round_trip(self):
        event = AllBlocksCleared()
        self.assertEqual(self._round_trip(event, AllBlocksCleared), event)

    def test_kv_event_batch_mixed_events_round_trip(self):
        batch = KVEventBatch(
            ts=123.5,
            events=[
                BlockStored(
                    block_hashes=[1, 2],
                    parent_block_hash=None,
                    token_ids=[3, 4],
                    block_size=8,
                    lora_id=None,
                    medium="GPU",
                ),
                BlockRemoved(block_hashes=[5], medium="DISK"),
                AllBlocksCleared(),
            ],
        )
        decoded = self._round_trip(batch, KVEventBatch)
        self.assertEqual(decoded, batch)
        self.assertEqual(
            [type(e).__name__ for e in decoded.events],
            ["BlockStored", "BlockRemoved", "AllBlocksCleared"],
        )


class TestNullEventPublisher(CustomTestCase):
    """Tests for the no-op NullEventPublisher."""

    def test_publish_accepts_event_batch(self):
        publisher = NullEventPublisher()
        batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
        self.assertIsNone(publisher.publish(batch))

    def test_shutdown_is_noop(self):
        publisher = NullEventPublisher()
        self.assertIsNone(publisher.shutdown())


class TestEventPublisherFactory(CustomTestCase):
    """Tests for EventPublisherFactory create/register behavior."""

    def setUp(self):
        # Snapshot the registry so register_publisher tests cannot leak state.
        self._saved_registry = dict(EventPublisherFactory._registry)

    def tearDown(self):
        EventPublisherFactory._registry = dict(self._saved_registry)

    def test_create_with_none_config_returns_null_publisher(self):
        self.assertIsInstance(EventPublisherFactory.create(None), NullEventPublisher)
        # Any falsy config short-circuits to the null publisher.
        self.assertIsInstance(EventPublisherFactory.create(""), NullEventPublisher)

    def test_null_kind_resolves_to_null_publisher(self):
        # The "null" kind maps to NullEventPublisher in the registry.
        self.assertIs(EventPublisherFactory._registry["null"], NullEventPublisher)

    def test_create_unknown_publisher_raises_value_error(self):
        with self.assertRaises(ValueError):
            EventPublisherFactory.create('{"publisher": "does-not-exist"}')

    def test_register_publisher_adds_to_registry(self):
        EventPublisherFactory.register_publisher("custom-null", NullEventPublisher)
        self.assertIn("custom-null", EventPublisherFactory._registry)
        self.assertIs(
            EventPublisherFactory._registry["custom-null"], NullEventPublisher
        )

    def test_register_duplicate_publisher_raises_key_error(self):
        with self.assertRaises(KeyError):
            EventPublisherFactory.register_publisher("null", NullEventPublisher)


if __name__ == "__main__":
    unittest.main()
