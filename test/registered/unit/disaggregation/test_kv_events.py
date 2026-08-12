"""Unit tests for srt/disaggregation/kv_events KV-event publisher rank selection.

Covers the data-parallel rank used to offset each scheduler's KV-event
publisher port, across pure DP, DP-attention, and single-replica modes. The
port offset must make every independent KV cache publish on a distinct port so
the router can subscribe per replica (the `dp_size` it reads from
`/server_info`).
"""

import json
import time
import unittest
import uuid
from unittest.mock import MagicMock, patch

import msgspec
import zmq
from pydantic import BaseModel, ConfigDict

from sglang.srt.disaggregation.kv_events import (
    SNAPSHOT_CHUNK,
    SNAPSHOT_CHUNK_RECORDS,
    SNAPSHOT_END,
    SNAPSHOT_HEADER,
    SNAPSHOT_REQUEST,
    SNAPSHOT_SEND_TIMEOUT_MS,
    BlockRemoved,
    BlockStored,
    BlockStoredMetadata,
    BlockStoredWithMetadata,
    EventPublisher,
    EventPublisherFactory,
    KVEventBatch,
    KVSnapshotBlock,
    KVSnapshotHeader,
    NullEventPublisher,
    StorageMedium,
    ZmqEventPublisher,
    select_kv_publisher_dp_rank,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


class TestEventPublisherFactory(CustomTestCase):
    def test_null_publisher_has_its_own_empty_config_schema(self):
        publisher = EventPublisherFactory.create('{"publisher": "null"}')
        self.assertIsInstance(publisher, NullEventPublisher)

        with self.assertRaisesRegex(ValueError, "endpoint"):
            EventPublisherFactory.create(
                '{"publisher": "null", "endpoint": "tcp://*:5557"}'
            )

    def test_legacy_custom_publisher_keeps_stable_constructor_contract(self):
        class LegacyPublisher(EventPublisher):
            def __init__(
                self,
                attn_dp_rank,
                endpoint,
                replay_endpoint,
                buffer_steps,
                hwm,
                max_queue_size,
                topic,
            ):
                self.attn_dp_rank = attn_dp_rank
                self.endpoint = endpoint
                self.replay_endpoint = replay_endpoint
                self.buffer_steps = buffer_steps
                self.hwm = hwm
                self.max_queue_size = max_queue_size
                self.topic = topic

            def publish(self, events):
                return

            def shutdown(self):
                return

        with patch.dict(EventPublisherFactory._registry):
            EventPublisherFactory.register_publisher("legacy-test", LegacyPublisher)
            publisher = EventPublisherFactory.create(
                '{"publisher": "legacy-test"}', attn_dp_rank=3
            )

            self.assertEqual(publisher.attn_dp_rank, 3)
            self.assertEqual(publisher.endpoint, "tcp://*:5557")
            self.assertIsNone(publisher.replay_endpoint)
            self.assertEqual(publisher.buffer_steps, 10_000)
            self.assertEqual(publisher.topic, "")

            # New ZMQ-only fields are rejected explicitly instead of being
            # forwarded into an unrelated legacy constructor.
            with self.assertRaisesRegex(ValueError, "snapshot_endpoint"):
                EventPublisherFactory.create(
                    json.dumps(
                        {
                            "publisher": "legacy-test",
                            "snapshot_endpoint": "tcp://*:5757",
                        }
                    )
                )

    def test_custom_publisher_owns_its_config_model(self):
        class CustomConfig(BaseModel):
            model_config = ConfigDict(extra="forbid")

            channel: str
            batch_size: int = 8

        class CustomPublisher(EventPublisher):
            def __init__(self, attn_dp_rank, config):
                self.attn_dp_rank = attn_dp_rank
                self.config = config

            def publish(self, events):
                return

            def shutdown(self):
                return

        def build_custom_publisher(attn_dp_rank, config):
            return CustomPublisher(attn_dp_rank, config)

        with patch.dict(EventPublisherFactory._registry):
            EventPublisherFactory.register_publisher_spec(
                "custom-test", CustomConfig, build_custom_publisher
            )
            publisher = EventPublisherFactory.create(
                json.dumps({"publisher": "custom-test", "channel": "cache"}),
                attn_dp_rank=2,
            )

            self.assertEqual(publisher.attn_dp_rank, 2)
            self.assertIsInstance(publisher.config, CustomConfig)
            self.assertEqual(publisher.config.channel, "cache")
            self.assertEqual(publisher.config.batch_size, 8)

            with self.assertRaisesRegex(ValueError, "snapshot_endpoint"):
                EventPublisherFactory.create(
                    json.dumps(
                        {
                            "publisher": "custom-test",
                            "channel": "cache",
                            "snapshot_endpoint": "tcp://*:5757",
                        }
                    )
                )


class TestZmqEventPublisherReplay(CustomTestCase):
    def test_pub_send_failure_keeps_event_in_replay_buffer(self):
        class FailingSendPublisher(ZmqEventPublisher):
            def _socket_setup(self):
                self._pub = MagicMock()
                self._pub.send_multipart.side_effect = RuntimeError(
                    "injected PUB failure"
                )

        publisher = FailingSendPublisher(
            attn_dp_rank=0,
            endpoint="inproc://unused-failing-publisher",
        )
        event = KVEventBatch(
            ts=1.0,
            events=[BlockRemoved(block_hashes=[101])],
        )
        try:
            with patch(
                "sglang.srt.disaggregation.kv_events.logger.exception"
            ) as log_exception:
                publisher.publish(event)
                publisher._event_queue.join()

            self.assertEqual(publisher._next_seq, 1)
            self.assertEqual(len(publisher._buffer), 1)
            seq, payload = publisher._buffer[0]
            self.assertEqual(seq, 0)
            self.assertEqual(msgspec.msgpack.decode(payload, type=KVEventBatch), event)
            log_exception.assert_called_once()
        finally:
            publisher.shutdown()


class TestKvPlacementSnapshotProtocol(CustomTestCase):
    def _fetch_header(self, endpoint: str) -> KVSnapshotHeader:
        dealer = zmq.Context.instance().socket(zmq.DEALER)
        try:
            dealer.connect(endpoint)
            dealer.send_multipart([b"", SNAPSHOT_REQUEST])
            frames = dealer.recv_multipart()
            self.assertEqual(frames[1], SNAPSHOT_HEADER)
            header = msgspec.msgpack.decode(frames[2], type=KVSnapshotHeader)
            while dealer.recv_multipart()[1] != SNAPSHOT_END:
                pass
            return header
        finally:
            dealer.close(linger=0)

    def test_epoch_is_scoped_to_replica_publisher_lifecycle(self):
        suffix = uuid.uuid4().hex
        event_endpoint = f"inproc://kv-events-restart-{suffix}"
        snapshot_endpoint = f"inproc://kv-snapshot-restart-{suffix}"
        config = json.dumps(
            {
                "publisher": "zmq",
                "endpoint": event_endpoint,
                "snapshot_endpoint": snapshot_endpoint,
            }
        )

        rank0 = EventPublisherFactory.create(config, attn_dp_rank=0)
        rank1 = EventPublisherFactory.create(config, attn_dp_rank=1)
        rank1_snapshot_endpoint = ZmqEventPublisher.offset_endpoint_port(
            snapshot_endpoint, 1
        )
        try:
            rank0_header = self._fetch_header(snapshot_endpoint)
            rank1_header = self._fetch_header(rank1_snapshot_endpoint)
            rank0.shutdown()

            restarted_rank0 = EventPublisherFactory.create(config, attn_dp_rank=0)
            try:
                restarted_rank0_header = self._fetch_header(snapshot_endpoint)
                unchanged_rank1_header = self._fetch_header(rank1_snapshot_endpoint)
            finally:
                restarted_rank0.shutdown()
        finally:
            rank0.shutdown()
            rank1.shutdown()

        self.assertNotEqual(rank0_header.epoch, rank1_header.epoch)
        self.assertNotEqual(rank0_header.epoch, restarted_rank0_header.epoch)
        self.assertEqual(rank1_header.epoch, unchanged_rank1_header.epoch)
        self.assertEqual(rank0_header.replica_rank, 0)
        self.assertEqual(rank1_header.replica_rank, 1)
        self.assertEqual(restarted_rank0_header.replica_rank, 0)

    def test_snapshot_is_consistent_with_live_barrier(self):
        suffix = uuid.uuid4().hex
        event_endpoint = f"inproc://kv-events-{suffix}"
        snapshot_endpoint = f"inproc://kv-snapshot-{suffix}"
        publisher = ZmqEventPublisher(
            attn_dp_rank=0,
            endpoint=event_endpoint,
            snapshot_endpoint=snapshot_endpoint,
            epoch="replica-epoch-1",
        )
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        dealer = ctx.socket(zmq.DEALER)
        try:
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.connect(event_endpoint)
            dealer.connect(snapshot_endpoint)
            # Give the inproc SUB subscription time to propagate before the
            # test's barrier is emitted. Production does not rely on this
            # sleep: the Router retries snapshots until it observes the
            # barrier carried in the header.
            time.sleep(0.05)

            publisher.publish(
                KVEventBatch(
                    ts=1.0,
                    events=[
                        BlockStored(
                            block_hashes=[11, 12],
                            parent_block_hash=None,
                            token_ids=[1, 2],
                            block_size=2,
                            lora_id=None,
                        ),
                        BlockStored(
                            block_hashes=[21],
                            parent_block_hash=12,
                            token_ids=[3],
                            block_size=1,
                            lora_id=None,
                        ),
                    ],
                )
            )
            publisher._event_queue.join()
            snapshot_blocks = publisher._snapshot_blocks
            self.assertEqual(set(snapshot_blocks), {11, 12, 21})

            publisher.publish(
                KVEventBatch(
                    ts=2.0,
                    events=[BlockRemoved(block_hashes=[21])],
                )
            )
            publisher._event_queue.join()
            # Block hashes are the event protocol's removal identity. Removal
            # must update the mirror in place instead of rebuilding and
            # scanning the complete snapshot dictionary.
            self.assertIs(publisher._snapshot_blocks, snapshot_blocks)
            self.assertEqual(set(snapshot_blocks), {11, 12})

            dealer.send_multipart([b"", SNAPSHOT_REQUEST])
            header_frames = dealer.recv_multipart()
            self.assertEqual(header_frames[0], b"")
            self.assertEqual(header_frames[1], SNAPSHOT_HEADER)
            header = msgspec.msgpack.decode(header_frames[2], type=KVSnapshotHeader)

            blocks = []
            while True:
                frames = dealer.recv_multipart()
                self.assertEqual(frames[0], b"")
                if frames[1] == SNAPSHOT_END:
                    break
                self.assertEqual(frames[1], SNAPSHOT_CHUNK)
                blocks.extend(
                    msgspec.msgpack.decode(frames[2], type=list[KVSnapshotBlock])
                )

            self.assertEqual(header.epoch, "replica-epoch-1")
            self.assertEqual(header.replica_rank, 0)
            self.assertEqual(header.barrier_seq, 2)
            self.assertEqual(header.resume_seq, 3)
            self.assertEqual(header.record_count, 2)
            self.assertEqual(
                [(b.parent_block_hash, b.block_hashes) for b in blocks],
                [(None, [11]), (11, [12])],
            )

            # Two data events precede the barrier. Its topic carries both the
            # lifecycle epoch and the unique id returned in the snapshot.
            for expected_seq in (0, 1):
                _topic, seq, _payload = sub.recv_multipart()
                self.assertEqual(int.from_bytes(seq, "big"), expected_seq)
            barrier_topic, barrier_seq, barrier_payload = sub.recv_multipart()
            self.assertIn(b"\x00sgl-kv-epoch=replica-epoch-1", barrier_topic)
            self.assertIn(
                f"\x00sgl-kv-snapshot={header.barrier_id}".encode(),
                barrier_topic,
            )
            self.assertEqual(int.from_bytes(barrier_seq, "big"), 2)
            barrier = msgspec.msgpack.decode(barrier_payload, type=KVEventBatch)
            self.assertEqual(barrier.events, [])
        finally:
            sub.close(linger=0)
            dealer.close(linger=0)
            publisher.shutdown()

    def test_snapshot_streams_more_than_one_chunk_without_losing_blocks(self):
        suffix = uuid.uuid4().hex
        event_endpoint = f"inproc://kv-events-multichunk-{suffix}"
        snapshot_endpoint = f"inproc://kv-snapshot-multichunk-{suffix}"
        publisher = ZmqEventPublisher(
            attn_dp_rank=0,
            endpoint=event_endpoint,
            snapshot_endpoint=snapshot_endpoint,
            epoch="replica-epoch-multichunk",
        )
        dealer = zmq.Context.instance().socket(zmq.DEALER)
        block_hashes = list(range(SNAPSHOT_CHUNK_RECORDS * 2 + 1))
        try:
            publisher.publish(
                KVEventBatch(
                    ts=1.0,
                    events=[
                        BlockStored(
                            block_hashes=block_hashes,
                            parent_block_hash=None,
                            token_ids=[],
                            block_size=1,
                            lora_id=None,
                        )
                    ],
                )
            )
            publisher._event_queue.join()

            dealer.connect(snapshot_endpoint)
            dealer.send_multipart([b"", SNAPSHOT_REQUEST])
            header_frames = dealer.recv_multipart()
            self.assertEqual(header_frames[1], SNAPSHOT_HEADER)
            header = msgspec.msgpack.decode(header_frames[2], type=KVSnapshotHeader)

            chunk_sizes = []
            received = []
            while True:
                frames = dealer.recv_multipart()
                if frames[1] == SNAPSHOT_END:
                    break
                self.assertEqual(frames[1], SNAPSHOT_CHUNK)
                chunk = msgspec.msgpack.decode(frames[2], type=list[KVSnapshotBlock])
                chunk_sizes.append(len(chunk))
                received.extend(chunk)

            self.assertEqual(
                chunk_sizes,
                [SNAPSHOT_CHUNK_RECORDS, SNAPSHOT_CHUNK_RECORDS, 1],
            )
            self.assertEqual(header.record_count, len(block_hashes))
            self.assertEqual(len(received), len(block_hashes))
            self.assertEqual(received[0].parent_block_hash, None)
            self.assertEqual(received[0].block_hashes, [0])
            self.assertEqual(received[-1].parent_block_hash, block_hashes[-2])
            self.assertEqual(received[-1].block_hashes, [block_hashes[-1]])
        finally:
            dealer.close(linger=0)
            publisher.shutdown()

    def test_snapshot_send_timeout_does_not_kill_service_thread(self):
        suffix = uuid.uuid4().hex
        event_endpoint = f"inproc://kv-events-send-timeout-{suffix}"
        snapshot_endpoint = f"inproc://kv-snapshot-send-timeout-{suffix}"
        publisher = ZmqEventPublisher(
            attn_dp_rank=0,
            endpoint=event_endpoint,
            snapshot_endpoint=snapshot_endpoint,
            epoch="replica-epoch-send-timeout",
        )
        first = zmq.Context.instance().socket(zmq.DEALER)
        second = zmq.Context.instance().socket(zmq.DEALER)
        original_send = publisher._send_snapshot_response
        attempts = 0

        def timeout_once(sock, client_id, encoder, snapshot):
            nonlocal attempts
            self.assertEqual(sock.getsockopt(zmq.SNDTIMEO), SNAPSHOT_SEND_TIMEOUT_MS)
            attempts += 1
            if attempts == 1:
                raise zmq.Again()
            return original_send(sock, client_id, encoder, snapshot)

        try:
            first.connect(snapshot_endpoint)
            second.connect(snapshot_endpoint)
            with patch.object(
                publisher,
                "_send_snapshot_response",
                side_effect=timeout_once,
            ):
                first.send_multipart([b"", SNAPSHOT_REQUEST])
                deadline = time.monotonic() + 2.0
                while attempts == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(attempts, 1)

                # The first response timed out, but the same ROUTER thread must
                # remain available to serve a later client.
                second.send_multipart([b"", SNAPSHOT_REQUEST])
                frames = second.recv_multipart()
                self.assertEqual(frames[1], SNAPSHOT_HEADER)
                while second.recv_multipart()[1] != SNAPSHOT_END:
                    pass
                self.assertEqual(attempts, 2)
                self.assertTrue(publisher._snapshot_thread.is_alive())
        finally:
            first.close(linger=0)
            second.close(linger=0)
            publisher.shutdown()


if __name__ == "__main__":
    unittest.main()
