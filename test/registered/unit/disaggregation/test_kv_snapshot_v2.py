"""Snapshot v2 and replay tests using the production publisher and ZMQ sockets."""

import time
import unittest
import uuid

import msgspec
import zmq

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStoredWithComponents,
    KVEventBatch,
    ZmqEventPublisher,
)


class TestSnapshotV2(unittest.TestCase):
    def setUp(self):
        prefix = f"inproc://v2-{uuid.uuid4().hex}"
        self.snapshot = prefix + "-snapshot"
        self.replay = prefix + "-replay"
        self.publisher = ZmqEventPublisher(
            0,
            endpoint=prefix,
            snapshot_endpoint=self.snapshot,
            replay_endpoint=self.replay,
            namespace="model-a",
            model="test-model",
            worker_id="worker-a",
            page_size=4,
            cache_spec={"version": 1, "components": 3, "swa_window_tokens": 8},
            buffer_steps=3,
        )

    def tearDown(self):
        self.publisher.shutdown()

    def send(self, *events):
        self.publisher.publish(KVEventBatch(ts=time.time(), events=list(events)))
        self.publisher._event_queue.join()

    def request(self, endpoint, *frames):
        socket = zmq.Context.instance().socket(zmq.DEALER)
        socket.setsockopt(zmq.RCVTIMEO, 3000)
        socket.connect(endpoint)
        try:
            socket.send_multipart([b"", *frames])
            result = []
            while True:
                frame = socket.recv_multipart()
                result.append(frame)
                if frame[1] in (b"end", b"error"):
                    return result
        finally:
            socket.close(linger=0)

    def store(self, tier, components, hashes=None):
        return BlockStoredWithComponents(
            block_hashes=hashes or [11],
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4],
            block_size=4,
            lora_id=None,
            medium=tier,
            component_types=components,
        )

    def cut(self):
        frames = self.request(self.snapshot, b"snapshot-v2")
        header = msgspec.msgpack.decode(frames[0][2])
        blocks = [
            block
            for frame in frames[1:-1]
            for block in msgspec.msgpack.decode(frame[2])
        ]
        self.assertEqual(header["record_count"], len(blocks))
        return header, blocks

    def test_tiers_components_and_current_state(self):
        self.send(self.store("GPU", ["full", "swa"]), self.store("CPU_PINNED", ["full"]))
        header, blocks = self.cut()
        self.assertEqual({(b["tier"], b["component_mask"]) for b in blocks}, {(1, 3), (2, 1)})
        self.assertEqual(header["worker_id"], "worker-a")
        self.assertEqual(header["namespace"], "model-a")
        self.assertEqual(header["model"], "test-model")
        self.assertEqual(header["page_size"], 4)
        self.assertEqual(header["resume_seq"], header["barrier_seq"] + 1)
        self.send(BlockRemoved([11], "GPU"))
        _, blocks = self.cut()
        self.assertEqual([(b["tier"], b["block_size"]) for b in blocks], [(2, 4)])
        self.send(self.store("CPU_PINNED", ["full", "mamba"]))
        _, blocks = self.cut()
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["component_mask"], 5)
        self.send(AllBlocksCleared())
        self.assertEqual(self.cut()[1], [])

    def test_replay_epoch_fencing_and_expired_window(self):
        header, _ = self.cut()
        epoch = header["worker_epoch"].encode()
        for _ in range(5):
            self.send(self.store("GPU", ["full"]))
        frames = self.request(self.replay, b"replay-v2", epoch, (0).to_bytes(8, "big"))
        replay_header = msgspec.msgpack.decode(frames[0][2])
        self.assertEqual(replay_header["worker_epoch"], epoch.decode())
        sequences = [int.from_bytes(f[1], "big") for f in frames[1:-1]]
        self.assertEqual(sequences, [3, 4, 5])
        self.assertEqual(replay_header["resume_seq"], 6)
        rejected = self.request(self.replay, b"replay-v2", b"old-epoch", bytes(8))
        self.assertEqual(rejected[0][1], b"error")

    def test_parent_links_and_legacy_snapshot_remain_usable(self):
        self.send(self.store("GPU", ["full"], [11, 12, 13]))
        _, blocks = self.cut()
        self.assertEqual([b["parent_block_hash"] for b in blocks], [None, 11, 12])
        legacy = self.request(self.snapshot, b"snapshot-v1")
        self.assertEqual(msgspec.msgpack.decode(legacy[0][2])[0], 1)

    def test_invalid_component_cannot_kill_publisher_thread(self):
        with self.assertRaises(ValueError):
            self.send(self.store("GPU", ["invalid"]))
        self.send(self.store("GPU", ["full"]))
        self.assertEqual(len(self.cut()[1]), 1)


if __name__ == "__main__":
    unittest.main()
