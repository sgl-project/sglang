"""Unit tests for srt/disaggregation/kv_events KV-event publisher rank selection.

Covers the data-parallel rank used to offset each scheduler's KV-event
publisher port, across pure DP, DP-attention, and single-replica modes. The
port offset must make every independent KV cache publish on a distinct port so
the router can subscribe per replica (the `dp_size` it reads from
`/server_info`).
"""

import atexit
import json
import threading
import time
import unittest
from collections import OrderedDict, deque
from queue import Queue
from unittest.mock import patch

import msgspec
import zmq

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventPublisherFactory,
    KVEventBatch,
    StorageMedium,
    ZmqEventPublisher,
    kv_event_namespace_seed,
    namespaced_block_hash,
    resolve_load_pub_range,
)
from sglang.srt.runtime_context import describe_kv_events_publisher
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestLocalKvEventSource(CustomTestCase):
    def _publisher(self, attn_dp_rank=0, **config):
        publisher = EventPublisherFactory.create(
            json.dumps({"publisher": "zmq", **config}), attn_dp_rank=attn_dp_rank
        )
        atexit.unregister(publisher.shutdown)
        self.addCleanup(publisher.shutdown)
        return publisher

    def test_advertised_source_delivers_events(self):
        for host, bind, topic in (("*", None, "kv"), ("127.0.0.1", True, "")):
            with self.subTest(host=host):
                with zmq.Context.instance().socket(zmq.PUB) as probe:
                    port = probe.bind_to_random_port("tcp://127.0.0.1")
                publisher = self._publisher(
                    attn_dp_rank=4,
                    endpoint=f"tcp://{host}:{port - 4}",
                    bind=bind,
                    topic=topic,
                )
                source = publisher.describe_local_source(64)
                self.assertEqual(
                    source,
                    dict(
                        dp_rank=4,
                        endpoint=f"tcp://127.0.0.1:{port}",
                        topic=topic,
                        block_size=64,
                    ),
                )
                self._assert_event_received(publisher, source)

    def _assert_event_received(self, publisher, source):
        with zmq.Context() as context, context.socket(zmq.SUB) as subscriber:
            subscriber.setsockopt_string(zmq.SUBSCRIBE, source["topic"])
            subscriber.connect(source["endpoint"])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                publisher.publish(KVEventBatch(ts=1.0, events=[AllBlocksCleared()]))
                if subscriber.poll(100):
                    topic, _, payload = subscriber.recv_multipart()
                    self.assertEqual(topic, source["topic"].encode())
                    batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
                    self.assertEqual(batch.attn_dp_rank, source["dp_rank"])
                    self.assertEqual(batch.events, [AllBlocksCleared()])
                    break
            else:
                self.fail("No event received from the advertised local source")


class TestOffsetEndpointPort(CustomTestCase):
    def test_ipc_endpoint_gets_a_dp_suffix(self):
        self.assertEqual(
            ZmqEventPublisher.offset_endpoint_port("ipc:///tmp/kv_events", 2),
            "ipc:///tmp/kv_events_dp2",
        )


class TestShutdown(CustomTestCase):
    def test_full_queue_does_not_skip_socket_cleanup(self):
        with patch.object(threading.Thread, "start"):
            publisher = ZmqEventPublisher(
                attn_dp_rank=0, endpoint="inproc://kv-shutdown-test", max_queue_size=1
            )
        atexit.unregister(publisher.shutdown)
        publisher.publish(KVEventBatch(ts=1.0, events=[]))
        publisher.shutdown()
        self.assertTrue(publisher._pub.closed)


class TestResolveLoadPubRange(CustomTestCase):
    """The single source of truth both the bind and /server_info route through."""

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
            attn_dp_rank=0,
            endpoint="inproc://kv-replay-test",
            replay_endpoint=replay,
            topic="kv@prefill@model",
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
        self.assertEqual([len(f) for f in frames], [4, 4, 4])
        seqs = [int.from_bytes(f[2], "big", signed=True) for f in frames]
        self.assertEqual(seqs, [1, 2, -1])
        self.assertEqual([f[0] for f in frames], [b""] * 3)
        self.assertEqual([f[1] for f in frames], [b"kv@prefill@model"] * 2 + [b""])
        for frame in frames[:2]:
            batch = msgspec.msgpack.decode(frame[3], type=KVEventBatch)
            self.assertEqual(batch.events, [AllBlocksCleared()])
            self.assertEqual(batch.attn_dp_rank, 0)
        self.assertEqual(frames[2][3], b"")


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
        self.publisher._bind = None
        self.publisher._topic_bytes = b"kv@prefill@model"
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
                if message == [b"", b"", self.publisher.END_SEQ, b""]:
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
            [
                [b"", self.publisher._topic_bytes, seq.to_bytes(8, "big"), payload]
                for seq, payload in batches
            ]
            + [[b"", b"", self.publisher.END_SEQ, b""]],
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
                [b"", self.publisher._topic_bytes, (0).to_bytes(8, "big"), b"first"],
                [b"", self.publisher._topic_bytes, (1).to_bytes(8, "big"), b"second"],
                [b"", b"", self.publisher.END_SEQ, b""],
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
            [int.from_bytes(frame[2], "big") for frame in frames[:-1]],
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
                [b"", self.publisher._topic_bytes, (1).to_bytes(8, "big"), b"first"],
                [b"", self.publisher._topic_bytes, (2).to_bytes(8, "big"), b"second"],
                [b"", b"", self.publisher.END_SEQ, b""],
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
