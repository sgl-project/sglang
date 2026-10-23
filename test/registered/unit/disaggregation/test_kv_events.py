"""Unit tests for srt/disaggregation/kv_events KV-event publisher rank selection.

Covers the data-parallel rank used to offset each scheduler's KV-event
publisher port, across pure DP, DP-attention, and single-replica modes. The
port offset must make every independent KV cache publish on a distinct port so
the router can subscribe per replica (the `dp_size` it reads from
`/server_info`).
"""

import atexit
import json
import time
import unittest
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
from sglang.srt.utils.common import build_server_info
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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


class TestAdvertisedKvEventSource(CustomTestCase):
    """The off-box view of a publisher, merged onto node 0 for multi-node DP."""

    def _publisher(self, **config):
        publisher = EventPublisherFactory.create(
            json.dumps({"publisher": "zmq", **config}), attn_dp_rank=2
        )
        atexit.unregister(publisher.shutdown)
        self.addCleanup(publisher.shutdown)
        return publisher

    def _ports(self):
        with zmq.Context.instance().socket(zmq.PUB) as probe:
            pub = probe.bind_to_random_port("tcp://127.0.0.1")
        with zmq.Context.instance().socket(zmq.ROUTER) as probe:
            replay = probe.bind_to_random_port("tcp://127.0.0.1")
        # The publisher offsets both by attn_dp_rank=2.
        return pub, replay

    def test_wildcard_bind_uses_advertise_host(self):
        pub, replay = self._ports()
        publisher = self._publisher(
            endpoint=f"tcp://*:{pub - 2}",
            replay_endpoint=f"tcp://*:{replay - 2}",
            topic="kv",
            advertise_host="10.0.0.5",
        )
        self.assertEqual(
            publisher.describe_advertised_source(64),
            dict(
                dp_rank=2,
                endpoint=f"tcp://10.0.0.5:{pub}",
                replay_endpoint=f"tcp://10.0.0.5:{replay}",
                topic="kv",
                block_size=64,
            ),
        )
        # The local view is unchanged: loopback for a separate local process.
        self.assertEqual(
            publisher.describe_local_source(64)["endpoint"], f"tcp://127.0.0.1:{pub}"
        )

    def test_wildcard_bind_falls_back_to_detected_ip(self):
        pub, _ = self._ports()
        publisher = self._publisher(endpoint=f"tcp://*:{pub - 2}")
        with patch(
            "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
            return_value="10.0.0.9",
        ):
            source = publisher.describe_advertised_source(64)
        self.assertEqual(source["endpoint"], f"tcp://10.0.0.9:{pub}")
        self.assertNotIn("replay_endpoint", source)

        with patch(
            "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
            side_effect=ValueError("no ip"),
        ):
            self.assertIsNone(publisher.describe_advertised_source(64))

    def test_loopback_bind_is_advertised_only_with_a_host(self):
        pub, _ = self._ports()
        publisher = self._publisher(endpoint=f"tcp://127.0.0.1:{pub - 2}", bind=True)
        with patch(
            "sglang.srt.disaggregation.kv_events.get_local_ip_auto",
            return_value="10.0.0.9",
        ):
            self.assertIsNone(publisher.describe_advertised_source(64))
        self.assertIsNotNone(publisher.describe_local_source(64))

        pub, _ = self._ports()
        publisher = self._publisher(
            endpoint=f"tcp://127.0.0.1:{pub - 2}", bind=True, advertise_host="h1"
        )
        self.assertEqual(
            publisher.describe_advertised_source(64)["endpoint"], f"tcp://h1:{pub}"
        )

    def test_connect_style_publisher_is_not_advertised(self):
        pub, _ = self._ports()
        publisher = self._publisher(
            endpoint=f"tcp://10.0.0.7:{pub - 2}", advertise_host="10.0.0.5"
        )
        self.assertIsNone(publisher.describe_advertised_source(64))
        self.assertIsNone(publisher.describe_local_source(64))

    def test_build_server_info_attaches_sources_to_the_descriptor(self):
        cfg = json.dumps({"publisher": "zmq", "endpoint": "tcp://*:5557"})
        args = ServerArgs(model_path="dummy", page_size=16, kv_events_config=cfg)
        sources = [
            {"dp_rank": 0, "node_rank": 0, "endpoint": "tcp://10.0.0.1:5557"},
            {"dp_rank": 1, "node_rank": 1, "endpoint": "tcp://10.0.0.2:5558"},
        ]
        info = build_server_info(
            args,
            {
                "max_total_num_tokens": 8,
                "kv_event_sources": [],
                "advertised_kv_event_sources": sources,
            },
        )
        self.assertEqual(info["kv_events"]["sources"], sources)
        self.assertEqual(info["kv_events"]["endpoint_port_base"], 5557)
        self.assertEqual(info["kv_event_sources"], [])
        self.assertNotIn("advertised_kv_event_sources", info)

        without = build_server_info(args, {"max_total_num_tokens": 8})
        self.assertNotIn("sources", without["kv_events"])

        disabled = ServerArgs(model_path="dummy", page_size=16)
        self.assertIsNone(
            build_server_info(disabled, {"advertised_kv_event_sources": sources})[
                "kv_events"
            ]
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


if __name__ == "__main__":
    unittest.main()
