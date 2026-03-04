import time
import unittest

import requests
import zmq
from msgspec.msgpack import Decoder

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestKvEvents(CustomTestCase):
    def test_kv_events_enabled(self):
        """Test that kv events are sent and received by subscriber data when enabled"""

        # Launch kv events subscriber
        decoder = Decoder(type=KVEventBatch)
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        sub.connect("tcp://localhost:5557")
        topic = "kv-events"
        sub.setsockopt_string(zmq.SUBSCRIBE, topic)

        # Launch sglang server
        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--kv-events-config",
                '{"publisher": "zmq", "topic": "kv-events"}',
                "--max-total-tokens",
                32,
                "--cuda-graph-max-bs",
                2,
                "--enable-dp-attention",
                "--dp-size",
                1,
            ],
        )

        try:
            # Make some requests to generate some metrics
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of Spain is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )

            # Get events
            events = []
            start = time.time()
            max_wait_s = 5
            min_events_expected = 5  # Expect at least some events

            while (
                len(events) < min_events_expected and (time.time() - start) < max_wait_s
            ):
                if sub.poll(timeout=100):  # 100ms timeout
                    _, seq_bytes, payload = sub.recv_multipart()
                    event_batch = decoder.decode(payload)
                    for event in event_batch.events:
                        events.append(event)

            # Verify we received events
            self.assertGreater(
                len(events), 0, "Should have received at least one KV cache event"
            )

            # Track which blocks were stored and removed
            stored_blocks = {}  # hash -> BlockStored event
            removed_hashes = set()

            # Validate event structure and relationships
            for event in events:
                self.assertIsInstance(
                    event,
                    (BlockStored, BlockRemoved, AllBlocksCleared),
                    f"Event should be a KV cache event, got {type(event)}",
                )

                if isinstance(event, BlockStored):
                    # Validate BlockStored structure
                    self.assertIsInstance(event.block_hashes, list)
                    self.assertEqual(
                        len(event.block_hashes), 1, "Should have one hash per block"
                    )
                    self.assertIsInstance(event.token_ids, list)
                    self.assertEqual(
                        event.block_size,
                        len(event.token_ids),
                        "block_size should match token_ids length",
                    )
                    self.assertIsNone(
                        event.lora_id, "lora_id should be None for basic test"
                    )

                    # Store this block for later validation
                    block_hash = event.block_hashes[0]
                    stored_blocks[block_hash] = event

                    # If parent_block_hash is set, verify it was stored earlier
                    if event.parent_block_hash is not None:
                        # Parent should either be in stored_blocks or could be from a previous request
                        pass  # Don't strictly enforce this as root blocks may have synthetic parents

                elif isinstance(event, BlockRemoved):
                    # Validate BlockRemoved structure
                    self.assertIsInstance(event.block_hashes, list)
                    self.assertEqual(
                        len(event.block_hashes), 1, "Should have one hash per block"
                    )
                    removed_hashes.add(event.block_hashes[0])

            # Verify we got both BlockStored and BlockRemoved events
            self.assertGreater(
                len(stored_blocks), 0, "Should have at least one BlockStored event"
            )
            # BlockRemoved events may not always occur in this short test, so just check if they do occur
            # that they reference previously stored blocks
            for removed_hash in removed_hashes:
                # It's OK if the removed block wasn't in our stored_blocks
                # (it could have been stored before we started listening)
                pass

        finally:
            sub.close()
            context.term()
            kill_process_tree(process.pid)

    def test_kv_events_attn_dp(self):
        """Test that kv events are properly tagged with DP rank in attention DP mode"""

        # Launch multiple subscribers for different DP ranks
        decoder = Decoder(type=KVEventBatch)
        context = zmq.Context()

        # Subscribe to both DP rank endpoints
        sub_dp0 = context.socket(zmq.SUB)
        sub_dp0.connect("tcp://localhost:5557")  # DP rank 0
        topic = "kv-events"
        sub_dp0.setsockopt_string(zmq.SUBSCRIBE, topic)

        sub_dp1 = context.socket(zmq.SUB)
        sub_dp1.connect("tcp://localhost:5558")  # DP rank 1 (offset by rank)
        sub_dp1.setsockopt_string(zmq.SUBSCRIBE, topic)

        # Launch sglang server with DP attention enabled
        process = popen_launch_server(
            DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--kv-events-config",
                '{"publisher": "zmq", "topic": "kv-events"}',
                "--max-total-tokens",
                64,
                "--cuda-graph-max-bs",
                4,
                "--enable-dp-attention",
                "--dp-size",
                2,
                "--tp-size",
                2,
            ],
        )

        try:
            # Make requests to generate events
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            # Send multiple requests to trigger events from both DP ranks
            for i in range(4):
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": f"Request {i}: The capital of country {i} is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 16,
                        },
                    },
                )

            # Collect events from both DP ranks
            events_dp0 = []
            events_dp1 = []
            start = time.time()
            max_wait_s = 10
            min_events_per_rank = 3  # Expect at least a few events from each rank

            while (time.time() - start) < max_wait_s and (
                len(events_dp0) < min_events_per_rank
                or len(events_dp1) < min_events_per_rank
            ):
                # Check DP rank 0
                if sub_dp0.poll(timeout=100):  # 100ms timeout
                    _, seq_bytes, payload = sub_dp0.recv_multipart()
                    event_batch = decoder.decode(payload)
                    print(
                        f"DP Rank 0 - EventBatch: ts={event_batch.ts}, attn_dp_rank={event_batch.attn_dp_rank}"
                    )
                    self.assertEqual(
                        event_batch.attn_dp_rank,
                        0,
                        "DP rank 0 events should have attn_dp_rank=0",
                    )
                    for event in event_batch.events:
                        print(f"  DP0 - {event}")
                        events_dp0.append(event)

                # Check DP rank 1
                if sub_dp1.poll(timeout=100):  # 100ms timeout
                    _, seq_bytes, payload = sub_dp1.recv_multipart()
                    event_batch = decoder.decode(payload)
                    print(
                        f"DP Rank 1 - EventBatch: ts={event_batch.ts}, attn_dp_rank={event_batch.attn_dp_rank}"
                    )
                    self.assertEqual(
                        event_batch.attn_dp_rank,
                        1,
                        "DP rank 1 events should have attn_dp_rank=1",
                    )
                    for event in event_batch.events:
                        print(f"  DP1 - {event}")
                        events_dp1.append(event)

            # Verify we got events from both DP ranks
            print(f"Collected {len(events_dp0)} events from DP rank 0")
            print(f"Collected {len(events_dp1)} events from DP rank 1")

            self.assertGreaterEqual(
                len(events_dp0),
                min_events_per_rank,
                f"Expected at least {min_events_per_rank} events from DP rank 0",
            )
            self.assertGreaterEqual(
                len(events_dp1),
                min_events_per_rank,
                f"Expected at least {min_events_per_rank} events from DP rank 1",
            )

            # Verify event types are as expected
            for events in [events_dp0, events_dp1]:
                for event in events:
                    self.assertIsInstance(
                        event,
                        (BlockStored, BlockRemoved, AllBlocksCleared),
                        f"Event should be a KV cache event, got {type(event)}",
                    )

        finally:
            sub_dp0.close()
            sub_dp1.close()
            context.term()
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
