import time
import unittest

import msgspec
import requests
import zmq
from msgspec.msgpack import Decoder

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventBatch,
    KVCacheEvent,
    KVEventBatch,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
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

            # Expected events. These may be dependent on model used (meta-llama/Llama-3.2-1B-Instruct)
            expected_events = [
                # <begin> The capital city of France is
                BlockStored(
                    block_hashes=[-6650323075460941099],
                    parent_block_hash=5740354900026072187,
                    token_ids=[128000, 791, 6864, 3363, 315, 9822, 374],
                    block_size=7,
                    lora_id=None,
                ),
                # Paris. The Eiffel Tower
                BlockStored(
                    block_hashes=[-7584018293207282755],
                    parent_block_hash=-6650323075460941099,
                    token_ids=[12366, 13, 578, 469, 3168, 301, 22703],
                    block_size=7,
                    lora_id=None,
                ),
                BlockStored(
                    block_hashes=[-8753497827991233192],
                    parent_block_hash=5740354900026072187,
                    token_ids=[0],
                    block_size=1,
                    lora_id=None,
                ),
                BlockRemoved(block_hashes=[-6650323075460941099]),
                # <begin> The capital
                BlockStored(
                    block_hashes=[-2697055055087824455],
                    parent_block_hash=5740354900026072187,
                    token_ids=[128000, 791, 6864],
                    block_size=3,
                    lora_id=None,
                ),
                # city of France is
                BlockStored(
                    block_hashes=[-7505627135785778022],
                    parent_block_hash=-2697055055087824455,
                    token_ids=[3363, 315, 9822, 374],
                    block_size=4,
                    lora_id=None,
                ),
                # of France is
                BlockStored(
                    block_hashes=[-3861108700662737012],
                    parent_block_hash=-2697055055087824455,
                    token_ids=[315, 9822, 374],
                    block_size=3,
                    lora_id=None,
                ),
                BlockRemoved(block_hashes=[-7584018293207282755]),
                BlockRemoved(block_hashes=[-8753497827991233192]),
                BlockRemoved(block_hashes=[-7505627135785778022]),
                # Paris. The Eiffel Tower is located in Paris. The Eiffel Tower is a famous landmark in Paris
                BlockStored(
                    block_hashes=[-3064341286825792715],
                    parent_block_hash=-3861108700662737012,
                    token_ids=[
                        12366,
                        13,
                        578,
                        469,
                        3168,
                        301,
                        22703,
                        374,
                        7559,
                        304,
                        12366,
                        13,
                        578,
                        469,
                        3168,
                        301,
                        22703,
                        374,
                        264,
                        11495,
                        38350,
                        304,
                        12366,
                    ],
                    block_size=23,
                    lora_id=None,
                ),
                BlockRemoved(block_hashes=[-3861108700662737012]),
                # of
                BlockStored(
                    block_hashes=[6115672085296369592],
                    parent_block_hash=-2697055055087824455,
                    token_ids=[315],
                    block_size=1,
                    lora_id=None,
                ),
                # France is
                BlockStored(
                    block_hashes=[4208810872343132234],
                    parent_block_hash=6115672085296369592,
                    token_ids=[9822, 374],
                    block_size=2,
                    lora_id=None,
                ),
                # Spain is
                BlockStored(
                    block_hashes=[1675819893649989955],
                    parent_block_hash=6115672085296369592,
                    token_ids=[18157, 374],
                    block_size=2,
                    lora_id=None,
                ),
                BlockRemoved(block_hashes=[-3064341286825792715]),
                # Madrid. The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
                BlockStored(
                    block_hashes=[-8505834929190027295],
                    parent_block_hash=1675819893649989955,
                    token_ids=[
                        25048,
                        13,
                        578,
                        6864,
                        315,
                        9822,
                        374,
                        12366,
                        13,
                        578,
                        6864,
                        315,
                        15704,
                        374,
                        22463,
                        13,
                        578,
                        6864,
                        315,
                        18157,
                        374,
                        25048,
                        13,
                    ],
                    block_size=23,
                    lora_id=None,
                ),
            ]

            # Get events
            events = []
            start = time.time()
            max_wait_s = 5
            while (
                len(events) < len(expected_events)
                and (time.time() - start) < max_wait_s
            ):
                _, seq_bytes, payload = sub.recv_multipart()
                event_batch = decoder.decode(payload)
                for event in event_batch.events:
                    print(f"  - {event}")
                    events.append(event)

            for expected in expected_events:
                self.assertIn(expected, events)

        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
