#!/usr/bin/env python3
"""
Test script to verify KV events contain the medium field.
Run with a server started with kv-events enabled:
    ./agg_hicache.sh
Then run this script:
    python test_kv_events_medium.py
"""

import time
import sys
import zmq
from msgspec.msgpack import Decoder

from sglang.srt.disaggregation.kv_events import (
    BlockStored, BlockRemoved, AllBlocksCleared, KVEventBatch,
    MEDIUM_GPU, MEDIUM_CPU_TIER1, MEDIUM_CPU_TIER2
)


def test_kv_events_medium(endpoint="tcp://localhost:5557", topic="kv-events", timeout_s=30):
    """
    Subscribe to KV events and verify they contain the medium field.
    """
    print(f"Connecting to ZMQ endpoint: {endpoint}")
    print(f"Subscribing to topic: {topic}")

    decoder = Decoder(type=KVEventBatch)
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    # Set receive timeout
    sub.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout

    print(f"\nWaiting for events (timeout: {timeout_s}s)...")
    print("Send some requests to the server to generate events.\n")

    events_received = 0
    stored_with_medium = 0
    stored_without_medium = 0
    removed_with_medium = 0
    removed_without_medium = 0
    all_cleared = 0

    start_time = time.time()

    while time.time() - start_time < timeout_s:
        try:
            parts = sub.recv_multipart()
            if len(parts) != 3:
                print(f"  Warning: Unexpected message format: {len(parts)} parts")
                continue

            _, seq_bytes, payload = parts
            seq = int.from_bytes(seq_bytes, "big")

            event_batch = decoder.decode(payload)

            for event in event_batch.events:
                events_received += 1

                if isinstance(event, BlockStored):
                    if event.medium is not None:
                        stored_with_medium += 1
                        print(f"  [seq={seq}] BlockStored: hashes={event.block_hashes}, medium={event.medium}")
                    else:
                        stored_without_medium += 1
                        print(f"  [seq={seq}] BlockStored: hashes={event.block_hashes}, medium=None (MISSING!)")

                elif isinstance(event, BlockRemoved):
                    if event.medium is not None:
                        removed_with_medium += 1
                        print(f"  [seq={seq}] BlockRemoved: hashes={event.block_hashes}, medium={event.medium}")
                    else:
                        removed_without_medium += 1
                        print(f"  [seq={seq}] BlockRemoved: hashes={event.block_hashes}, medium=None (MISSING!)")

                elif isinstance(event, AllBlocksCleared):
                    all_cleared += 1
                    print(f"  [seq={seq}] AllBlocksCleared")

        except zmq.Again:
            # Timeout waiting for message
            if events_received > 0:
                print("  (no more events)")
                break
            else:
                print("  (waiting for events...)")

    sub.close()
    context.term()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total events received: {events_received}")
    print(f"  BlockStored with medium:    {stored_with_medium}")
    print(f"  BlockStored without medium: {stored_without_medium}")
    print(f"  BlockRemoved with medium:   {removed_with_medium}")
    print(f"  BlockRemoved without medium: {removed_without_medium}")
    print(f"  AllBlocksCleared:           {all_cleared}")

    # Validation
    if events_received == 0:
        print("\nWARNING: No events received. Make sure the server is running and accepting requests.")
        return False

    if stored_without_medium > 0 or removed_without_medium > 0:
        print("\nFAILED: Some events are missing the medium field!")
        return False

    if stored_with_medium > 0:
        print(f"\nSUCCESS: All BlockStored events have medium field set!")

    if removed_with_medium > 0:
        print(f"SUCCESS: All BlockRemoved events have medium field set!")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test KV events medium field")
    parser.add_argument("--endpoint", default="tcp://localhost:5557", help="ZMQ endpoint")
    parser.add_argument("--topic", default="kv-events", help="ZMQ topic")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    args = parser.parse_args()

    success = test_kv_events_medium(args.endpoint, args.topic, args.timeout)
    sys.exit(0 if success else 1)
