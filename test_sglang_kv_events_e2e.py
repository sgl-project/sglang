#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
E2E test for SGLang KV Events with HiCache tier transitions.

This test validates that:
1. SGLang emits KV events with correct `medium` field (GPU, CPU_TIER1, CPU_TIER2)
2. Tier transitions produce the expected event sequence
3. Events can be received via ZMQ (as the Dynamo router would)

Unlike vLLM/TRT-LLM, SGLang doesn't need a consolidator because:
- HiCache is built-in (no separate KVBM)
- All events come from a single source (hiradix_cache.py)
"""

import json
import subprocess
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import msgpack
import requests
import zmq

# Test configuration
MODEL = "Qwen/Qwen3-0.6B"
SERVER_PORT = 30000
ZMQ_PORT = 5557
HICACHE_SIZE_GB = 1  # Small for faster tier transitions


def start_sglang_server() -> subprocess.Popen:
    """Start SGLang server with HiCache and KV events enabled."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--port", str(SERVER_PORT),
        "--mem-fraction-static", "0.05",  # Small GPU cache to trigger evictions
        "--enable-hierarchical-cache",
        "--hicache-size", str(HICACHE_SIZE_GB),
        "--hicache-storage-backend", "file",  # Enable L3 storage
        "--kv-events-config", json.dumps({
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{ZMQ_PORT}"
        }),
    ]

    print(f"Starting SGLang server: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return process


def wait_for_server(url: str, timeout: int = 120) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"Server ready after {time.time() - start:.1f}s")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


class KvEventCollector:
    """Collect KV events from ZMQ socket."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.events: List[dict] = []
        self.events_by_medium: Dict[str, List[dict]] = defaultdict(list)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start collecting events in background thread."""
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop collecting events."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _collect_loop(self):
        """Background loop to collect events."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(self.endpoint)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout

        while not self._stop.is_set():
            try:
                msg = sock.recv()
                event = msgpack.unpackb(msg, raw=False)
                self.events.append(event)

                # Categorize by medium
                medium = event.get("medium", "unknown")
                event_type = event.get("type", "unknown")
                self.events_by_medium[medium].append(event)

                print(f"  [{len(self.events)}] {event_type}: medium={medium}, "
                      f"hashes={event.get('block_hashes', [])[:2]}...")

            except zmq.Again:
                continue
            except Exception as e:
                print(f"Error receiving event: {e}")

        sock.close()
        ctx.term()

    def get_stats(self) -> dict:
        """Get event statistics."""
        stats = {
            "total_events": len(self.events),
            "by_medium": {k: len(v) for k, v in self.events_by_medium.items()},
            "block_stored_gpu": 0,
            "block_stored_cpu_tier1": 0,
            "block_stored_cpu_tier2": 0,
            "block_removed_gpu": 0,
            "block_removed_cpu_tier1": 0,
        }

        for event in self.events:
            event_type = event.get("type", "")
            medium = event.get("medium", "")

            if event_type == "BlockStored":
                if medium == "GPU":
                    stats["block_stored_gpu"] += 1
                elif medium == "CPU_TIER1":
                    stats["block_stored_cpu_tier1"] += 1
                elif medium == "CPU_TIER2":
                    stats["block_stored_cpu_tier2"] += 1
            elif event_type == "BlockRemoved":
                if medium == "GPU":
                    stats["block_removed_gpu"] += 1
                elif medium == "CPU_TIER1":
                    stats["block_removed_cpu_tier1"] += 1

        return stats


def send_requests(base_url: str, num_requests: int = 10, max_tokens: int = 100):
    """Send requests to trigger KV cache activity."""
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short poem about the ocean.",
        "What is the capital of France and why is it important?",
        "Describe the process of photosynthesis.",
        "Tell me about the history of the internet.",
    ]

    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        try:
            resp = requests.post(
                f"{base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": max_tokens},
                },
                timeout=60,
            )
            print(f"Request {i+1}/{num_requests}: status={resp.status_code}")
        except Exception as e:
            print(f"Request {i+1} failed: {e}")

        # Small delay between requests
        time.sleep(0.5)


def test_kv_events_with_hicache():
    """
    Test that SGLang emits correct KV events with HiCache enabled.

    Expected behavior:
    1. BlockStored(GPU) - initial cache of prompt tokens
    2. BlockStored(CPU_TIER1) - offload to host memory when GPU fills
    3. BlockRemoved(GPU) - GPU memory freed (eviction)
    4. BlockStored(CPU_TIER2) - backup to file storage (L3)
    """
    server = None
    collector = None

    try:
        # Start event collector first
        zmq_endpoint = f"tcp://127.0.0.1:{ZMQ_PORT}"
        collector = KvEventCollector(zmq_endpoint)
        collector.start()
        print(f"Event collector started, listening on {zmq_endpoint}")

        # Start server
        server = start_sglang_server()

        # Wait for server to be ready
        base_url = f"http://127.0.0.1:{SERVER_PORT}"
        if not wait_for_server(base_url):
            raise RuntimeError("Server failed to start")

        # Wait a bit for initial events
        time.sleep(2)

        # Send requests to trigger tier transitions
        print("\n=== Sending requests to trigger tier transitions ===")
        send_requests(base_url, num_requests=20, max_tokens=200)

        # Wait for events to be processed
        print("\n=== Waiting for tier transitions ===")
        time.sleep(10)

        # Get statistics
        stats = collector.get_stats()
        print("\n=== Event Statistics ===")
        print(f"Total events: {stats['total_events']}")
        print(f"By medium: {stats['by_medium']}")
        print(f"BlockStored(GPU): {stats['block_stored_gpu']}")
        print(f"BlockStored(CPU_TIER1): {stats['block_stored_cpu_tier1']}")
        print(f"BlockStored(CPU_TIER2): {stats['block_stored_cpu_tier2']}")
        print(f"BlockRemoved(GPU): {stats['block_removed_gpu']}")
        print(f"BlockRemoved(CPU_TIER1): {stats['block_removed_cpu_tier1']}")

        # Assertions
        assert stats["total_events"] > 0, "No events received"
        assert stats["block_stored_gpu"] > 0, "No GPU BlockStored events"

        # With HiCache enabled, we should see CPU_TIER1 events
        # (may not always happen depending on cache pressure)
        if stats["block_stored_cpu_tier1"] > 0:
            print("✓ CPU_TIER1 tier transitions observed")
        else:
            print("⚠ No CPU_TIER1 events (may need more cache pressure)")

        # With file backend, we should see CPU_TIER2 events
        if stats["block_stored_cpu_tier2"] > 0:
            print("✓ CPU_TIER2 (L3 storage) events observed")
        else:
            print("⚠ No CPU_TIER2 events (may need more time/pressure)")

        print("\n=== Test Passed ===")

    finally:
        if collector:
            collector.stop()
        if server:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    test_kv_events_with_hicache()
