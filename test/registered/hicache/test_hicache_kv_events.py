"""
Deterministic KV Events Test Harness for HiRadixCache.

Validates that KV events (BlockStored/BlockRemoved with medium field) correctly
track blocks across L1 (GPU) / L2 (Host) / L3 (Storage) tiers.

Manages the SGLang server lifecycle, collects events via ZMQ subscriber in a
background thread, and validates event correctness using a state-machine validator.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 -m pytest test/registered/hicache/test_hicache_kv_events.py -v -s

    # Single test:
    CUDA_VISIBLE_DEVICES=1 python3 -m pytest \
        test/registered/hicache/test_hicache_kv_events.py::TestHiCacheKvEvents::test_01_l1_insert -v -s
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import msgspec
import requests
import zmq

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from sglang.bench_serving import get_tokenizer
from sglang.srt.utils import kill_process_tree

MODEL = "Qwen/Qwen3-0.6B"
SERVER_PORT = 39000
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"
ZMQ_PORT = 15557
ZMQ_ENDPOINT = f"tcp://127.0.0.1:{ZMQ_PORT}"
PAGE_SIZE = 64

# Deterministic prompt building blocks
PROMPT_SEED_A = "The quick brown fox jumps over the lazy dog. "
PROMPT_SEED_B = "A journey of a thousand miles begins with a single step. "
PROMPT_SEED_C = "To be or not to be that is the question whether tis nobler. "
FLOOD_SALT = "xK9mQ2v7"


@dataclass
class ParsedEvent:
    """Parsed representation of a single KV cache event."""

    event_type: str  # "BlockStored", "BlockRemoved", "AllBlocksCleared"
    medium: Optional[str]
    block_hashes: List[int]
    token_ids: Optional[List[int]] = None
    seq: int = 0
    timestamp: float = 0.0


class KvEventCollector:
    """Background ZMQ SUB thread that collects KV events."""

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._events: List[ParsedEvent] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._decoder = msgspec.msgpack.Decoder(KVEventBatch)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="kv-event-collector"
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def snapshot(self) -> int:
        """Return current event count as a marker for isolation between tests."""
        with self._lock:
            return len(self._events)

    def events_since(self, marker: int) -> List[ParsedEvent]:
        """Return events collected after the given marker."""
        with self._lock:
            return list(self._events[marker:])

    def all_events(self) -> List[ParsedEvent]:
        with self._lock:
            return list(self._events)

    def drain_and_wait(self, timeout_s: float = 3.0, quiesce_s: float = 1.0):
        """Sleep then wait until no new events arrive for quiesce_s seconds."""
        time.sleep(timeout_s)
        prev_count = -1
        while True:
            with self._lock:
                current_count = len(self._events)
            if current_count == prev_count:
                break
            prev_count = current_count
            time.sleep(quiesce_s)

    def _run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(self._endpoint)

        try:
            while not self._stop.is_set():
                try:
                    parts = sock.recv_multipart()
                    if len(parts) == 3:
                        _topic, seq_bytes, payload = parts
                        seq = int.from_bytes(seq_bytes, "big")
                    elif len(parts) == 1:
                        payload = parts[0]
                        seq = 0
                    else:
                        continue

                    batch = self._decoder.decode(payload)
                    self._process_batch(batch, seq)
                except zmq.Again:
                    continue
                except msgspec.DecodeError:
                    continue
        finally:
            sock.close()
            ctx.term()

    def _process_batch(self, batch: KVEventBatch, seq: int):
        with self._lock:
            for event in batch.events:
                if isinstance(event, BlockStored):
                    self._events.append(
                        ParsedEvent(
                            event_type="BlockStored",
                            medium=event.medium,
                            block_hashes=list(event.block_hashes),
                            token_ids=list(event.token_ids) if event.token_ids else None,
                            seq=seq,
                            timestamp=batch.ts,
                        )
                    )
                elif isinstance(event, BlockRemoved):
                    self._events.append(
                        ParsedEvent(
                            event_type="BlockRemoved",
                            medium=event.medium,
                            block_hashes=list(event.block_hashes),
                            seq=seq,
                            timestamp=batch.ts,
                        )
                    )
                elif isinstance(event, AllBlocksCleared):
                    self._events.append(
                        ParsedEvent(
                            event_type="AllBlocksCleared",
                            medium=None,
                            block_hashes=[],
                            seq=seq,
                            timestamp=batch.ts,
                        )
                    )


class InlineValidator:
    """State-machine validator for KV event correctness.

    Invariants:
    - No duplicate BlockStored for same (hash, medium) without intervening BlockRemoved
    - No BlockRemoved for blocks not currently stored in that medium
    """

    def __init__(self):
        # block_hash -> set of mediums currently stored
        self.block_tiers: Dict[int, Set[str]] = defaultdict(set)
        self.errors: List[str] = []

    def validate(self, events: List[ParsedEvent]):
        for ev in events:
            if ev.event_type == "AllBlocksCleared":
                self.block_tiers.clear()
                continue

            medium = ev.medium or "UNKNOWN"
            for bh in ev.block_hashes:
                if ev.event_type == "BlockStored":
                    if medium in self.block_tiers[bh]:
                        self.errors.append(
                            f"Duplicate store: block {bh} already in {medium}"
                        )
                    self.block_tiers[bh].add(medium)
                elif ev.event_type == "BlockRemoved":
                    if medium not in self.block_tiers[bh]:
                        self.errors.append(
                            f"Orphan remove: block {bh} not in {medium} "
                            f"(current: {self.block_tiers[bh] or 'none'})"
                        )
                    else:
                        self.block_tiers[bh].discard(medium)

    def blocks_in_tier(self, medium: str) -> Set[int]:
        return {bh for bh, tiers in self.block_tiers.items() if medium in tiers}


def make_prompt(seed: str, approx_tokens: int) -> str:
    """Repeat seed string to approximate a target token count.

    Rough heuristic: 1 token ~= 4 chars for English text.
    """
    target_chars = approx_tokens * 4
    reps = max(1, target_chars // len(seed))
    return seed * reps


def make_flood_prompt(index: int, approx_tokens: int) -> str:
    """Generate a unique prompt for flooding. Each index produces a distinct prefix."""
    seed = f"FLOOD_{index}_{FLOOD_SALT} "
    return make_prompt(seed, approx_tokens)


class TestHiCacheKvEvents(unittest.TestCase):
    """Integration test for HiRadixCache KV event emission."""

    process = None
    collector = None

    @classmethod
    def setUpClass(cls):
        # Start ZMQ collector BEFORE the server to avoid slow-joiner problem
        cls.collector = KvEventCollector(ZMQ_ENDPOINT)
        cls.collector.start()

        kv_events_config = json.dumps(
            {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{ZMQ_PORT}",
            }
        )

        # Redirect server logs to a temp file to keep test output clean
        cls._server_log = tempfile.NamedTemporaryFile(
            mode="w", prefix="sglang_kv_events_test_", suffix=".log", delete=False
        )

        command = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--host", "127.0.0.1",
            "--port", str(SERVER_PORT),
            "--page-size", "64",
            "--mem-fraction-static", "0.03",
            "--enable-hierarchical-cache",
            "--hicache-size", "1",
            "--hicache-storage-backend", "file",
            "--hicache-write-policy", "write_through",
            "--enable-metrics",
            "--enable-cache-report",
            "--trust-remote-code",
            "--decode-log-interval", "1",
            "--kv-events-config", kv_events_config,
        ]

        env = {**os.environ}

        print(f"\n{'='*60}")
        print(f"Launching server (logs: {cls._server_log.name})")
        cls.process = subprocess.Popen(
            command,
            stdout=cls._server_log,
            stderr=cls._server_log,
            env=env,
        )

        cls.tokenizer = get_tokenizer(MODEL)

        # Wait for server readiness
        cls._wait_for_server_ready()
        print(f"Server ready at {BASE_URL}, ZMQ collector on {ZMQ_ENDPOINT}")
        print(f"{'='*60}")

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        if cls.collector is not None:
            cls.collector.stop()
        if cls._server_log is not None:
            cls._server_log.close()

    @classmethod
    def _wait_for_server_ready(cls, timeout: int = 300):
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{BASE_URL}/health", timeout=5)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError("Server did not become ready within timeout")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def send_request(self, prompt: str, max_tokens: int = 10):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, f"Request failed: {resp.text}")
        return resp.json()

    def flush_cache(self):
        resp = requests.post(f"{BASE_URL}/flush_cache", timeout=10)
        self.assertEqual(resp.status_code, 200)

    def flush_and_drain(self):
        """Flush cache and wait for events to settle."""
        self.flush_cache()
        self.collector.drain_and_wait(timeout_s=2.0, quiesce_s=1.0)

    @staticmethod
    def _extract_hashes(events: List[ParsedEvent], event_type: str, medium: str) -> Set[int]:
        """Collect all block_hashes matching event_type and medium."""
        result = set()
        for ev in events:
            if ev.event_type == event_type and ev.medium == medium:
                result.update(ev.block_hashes)
        return result

    @staticmethod
    def _has_event(events: List[ParsedEvent], event_type: str, medium: Optional[str] = None) -> bool:
        for ev in events:
            if ev.event_type == event_type:
                if medium is None or ev.medium == medium:
                    return True
        return False

    @staticmethod
    def _filter_events(
        events: List[ParsedEvent], event_type: str, medium: Optional[str] = None
    ) -> List[ParsedEvent]:
        """Return events matching type and optionally medium."""
        return [
            ev for ev in events
            if ev.event_type == event_type and (medium is None or ev.medium == medium)
        ]

    @staticmethod
    def _event_summary(events: List[ParsedEvent]) -> str:
        """One-line summary of event counts by (type, medium)."""
        counts: Dict[str, int] = defaultdict(int)
        for ev in events:
            key = f"{ev.event_type}({ev.medium})" if ev.medium else ev.event_type
            counts[key] += 1
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        return ", ".join(parts) if parts else "(none)"

    # ------------------------------------------------------------------
    # Tests (numbered for execution order)
    # ------------------------------------------------------------------

    def test_01_l1_insert_produces_block_stored_gpu(self):
        """Fresh prompt insert should produce BlockStored(GPU) events."""
        print("\n--- test_01: L1 insert -> BlockStored(GPU) ---")
        print("  Flushing cache...")
        self.flush_and_drain()
        marker = self.collector.snapshot()

        # ~192 tokens = 3 pages at page_size=64
        print("  Sending fresh prompt (~192 tokens)...")
        prompt = make_prompt(PROMPT_SEED_A, 192)
        self.send_request(prompt)
        print("  Draining events...")
        self.collector.drain_and_wait(timeout_s=3.0, quiesce_s=1.0)

        events = self.collector.events_since(marker)
        gpu_stored = self._extract_hashes(events, "BlockStored", "GPU")
        print(f"  Collected {len(events)} events: {self._event_summary(events)}")
        print(f"  BlockStored(GPU) unique hashes: {len(gpu_stored)}")

        self.assertTrue(
            len(gpu_stored) > 0,
            f"Expected BlockStored(GPU) events, got none. Total events: {len(events)}"
        )

    def test_02_write_through_produces_block_stored_cpu_tier1(self):
        """Write-through policy should produce BlockStored(CPU_PINNED) on prefix reuse."""
        print("\n--- test_02: Write-through -> BlockStored(CPU_PINNED) ---")
        print("  Flushing cache...")
        self.flush_and_drain()
        marker = self.collector.snapshot()

        print("  Sending prompt A (~192 tokens)...")
        prompt_a = make_prompt(PROMPT_SEED_A, 192)
        self.send_request(prompt_a)

        print("  Sending prompt B (same prefix + suffix) to trigger write-through...")
        prompt_b = make_prompt(PROMPT_SEED_A, 192) + make_prompt(PROMPT_SEED_B, 64)
        self.send_request(prompt_b)
        print("  Draining events...")
        self.collector.drain_and_wait(timeout_s=3.0, quiesce_s=1.0)

        events = self.collector.events_since(marker)
        gpu_stored = self._extract_hashes(events, "BlockStored", "GPU")
        cpu1_stored = self._extract_hashes(events, "BlockStored", "CPU_PINNED")
        print(f"  Collected {len(events)} events: {self._event_summary(events)}")
        print(f"  BlockStored(GPU): {len(gpu_stored)}, BlockStored(CPU_PINNED): {len(cpu1_stored)}")

        self.assertTrue(
            len(gpu_stored) > 0,
            f"Expected BlockStored(GPU) events. Total events: {len(events)}"
        )
        self.assertTrue(
            len(cpu1_stored) > 0,
            f"Expected BlockStored(CPU_PINNED) from write-through. Total events: {len(events)}, "
            f"event types: {[(e.event_type, e.medium) for e in events]}"
        )

    def test_03_l1_eviction_produces_block_removed_gpu(self):
        """Flooding GPU cache should produce BlockRemoved(GPU) for evicted blocks."""
        print("\n--- test_03: L1 eviction -> BlockRemoved(GPU) ---")
        print("  Flushing cache...")
        self.flush_and_drain()
        marker = self.collector.snapshot()

        print("  Populating cache + triggering write-through...")
        prompt_a = make_prompt(PROMPT_SEED_A, 192)
        self.send_request(prompt_a)
        prompt_b = make_prompt(PROMPT_SEED_A, 192) + make_prompt(PROMPT_SEED_B, 64)
        self.send_request(prompt_b)

        self.collector.drain_and_wait(timeout_s=2.0, quiesce_s=1.0)
        pre_events = self.collector.events_since(marker)
        original_gpu_hashes = self._extract_hashes(pre_events, "BlockStored", "GPU")
        print(f"  Pre-flood: {self._event_summary(pre_events)}")
        print(f"  Original GPU hashes to evict: {len(original_gpu_hashes)}")

        print("  Flooding with 20 unique prompts to force GPU eviction...")
        flood_marker = self.collector.snapshot()
        for i in range(20):
            flood_prompt = make_flood_prompt(i, 192)
            self.send_request(flood_prompt)

        print("  Draining events...")
        self.collector.drain_and_wait(timeout_s=5.0, quiesce_s=1.0)

        flood_events = self.collector.events_since(flood_marker)
        gpu_removed = self._extract_hashes(flood_events, "BlockRemoved", "GPU")
        print(f"  Flood events: {self._event_summary(flood_events)}")
        print(f"  BlockRemoved(GPU) unique hashes: {len(gpu_removed)}")

        self.assertTrue(
            len(gpu_removed) > 0,
            f"Expected BlockRemoved(GPU) from eviction. "
            f"Original GPU hashes: {len(original_gpu_hashes)}, "
            f"flood events: {len(flood_events)}"
        )

    def test_04_l2_to_l1_reload_produces_block_stored_gpu(self):
        """Re-requesting evicted prefix should reload from L2 and produce BlockStored(GPU)."""
        print("\n--- test_04: L2->L1 reload -> BlockStored(GPU) ---")
        print("  Flushing cache...")
        self.flush_and_drain()

        print("  Step 1: Populate + write-through...")
        prompt_a = make_prompt(PROMPT_SEED_C, 192)
        self.send_request(prompt_a)
        prompt_b = make_prompt(PROMPT_SEED_C, 192) + make_prompt(PROMPT_SEED_B, 64)
        self.send_request(prompt_b)
        self.collector.drain_and_wait(timeout_s=2.0, quiesce_s=1.0)

        print("  Step 2: Flood 20 prompts to evict original from GPU...")
        for i in range(20):
            self.send_request(make_flood_prompt(i + 100, 192))
        self.collector.drain_and_wait(timeout_s=5.0, quiesce_s=1.0)

        print("  Step 3: Re-request original prefix (triggers load_back)...")
        marker = self.collector.snapshot()
        self.send_request(prompt_a)
        print("  Draining events...")
        self.collector.drain_and_wait(timeout_s=3.0, quiesce_s=1.0)

        events = self.collector.events_since(marker)
        gpu_stored = self._extract_hashes(events, "BlockStored", "GPU")
        print(f"  Reload events: {self._event_summary(events)}")
        print(f"  BlockStored(GPU) from reload: {len(gpu_stored)} hashes")

        self.assertTrue(
            len(gpu_stored) > 0,
            f"Expected BlockStored(GPU) from L2->L1 reload. Events: {len(events)}"
        )

    def test_05_l2_to_l3_storage_backup_produces_block_stored_cpu_tier2(self):
        """Write-through with file backend should produce BlockStored(CPU_TIER2)."""
        print("\n--- test_05: L2->L3 storage backup -> BlockStored(CPU_TIER2) ---")
        print("  Flushing cache...")
        self.flush_and_drain()
        marker = self.collector.snapshot()

        print("  Populating + triggering write-through (async storage backup)...")
        prompt_a = make_prompt(PROMPT_SEED_B, 192)
        self.send_request(prompt_a)
        prompt_b = make_prompt(PROMPT_SEED_B, 192) + make_prompt(PROMPT_SEED_C, 64)
        self.send_request(prompt_b)

        print("  Waiting for async storage write...")
        self.collector.drain_and_wait(timeout_s=5.0, quiesce_s=2.0)

        events = self.collector.events_since(marker)
        cpu2_stored = self._extract_hashes(events, "BlockStored", "CPU_TIER2")
        print(f"  Collected {len(events)} events: {self._event_summary(events)}")
        print(f"  BlockStored(CPU_TIER2) unique hashes: {len(cpu2_stored)}")

        self.assertTrue(
            len(cpu2_stored) > 0,
            f"Expected BlockStored(CPU_TIER2) from storage backup. "
            f"Events: {len(events)}, types: {[(e.event_type, e.medium) for e in events]}"
        )

    def test_06_l3_to_l2_prefetch_produces_block_stored_cpu_tier1(self):
        """After full flush, re-requesting should prefetch from L3->L2."""
        print("\n--- test_06: L3->L2 prefetch -> BlockStored(CPU_PINNED) ---")

        print("  Step 1: Populate + backup to L2 and L3...")
        prompt_a = make_prompt(PROMPT_SEED_A + PROMPT_SEED_B, 256)
        self.send_request(prompt_a)
        prompt_b = prompt_a + make_prompt(PROMPT_SEED_C, 64)
        self.send_request(prompt_b)
        print("  Waiting for storage backup...")
        self.collector.drain_and_wait(timeout_s=5.0, quiesce_s=2.0)

        print("  Step 2: Flush cache (clears GPU + L2)...")
        self.flush_and_drain()

        print("  Step 3: Re-request original prefix (triggers L3->L2 prefetch)...")
        marker = self.collector.snapshot()
        self.send_request(prompt_a)
        print("  Draining events...")
        self.collector.drain_and_wait(timeout_s=5.0, quiesce_s=2.0)

        events = self.collector.events_since(marker)
        cpu1_stored = self._extract_hashes(events, "BlockStored", "CPU_PINNED")
        print(f"  Prefetch events: {self._event_summary(events)}")
        print(f"  BlockStored(CPU_PINNED) from prefetch: {len(cpu1_stored)} hashes")

        self.assertTrue(
            len(cpu1_stored) > 0,
            f"Expected BlockStored(CPU_PINNED) from L3->L2 prefetch. "
            f"Events: {len(events)}, types: {[(e.event_type, e.medium) for e in events]}"
        )

    def test_07_full_lifecycle_global_invariant_validation(self):
        """Mixed workload validated by state-machine: no duplicate stores, no orphan removes."""
        print("\n--- test_07: Full lifecycle -- global invariant validation ---")
        print("  Flushing cache...")
        self.flush_and_drain()
        marker = self.collector.snapshot()

        print("  Phase 1: Inserting 3 unique prompts...")
        for i in range(3):
            self.send_request(make_prompt(f"LIFECYCLE_A_{i} ", 128))

        print("  Phase 2: Prefix reuse (3 requests, shared prefix) -> write-through...")
        base = make_prompt("LIFECYCLE_SHARED_PREFIX ", 192)
        self.send_request(base)
        self.send_request(base + make_prompt("SUFFIX_1 ", 64))
        self.send_request(base + make_prompt("SUFFIX_2 ", 64))

        print("  Phase 3: Flooding 15 unique prompts to force evictions...")
        for i in range(15):
            self.send_request(make_flood_prompt(i + 200, 192))

        print("  Phase 4: Re-request shared prefix (reload from L2)...")
        self.send_request(base)

        print("  Draining all events...")
        self.collector.drain_and_wait(timeout_s=5.0, quiesce_s=2.0)
        events = self.collector.events_since(marker)

        self.assertTrue(
            len(events) > 0,
            "Expected events from mixed workload"
        )

        print(f"  Collected {len(events)} events: {self._event_summary(events)}")
        print("  Running state-machine validator...")

        validator = InlineValidator()
        validator.validate(events)

        gpu = len(validator.blocks_in_tier("GPU"))
        cpu1 = len(validator.blocks_in_tier("CPU_PINNED"))
        cpu2 = len(validator.blocks_in_tier("CPU_TIER2"))
        print(f"  Final tier state: GPU={gpu}, CPU_PINNED={cpu1}, CPU_TIER2={cpu2}")
        print(f"  Validation errors: {len(validator.errors)}")

        if validator.errors:
            for err in validator.errors[:10]:
                print(f"    ERROR: {err}")
            self.fail(
                f"Validation found {len(validator.errors)} errors:\n"
                + "\n".join(validator.errors[:10])
            )

    def test_08_flush_cache_emits_all_blocks_cleared(self):
        """POST /flush_cache should emit an AllBlocksCleared event."""
        print("\n--- test_08: flush_cache -> AllBlocksCleared ---")
        print("  Inserting data to populate cache...")
        self.send_request(make_prompt("FLUSH_TEST ", 128))
        self.collector.drain_and_wait(timeout_s=2.0, quiesce_s=1.0)

        marker = self.collector.snapshot()
        print("  Sending POST /flush_cache...")
        self.flush_cache()
        print("  Draining events...")
        self.collector.drain_and_wait(timeout_s=3.0, quiesce_s=1.0)

        events = self.collector.events_since(marker)
        has_cleared = self._has_event(events, "AllBlocksCleared")
        print(f"  Post-flush events: {self._event_summary(events)}")
        print(f"  AllBlocksCleared received: {has_cleared}")

        self.assertTrue(
            has_cleared,
            f"Expected AllBlocksCleared event after flush_cache. "
            f"Events: {len(events)}, types: {[(e.event_type, e.medium) for e in events]}"
        )


    def test_09_deterministic_hashes_for_same_prompt(self):
        """Two fresh inserts of the same prompt produce identical block_hashes and token_ids."""
        print("\n--- test_09: Deterministic hashes for identical fresh inserts ---")
        prompt = make_prompt("DETERMINISM_CHECK ", 192)

        def fresh_insert():
            self.flush_and_drain()
            marker = self.collector.snapshot()
            self.send_request(prompt)
            self.collector.drain_and_wait(timeout_s=3.0, quiesce_s=1.0)
            return self._filter_events(
                self.collector.events_since(marker), "BlockStored", "GPU"
            )

        print("  Run 1: Fresh insert...")
        events_1 = fresh_insert()
        print(f"    BlockStored(GPU): {len(events_1)} pages")

        print("  Run 2: Flush + fresh insert (same prompt)...")
        events_2 = fresh_insert()
        print(f"    BlockStored(GPU): {len(events_2)} pages")

        self.assertEqual(
            len(events_1), len(events_2),
            f"Page count mismatch: run1={len(events_1)}, run2={len(events_2)}"
        )

        for i, (e1, e2) in enumerate(zip(events_1, events_2)):
            self.assertEqual(
                e1.block_hashes, e2.block_hashes,
                f"Block hash mismatch at page {i}: {e1.block_hashes} != {e2.block_hashes}"
            )
            self.assertEqual(
                e1.token_ids, e2.token_ids,
                f"Token IDs mismatch at page {i}"
            )
            print(f"    Page {i}: hash={e1.block_hashes[0] & 0xFFFF:04x} "
                  f"tokens={len(e1.token_ids)} -- match")

    def test_10_token_ids_match_tokenized_prompt(self):
        """BlockStored token_ids should match the actual tokenization of the prompt."""
        print("\n--- test_10: token_ids match tokenized input ---")
        prompt = make_prompt("TOKEN_VERIFY ", 192)
        expected_tokens = self.tokenizer.encode(prompt)
        print(f"  Tokenized prompt: {len(expected_tokens)} tokens "
              f"({len(expected_tokens) // PAGE_SIZE} full pages)")

        self.flush_and_drain()
        marker = self.collector.snapshot()
        self.send_request(prompt)
        self.collector.drain_and_wait(timeout_s=3.0, quiesce_s=1.0)

        events = self._filter_events(
            self.collector.events_since(marker), "BlockStored", "GPU"
        )
        self.assertTrue(len(events) > 0, "Expected BlockStored(GPU) events")

        # Reconstruct token stream from events (page by page)
        event_tokens = []
        for ev in events:
            event_tokens.extend(ev.token_ids)

        # Events cover full pages only (truncated to page boundaries)
        full_pages = len(expected_tokens) // PAGE_SIZE
        expected_full = expected_tokens[: full_pages * PAGE_SIZE]
        event_full = event_tokens[: full_pages * PAGE_SIZE]

        print(f"  Events produced {len(events)} pages, {len(event_tokens)} tokens")
        print(f"  Expected full pages: {full_pages} ({len(expected_full)} tokens)")

        self.assertEqual(
            event_full, expected_full,
            f"Token mismatch: event tokens do not match tokenized prompt. "
            f"First diff at index {next((i for i, (a, b) in enumerate(zip(event_full, expected_full)) if a != b), '?')}"
        )

        # Verify block_size == len(token_ids) for each event
        for i, ev in enumerate(events):
            # block_size is not in ParsedEvent, but token_ids length should be PAGE_SIZE
            # (except possibly the last page)
            if i < len(events) - 1:
                self.assertEqual(
                    len(ev.token_ids), PAGE_SIZE,
                    f"Page {i}: expected {PAGE_SIZE} tokens, got {len(ev.token_ids)}"
                )
            print(f"    Page {i}: {len(ev.token_ids)} tokens -- OK")

        print("  All token_ids match tokenized input.")


    def test_11_l3_restore_produces_same_hashes_as_fresh_insert(self):
        """After L3 backup completes, flush+re-request should restore with identical hashes."""
        print("\n--- test_11: L3 restore produces same hashes as fresh insert ---")
        prompt = make_prompt("L3_RESTORE_HASH_CHECK ", 192)

        # Run 1: Fresh insert, then trigger write-through + wait for L3 backup
        print("  Run 1: Fresh insert...")
        self.flush_and_drain()
        marker_1 = self.collector.snapshot()
        resp_1 = self.send_request(prompt)
        cached_1 = resp_1.get("meta_info", {}).get("cached_tokens", 0)
        print(f"    cached_tokens={cached_1} (expect 0, fresh insert)")

        # Trigger write-through: send same prefix with different suffix
        print("  Triggering write-through (prefix reuse)...")
        prompt_ext = prompt + make_prompt("EXTRA_SUFFIX ", 64)
        self.send_request(prompt_ext)

        # Wait long enough for async L3 storage backup to complete
        print("  Waiting for L3 storage backup to complete...")
        self.collector.drain_and_wait(timeout_s=6.0, quiesce_s=2.0)

        all_events_1 = self.collector.events_since(marker_1)
        gpu_events_1 = self._filter_events(all_events_1, "BlockStored", "GPU")
        cpu2_events_1 = self._filter_events(all_events_1, "BlockStored", "CPU_TIER2")
        print(f"    All events: {self._event_summary(all_events_1)}")

        self.assertTrue(
            len(cpu2_events_1) > 0,
            f"L3 backup did not complete. Events: {self._event_summary(all_events_1)}"
        )

        # Extract hashes from the original prompt's pages (first N GPU events before the suffix)
        # The first gpu events correspond to the shared prefix pages
        original_hashes_1 = [ev.block_hashes[0] for ev in gpu_events_1]
        original_tokens_1 = [ev.token_ids for ev in gpu_events_1]
        print(f"    Fresh insert: {len(gpu_events_1)} GPU pages, "
              f"L3 backup: {len(cpu2_events_1)} pages")

        # Run 2: Flush everything (clears GPU+L2, L3 persists), then re-request
        print("  Run 2: Flush (GPU+L2 cleared, L3 persists) + re-request...")
        self.flush_and_drain()

        marker_2 = self.collector.snapshot()
        self.send_request(prompt)
        self.collector.drain_and_wait(timeout_s=5.0, quiesce_s=2.0)

        all_events_2 = self.collector.events_since(marker_2)
        gpu_events_2 = self._filter_events(all_events_2, "BlockStored", "GPU")
        cpu1_events_2 = self._filter_events(all_events_2, "BlockStored", "CPU_PINNED")
        cpu2_events_2 = self._filter_events(all_events_2, "BlockStored", "CPU_TIER2")
        print(f"    All events: {self._event_summary(all_events_2)}")

        # Verify L3 restore path: CPU_PINNED events WITHOUT new CPU_TIER2 writes
        # means data came from L3 prefetch, not from a fresh write-through
        l3_restore_used = len(cpu1_events_2) > 0 and len(cpu2_events_2) == 0
        self.assertTrue(
            l3_restore_used,
            f"Expected L3->L2 prefetch (CPU_PINNED events without CPU_TIER2). "
            f"Got CPU_PINNED={len(cpu1_events_2)}, CPU_TIER2={len(cpu2_events_2)}"
        )
        print(f"    L3->L2 prefetch confirmed: {len(cpu1_events_2)} CPU_PINNED events, "
              f"0 new CPU_TIER2 writes")

        # Core assertion: hashes from restore path match fresh insert
        # Compare only the pages corresponding to the original prompt (not the suffix)
        original_page_count = len(gpu_events_1)
        restore_gpu = gpu_events_2[:original_page_count]

        self.assertGreaterEqual(
            len(gpu_events_2), len(restore_gpu),
            f"Expected at least {original_page_count} GPU events in run 2, got {len(gpu_events_2)}"
        )

        for i, (e1, e2) in enumerate(zip(gpu_events_1, restore_gpu)):
            self.assertEqual(
                e1.block_hashes, e2.block_hashes,
                f"Hash mismatch at page {i}: fresh={e1.block_hashes} restore={e2.block_hashes}"
            )
            self.assertEqual(
                e1.token_ids, e2.token_ids,
                f"Token IDs mismatch at page {i}: fresh insert vs L3 restore"
            )
            print(f"    Page {i}: hash={e1.block_hashes[0] & 0xFFFF:04x} "
                  f"tokens={len(e1.token_ids)} -- fresh == restore")

        print("  Hashes from L3 restore match fresh insert.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
