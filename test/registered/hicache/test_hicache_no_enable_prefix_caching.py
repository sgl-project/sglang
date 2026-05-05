"""
Test HiCache behavior with the --no-enable-prefix-caching flag (disables GPU L1 reuse, but allows CPU L2 reuse).

The test verifies that prefix blocks are
(1) still written through to CPU L2, but are not reused from GPU L1,
(2) but instead loaded back from CPU L2 when needed, resulting in expected cache hit counts and KV event patterns.

Usage:
    python3 test/registered/hicache/test_hicache_no_enable_prefix_caching.py
"""

"""
Test scenario outline in steps:

1. Warmup:
  put prefix in CPU L2 and demote GPU L1

2. Request A:
  same prefix
  load CPU L2 -> GPU
  keep request running

3. Request B:
  arrives while Request A still holds the prefix in GPU

Expected strict behavior:
  Request B must not reuse Request A's GPU L1-loaded prefix
  It must either:
    load its own L2 copy, or
    wait until A releases/demotes and then load from L2
"""

import json
import socket
import time
import unittest
from collections import Counter
from typing import Callable, Iterable, List, Set

import msgspec
import requests
import zmq

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
)
from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=360, suite="stage-b-test-1-gpu-small-amd")


GPU = "GPU"
CPU = "CPU_PINNED"

PAGE_SIZE = 64
BASE_PREFIX_TOKENS = 768
MIN_CACHE_HIT_TOKENS = 512
MIN_SHARED_BLOCKS = 4


def _get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _medium_value(medium) -> str:
    if medium is None:
        return ""
    if hasattr(medium, "value"):
        return str(medium.value)
    return str(medium)


class KVEventSubscriber:
    def __init__(self, port: int):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(f"tcp://127.0.0.1:{port}")

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def close(self):
        self.poller.unregister(self.sock)
        self.sock.close(linger=0)

    def drain(self, timeout: float = 0.2) -> List[KVCacheEvent]:
        """Drain all currently available KV events until timeout expires."""
        out = []
        deadline = time.time() + timeout

        while time.time() < deadline:
            poll_ms = max(1, int((deadline - time.time()) * 1000))
            ready = dict(self.poller.poll(poll_ms))
            if self.sock not in ready:
                break

            parts = self.sock.recv_multipart()
            # ZmqEventPublisher sends: topic, seq_bytes, msgpack_payload.
            if len(parts) != 3:
                continue

            _topic, _seq_bytes, payload = parts
            batch = self.decoder.decode(payload)
            out.extend(batch.events)

        return out

    def drain_until(
        self,
        predicate: Callable[[List[KVCacheEvent]], bool],
        timeout: float = 30.0,
        step: float = 0.2,
    ) -> List[KVCacheEvent]:
        out = []
        deadline = time.time() + timeout

        while time.time() < deadline:
            out.extend(self.drain(timeout=step))
            if predicate(out):
                return out

        return out


def _stored_hashes(events: Iterable[KVCacheEvent], medium: str) -> Set[int]:
    hashes = set()
    for ev in events:
        if isinstance(ev, BlockStored) and _medium_value(ev.medium) == medium:
            hashes.update(int(h) for h in ev.block_hashes)
    return hashes


def _removed_hashes(events: Iterable[KVCacheEvent], medium: str) -> Set[int]:
    hashes = set()
    for ev in events:
        if isinstance(ev, BlockRemoved) and _medium_value(ev.medium) == medium:
            hashes.update(int(h) for h in ev.block_hashes)
    return hashes


def _gpu_store_counts(
    events: Iterable[KVCacheEvent],
    candidates: Set[int],
) -> Counter:
    counts = Counter()
    for ev in events:
        if isinstance(ev, BlockStored) and _medium_value(ev.medium) == GPU:
            for h in ev.block_hashes:
                h = int(h)
                if h in candidates:
                    counts[h] += 1
    return counts


def _cpu_backed_and_gpu_demoted_hashes(events: Iterable[KVCacheEvent]) -> Set[int]:
    """Hashes that were stored to CPU L2 and then removed from GPU L1."""
    return _stored_hashes(events, CPU) & _removed_hashes(events, GPU)


class TestHiCacheNoEnablePrefixCaching(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.model = "/images/nixl/models/Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = get_tokenizer(cls.model)

        cls.kv_event_port = _get_free_tcp_port()
        kv_events_config = {
            "publisher": "zmq",
            # Publisher binds wildcard endpoints; subscriber connects to 127.0.0.1.
            "endpoint": f"tcp://*:{cls.kv_event_port}",
            "topic": "",
            "buffer_steps": 10000,
            "hwm": 100000,
            "max_queue_size": 100000,
        }

        other_args = [
            "--enable-hierarchical-cache",
            "--hicache-write-policy",
            "write_through",
            "--hicache-size",
            str(200 if is_hip() else 100),
            "--page-size",
            str(PAGE_SIZE),
            "--enable-cache-report",
            "--max-running-requests",
            "8",
            "--log-level",
            "debug",
            "--kv-events-config",
            json.dumps(kv_events_config),
            # This is the new flag under test.
            "--no-enable-prefix-caching",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        cls.events = KVEventSubscriber(cls.kv_event_port)

        # Avoid ZMQ slow-joiner loss before the first request in this test file.
        time.sleep(0.5)
        cls.events.drain(timeout=0.5)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "events"):
            cls.events.close()
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def _make_shared_prefix(self, min_tokens: int = BASE_PREFIX_TOKENS) -> str:
        seed = (
            "This is a deterministic shared prefix for HiCache L2 load-back "
            "testing. It is intentionally repetitive and contains no special "
            "control instructions. "
        )
        text = seed
        while True:
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                ids = self.tokenizer.encode(text)

            if len(ids) >= min_tokens:
                return self.tokenizer.decode(ids[:min_tokens]) + "\n\n"

            text += seed

    def _send_generate(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        timeout: float = 120.0,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=timeout,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def _cached_tokens(self, response_json: dict) -> int:
        return int(response_json.get("meta_info", {}).get("cached_tokens", 0))

    def _assert_server_process_alive(self):
        rc = self.process.poll()
        self.assertIsNone(
            rc,
            f"SGLang server process exited unexpectedly with code {rc}. "
            "Check scheduler logs above for memory-pool checker failures.",
        )

    def _assert_scheduler_healthy(self):
        self._assert_server_process_alive()
        try:
            response = requests.get(
                f"{self.base_url}/health_generate",
                timeout=30,
            )
        except Exception as exc:
            self.fail(f"SGLang scheduler is not reachable after workload: {exc}")

        self.assertEqual(
            response.status_code,
            200,
            f"health_generate failed: {response.status_code} {response.text}",
        )

    def _assert_no_idle_pool_checker_crash(self, idle_seconds: float = 3.0):
        deadline = time.time() + idle_seconds
        while time.time() < deadline:
            self._assert_server_process_alive()
            time.sleep(0.2)

        self._assert_scheduler_healthy()

    def test_repeated_prefix_reuses_l2_not_l1(self):
        shared_prefix = self._make_shared_prefix()

        # Request 1: populate the tree, back up prefix blocks to CPU L2,
        # then the new mode should demote those backed-up blocks from GPU L1.
        response1 = self._send_generate(
            shared_prefix + "Question A: answer with one short sentence.",
            max_new_tokens=8,
        )
        self.assertIsNotNone(response1)

        cached1 = self._cached_tokens(response1)

        events_after_first = self.events.drain_until(
            lambda evs: len(_cpu_backed_and_gpu_demoted_hashes(evs))
            >= MIN_SHARED_BLOCKS,
            timeout=30.0,
        )

        demoted_hashes = _cpu_backed_and_gpu_demoted_hashes(events_after_first)

        print(
            f"Request 1: \n\tcached_tokens={cached1}, \n\tdemoted_hashes={demoted_hashes}"
        )

        meta1 = response1.get("meta_info", {})
        print(f"\nRequest 1 meta_info: {meta1}")

        print(
            f"\nRequest 1 cached tokens (should be None): \n\t{meta1.get('cached_tokens_details', {})}"
        )

        self.assertGreaterEqual(
            len(demoted_hashes),
            MIN_SHARED_BLOCKS,
            "Expected the first request to write shared prefix blocks to CPU L2 "
            "and remove their GPU L1 copies. If this fails, the new flag is not "
            "demoting L1-backed prefix nodes after write-through.",
        )

        # Drop any leftover events so Request 2 assertions only inspect the
        # second request's load-back path.
        self.events.drain(timeout=0.5)

        # Request 2: same prefix should be reusable, but since the GPU copy was
        # demoted, the reuse must come through CPU L2 -> GPU load-back.
        response2 = self._send_generate(
            shared_prefix + "Question B: answer with one short sentence.",
            max_new_tokens=8,
        )
        cached2 = self._cached_tokens(response2)

        events_after_second = self.events.drain_until(
            lambda evs: len(_stored_hashes(evs, GPU) & demoted_hashes)
            >= MIN_SHARED_BLOCKS,
            timeout=30.0,
        )
        loaded_back_hashes = _stored_hashes(events_after_second, GPU) & demoted_hashes

        # check: cached2 >= MIN_CACHE_HIT_TOKENS, which indicates a large prefix hit from CPU L2, but not GPU L1.
        print(
            f"Request 2: \n\tcached_tokens={cached2}, \n\tloaded_back_hashes={loaded_back_hashes}, \n\tdemoted_hashes={demoted_hashes}"
        )

        self.assertGreaterEqual(
            cached2,
            MIN_CACHE_HIT_TOKENS,
            f"Expected a large repeated-prefix cache hit from CPU L2, got "
            f"cached_tokens={cached2}.",
        )

        # check more granular cache hit details: evidence of L2 hits and no L1 hits.
        meta2 = response2.get("meta_info", {})
        cached_tokens_host = meta2.get("cached_tokens_details", {}).get("host")
        cached_tokens_device = meta2.get("cached_tokens_details", {}).get("device")

        print(f"\nRequest 2 meta_info: {meta2}")
        print(
            f"\nRequest 2 cached tokens: \n\tcache_tokens_host={cached_tokens_host}, \n\tcache_tokens_device={cached_tokens_device}"
        )

        if cached_tokens_host is not None:
            self.assertGreaterEqual(
                int(cached_tokens_host),
                MIN_CACHE_HIT_TOKENS,
                f"Expected host/L2 cached tokens, meta_info={meta2}",
            )

        if cached_tokens_device is not None:
            self.assertEqual(
                int(cached_tokens_device),
                0,
                f"Expected no device/L1 cached tokens, meta_info={meta2}",
            )

        # check if the prefix blocks were loaded back from CPU L2 to GPU, rather than reused from GPU L1.
        self.assertGreaterEqual(
            len(loaded_back_hashes),
            MIN_SHARED_BLOCKS,
            "Expected Request 2 to load previously-demoted CPU L2 prefix blocks "
            "back to GPU. If cached_tokens is high but no matching GPU store "
            "event appears, the request likely reused an L1 GPU block instead "
            "of exercising L2 load-back.",
        )

        # Important: the scheduler may crash only after the response, when the
        # runtime checker runs on idle. Without this, unittest can print OK
        # even though the child server died.
        self._assert_no_idle_pool_checker_crash()


if __name__ == "__main__":
    unittest.main()
