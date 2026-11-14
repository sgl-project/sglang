# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.moe.expert_pack import (  # noqa: E402
    ExpertPackStore,
    _CacheSlot,
)


def _make_mock_store(
    *,
    num_experts: int = 8,
    top_k: int = 2,
    num_slots: int = 4,
    resident_experts: list[int] | None = None,
) -> ExpertPackStore:
    store = object.__new__(ExpertPackStore)
    store._cache = torch.empty((num_slots, 1), dtype=torch.uint8)
    store._transfer_stream = SimpleNamespace(wait_event=lambda event: None)
    store._read_executor = SimpleNamespace()
    store.active_moe_layer_ids = frozenset({0})
    store.header = SimpleNamespace(num_experts=num_experts, top_k=top_k)
    store._lock = threading.Lock()
    store._key_frequency = {}
    store._active_keys = set()
    store._route_calls_by_layer = {0: 0}
    store._route_tokens_by_layer = {0: 0}
    store._cache_slots = [_CacheSlot() for _ in range(num_slots)]
    store._key_to_slot = {}
    store._lru = OrderedDict()
    store._staging = [torch.empty(1, dtype=torch.uint8)]
    store._stage_events = [None]
    store._stage_cursor = 0
    store.object_payload_bytes = 64
    store.stats = {
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_evictions": 0,
        "h2d_bytes": 0,
        "io_errors": 0,
        "pack_read_ns": 0,
        "resident_experts": 0,
        "resident_bytes": 0,
    }
    store.entries = {
        (0, expert, 0): SimpleNamespace(generation=100 + expert)
        for expert in range(num_experts)
    }
    store._object_read_ranges = lambda: ()
    store._install_staging = lambda staging, slot_index, stream: None

    for slot_index, expert in enumerate(resident_experts or []):
        key = (0, expert)
        slot = store._cache_slots[slot_index]
        slot.key = key
        slot.generation = 100 + expert
        slot.frequency = 1
        store._key_to_slot[key] = slot_index
        store._lru[key] = None
        store._key_frequency[key] = 1

    return store


class TestExpertPackAcquire(unittest.TestCase):
    def test_acquire_maps_duplicate_routes_and_updates_stats(self):
        store = _make_mock_store(
            num_experts=8,
            top_k=2,
            num_slots=4,
            resident_experts=[1, 3, 5],
        )
        topk_ids = torch.tensor(
            [[5, 1], [3, 5], [1, 3], [5, 3]],
            dtype=torch.int64,
        )
        fake_stream = SimpleNamespace(wait_event=lambda event: None)
        with patch("torch.cuda.current_stream", return_value=fake_stream):
            slot_tensor, slots = store.acquire(0, topk_ids)

        expected_slots = [2, 0, 1, 2, 0, 1, 2, 1]
        self.assertEqual(slots, expected_slots)
        self.assertEqual(slot_tensor.tolist(), expected_slots)
        self.assertEqual(slot_tensor.dtype, torch.int32)
        self.assertEqual(store.stats["cache_hits"], 3)
        self.assertEqual(store.stats["cache_misses"], 0)
        self.assertEqual(store._route_calls_by_layer[0], 1)
        self.assertEqual(store._route_tokens_by_layer[0], 4)
        self.assertEqual(
            list(store._lru.keys()),
            [(0, 1), (0, 3), (0, 5)],
        )

    def test_acquire_rejects_out_of_range_expert_ids(self):
        store = _make_mock_store(num_experts=4, top_k=2, num_slots=4)
        for invalid in (
            torch.tensor([[0, -1]], dtype=torch.int64),
            torch.tensor([[0, 4]], dtype=torch.int64),
        ):
            with self.assertRaisesRegex(
                ValueError, "route contains an out-of-range expert id"
            ):
                store.acquire(0, invalid)


if __name__ == "__main__":
    unittest.main()
