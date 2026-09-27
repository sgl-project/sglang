# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_spec_lifecycle import (
    CompletionFence,
    HostReservation,
    SpeculativeKVLifecycle,
)
from sglang.srt.mem_cache.hisparse_spec_state import SpecTxnKey
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin


class Event:
    def __init__(self, done=False):
        self.done = done

    def record(self):
        pass

    def query(self):
        return self.done


class Backend:
    def __init__(self):
        self.live = set()
        self.published = []
        self.event = Event()
        self.fail = None

    def allocate(self, key, positions):
        count = len(positions)
        if self.fail == "allocate":
            return None
        ids = tuple(range(count))
        self.live.update(ids)
        return HostReservation(key, positions, ids, ids)

    def copy(self, plan):
        if self.fail == "copy":
            raise RuntimeError("copy failed before submission")
        self.plan = plan
        fence = CompletionFence(self.event)
        fence.record()
        return fence

    def publish(self, plan, reservation):
        self.published.append(plan)

    def release(self, reservation):
        for i in reservation.new_page_rows:
            self.live.remove(i)


class PageHostBackend(Backend, HiSparseHostPoolMixin):
    """Real host mixin allocation on an unpublished per-request snapshot."""

    page_size = 64

    def __init__(self):
        super().__init__()
        self.mapping = torch.full((1, 256), -1, dtype=torch.int64)
        self.mapping[0, :64] = torch.arange(320, 384)
        self.allocated = torch.tensor([64])
        self.next_page = 9
        self.pending = {}
        self.live = set(range(320, 384))

    def alloc(self, count):
        rows = torch.arange(self.next_page * 64, self.next_page * 64 + count)
        self.next_page += count // 64
        self.live.update(rows.tolist())
        return rows

    def allocate(self, key, positions):
        shadow = self.mapping.clone()
        allocated = self.allocated.clone()
        prior = int(allocated[0])
        ids = self.alloc_paged_token_slots(
            shadow, allocated, 0, positions[0], len(positions)
        )
        reservation = HostReservation(
            key,
            positions,
            tuple(ids.tolist()),
            tuple(shadow[0, prior : int(allocated[0])].tolist()),
        )
        self.pending[key] = (shadow, allocated)
        return reservation

    def publish(self, plan, reservation):
        shadow, allocated = self.pending.pop(plan.key)
        self.mapping.copy_(shadow)
        self.allocated.copy_(allocated)
        super().publish(plan, reservation)

    def release(self, reservation):
        self.pending.pop(reservation.key)
        super().release(reservation)


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        self.allocator = HiSparseTokenToKVPoolAllocator(
            1024, 64, torch.float32, "cpu", MagicMock(), False, 4
        )
        # Real noncontiguous physical allocator page order, not a fake allocator.
        self.allocator.hisparse_attn_allocator.free_pages = torch.tensor(
            [3, 7, 2, 9, 1, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
        )
        self.backend = Backend()
        self.manager = SpeculativeKVLifecycle(self.allocator, self.backend)
        self.mapping = self.allocator.full_to_hisparse_device_index_mapping
        self.allocator._kvcache._translate_loc_to_hisparse_device.side_effect = (
            lambda ids: self.mapping[ids]
        )

    def test_boundaries_rounding_and_rearm(self):
        iteration = 0
        for prefix in (63, 64, 65, 127):
            for reserve in (1, 63, 64, 65, 129):
                for q in (1, 2, 4):
                    if q > reserve:
                        continue
                    iteration += 1
                    key = SpecTxnKey(0, 0, iteration)
                    ids = tuple(range(64 + prefix, 64 + prefix + q))
                    before = self.allocator.logical_attn_allocator.available_size()
                    arena = self.manager.begin(key, prefix, ids, reserve)
                    self.assertEqual(len(arena.page_ids), (reserve + 63) // 64)
                    self.assertTrue(self.manager.cancel(key))
                    self.assertTrue(torch.all(self.mapping[list(ids)] == 0))
                    self.assertEqual(
                        self.allocator.logical_attn_allocator.available_size(), before
                    )
                    self.assertEqual(
                        self.allocator.hisparse_attn_allocator.available_size(), 1024
                    )

    def test_fences_acceptance_and_generation(self):
        key = SpecTxnKey(0, 2, 0)
        arena = self.manager.begin(key, 63, (127, 128, 129, 130), 65)
        reader = Event()
        fence = CompletionFence(reader)
        fence.record()
        self.manager.add_reader(key, fence)
        self.manager.verified(key)
        plan = self.manager.commit(key, 2)
        self.assertEqual(plan.positions, (63, 64))
        self.assertEqual(plan.logical_ids, (127, 128))
        self.assertFalse(self.manager.release(key))
        self.assertEqual(self.backend.published, [])
        self.backend.event.done = True
        self.assertFalse(self.manager.poll(key))
        self.assertEqual(self.backend.published, [plan])
        reader.done = True
        self.assertTrue(self.manager.cancel(key))
        self.assertEqual(self.backend.live, {0, 1})  # Published host KV survives.
        newer = SpecTxnKey(0, 3, 0)
        self.manager.begin(newer, 65, (127, 128), 2)
        with self.assertRaises(ValueError):
            self.manager.cancel(key)
        with self.assertRaises(ValueError):
            self.allocator.retire_provisional_mapping(arena, (127, 128, 129, 130))
        self.manager.cancel(newer)

    def test_failed_setup_and_generic_free_protection(self):
        self.mapping[127] = 999
        before = self.mapping.clone()
        with self.assertRaises(ValueError):
            self.manager.begin(SpecTxnKey(0, 0, 0), 63, (127, 128), 64)
        self.assertTrue(torch.equal(before, self.mapping))
        self.mapping[127] = 0
        key = SpecTxnKey(0, 0, 0)
        self.manager.begin(key, 63, (127, 128), 64)
        for ids in ([128], [129], [126]):
            with self.assertRaises(ValueError):
                self.allocator.free(torch.tensor(ids))
        with self.assertRaises(ValueError):
            self.allocator.free_hisparse(torch.tensor([127]))
        with self.assertRaises(ValueError):
            self.allocator.clear()
        self.assertEqual(self.allocator.logical_attn_allocator.available_size(), 4096)
        self.manager.cancel(key)

    def test_copy_and_allocation_failure(self):
        for iteration, fail in enumerate(("allocate", "copy")):
            key = SpecTxnKey(0, 0, iteration)
            self.manager.begin(key, 63, (127, 128), 64)
            self.manager.verified(key)
            self.backend.fail = fail
            with self.assertRaises((MemoryError, RuntimeError)):
                self.manager.commit(key, 1)
            self.assertEqual(self.backend.live, set())
            self.assertEqual(
                self.allocator.hisparse_attn_allocator.available_size(), 1024
            )

    def test_cancel_during_backup(self):
        key = SpecTxnKey(0, 0, 0)
        self.manager.begin(key, 127, (191, 192), 2)
        self.manager.verified(key)
        self.manager.commit(key, 1)
        self.assertFalse(self.manager.cancel(key))
        self.backend.event.done = True
        self.assertTrue(self.manager.poll(key))
        self.assertEqual(self.backend.live, set())
        self.assertEqual(self.backend.published, [])

    def test_unarmed_reader_and_publish_failure(self):
        key = SpecTxnKey(0, 0, 0)
        self.manager.begin(key, 63, (127, 128), 64)
        reader = CompletionFence(Event(True))
        self.manager.add_reader(key, reader)
        self.manager.verified(key)
        self.backend.event.done = True
        self.manager.commit(key, 1)
        original = self.backend.publish
        self.backend.publish = MagicMock(side_effect=RuntimeError("atomic failure"))
        with self.assertRaises(RuntimeError):
            self.manager.release(key)
        self.backend.publish = original
        self.assertFalse(self.manager.poll(key))
        reader.record()
        self.assertTrue(self.manager.poll(key))

    def test_allocation_exception_and_alias_corruption(self):
        key = SpecTxnKey(0, 0, 0)
        arena = self.manager.begin(key, 63, (127, 128), 64)
        self.mapping[200] = arena.position_slots[0][1]
        with self.assertRaises(ValueError):
            self.manager.cancel(key)
        self.mapping[200] = 0
        self.assertTrue(self.manager.poll(key))
        key = SpecTxnKey(0, 0, 1)
        self.manager.begin(key, 63, (127, 128), 64)
        self.manager.verified(key)
        self.backend.allocate = MagicMock(side_effect=MemoryError("allocation"))
        with self.assertRaises(MemoryError):
            self.manager.commit(key, 1)
        self.assertEqual(self.allocator.hisparse_attn_allocator.available_size(), 1024)

    def test_real_host_page_tail_reservation(self):
        backend = PageHostBackend()
        self.manager.backend = backend
        # Cancellation across boundary frees new page, never old committed tail.
        key = SpecTxnKey(0, 0, 0)
        self.manager.begin(key, 63, (127, 128), 64)
        self.manager.verified(key)
        plan = self.manager.commit(key, 2)
        self.assertEqual(plan.host_ids, (383, 576))
        self.assertEqual(int(backend.allocated[0]), 64)
        self.assertFalse(self.manager.cancel(key))
        backend.event.done = True
        self.assertTrue(self.manager.poll(key))
        self.assertEqual(backend.live, set(range(320, 384)))
        # Commit publishes full new page for reuse; next commit allocates none.
        for iteration, old_len in ((1, 63), (2, 65)):
            key = SpecTxnKey(0, 0, iteration)
            self.manager.begin(key, old_len, (127, 128), 64)
            self.manager.verified(key)
            self.manager.commit(key, 2)
            self.assertTrue(self.manager.release(key))
            self.assertEqual(int(backend.allocated[0]), 128)
            self.assertEqual(len(backend.live), 128)

    def test_invalid_copy_fence_quarantines_ownership(self):
        key = SpecTxnKey(0, 0, 0)
        self.manager.begin(key, 63, (127, 128), 64)
        self.manager.verified(key)
        self.backend.copy = MagicMock(return_value=Event(True))
        with self.assertRaises(TypeError):
            self.manager.commit(key, 2)
        self.assertFalse(self.manager.cancel(key))
        self.assertFalse(self.manager.poll(key))
        self.assertEqual(self.backend.live, {0, 1})
        self.assertEqual(self.allocator.hisparse_attn_allocator.available_size(), 960)
        with self.assertRaises(ValueError):
            self.manager.resolve_quarantine(key, Event(True))
        drain = CompletionFence(Event(True))
        self.assertFalse(self.manager.resolve_quarantine(key, drain))
        self.assertEqual(self.backend.live, {0, 1})
        drain.record()
        self.assertTrue(self.manager.poll(key))
        self.assertEqual(self.backend.live, set())
        self.assertEqual(self.backend.published, [])
        self.assertEqual(self.allocator.hisparse_attn_allocator.available_size(), 1024)

    def test_foreign_host_reservation_is_never_released(self):
        key = SpecTxnKey(0, 0, 0)
        foreign = HostReservation(
            SpecTxnKey(1, 0, 0), (63,), (320,), tuple(range(320, 384))
        )
        self.manager.begin(key, 63, (127, 128), 64)
        self.manager.verified(key)
        self.backend.live.update(foreign.new_page_rows)
        self.backend.allocate = lambda _key, _positions: foreign
        self.backend.release = MagicMock()
        with self.assertRaisesRegex(ValueError, "another transaction"):
            self.manager.commit(key, 1)
        self.backend.release.assert_not_called()
        self.assertEqual(self.backend.live, set(foreign.new_page_rows))
        self.assertEqual(self.allocator.hisparse_attn_allocator.available_size(), 1024)

    def test_1000_bounded_random_transactions(self):
        rng = random.Random(51)
        for iteration in range(1000):
            self.backend = Backend()
            self.manager.backend = self.backend
            key = SpecTxnKey(0, iteration // 20, iteration % 20)
            q = rng.choice((2, 4))
            reserve = rng.choice((63, 64, 65, 129))
            prefix = rng.choice((63, 64, 65, 127))
            ids = tuple(range(64 + prefix, 64 + prefix + q))
            self.manager.begin(key, prefix, ids, reserve)
            reader = Event()
            fence = CompletionFence(reader)
            fence.record()
            self.manager.add_reader(key, fence)
            if rng.randrange(2):
                self.manager.verified(key)
                self.manager.commit(key, rng.randint(1, q))
            self.assertFalse(self.manager.cancel(key))
            reader.done = self.backend.event.done = True
            self.assertTrue(self.manager.poll(key))
            self.assertEqual(self.backend.live, set())
            self.assertEqual(
                self.allocator.hisparse_attn_allocator.available_size(), 1024
            )
            self.assertTrue(torch.all(self.mapping[:-1] == 0))


if __name__ == "__main__":
    unittest.main()
