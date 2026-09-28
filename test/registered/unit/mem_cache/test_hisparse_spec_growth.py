"""Actual eager adapter + ordinary coordinator growth; CPU allocation/copy leaves."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.hisparse_spec_state import UnionCapacityError, VerifyRow
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Load the neighboring real fixture module under direct-file, unittest and
# pytest discovery without relying on the caller's test-directory PYTHONPATH.
_spec = importlib.util.spec_from_file_location(
    "hisparse_growth_coordinator_fixtures",
    Path(__file__).with_name("test_hisparse_spec_coordinator.py"),
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


def setup():
    fixture = fixtures.TestCoordinator()
    fixture.setUp()
    fixture.complete()
    assert fixture.m.retire(fixture.req, fixture.key, cancel=True)
    c = fixture.c
    c.device_buffer_size = 256
    c.padded_buffer_size = 320
    c.req_to_device_buffer = torch.zeros((1, 320), dtype=torch.int32)
    c.req_to_device_buffer[0, :64] = torch.arange(64, 128)
    c.req_device_buffer_tokens = torch.arange(256).view(1, 1, 256).repeat(2, 1, 1)
    c.req_device_buffer_token_locs = torch.zeros((2, 1, 320), dtype=torch.int32)
    c.req_device_buffer_token_locs[:, :, :64] = torch.arange(64, 128)
    c.lru_slots = torch.arange(256).view(1, 1, 256).repeat(2, 1, 1)
    c.req_to_host_pool = torch.arange(128, 640).view(1, 512)
    c._grow_device_buffers = HiSparseCoordinator._grow_device_buffers.__get__(c)
    fixture.m.teardown(fixture.req)
    fixture.m = type(fixture.m)(c, fixture.dm)
    return fixture


class TestGrowth(unittest.TestCase):
    def test_cross_page_new_committed_position_is_host_miss(self):
        f = setup()
        key = f.m.prepare(f.req, 65, (1000, 1001, 1002, 1003), 4)
        f.key = key
        table = f.m.stage_layer(f.req, key, 0, [VerifyRow(key, 65, tuple(range(65)))])
        plan = f.m.owners[0].tables[0][1]
        self.assertEqual(plan.miss_src, (192,))
        self.assertEqual(table.shape, (1, 65))
        self.assertEqual(f.c.req_device_buffer_size.tolist(), [128])
        self.assertEqual(f.c.req_device_buffer_tokens[1, 0, 64:128].tolist(), [-1] * 64)
        self.assertEqual(
            f.c.req_device_buffer_token_locs[0, 0, :64].tolist(), list(range(64, 128))
        )

    def test_multitoken_jump_and_reserved_page(self):
        for length, capacity in [(64, 64), (129, 192), (256, 320), (300, 320)]:
            with self.subTest(length=length):
                f = setup()
                f.m.prepare(f.req, length, (1000, 1001, 1002, 1003), 4)
                self.assertEqual(int(f.c.req_device_buffer_size[0]), capacity)
                self.assertEqual(
                    f.c.req_device_buffer_tokens[0, 0, :64].tolist(), list(range(64))
                )
                if capacity > 64:
                    self.assertTrue(
                        torch.all(
                            f.c.req_device_buffer_tokens[:, :, 64 : min(capacity, 256)]
                            == -1
                        )
                    )
                owned = set(f.m.owners[0].arena.page_ids)
                hot = f.c.req_to_device_buffer[0, :capacity]
                self.assertFalse(owned & set((hot // 64).tolist()))
                self.assertEqual(len(set(hot.tolist())), capacity)
                f.m.teardown(f.req)
                f.allocator.hisparse_attn_allocator.free(hot)
                self.assertEqual(
                    f.allocator.hisparse_attn_allocator.available_size(), 1024
                )

    def test_allocator_failure_never_binds_or_leaks_arena(self):
        f = setup()
        free = f.allocator.hisparse_attn_allocator.available_size()
        with patch.object(
            f.allocator.hisparse_attn_allocator, "alloc", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "grow_device_buffers failed"):
                f.m.prepare(f.req, 65, (1000, 1001, 1002, 1003), 4)
        self.assertEqual(f.allocator.hisparse_attn_allocator.available_size(), free)
        self.assertEqual(int(f.c.req_device_buffer_size[0]), 64)
        self.assertIsNone(f.m.owners[0].key)
        self.assertFalse(f.m.backend._bindings)
        f.m.prepare(f.req, 65, (1000, 1001, 1002, 1003), 4)

    def test_arena_failure_retains_request_owned_growth(self):
        f = setup()
        allocator = f.allocator.hisparse_attn_allocator
        original = allocator.alloc
        calls = []

        def allocate(size):
            calls.append(size)
            return original(size) if len(calls) == 1 else None

        with patch.object(allocator, "alloc", side_effect=allocate):
            with self.assertRaisesRegex(
                MemoryError, "provisional physical page allocation failed"
            ):
                f.m.prepare(f.req, 65, (1000, 1001, 1002, 1003), 4)
        self.assertEqual(int(f.c.req_device_buffer_size[0]), 128)
        self.assertEqual(allocator.available_size(), 1024 - 128)
        self.assertFalse(f.m.backend._bindings)
        f.m.teardown(f.req)
        allocator.free(f.c.req_to_device_buffer[0, :128])
        self.assertEqual(allocator.available_size(), 1024)

    def test_prior_transaction_and_stale_owner_refuse_growth(self):
        f = setup()
        key = f.m.prepare(f.req, 64, (1000, 1001, 1002, 1003), 4)
        with self.assertRaisesRegex(ValueError, "previous verifier readers"):
            f.m.prepare(f.req, 65, (1000, 1001, 1002, 1003), 4)
        stale = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=0), hisparse_staging=False
        )
        with self.assertRaisesRegex(ValueError, "slot reused"):
            f.m.prepare(stale, 65, (1000, 1001, 1002, 1003), 4)
        self.assertEqual(int(f.c.req_device_buffer_size[0]), 64)
        self.assertEqual(f.m.owners[0].key, key)

    def test_true_oversized_union_still_fails(self):
        f = setup()
        key = f.m.prepare(f.req, 300, (1000, 1001, 1002, 1003), 4)
        with self.assertRaises(UnionCapacityError):
            f.m.stage_layer(f.req, key, 0, [VerifyRow(key, 300, tuple(range(257)))])
        f.c._copy_speculative_union.assert_not_called()

    def test_actual_request_teardown_reclaims_grown_pages_once(self):
        from unittest.mock import MagicMock

        f = setup()
        c = f.c
        ids = f.allocator.logical_attn_allocator.alloc(128)
        logical_free = f.allocator.logical_attn_allocator.available_size()
        f.req.kv.kv_allocated_len = 128
        c.req_to_token_pool = SimpleNamespace(req_to_token=ids.view(1, -1))
        c.mem_pool_device.translate_loc_from_full_to_compressed = lambda x: x
        c.mem_pool_device.full_to_hisparse_device_index_mapping = (
            f.allocator.full_to_hisparse_device_index_mapping
        )
        c._speculative_verifier = f.m
        c._speculative_teardown = HiSparseCoordinator._speculative_teardown.__get__(c)
        c.wait_for_pending_backup = MagicMock()
        c._lru_init = torch.arange(256)
        c._skip_first_backup = torch.tensor([False])
        f.host.allocated_host_indices.return_value = torch.arange(128, 256)
        f.m.prepare(f.req, 65, (1000, 1001, 1002, 1003), 4)
        HiSparseCoordinator.request_finished(c, f.req)
        self.assertEqual(f.allocator.hisparse_attn_allocator.available_size(), 1024)
        self.assertEqual(int(c.req_device_buffer_size[0]), 0)
        self.assertEqual(
            f.allocator.logical_attn_allocator.available_size(), logical_free
        )
        f.allocator.get_kvcache()._translate_loc_to_hisparse_device.side_effect = (
            lambda x: f.allocator.full_to_hisparse_device_index_mapping[x]
        )
        f.allocator.free(ids)
        self.assertEqual(f.allocator.hisparse_attn_allocator.available_size(), 1024)
        self.assertEqual(
            f.allocator.logical_attn_allocator.available_size(), logical_free + 128
        )


if __name__ == "__main__":
    unittest.main()
