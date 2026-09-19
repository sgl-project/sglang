"""Unit tests for the minimal HiSparse spec coordinator lifecycle."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseSpecSwapManager,
)
from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiSparseSpecCoordinator(CustomTestCase):
    def test_spec_decode_reserve_allocates_logical_pages_only(self):
        logical_locs = torch.tensor([64, 65, 66, 67], dtype=torch.int64)
        allocator = SimpleNamespace(
            page_size=64,
            alloc_logical_only=MagicMock(return_value=logical_locs),
        )
        tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        req_to_token = torch.full((1, 16), -1, dtype=torch.int64)
        req_to_token[0, 1] = 63
        req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=2))
        batch = SimpleNamespace(
            device=torch.device("cpu"),
            hisparse_coordinator=object(),
        )

        def assign_on_cpu(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            batch_size,
        ):
            del end_offset, batch_size
            start = int(start_offset[0])
            req_to_token[
                int(req_pool_indices[0]), start : start + out_cache_loc.numel()
            ] = out_cache_loc

        with (
            patch.object(allocation, "evict_from_tree_cache"),
            patch.object(allocation, "get_last_loc", return_value=torch.tensor([63])),
            patch.object(
                allocation,
                "assign_req_to_token_pool_func",
                side_effect=assign_on_cpu,
            ),
        ):
            allocation.alloc_for_spec_decode(
                tree_cache,
                req_to_token_pool,
                reqs=[req],
                req_pool_indices=torch.tensor([0]),
                cur_kv_lens=torch.tensor([2]),
                cur_kv_lens_cpu=torch.tensor([2]),
                nxt_kv_lens=torch.tensor([6]),
                nxt_kv_lens_cpu=torch.tensor([6]),
                num_needed_tokens=4,
                batch=batch,
            )

        allocator.alloc_logical_only.assert_called_once()
        torch.testing.assert_close(req_to_token[0, 2:6], logical_locs)

    def test_physical_free_excludes_reserved_page_zero(self):
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
        allocator.page_size = 64
        allocator.hisparse_attn_allocator = MagicMock()

        allocator.free_hisparse_indices(torch.tensor([0, 1, 63, 64, 65]))

        freed = allocator.hisparse_attn_allocator.free.call_args.args[0]
        torch.testing.assert_close(freed, torch.tensor([64, 65]))

    def test_spec_swap_manager_owns_runtime_state(self):
        coordinator = MagicMock()
        coordinator.mem_pool_device.layer_num = 2
        coordinator.req_to_token_pool.req_to_token = torch.empty((3, 16))
        coordinator.device = "cpu"
        coordinator.is_dsv4_hisparse = False
        coordinator.top_k = 1024
        coordinator.device_buffer_size = 4096

        spec_swap = HiSparseSpecSwapManager(
            coordinator,
            num_draft_tokens=2,
        )

        self.assertTrue(spec_swap.enabled)
        self.assertEqual(spec_swap.num_draft_tokens, 2)
        self.assertEqual(spec_swap.scratch_capacity, 1024)
        self.assertEqual(len(spec_swap.states), 2)
        self.assertEqual(spec_swap.req_to_scratch.shape, (2, 3, 1024))
        self.assertEqual(spec_swap.top_k_device_locs.shape, (3, 2, 1024))
        self.assertIs(
            spec_swap.states[0].scratch_state, spec_swap.states[1].scratch_state
        )

    def test_speculative_reserve_releases_transient_device_pages(self):
        coordinator = object.__new__(HiSparseCoordinator)
        mapping = torch.zeros(256, dtype=torch.int64)
        mapping[130:134] = torch.arange(66, 70)
        req_to_token = torch.zeros((1, 16), dtype=torch.int64)
        req_to_token[0, 2:6] = torch.arange(130, 134)
        coordinator.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        freed_locs = []
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            full_to_hisparse_device_index_mapping=mapping,
            free_hisparse_indices=lambda locs: freed_locs.append(locs.clone()),
        )

        batch = SimpleNamespace(req_pool_indices=torch.tensor([0]))
        spec_swap = object.__new__(HiSparseSpecSwapManager)
        spec_swap._coordinator = coordinator
        spec_swap.release_decode_reserve(
            batch=batch,
            current_kv_lens_cpu=torch.tensor([2]),
            next_kv_lens_cpu=torch.tensor([6]),
        )

        torch.testing.assert_close(freed_locs[0], torch.arange(66, 70))
        self.assertEqual(mapping[130:134].count_nonzero(), 0)

    def test_scratch_allocation_has_per_layer_location_views(self):
        scratch_allocator = MagicMock()
        scratch_allocator.alloc.return_value = torch.arange(9, 13)
        freed_locs = []
        coordinator = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                hisparse_attn_allocator=scratch_allocator,
                free_hisparse_indices=lambda locs: freed_locs.append(locs.clone()),
            )
        )
        spec_swap = object.__new__(HiSparseSpecSwapManager)
        spec_swap._coordinator = coordinator
        spec_swap.enabled = True
        spec_swap.scratch_capacity = 4
        spec_swap._scratch_reqs = set()
        spec_swap.req_to_scratch = torch.zeros((3, 2, 4), dtype=torch.int32)

        spec_swap.ensure_scratch(torch.tensor([1]))

        expected = torch.arange(9, 13, dtype=torch.int32)
        for layer_id in range(3):
            torch.testing.assert_close(spec_swap.req_to_scratch[layer_id, 1], expected)
        self.assertIn(1, spec_swap._scratch_reqs)

        spec_swap.free_scratch(1)

        torch.testing.assert_close(freed_locs[0], expected)
        self.assertNotIn(1, spec_swap._scratch_reqs)
        self.assertEqual(spec_swap.req_to_scratch[:, 1].count_nonzero(), 0)

    def test_accept_commit_keeps_hot_and_newest_device_slots(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.device_buffer_size = 5
        coordinator.req_to_device_buffer = torch.tensor(
            [[20, 21, 22, 23, 24, 25, 26]], dtype=torch.int64
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (2, 1, 7), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full(
            (2, 1, 7), -1, dtype=torch.int32
        )
        coordinator._skip_first_backup = [False]

        verify_cache_locs = torch.tensor([50, 51, 52, 53], dtype=torch.int64)
        req_to_token = torch.zeros((1, 16), dtype=torch.int64)
        req_to_token[0, 4:8] = verify_cache_locs
        coordinator.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        full_to_device_mapping = torch.zeros(128, dtype=torch.int64)
        full_to_device_mapping[verify_cache_locs] = torch.tensor([10, 11, 12, 13])
        transfer_values = MagicMock()
        coordinator.mem_pool_device = SimpleNamespace(
            transfer_values_on_device=transfer_values
        )
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            full_to_hisparse_device_index_mapping=full_to_device_mapping
        )

        alloc_host = MagicMock(
            side_effect=lambda _pool, _allocated, _req, start, count: torch.arange(
                1000 + start, 1000 + start + count, dtype=torch.int64
            )
        )
        coordinator.mem_pool_host = SimpleNamespace(alloc_paged_token_slots=alloc_host)
        coordinator.req_to_host_pool = torch.full((1, 16), -1, dtype=torch.int64)
        coordinator.req_to_host_pool_allocated_len = torch.zeros(1, dtype=torch.int64)

        spec_swap = object.__new__(HiSparseSpecSwapManager)
        spec_swap._coordinator = coordinator
        spec_swap._backup_device_locs_to_host = MagicMock()

        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            req_pool_indices=torch.tensor([0]),
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([4]),
            seq_lens_cpu=torch.tensor([4]),
            out_cache_loc=verify_cache_locs,
        )
        spec_swap.commit_accept_tokens(
            batch=batch,
            accept_indices=torch.tensor([[0, 1, 2, -1]], dtype=torch.int32),
        )

        transfer_call = transfer_values.call_args
        torch.testing.assert_close(
            transfer_call.kwargs["src_indices"], torch.tensor([10, 12])
        )
        torch.testing.assert_close(
            transfer_call.kwargs["dst_indices"], torch.tensor([24, 25])
        )
        backup_call = spec_swap._backup_device_locs_to_host.call_args
        torch.testing.assert_close(
            backup_call.args[0], torch.tensor([1004, 1005, 1006])
        )
        torch.testing.assert_close(backup_call.args[1], torch.tensor([24, 11, 25]))
        self.assertTrue(backup_call.kwargs["wait"])

        self.assertEqual(full_to_device_mapping[50], 24)
        self.assertEqual(full_to_device_mapping[51], 0)
        self.assertEqual(full_to_device_mapping[52], 25)
        self.assertEqual(full_to_device_mapping[53], 0)
        torch.testing.assert_close(
            coordinator.req_device_buffer_tokens[:, 0, 4],
            torch.tensor([4, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            coordinator.req_device_buffer_tokens[:, 0, 5],
            torch.tensor([6, 6], dtype=torch.int32),
        )
        self.assertTrue(coordinator._skip_first_backup[0])


if __name__ == "__main__":
    unittest.main()
