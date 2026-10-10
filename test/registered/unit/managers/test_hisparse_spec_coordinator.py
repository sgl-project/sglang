"""Unit tests for the minimal HiSparse spec coordinator lifecycle."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseSpecSwapManager,
)
from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
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

        def assign_on_cpu(req_indices, table, start, end, locs, batch_size):
            table[int(req_indices[0]), int(start[0]) : int(end[0])] = locs

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

    def test_scratch_allocation_has_per_layer_location_views(self):
        scratch_allocator = MagicMock()
        scratch_allocator.alloc.return_value = torch.arange(64, 1088)
        token_allocator = SimpleNamespace(
            hisparse_attn_allocator=scratch_allocator,
            free_hisparse_indices=MagicMock(),
        )
        coordinator = MagicMock(
            token_to_kv_pool_allocator=token_allocator,
            mem_pool_device=SimpleNamespace(layer_num=3),
            req_to_token_pool=SimpleNamespace(req_to_token=torch.empty((2, 16))),
            device="cpu",
            is_dsv4_hisparse=False,
            top_k=1024,
            device_buffer_size=4096,
        )
        spec_swap = HiSparseSpecSwapManager(coordinator, num_draft_tokens=2)

        spec_swap.allocate_scratch(1)

        expected = torch.arange(64, 1088, dtype=torch.int32)
        for layer_id in range(3):
            torch.testing.assert_close(spec_swap.req_to_scratch[layer_id, 1], expected)
        self.assertIn(1, spec_swap._scratch_reqs)
        self.assertIs(
            spec_swap.states[0].scratch_state, spec_swap.states[1].scratch_state
        )

        owned_locs = spec_swap.extend_owned_locs(
            1, torch.tensor([20], dtype=torch.int32)
        )
        torch.testing.assert_close(
            owned_locs,
            torch.cat([torch.tensor([20], dtype=torch.int32), expected.repeat(3)]),
        )

        spec_swap.free_unrotated_scratch(1)
        token_allocator.free_hisparse_indices.assert_called_once()
        torch.testing.assert_close(
            token_allocator.free_hisparse_indices.call_args.args[0], expected
        )
        self.assertNotIn(1, spec_swap._scratch_reqs)
        self.assertEqual(spec_swap.req_to_scratch[:, 1].count_nonzero(), 0)

    def test_spec_workspace_is_reserved_before_request_becomes_runnable(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.is_dsv4_hisparse = False
        coordinator.device_buffer_size = 4096
        coordinator.padded_buffer_size = 4160
        coordinator.mem_pool_device = SimpleNamespace(
            page_size=64,
            translate_loc_from_full_to_compressed=lambda locs: locs,
        )
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(1, -1)
        )
        buffer_indices = torch.arange(64, 4224, dtype=torch.int64)
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            alloc_device_buffer=MagicMock(return_value=buffer_indices)
        )
        coordinator.req_to_device_buffer = torch.zeros((1, 4160), dtype=torch.int32)
        coordinator.req_device_buffer_size = torch.zeros(1, dtype=torch.int32)
        coordinator.req_device_buffer_tokens = torch.full(
            (2, 1, 4160), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full(
            (2, 1, 4160), -1, dtype=torch.int32
        )
        coordinator._device_buffer_arange_i32 = torch.arange(4096, dtype=torch.int32)
        coordinator.spec_swap = SimpleNamespace(
            enabled=True,
            allocate_scratch=MagicMock(),
            reset=MagicMock(),
        )
        req = SimpleNamespace(
            rid="req-0",
            kv=SimpleNamespace(req_pool_idx=0, kv_allocated_len=8),
        )

        coordinator.alloc_device_buffer(req)

        self.assertEqual(int(coordinator.req_device_buffer_size[0]), 4160)
        torch.testing.assert_close(
            coordinator.req_to_device_buffer[0], buffer_indices.to(torch.int32)
        )
        coordinator.spec_swap.allocate_scratch.assert_called_once_with(0)
        coordinator.spec_swap.reset.assert_called_once_with(0)

    def test_accept_commit_keeps_hot_and_newest_device_slots(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.device_buffer_size = 5
        coordinator.page_size = 8
        coordinator.req_to_device_buffer = torch.tensor(
            [list(range(20, 33))], dtype=torch.int64
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (2, 1, 13), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full(
            (2, 1, 13), -1, dtype=torch.int32
        )
        coordinator._skip_first_backup = [False]

        verify_cache_locs = torch.tensor([50, 51, 52, 53], dtype=torch.int64)
        req_to_token = torch.zeros((1, 16), dtype=torch.int64)
        req_to_token[0, 4:8] = verify_cache_locs
        coordinator.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        full_to_device_mapping = torch.zeros(128, dtype=torch.int64)
        transfer_values = MagicMock()
        coordinator.mem_pool_device = SimpleNamespace(
            transfer_values_on_device=transfer_values
        )
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            full_to_hisparse_device_index_mapping=full_to_device_mapping
        )

        coordinator.mem_pool_host = HiSparseHostPoolMixin()
        coordinator.mem_pool_host.page_size = 8
        coordinator.mem_pool_host.alloc_page = MagicMock(
            return_value=torch.arange(1000, 1008)
        )
        coordinator.req_to_host_pool = torch.full((1, 16), -1, dtype=torch.int64)
        coordinator.req_to_host_pool_allocated_len = torch.zeros(1, dtype=torch.int64)

        spec_swap = object.__new__(HiSparseSpecSwapManager)
        spec_swap._coordinator = coordinator
        spec_swap.enabled = True
        spec_swap.num_draft_tokens = 4
        spec_swap._backup_device_locs_to_host = MagicMock()

        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0, kv_allocated_len=8))
            ],
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            req_pool_indices=torch.tensor([0]),
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([4]),
            seq_lens_cpu=torch.tensor([2]),  # CPU metadata can lag during overlap.
            out_cache_loc=verify_cache_locs,
        )
        spec_swap.prepare_verify(batch)
        spec_swap.prepare_verify(batch)
        coordinator.mem_pool_host.alloc_page.assert_called_once_with(1)
        self.assertEqual(int(coordinator.req_to_host_pool_allocated_len[0]), 8)
        accept_indices = torch.tensor([[0, 1, 2, -1]], dtype=torch.int32)

        class NoDataDependentOps(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func in (
                    torch.ops.aten._local_scalar_dense.default,
                    torch.ops.aten.nonzero.default,
                    torch.ops.aten.repeat_interleave.Tensor,
                ):
                    raise AssertionError(f"Synchronizing operation: {func}")
                if func in (
                    torch.ops.aten.index.Tensor,
                    torch.ops.aten.index_put_.default,
                ):
                    if any(
                        idx is not None and idx.dtype == torch.bool for idx in args[1]
                    ):
                        raise AssertionError("Dynamically sized boolean indexing")
                return func(*args, **(kwargs or {}))

        with (
            patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("CPU readback")
            ),
            NoDataDependentOps(),
        ):
            spec_swap.commit_accept_tokens(batch=batch, accept_indices=accept_indices)

        transfer_call = transfer_values.call_args
        torch.testing.assert_close(
            transfer_call.kwargs["src_indices"], torch.tensor([26, 27, 28, 29])
        )
        torch.testing.assert_close(
            transfer_call.kwargs["dst_indices"], torch.tensor([24, 27, 25, 29])
        )
        backup_call = spec_swap._backup_device_locs_to_host.call_args
        torch.testing.assert_close(
            backup_call.args[0], torch.tensor([1004, 1005, 1006, 1007])
        )
        torch.testing.assert_close(backup_call.args[1], torch.tensor([24, 27, 25, 29]))
        torch.testing.assert_close(
            full_to_device_mapping[verify_cache_locs], torch.tensor([24, 0, 25, 0])
        )
        torch.testing.assert_close(
            coordinator.req_device_buffer_tokens[:, 0, 4],
            torch.tensor([4, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            coordinator.req_device_buffer_tokens[:, 0, 5],
            torch.tensor([6, 6], dtype=torch.int32),
        )
        torch.testing.assert_close(
            coordinator.req_device_buffer_tokens[:, 0, 6:10],
            torch.tensor([[4, 5, 6, 7], [4, 5, 6, 7]], dtype=torch.int32),
        )
        self.assertTrue(coordinator._skip_first_backup[0])


if __name__ == "__main__":
    unittest.main()
