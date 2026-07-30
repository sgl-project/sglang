"""CPU unit test for the decode-context-parallel (DCP) per-rank KV-length math.

Pins ``get_dcp_lens`` (the single, superset implementation in
``layers/dcp/layout.py``) to a brute-force owner-count reference, and proves
it is bit-identical to the legacy in-place formula that
``update_local_kv_lens_for_dcp`` used before it was collapsed into a wrapper:

    floor((len - rank - 1) / N) + 1   ==   len // N + (rank < len % N)   (len >= 0)

Usage:
    python -m pytest test_dcp_layout_unit.py -v
    python test_dcp_layout_unit.py
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 8]
LENS = list(range(0, 41))
STARTS = [0, 1, 2, 5, 7, 13, 31]


def _owner_count(length: int, n: int, rank: int, start: int) -> int:
    """Ground truth: # of absolute positions p in [start, start+length) with p % n == rank."""
    return sum(1 for p in range(start, start + length) if p % n == rank)


def _legacy_inplace_formula(length: int, n: int, rank: int) -> int:
    """The pre-refactor update_local_kv_lens_for_dcp body (start == 0 case)."""
    return (length - rank - 1) // n + 1


class TestGetDcpLens(CustomTestCase):
    def test_start_none_matches_owner_count(self):
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int32)
                got = get_dcp_lens(lens, n, rank)
                expected = torch.tensor(
                    [_owner_count(L, n, rank, 0) for L in LENS], dtype=torch.int32
                )
                self.assertTrue(
                    torch.equal(got.to(torch.int32), expected),
                    f"start=None mismatch at n={n}, rank={rank}: {got.tolist()} != {expected.tolist()}",
                )

    def test_start_none_matches_legacy_inplace_formula(self):
        # The collapse claim: get_dcp_lens (start=None) == legacy floor((L-rank-1)/N)+1.
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int64)
                got = get_dcp_lens(lens, n, rank)
                legacy = torch.tensor(
                    [_legacy_inplace_formula(L, n, rank) for L in LENS],
                    dtype=torch.int64,
                )
                self.assertTrue(
                    torch.equal(got.to(torch.int64), legacy),
                    f"legacy-formula mismatch at n={n}, rank={rank}",
                )

    def test_start_tensor_matches_owner_count(self):
        for n in DCP_SIZES:
            for rank in range(n):
                for start in STARTS:
                    lens = torch.tensor(LENS, dtype=torch.int64)
                    start_t = torch.full_like(lens, start)
                    got = get_dcp_lens(lens, n, rank, start=start_t)
                    expected = torch.tensor(
                        [_owner_count(L, n, rank, start) for L in LENS],
                        dtype=torch.int64,
                    )
                    self.assertTrue(
                        torch.equal(got.to(torch.int64), expected),
                        f"start={start} mismatch at n={n}, rank={rank}: "
                        f"{got.tolist()} != {expected.tolist()}",
                    )

    def test_dcp_size_one_is_identity(self):
        lens = torch.tensor(LENS, dtype=torch.int32)
        self.assertTrue(torch.equal(get_dcp_lens(lens, 1, 0), lens))

    def test_paged_allocator_exposes_dcp_virtual_capacity(self):
        real_kv_size = 1024
        dcp_size = 4
        physical_page_size = 64
        allocator = PagedTokenToKVPoolAllocator(
            size=real_kv_size * dcp_size,
            page_size=physical_page_size * dcp_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=object(),
            need_sort=False,
        )

        allocations = [allocator.alloc(physical_page_size * dcp_size) for _ in range(4)]
        self.assertTrue(all(indices is not None for indices in allocations))
        virtual_indices = torch.cat(allocations)

        self.assertEqual(allocator.size, real_kv_size * dcp_size)
        self.assertEqual(allocator.page_size, physical_page_size * dcp_size)
        self.assertEqual(allocator.num_pages, real_kv_size // physical_page_size)
        self.assertEqual(
            len(torch.unique(virtual_indices // dcp_size)),
            len(virtual_indices) // dcp_size,
        )
        self.assertLess(
            int((virtual_indices // dcp_size).max()),
            real_kv_size + physical_page_size,
        )

    def test_configurator_scales_only_the_virtual_dcp_allocator(self):
        physical_kv_size = 1024
        physical_page_size = 64
        physical_kv_cache = SimpleNamespace(
            size=physical_kv_size,
            page_size=physical_page_size,
        )
        sizes = SimpleNamespace(
            max_total_num_tokens=physical_kv_size,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )
        allocators = {}

        for dcp_size in (1, 4):
            configurator = SimpleNamespace(
                server_args=SimpleNamespace(
                    disaggregation_mode="null",
                    enable_hisparse=False,
                    page_size=physical_page_size,
                    dcp_size=dcp_size,
                ),
                hybrid_gdn_config=None,
                is_hybrid_swa=False,
                kv_cache_dtype=torch.bfloat16,
                device="cpu",
                is_draft_worker=False,
            )
            with patch(
                "sglang.srt.mem_cache.kv_cache_configurator.current_platform.is_out_of_tree",
                return_value=False,
            ):
                allocators[dcp_size] = (
                    KVCacheConfigurator._build_token_to_kv_pool_allocator(
                        configurator,
                        sizes=sizes,
                        token_to_kv_pool=physical_kv_cache,
                        is_dsv4_model=False,
                        req_to_token_pool=object(),
                        token_to_kv_pool_allocator=None,
                    )
                )

        dcp1_allocator = allocators[1]
        dcp4_allocator = allocators[4]
        self.assertIs(dcp1_allocator.get_kvcache(), physical_kv_cache)
        self.assertIs(dcp4_allocator.get_kvcache(), physical_kv_cache)
        self.assertEqual(dcp1_allocator.size, 1024)
        self.assertEqual(dcp1_allocator.page_size, 64)
        self.assertEqual(dcp1_allocator.num_pages, 16)
        self.assertEqual(dcp4_allocator.size, 4096)
        self.assertEqual(dcp4_allocator.page_size, 256)
        self.assertEqual(dcp4_allocator.num_pages, 16)

    def test_live_cell_and_page_ownership_formulas(self):
        dcp_size = 4
        physical_page_size = 64
        ragged_lengths = (0, 1, 2, 3, 4, 63, 64, 65, 255, 256, 257, 515)

        per_rank_counts = []
        for rank in range(dcp_size):
            expected_counts = [
                length // dcp_size + int(rank < length % dcp_size)
                for length in ragged_lengths
            ]
            actual_counts = [
                _owner_count(length, dcp_size, rank, 0) for length in ragged_lengths
            ]
            self.assertEqual(actual_counts, expected_counts)
            per_rank_counts.append(sum(actual_counts))

            allocated_pages = [
                math.ceil(length / (physical_page_size * dcp_size))
                for length in ragged_lengths
            ]
            active_pages = [
                math.ceil(count / physical_page_size) for count in actual_counts
            ]
            self.assertTrue(
                all(
                    active <= allocated
                    for active, allocated in zip(active_pages, allocated_pages)
                )
            )
            self.assertTrue(
                all(
                    allocated - active <= 1
                    for active, allocated in zip(active_pages, allocated_pages)
                )
            )

        self.assertEqual(sum(per_rank_counts), sum(ragged_lengths))

        aligned_lengths = (256, 512, 768, 1024)
        full_replica_cells = sum(aligned_lengths)
        full_replica_pages = sum(
            length // physical_page_size for length in aligned_lengths
        )
        for rank in range(dcp_size):
            local_cells = sum(
                _owner_count(length, dcp_size, rank, 0) for length in aligned_lengths
            )
            local_pages = sum(
                math.ceil(_owner_count(length, dcp_size, rank, 0) / physical_page_size)
                for length in aligned_lengths
            )
            self.assertEqual(local_cells * dcp_size, full_replica_cells)
            self.assertEqual(local_pages * dcp_size, full_replica_pages)

    def test_hybrid_pool_reports_the_backing_attention_shape(self):
        pool = object.__new__(HybridLinearKVPool)
        pool.start_layer = 0
        pool.layer_transfer_counter = None
        pool.full_attention_layer_id_mapping = {3: 0, 7: 1}
        pool.full_kv_pool = MagicMock()
        expected = (torch.Size([1024, 1, 576]), torch.Size([1024, 1, 576]))
        pool.full_kv_pool.get_kv_buffer_shape.return_value = expected

        self.assertEqual(pool.get_kv_buffer_shape(), expected)
        pool.full_kv_pool.get_kv_buffer_shape.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
