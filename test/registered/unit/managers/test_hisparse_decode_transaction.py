import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


class _FakeChildAllocator:
    def __init__(self, *, page_size: int, available: int = 4096) -> None:
        self.page_size = page_size
        self.available = available
        self.next_coordinate = 512
        self.alloc_requests: list[int] = []
        self.free_requests: list[torch.Tensor] = []
        self.fail_next = False

    def available_size(self) -> int:
        return self.available

    def alloc(self, need_size: int) -> torch.Tensor | None:
        self.alloc_requests.append(need_size)
        if self.fail_next:
            self.fail_next = False
            return None
        if need_size > self.available:
            return None

        assert self.next_coordinate % self.page_size == 0
        indices = torch.arange(
            self.next_coordinate,
            self.next_coordinate + need_size,
            dtype=torch.int64,
        )
        self.next_coordinate += need_size
        self.available -= need_size
        return indices

    def free(self, indices: torch.Tensor) -> None:
        self.free_requests.append(indices.clone())
        self.available += indices.numel()


class _FakeHiSparseAllocator:
    def __init__(
        self,
        *,
        page_size: int,
        device_page_size: int,
        compress_ratio: int,
        supports_page_aligned_alloc: bool = True,
    ) -> None:
        self.page_size = page_size
        self.hisparse_device_page_size = device_page_size
        self.compress_ratio = compress_ratio
        self.supports_page_aligned_alloc = supports_page_aligned_alloc
        self.logical_attn_allocator = _FakeChildAllocator(page_size=page_size)
        self.hisparse_attn_allocator = _FakeChildAllocator(page_size=device_page_size)
        self.full_to_hisparse_device_index_mapping = torch.cat(
            [
                torch.zeros(2048, dtype=torch.int64),
                torch.tensor([-1], dtype=torch.int64),
            ]
        )
        self.clear_calls: list[torch.Tensor] = []
        self.collect_calls: list[torch.Tensor] = []
        self.release_calls: list[torch.Tensor] = []
        self.materialize_calls: list[torch.Tensor] = []
        self.get_last_calls = 0

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.hisparse_attn_allocator.available_size() * self.compress_ratio,
        )

    def get_last_loc_compressed(self, last_locs: torch.Tensor) -> torch.Tensor:
        self.get_last_calls += 1
        if self.compress_ratio == 1:
            return last_locs
        return (last_locs - (self.compress_ratio - 1)) // self.compress_ratio

    def collect_owned_hisparse_page_ids(
        self,
        *,
        mapping_indices: torch.Tensor,
        extra_owned_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.collect_calls.append(mapping_indices.clone())
        coordinates = self.full_to_hisparse_device_index_mapping[mapping_indices]
        if extra_owned_coordinates is not None:
            coordinates = torch.cat([coordinates, extra_owned_coordinates])
        positive_coordinates = coordinates[coordinates > 0]
        if positive_coordinates.numel() == 0:
            return torch.empty((0,), dtype=torch.int64)
        return torch.unique(
            positive_coordinates // self.hisparse_device_page_size,
            sorted=True,
        )

    def clear_hisparse_mapping(self, *, mapping_indices: torch.Tensor) -> None:
        self.clear_calls.append(mapping_indices.clone())
        self.full_to_hisparse_device_index_mapping[mapping_indices] = 0

    def release_owned_hisparse_pages(self, *, owned_page_ids: torch.Tensor) -> None:
        self.release_calls.append(owned_page_ids.clone())
        self.hisparse_attn_allocator.available += (
            owned_page_ids.numel() * self.hisparse_device_page_size
        )

    def materialize_owned_hisparse_page_blocks(
        self, *, owned_page_ids: torch.Tensor
    ) -> torch.Tensor:
        self.materialize_calls.append(owned_page_ids.clone())
        offsets = torch.arange(self.hisparse_device_page_size, dtype=torch.int64)
        return (
            owned_page_ids[:, None] * self.hisparse_device_page_size + offsets
        ).reshape(-1)


def _make_coordinator(
    *,
    page_size: int = 4,
    device_page_size: int | None = None,
    compress_ratio: int = 1,
    device_buffer_size: int = 8,
    supports_page_aligned_alloc: bool = True,
) -> HiSparseCoordinator:
    if device_page_size is None:
        device_page_size = page_size
    allocator = _FakeHiSparseAllocator(
        page_size=page_size,
        device_page_size=device_page_size,
        compress_ratio=compress_ratio,
        supports_page_aligned_alloc=supports_page_aligned_alloc,
    )
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.token_to_kv_pool_allocator = allocator
    coordinator.compress_ratio = compress_ratio
    coordinator.is_dsv4_hisparse = compress_ratio != 1
    coordinator.device_buffer_size = device_buffer_size
    coordinator.padded_buffer_size = device_buffer_size + device_page_size
    coordinator.device = "cpu"
    coordinator.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((4, 128), dtype=torch.int64)
    )
    coordinator.req_to_device_buffer = torch.zeros(
        (4, coordinator.padded_buffer_size), dtype=torch.int64
    )
    coordinator.req_device_buffer_size = torch.zeros(4, dtype=torch.int64)
    coordinator.req_device_buffer_token_locs = torch.full(
        (2, 4, coordinator.padded_buffer_size), -1, dtype=torch.int32
    )
    coordinator.mem_pool_device = SimpleNamespace(
        page_size=device_page_size,
        full_to_hisparse_device_index_mapping=(
            allocator.full_to_hisparse_device_index_mapping
        ),
        _translate_loc_to_hisparse_device=lambda locs: (
            allocator.full_to_hisparse_device_index_mapping[locs]
        ),
    )
    coordinator._eager_backup_previous_token = Mock()
    return coordinator


def _seed_generic_page(
    coordinator: HiSparseCoordinator,
    *,
    req_pool_idx: int,
    semantic_start: int,
    logical_start: int,
    temporary_start: int,
    buffer_cap: int,
    buffer_start: int,
) -> torch.Tensor:
    allocator = coordinator.token_to_kv_pool_allocator
    page_size = allocator.page_size
    logical_page = torch.arange(
        logical_start,
        logical_start + page_size,
        dtype=torch.int64,
    )
    temporary_page = torch.arange(
        temporary_start,
        temporary_start + page_size,
        dtype=torch.int64,
    )
    coordinator.req_to_token_pool.req_to_token[
        req_pool_idx, semantic_start : semantic_start + page_size
    ] = logical_page
    allocator.full_to_hisparse_device_index_mapping[logical_page] = temporary_page
    coordinator.req_to_device_buffer[req_pool_idx, :buffer_cap] = torch.arange(
        buffer_start,
        buffer_start + buffer_cap,
        dtype=torch.int64,
    )
    coordinator.req_device_buffer_size[req_pool_idx] = buffer_cap
    return logical_page


def _run_page_aligned_map(
    coordinator: HiSparseCoordinator,
    *,
    seq_lens: list[int],
    out_cache_locs: list[int],
    req_pool_indices: list[int],
) -> None:
    coordinator._map_page_aligned_last_loc_to_buffer(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        out_cache_loc=torch.tensor(out_cache_locs, dtype=torch.int64),
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor(req_pool_indices, dtype=torch.int64),
    )


class TestHiSparseDecodeTransaction(unittest.TestCase):
    def test_padded_growth_transfers_temporary_page_and_allocates_only_net_extra(
        self,
    ) -> None:
        """A 2P gross grow transfers P and allocates only the remaining P."""
        coordinator = _make_coordinator()
        allocator = coordinator.token_to_kv_pool_allocator
        logical_page = _seed_generic_page(
            coordinator,
            req_pool_idx=0,
            semantic_start=4,
            logical_start=16,
            temporary_start=200,
            buffer_cap=4,
            buffer_start=100,
        )
        req = SimpleNamespace(
            req_pool_idx=0,
            kv_committed_len=4,
            kv=SimpleNamespace(kv_allocated_len=4),
        )

        requirements = coordinator.next_decode_allocation_requirements([req])
        _run_page_aligned_map(
            coordinator,
            seq_lens=[5],
            out_cache_locs=[16],
            req_pool_indices=[0],
        )

        self.assertEqual(requirements.logical_need, 4)
        self.assertEqual(requirements.device_need, 8)
        self.assertEqual(allocator.hisparse_attn_allocator.alloc_requests, [4])
        self.assertTrue(
            torch.equal(
                coordinator.req_to_device_buffer[0, 4:8],
                torch.arange(200, 204, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                coordinator.req_to_device_buffer[0, 8:12],
                torch.arange(512, 516, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping[logical_page],
                coordinator.req_to_device_buffer[0, 4:8],
            )
        )
        self.assertEqual(int(coordinator.req_device_buffer_size[0]), 12)
        self.assertEqual(allocator.collect_calls, [])
        self.assertEqual(allocator.release_calls, [])

    def test_extra_allocation_failure_preserves_mapping_buffer_and_capacity(
        self,
    ) -> None:
        """Net-extra OOM leaves the temporary owner transaction untouched."""
        coordinator = _make_coordinator()
        allocator = coordinator.token_to_kv_pool_allocator
        logical_page = _seed_generic_page(
            coordinator,
            req_pool_idx=0,
            semantic_start=4,
            logical_start=16,
            temporary_start=200,
            buffer_cap=4,
            buffer_start=100,
        )
        allocator.hisparse_attn_allocator.fail_next = True
        mapping_before = allocator.full_to_hisparse_device_index_mapping[
            logical_page
        ].clone()
        buffer_before = coordinator.req_to_device_buffer.clone()

        with self.assertRaisesRegex(RuntimeError, "net allocation failed"):
            _run_page_aligned_map(
                coordinator,
                seq_lens=[5],
                out_cache_locs=[16],
                req_pool_indices=[0],
            )

        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping[logical_page],
                mapping_before,
            )
        )
        self.assertTrue(torch.equal(coordinator.req_to_device_buffer, buffer_before))
        self.assertEqual(int(coordinator.req_device_buffer_size[0]), 4)
        self.assertEqual(allocator.clear_calls, [])
        self.assertEqual(allocator.release_calls, [])

    def test_generic_page_internal_decode_skips_owner_transaction(self) -> None:
        """Generic page-internal decode only validates and publishes its current slot."""
        coordinator = _make_coordinator()
        allocator = coordinator.token_to_kv_pool_allocator
        coordinator.req_to_device_buffer[0, :8] = torch.arange(
            100, 108, dtype=torch.int64
        )
        coordinator.req_device_buffer_size[0] = 8
        allocator.full_to_hisparse_device_index_mapping[17] = 105
        coordinator._plan_device_buffer_growth = Mock(
            side_effect=AssertionError("no-crossing path must not build a growth plan")
        )

        with patch(
            "sglang.srt.managers.hisparse_coordinator.torch.div",
            side_effect=AssertionError("no-crossing path must not scan owner tables"),
        ):
            _run_page_aligned_map(
                coordinator,
                seq_lens=[6],
                out_cache_locs=[17],
                req_pool_indices=[0],
            )

        self.assertEqual(int(coordinator.req_device_buffer_token_locs[0, 0, 8]), 105)
        self.assertEqual(allocator.clear_calls, [])
        self.assertEqual(allocator.collect_calls, [])
        self.assertEqual(allocator.materialize_calls, [])

    def test_release_replay_and_threshold_clamp_publish_complete_pages(self) -> None:
        """No-grow retirement releases once and publishes D-wide clamp mappings."""
        coordinator = _make_coordinator()
        allocator = coordinator.token_to_kv_pool_allocator
        logical_page = _seed_generic_page(
            coordinator,
            req_pool_idx=0,
            semantic_start=4,
            logical_start=16,
            temporary_start=200,
            buffer_cap=8,
            buffer_start=100,
        )

        _run_page_aligned_map(
            coordinator,
            seq_lens=[5],
            out_cache_locs=[16],
            req_pool_indices=[0],
        )
        release_call_count = len(allocator.release_calls)
        self.assertEqual(len(allocator.collect_calls), 1)
        allocator.collect_calls.clear()
        published = allocator.full_to_hisparse_device_index_mapping[
            logical_page
        ].clone()
        _run_page_aligned_map(
            coordinator,
            seq_lens=[5],
            out_cache_locs=[16],
            req_pool_indices=[0],
        )

        self.assertTrue(
            torch.equal(published, coordinator.req_to_device_buffer[0, 4:8])
        )
        self.assertEqual(len(allocator.release_calls), release_call_count)
        self.assertEqual(allocator.collect_calls, [])
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping[logical_page],
                published,
            )
        )

        threshold_coordinator = _make_coordinator()
        threshold_allocator = threshold_coordinator.token_to_kv_pool_allocator
        threshold_page = _seed_generic_page(
            threshold_coordinator,
            req_pool_idx=0,
            semantic_start=8,
            logical_start=24,
            temporary_start=220,
            buffer_cap=12,
            buffer_start=100,
        )
        _run_page_aligned_map(
            threshold_coordinator,
            seq_lens=[9],
            out_cache_locs=[24],
            req_pool_indices=[0],
        )

        reserved = threshold_coordinator.req_to_device_buffer[0, 8]
        self.assertTrue(
            torch.equal(
                threshold_allocator.full_to_hisparse_device_index_mapping[
                    threshold_page
                ],
                reserved.expand(4),
            )
        )

    def test_partial_duplicate_and_transfer_release_intersection_fail_before_clear(
        self,
    ) -> None:
        """Partial, duplicate, and intersecting owners fail before mutation."""
        partial = _make_coordinator()
        partial_allocator = partial.token_to_kv_pool_allocator
        partial_page = _seed_generic_page(
            partial,
            req_pool_idx=0,
            semantic_start=4,
            logical_start=16,
            temporary_start=200,
            buffer_cap=8,
            buffer_start=100,
        )
        partial_allocator.full_to_hisparse_device_index_mapping[partial_page[1]] = 105
        with self.assertRaises((RuntimeError, AssertionError)):
            _run_page_aligned_map(
                partial,
                seq_lens=[5],
                out_cache_locs=[16],
                req_pool_indices=[0],
            )
        self.assertEqual(partial_allocator.clear_calls, [])

        duplicate = _make_coordinator()
        duplicate_allocator = duplicate.token_to_kv_pool_allocator
        _seed_generic_page(
            duplicate,
            req_pool_idx=0,
            semantic_start=4,
            logical_start=16,
            temporary_start=200,
            buffer_cap=4,
            buffer_start=100,
        )
        _seed_generic_page(
            duplicate,
            req_pool_idx=1,
            semantic_start=4,
            logical_start=24,
            temporary_start=200,
            buffer_cap=4,
            buffer_start=120,
        )
        with self.assertRaises((RuntimeError, AssertionError)):
            _run_page_aligned_map(
                duplicate,
                seq_lens=[5, 5],
                out_cache_locs=[16, 24],
                req_pool_indices=[0, 1],
            )
        self.assertEqual(duplicate_allocator.clear_calls, [])

        intersection = _make_coordinator()
        intersection_allocator = intersection.token_to_kv_pool_allocator
        _seed_generic_page(
            intersection,
            req_pool_idx=0,
            semantic_start=4,
            logical_start=16,
            temporary_start=200,
            buffer_cap=4,
            buffer_start=100,
        )
        _seed_generic_page(
            intersection,
            req_pool_idx=1,
            semantic_start=4,
            logical_start=24,
            temporary_start=200,
            buffer_cap=8,
            buffer_start=120,
        )
        with self.assertRaises((RuntimeError, AssertionError)):
            _run_page_aligned_map(
                intersection,
                seq_lens=[5, 5],
                out_cache_locs=[16, 24],
                req_pool_indices=[0, 1],
            )
        self.assertEqual(intersection_allocator.clear_calls, [])

    def test_release_only_duplicate_aliases_release_canonical_page_once(self) -> None:
        """Release-only duplicate aliases canonicalize one temporary page owner."""
        coordinator = _make_coordinator()
        allocator = coordinator.token_to_kv_pool_allocator
        first_logical_page = _seed_generic_page(
            coordinator,
            req_pool_idx=0,
            semantic_start=4,
            logical_start=16,
            temporary_start=200,
            buffer_cap=8,
            buffer_start=100,
        )
        second_logical_page = _seed_generic_page(
            coordinator,
            req_pool_idx=1,
            semantic_start=4,
            logical_start=24,
            temporary_start=200,
            buffer_cap=8,
            buffer_start=120,
        )

        _run_page_aligned_map(
            coordinator,
            seq_lens=[5, 5],
            out_cache_locs=[16, 24],
            req_pool_indices=[0, 1],
        )

        self.assertEqual(len(allocator.collect_calls), 1)
        self.assertEqual(allocator.collect_calls[0].numel(), 8)
        self.assertEqual(len(allocator.release_calls), 1)
        self.assertTrue(
            torch.equal(
                allocator.release_calls[0],
                torch.tensor([50], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping[first_logical_page],
                coordinator.req_to_device_buffer[0, 4:8],
            )
        )
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping[second_logical_page],
                coordinator.req_to_device_buffer[1, 4:8],
            )
        )

    def test_dsv4_active_nonactive_empty_and_threshold_paths(self) -> None:
        """DSV4 handles active H-wide retirement and empty/nonactive no-ops."""
        empty = _make_coordinator(
            page_size=8,
            device_page_size=2,
            compress_ratio=4,
            device_buffer_size=4,
        )
        _run_page_aligned_map(
            empty,
            seq_lens=[],
            out_cache_locs=[],
            req_pool_indices=[],
        )
        _run_page_aligned_map(
            empty,
            seq_lens=[9],
            out_cache_locs=[32],
            req_pool_indices=[0],
        )
        self.assertEqual(empty.token_to_kv_pool_allocator.get_last_calls, 0)
        self.assertEqual(empty.token_to_kv_pool_allocator.collect_calls, [])
        self.assertEqual(empty.token_to_kv_pool_allocator.materialize_calls, [])

        page_internal = _make_coordinator(
            page_size=8,
            device_page_size=2,
            compress_ratio=4,
            device_buffer_size=4,
        )
        page_internal.req_to_device_buffer[0, :6] = torch.arange(
            100, 106, dtype=torch.int64
        )
        page_internal.req_device_buffer_size[0] = 6
        page_internal.req_to_token_pool.req_to_token[0, 15] = 39
        page_internal.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
            9
        ] = 103
        page_internal._plan_device_buffer_growth = Mock(
            side_effect=AssertionError("no-crossing path must not build a growth plan")
        )
        with patch(
            "sglang.srt.managers.hisparse_coordinator.torch.div",
            side_effect=AssertionError("no-crossing path must not scan owner tables"),
        ):
            _run_page_aligned_map(
                page_internal,
                seq_lens=[16],
                out_cache_locs=[39],
                req_pool_indices=[0],
            )
        self.assertEqual(int(page_internal.req_device_buffer_token_locs[0, 0, 4]), 103)
        self.assertEqual(page_internal.token_to_kv_pool_allocator.clear_calls, [])
        self.assertEqual(page_internal.token_to_kv_pool_allocator.collect_calls, [])
        self.assertEqual(page_internal.token_to_kv_pool_allocator.materialize_calls, [])

        active = _make_coordinator(
            page_size=8,
            device_page_size=2,
            compress_ratio=4,
            device_buffer_size=4,
        )
        active.req_to_device_buffer[0, :6] = torch.arange(100, 106, dtype=torch.int64)
        active.req_device_buffer_size[0] = 6
        active.req_to_token_pool.req_to_token[0, 11] = 35
        active.req_to_token_pool.req_to_token[0, 15] = 39
        active_allocator = active.token_to_kv_pool_allocator
        active_allocator.full_to_hisparse_device_index_mapping[8:10] = torch.arange(
            200, 202, dtype=torch.int64
        )
        _run_page_aligned_map(
            active,
            seq_lens=[12],
            out_cache_locs=[35],
            req_pool_indices=[0],
        )
        self.assertTrue(
            torch.equal(
                active_allocator.full_to_hisparse_device_index_mapping[8:10],
                torch.arange(102, 104, dtype=torch.int64),
            )
        )
        self.assertEqual(len(active_allocator.collect_calls), 1)
        self.assertEqual(len(active_allocator.release_calls), 1)
        self.assertTrue(
            torch.equal(
                active_allocator.release_calls[0],
                torch.tensor([100], dtype=torch.int64),
            )
        )

        threshold = _make_coordinator(
            page_size=8,
            device_page_size=2,
            compress_ratio=4,
            device_buffer_size=4,
        )
        threshold.req_to_device_buffer[0, :6] = torch.arange(
            100, 106, dtype=torch.int64
        )
        threshold.req_device_buffer_size[0] = 6
        threshold.req_to_token_pool.req_to_token[0, 19] = 51
        threshold.req_to_token_pool.req_to_token[0, 23] = 55
        threshold_allocator = threshold.token_to_kv_pool_allocator
        threshold_allocator.full_to_hisparse_device_index_mapping[12:14] = torch.arange(
            220, 222, dtype=torch.int64
        )
        _run_page_aligned_map(
            threshold,
            seq_lens=[20],
            out_cache_locs=[51],
            req_pool_indices=[0],
        )
        reserved = threshold.req_to_device_buffer[0, 4]
        self.assertTrue(
            torch.equal(
                threshold_allocator.full_to_hisparse_device_index_mapping[12:14],
                reserved.expand(2),
            )
        )

    def test_dsv4_requirements_and_scheduler_gate_use_h_sized_peak(self) -> None:
        """DSV4 decode requires P logical slots and H device slots."""
        coordinator = _make_coordinator(
            page_size=8,
            device_page_size=2,
            compress_ratio=4,
            device_buffer_size=4,
        )
        allocator = coordinator.token_to_kv_pool_allocator
        req = SimpleNamespace(
            req_pool_idx=0,
            kv_committed_len=8,
            kv=SimpleNamespace(kv_allocated_len=8),
        )
        requirements = coordinator.next_decode_allocation_requirements([req])
        batch = SimpleNamespace(
            reqs=[req],
            token_to_kv_pool_allocator=allocator,
            hisparse_coordinator=coordinator,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            tree_cache=object(),
            new_tokens_required_next_decode=lambda selected: 8,
        )
        allocator.logical_attn_allocator.available = 8
        allocator.hisparse_attn_allocator.available = 2

        with patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache") as evict:
            self.assertTrue(ScheduleBatch.check_decode_mem(batch))
            allocator.hisparse_attn_allocator.available = 1
            self.assertFalse(ScheduleBatch.check_decode_mem(batch))

        self.assertEqual(requirements.logical_need, 8)
        self.assertEqual(requirements.device_need, 2)
        self.assertEqual(evict.call_args_list[0].args[1], 8)
        self.assertEqual(evict.call_args_list[1].args[1], 8)

    def test_selected_subset_retraction_gate_checks_both_children(self) -> None:
        """Selected retraction subsets use independent logical and device peaks."""
        coordinator = _make_coordinator()
        allocator = coordinator.token_to_kv_pool_allocator
        coordinator.req_device_buffer_size[0] = 4
        coordinator.req_device_buffer_size[1] = 12
        reqs = [
            SimpleNamespace(
                req_pool_idx=0,
                kv_committed_len=4,
                kv=SimpleNamespace(kv_allocated_len=4),
            ),
            SimpleNamespace(
                req_pool_idx=1,
                kv_committed_len=8,
                kv=SimpleNamespace(kv_allocated_len=8),
            ),
        ]
        batch = SimpleNamespace(
            reqs=reqs,
            token_to_kv_pool_allocator=allocator,
            hisparse_coordinator=coordinator,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            tree_cache=object(),
            new_tokens_required_next_decode=lambda selected: len(selected) * 4,
        )
        allocator.logical_attn_allocator.available = 4
        allocator.hisparse_attn_allocator.available = 7

        with patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache") as evict:
            self.assertTrue(ScheduleBatch.check_decode_mem(batch, selected_indices=[1]))
            self.assertFalse(
                ScheduleBatch.check_decode_mem(batch, selected_indices=[0])
            )
            allocator.hisparse_attn_allocator.available = 8
            self.assertTrue(ScheduleBatch.check_decode_mem(batch, selected_indices=[0]))

        self.assertEqual(evict.call_args_list[0].args[1], 4)
        self.assertEqual(evict.call_args_list[1].args[1], 8)
        self.assertEqual(evict.call_args_list[2].args[1], 8)

    def test_page_one_and_capability_false_use_legacy_remap(self) -> None:
        """Page-one and capability-false allocators retain the legacy remap path."""
        for page_size, capability in ((1, True), (4, False)):
            with self.subTest(page_size=page_size, capability=capability):
                coordinator = _make_coordinator(
                    page_size=page_size,
                    device_page_size=page_size,
                    supports_page_aligned_alloc=capability,
                )
                allocator = coordinator.token_to_kv_pool_allocator
                temporary_coordinate = 200
                allocator.full_to_hisparse_device_index_mapping[16] = (
                    temporary_coordinate
                )
                coordinator._grow_device_buffers = Mock(
                    return_value=torch.tensor([100], dtype=torch.int64)
                )
                coordinator._map_page_aligned_last_loc_to_buffer = Mock(
                    side_effect=AssertionError("direct path must remain disabled")
                )

                with patch("sglang.srt.managers.hisparse_coordinator._is_hip", False):
                    coordinator.map_last_loc_to_buffer(
                        seq_lens=torch.tensor([1], dtype=torch.int64),
                        out_cache_loc=torch.tensor([16], dtype=torch.int64),
                        req_pool_indices=torch.tensor([0], dtype=torch.int64),
                        seq_lens_cpu=torch.tensor([1], dtype=torch.int64),
                        req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
                    )

                coordinator._grow_device_buffers.assert_called_once()
                coordinator._map_page_aligned_last_loc_to_buffer.assert_not_called()
                self.assertEqual(
                    int(allocator.full_to_hisparse_device_index_mapping[16]), 100
                )
                self.assertNotEqual(
                    int(allocator.full_to_hisparse_device_index_mapping[16]),
                    temporary_coordinate,
                )

                coordinator.next_decode_allocation_requirements = Mock(
                    side_effect=AssertionError("legacy budget must remain selected")
                )
                batch = SimpleNamespace(
                    reqs=[],
                    token_to_kv_pool_allocator=allocator,
                    hisparse_coordinator=coordinator,
                    spec_algorithm=SimpleNamespace(is_none=lambda: True),
                    tree_cache=object(),
                    new_tokens_required_next_decode=lambda selected: 3,
                )
                with patch(
                    "sglang.srt.managers.schedule_batch.evict_from_tree_cache"
                ) as evict:
                    self.assertTrue(ScheduleBatch.check_decode_mem(batch))
                coordinator.next_decode_allocation_requirements.assert_not_called()
                self.assertEqual(evict.call_args.args[1], 3)


if __name__ == "__main__":
    unittest.main()
