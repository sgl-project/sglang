import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import get_dsa_state_page_indices
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.allocator.radix_hisparse import (
    RadixHiSparseTokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ReqToTokenPool:
    def __init__(self, size=1, max_context_len=256):
        self.size = size
        self.req_to_token = torch.full((size, max_context_len), -1, dtype=torch.int64)
        self.writes = []

    def alloc(self, reqs):
        for req in reqs:
            req.req_pool_idx = 0
        return torch.tensor([0], dtype=torch.int64)

    def write(self, indices, values):
        self.writes.append((indices, values))
        self.req_to_token[indices] = values


def _make_req(fill_len):
    req = SimpleNamespace(
        rid="radix-hisparse-req",
        origin_input_ids=list(range(fill_len)),
        output_ids=[],
        kv=None,
        kv_committed_len=None,
        time_stats=MagicMock(),
    )

    def set_extend_range(start, end):
        req.extend_range = SimpleNamespace(start=start, end=end, length=end - start)

    req.set_extend_range = set_extend_range
    return req


class TestRadixHiSparseDecodeAdmission(CustomTestCase):
    def test_l1_budget_includes_radix_evictable_pages(self):
        allocator = object.__new__(RadixHiSparseTokenToKVPoolAllocator)
        allocator.logical_attn_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=32)
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = allocator
        queue.tree_cache = SimpleNamespace(evictable_size=MagicMock(return_value=96))
        queue.scheduler = SimpleNamespace(enable_hisparse=True, last_batch=None)
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._need_space_for_single_req = MagicMock(return_value=0)
        queue._active_reserved_tokens = MagicMock(return_value=0)

        budget = queue._allocatable_token_budgets()

        self.assertEqual(budget, 128)
        queue.tree_cache.evictable_size.assert_called_once_with()

    def test_prealloc_reuses_prefix_and_allocates_only_l1_suffix(self):
        page_size = 64
        fill_len = 128
        prefix_len = 64
        prefix_indices = torch.arange(64, 128, dtype=torch.int64)
        suffix_indices = torch.arange(256, 320, dtype=torch.int64)
        host_indices = suffix_indices.clone()
        req = _make_req(fill_len)
        req_to_token_pool = _ReqToTokenPool()

        allocator = object.__new__(RadixHiSparseTokenToKVPoolAllocator)
        allocator.device = torch.device("cpu")
        allocator.page_size = page_size
        allocator.available_size = MagicMock(return_value=fill_len)
        allocator.alloc_logical_only = MagicMock(return_value=suffix_indices)
        coordinator = SimpleNamespace(
            is_radix_hisparse=True,
            bind_l1_host_locs=MagicMock(return_value=host_indices),
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = req_to_token_pool
        queue.token_to_kv_pool_allocator = allocator
        queue.tree_cache = SimpleNamespace(
            evictable_size=MagicMock(return_value=0),
            protected_size=MagicMock(return_value=0),
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=coordinator,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
        )

        result = queue._pre_alloc(
            req,
            prefix_indices=prefix_indices,
            prefix_len=prefix_len,
            total_prefix_len=prefix_len,
        )

        self.assertIs(result, host_indices)
        allocator.alloc_logical_only.assert_called_once()
        kwargs = allocator.alloc_logical_only.call_args.kwargs
        self.assertEqual(kwargs["prefix_lens"].tolist(), [prefix_len])
        self.assertEqual(kwargs["seq_lens"].tolist(), [fill_len])
        self.assertEqual(kwargs["last_loc"].tolist(), [127])
        self.assertEqual(kwargs["extend_num_tokens"], fill_len - prefix_len)
        coordinator.bind_l1_host_locs.assert_called_once_with(
            0, prefix_len, suffix_indices
        )
        self.assertTrue(
            torch.equal(req_to_token_pool.req_to_token[0, :prefix_len], prefix_indices)
        )
        self.assertTrue(
            torch.equal(
                req_to_token_pool.req_to_token[0, prefix_len:fill_len],
                suffix_indices,
            )
        )

    def test_direct_admission_rebinds_canonical_l1_row_before_l0(self):
        canonical_l1 = torch.tensor([[7, 8, 65, 66]], dtype=torch.int32)
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_radix_hisparse = True
        coordinator.is_dsv4_hisparse = False
        coordinator.req_to_token_pool = SimpleNamespace(req_to_token=canonical_l1)
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            full_kv_host_locs=lambda l1_indices: l1_indices
        )
        coordinator.req_to_host_pool = torch.full((1, 8), -1, dtype=torch.int64)
        coordinator.req_to_host_pool_allocated_len = torch.zeros(1, dtype=torch.int64)
        coordinator.device_buffer_size = 8
        coordinator._skip_first_backup = [False]

        observed = []
        coordinator.alloc_device_buffer = MagicMock(
            side_effect=lambda req: observed.append(
                ("alloc_l0", coordinator.req_to_host_pool[0, :4].clone())
            )
        )
        coordinator._preload_to_device_buffer = MagicMock(
            side_effect=lambda req: observed.append(
                ("preload", coordinator.req_to_host_pool[0, :4].clone())
            )
        )
        req = SimpleNamespace(
            rid="canonical",
            req_pool_idx=0,
            kv=SimpleNamespace(kv_allocated_len=4),
            hisparse_staging=False,
        )

        coordinator.admit_request_direct(req)

        expected = canonical_l1.to(torch.int64).squeeze(0)
        self.assertEqual([name for name, _ in observed], ["alloc_l0", "preload"])
        for _, host_row in observed:
            self.assertTrue(torch.equal(host_row, expected))
        self.assertEqual(int(coordinator.req_to_host_pool_allocated_len[0]), 4)

    def test_prebuilt_canonicalization_happens_before_radix_l0_admission(self):
        scheduler = SimpleNamespace()
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        req = MagicMock()
        req.last_node = object()
        req.kv_committed_len = None
        req.time_stats = MagicMock()
        scheduler.waiting_queue = [req]
        scheduler.enable_priority_scheduling = False
        scheduler.req_to_token_pool = MagicMock(size=1)
        scheduler.max_running_requests = 1
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.model_config = MagicMock()
        scheduler.enable_overlap = False
        scheduler.enable_hisparse = True
        scheduler.spec_algorithm = MagicMock()
        scheduler.server_args = SimpleNamespace()
        scheduler.future_map = MagicMock()

        order = []
        scheduler.hisparse_coordinator = SimpleNamespace(
            is_radix_hisparse=True,
            admit_request_direct=MagicMock(
                side_effect=lambda admitted_req: order.append("admit_l0")
            ),
        )
        running_batch = MagicMock()
        running_batch.batch_size.return_value = 0
        new_batch = MagicMock()
        new_batch.process_prebuilt.side_effect = lambda *args: order.append(
            "canonicalize"
        )

        with (
            patch(
                "sglang.srt.disaggregation.decode.ScheduleBatch.init_new",
                return_value=new_batch,
            ),
            patch(
                "sglang.srt.disaggregation.decode.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_enable_radix_cache=True
                ),
            ),
        ):
            result = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
                scheduler, running_batch
            )

        self.assertIs(result, new_batch)
        self.assertEqual(order, ["canonicalize", "admit_l0"])


class _FakeEvent:
    def __init__(self, order=None):
        self.order = order if order is not None else []

    def record(self):
        self.order.append("record")

    def wait(self, stream):
        self.order.append(("wait", stream))

    def synchronize(self):
        self.order.append("synchronize")


class TestRadixHiSparseDecodeWriteBack(CustomTestCase):
    def test_current_decode_rows_are_backed_up_and_event_is_retained(self):
        l1_locs = torch.tensor([7, 11], dtype=torch.int64)
        host_locs = torch.tensor([107, 111], dtype=torch.int64)
        l0_locs = torch.tensor([3, 5], dtype=torch.int64)
        event = _FakeEvent()

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            full_kv_host_locs=MagicMock(return_value=host_locs)
        )
        coordinator.mem_pool_device = SimpleNamespace(
            translate_loc_from_full_to_hisparse_device=MagicMock(return_value=l0_locs)
        )
        coordinator.mem_pool_host = SimpleNamespace(
            backup_from_device_all_layer=MagicMock()
        )

        fake_device_module = SimpleNamespace(
            Event=MagicMock(return_value=event),
            current_stream=MagicMock(return_value="copy-stream"),
        )
        with patch(
            "sglang.srt.managers.hisparse_coordinator.device_module",
            fake_device_module,
        ):
            returned_event = coordinator.backup_radix_decode_batch(l1_locs)

        self.assertIs(returned_event, event)
        self.assertIs(coordinator._radix_decode_backup_event, event)
        self.assertEqual(event.order, ["record"])
        args = coordinator.mem_pool_host.backup_from_device_all_layer.call_args
        self.assertIs(args.args[0], coordinator.mem_pool_device)
        self.assertTrue(torch.equal(args.args[1], host_locs))
        self.assertTrue(torch.equal(args.args[2], l0_locs))
        self.assertEqual(args.kwargs, {"io_backend": "kernel"})

    def test_next_l0_mapping_does_not_wait_for_previous_write_back(self):
        order = []
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_radix_hisparse = True
        coordinator.is_dsv4_hisparse = False
        coordinator.device_buffer_size = 2
        coordinator.wait_for_radix_decode_backup = MagicMock(
            side_effect=lambda: order.append("wait")
        )
        coordinator._eager_backup_previous_token = MagicMock()
        coordinator._grow_device_buffers = MagicMock(
            side_effect=lambda *args: order.append("map")
            or torch.tensor([3], dtype=torch.int64)
        )
        coordinator.req_device_buffer_token_locs = torch.zeros(
            (1, 1, 3), dtype=torch.int32
        )
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            get_last_loc_compressed=lambda locs: locs
        )
        coordinator.mem_pool_device = SimpleNamespace(
            full_to_hisparse_device_index_mapping=torch.zeros(16, dtype=torch.int64)
        )

        coordinator.map_last_loc_to_buffer(
            seq_lens=torch.tensor([5], dtype=torch.int64),
            out_cache_loc=torch.tensor([7], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
        )

        self.assertEqual(order, ["map"])
        coordinator.wait_for_radix_decode_backup.assert_not_called()
        coordinator._eager_backup_previous_token.assert_not_called()
        self.assertEqual(
            int(coordinator.mem_pool_device.full_to_hisparse_device_index_mapping[7]),
            3,
        )

    def test_model_forward_wait_dispatches_to_radix_write_back(self):
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_radix_hisparse = True
        coordinator.wait_for_radix_decode_backup = MagicMock()

        coordinator.wait_for_pending_backup()

        coordinator.wait_for_radix_decode_backup.assert_called_once_with()

    def test_model_forward_wait_preserves_legacy_backup_event(self):
        event = _FakeEvent()
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_radix_hisparse = False
        coordinator._has_pending_backup = True
        coordinator._backup_done_event = event

        fake_device_module = SimpleNamespace(
            current_stream=MagicMock(return_value="forward-stream")
        )
        with patch(
            "sglang.srt.managers.hisparse_coordinator.device_module",
            fake_device_module,
        ):
            coordinator.wait_for_pending_backup()

        self.assertEqual(event.order, [("wait", "forward-stream")])
        self.assertFalse(coordinator._has_pending_backup)

    def test_finish_fences_write_back_before_l0_release(self):
        order = []
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.decode_producer_stream = None
        coordinator.is_radix_hisparse = True
        coordinator.synchronize_radix_decode_backup = MagicMock(
            side_effect=lambda: order.append("fence")
        )
        coordinator.req_device_buffer_size = torch.tensor([1], dtype=torch.int64)
        coordinator.req_to_device_buffer = torch.tensor([[3, 0]], dtype=torch.int64)
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            free_hisparse_indices=MagicMock(
                side_effect=lambda locs: order.append("free-l0")
            )
        )
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[7, -1]], dtype=torch.int64)
        )
        coordinator.mem_pool_device = SimpleNamespace(
            translate_loc_from_full_to_compressed=lambda locs: locs,
            full_to_hisparse_device_index_mapping=torch.ones(16, dtype=torch.int64),
        )
        coordinator._clear_request_host_locs = MagicMock()
        coordinator.req_device_buffer_tokens = torch.zeros((1, 1, 2), dtype=torch.int32)
        coordinator.req_device_buffer_token_locs = torch.zeros(
            (1, 1, 2), dtype=torch.int32
        )
        coordinator.lru_slots = torch.zeros((1, 1, 2), dtype=torch.int32)
        coordinator._lru_init = torch.zeros((1, 2), dtype=torch.int32)
        coordinator._skip_first_backup = [False]
        req = SimpleNamespace(
            req_pool_idx=0,
            kv=SimpleNamespace(kv_allocated_len=1),
        )

        coordinator.request_finished(req)

        self.assertEqual(order[:2], ["fence", "free-l0"])


class TestRadixHiSparseDSAStateRange(CustomTestCase):
    def test_state_pages_use_decode_suffix_and_allow_full_hit(self):
        page_size = 64
        pool = SimpleNamespace(
            req_to_token=torch.cat(
                [
                    torch.arange(64, 128, dtype=torch.int64),
                    torch.arange(256, 320, dtype=torch.int64),
                ]
            ).reshape(1, -1)
        )

        suffix_pages = get_dsa_state_page_indices(
            pool, 0, page_size, 2 * page_size, page_size
        )
        full_hit_pages = get_dsa_state_page_indices(
            pool, 0, 2 * page_size, 2 * page_size, page_size
        )

        np.testing.assert_array_equal(suffix_pages, np.array([4]))
        self.assertEqual(full_hit_pages.size, 0)


if __name__ == "__main__":
    unittest.main()
