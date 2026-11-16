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
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_buffer_info import DSV4PagedBufferInfo
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hisparse_retraction import DSV4RetractionSnapshot
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestHiSparseSpecCoordinator(CustomTestCase):
    def _retraction_fixture(self):
        coordinator = object.__new__(HiSparseCoordinator)
        coordinator.device = "cpu"
        coordinator.is_dsv4_hisparse = False
        coordinator.compress_ratio = 1
        coordinator.sliding_window_size = 128
        coordinator.decode_producer_stream = None
        coordinator.wait_for_pending_backup = MagicMock()
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[2, 3, 4]])
        )
        mapping = torch.zeros(16, dtype=torch.int64)
        mapping[2:5] = torch.tensor([8, 9, 10])
        coordinator.mem_pool_device = SimpleNamespace(
            layer_num=2,
            translate_loc_from_full_to_compressed=lambda ids: ids,
            full_to_hisparse_device_index_mapping=mapping,
        )
        freed_device, freed_host = set(), set()
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1,
            translate_loc_from_full_to_swa=lambda ids: ids,
            free_hisparse_indices=lambda ids: freed_device.update(ids.tolist()),
            free_hisparse_mapping_only=lambda ids: mapping.index_fill_(0, ids, 0),
        )
        coordinator.mem_pool_host = SimpleNamespace(
            allocated_host_indices=lambda table, row, length: table[row, : int(length)],
            free=lambda ids: freed_host.update(ids.tolist()),
        )
        coordinator.req_to_host_pool = torch.tensor([[20, 21, 22]])
        coordinator.req_to_host_pool_allocated_len = torch.tensor([3])
        coordinator.req_to_device_buffer = torch.tensor([[8, 9, 10]])
        coordinator.req_device_buffer_size = torch.tensor([3])
        coordinator.req_device_buffer_tokens = torch.tensor([[[0, 1, 2]]] * 2)
        coordinator.req_device_buffer_token_locs = torch.tensor([[[8, 9, 10]]] * 2)
        coordinator._lru_init = torch.arange(3)
        coordinator.lru_slots = torch.zeros((2, 1, 3), dtype=torch.int64)
        coordinator._skip_first_backup = [False]
        coordinator.spec_swap = HiSparseSpecSwapManager(coordinator, num_draft_tokens=0)
        req = SimpleNamespace(
            kv=ReqKvInfo(req_pool_idx=0, kv_allocated_len=3, kv_committed_len=3),
            hisparse_staging=False,
            hisparse_retraction_pending=False,
        )
        return coordinator, req, freed_device, freed_host

    def test_release_request_reclaims_decode_host_and_device_state(self):
        """Normal completion must reach HiSparse cleanup before the row is freed."""
        coordinator, req, freed_device, freed_host = self._retraction_fixture()
        coordinator.release_request(req)
        self.assertEqual(freed_device, {8, 9, 10})
        self.assertEqual(freed_host, {20, 21, 22})
        self.assertEqual(int(coordinator.req_device_buffer_size[0]), 0)
        self.assertTrue(torch.all(coordinator.req_to_host_pool == -1))
        self.assertEqual(
            coordinator.mem_pool_device.full_to_hisparse_device_index_mapping.count_nonzero(),
            0,
        )

    def test_retract_flushes_the_committed_tail_before_free(self):
        """A forward's last KV token may not have reached eager backup yet."""
        for ratio, committed, already_backed in (
            (1, 3, False),
            (4, 12, False),
            (4, 11, False),
            (4, 12, True),
        ):
            with self.subTest(
                ratio=ratio, committed=committed, already_backed=already_backed
            ):
                coordinator, req, freed_device, freed_host = self._retraction_fixture()
                coordinator.is_dsv4_hisparse = ratio == 4
                coordinator.compress_ratio = ratio
                coordinator.device_buffer_size = 2
                coordinator._skip_first_backup[0] = already_backed
                req.kv.kv_allocated_len = req.kv.kv_committed_len = committed
                coordinator.req_to_token_pool.req_to_token = torch.arange(
                    committed
                ).reshape(1, -1)
                coordinator.token_to_kv_pool_allocator.get_kvcache = lambda: (
                    SimpleNamespace(get_device_pool_infos=lambda: ())
                )
                host_values = torch.full((32,), -1)
                device_values = torch.arange(16) + 100
                pending = []

                def backup(host_ids, device_ids):
                    pending.append(
                        lambda: host_values.index_copy_(
                            0, host_ids, device_values[device_ids]
                        )
                    )

                def synchronize():
                    for copy in pending:
                        copy()
                    pending.clear()

                coordinator.mem_pool_host.alloc_paged_token_slots = (
                    lambda table, lengths, row, start, size: table[
                        row, start : start + size
                    ]
                )
                coordinator.mem_pool_host.backup_tokens_from_device_all_layer = backup
                coordinator.mem_pool_host.backup_from_device_all_layer = (
                    lambda pool, host_ids, device_ids, **kw: backup(
                        host_ids, device_ids
                    )
                )
                needs_flush = committed % ratio == 0 and not already_backed

                def free(ids):
                    if needs_flush:
                        self.assertEqual(int(host_values[22]), int(device_values[10]))
                    freed_device.update(ids.tolist())

                coordinator.token_to_kv_pool_allocator.free_hisparse_indices = free
                event = SimpleNamespace(
                    record=lambda stream: None, synchronize=synchronize
                )
                with patch(
                    "sglang.srt.managers.hisparse_coordinator.device_module",
                    SimpleNamespace(Event=lambda: event, current_stream=lambda: None),
                ):
                    coordinator.retract_req_state_preserving(req)
                self.assertEqual(
                    int(host_values[22]), int(device_values[10]) if needs_flush else -1
                )
                self.assertEqual(freed_device, {8, 9, 10})
                self.assertEqual(freed_host, set())
                self.assertTrue(req.hisparse_retraction_pending)

    def test_retract_releases_rotated_ownership_and_scratch(self):
        """Free the current cache/scratch union, including per-layer rotations."""
        coordinator, req, freed_device, freed_host = self._retraction_fixture()
        coordinator.spec_swap.enabled = True
        coordinator.spec_swap._scratch_reqs = {0}
        coordinator.spec_swap.req_to_scratch = torch.tensor([[[8, 12]], [[9, 11]]])
        coordinator.req_device_buffer_token_locs = torch.tensor(
            [[[11, 9, 10]], [[8, 12, 10]]]
        )
        coordinator.retract_req_state_preserving(req)
        self.assertEqual(freed_device, {8, 9, 10, 11, 12})
        self.assertEqual(freed_host, set())
        self.assertNotIn(0, coordinator.spec_swap._scratch_reqs)
        self.assertEqual(coordinator.spec_swap.req_to_scratch.count_nonzero(), 0)

    def test_long_resume_invalidates_spec_cache_index(self):
        """A fresh all-miss buffer must not retain the reset's hot-token index."""
        coordinator, req, _, _ = self._retraction_fixture()
        coordinator.top_k = 1024
        coordinator.device_buffer_size = 4096
        coordinator.padded_buffer_size = 4160
        coordinator.mem_pool_device.page_size = 64
        coordinator.req_to_token_pool.req_to_token = torch.arange(4097).reshape(1, -1)
        coordinator.req_to_device_buffer = torch.zeros((1, 4160), dtype=torch.int64)
        coordinator.req_device_buffer_tokens = torch.full(
            (2, 1, 4160), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.full_like(
            coordinator.req_device_buffer_tokens, -1
        )
        coordinator._device_buffer_arange_i32 = torch.arange(4096, dtype=torch.int32)
        coordinator.token_to_kv_pool_allocator.alloc_device_buffer = lambda *args: (
            torch.arange(64, 4224)
        )
        coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator = (
            SimpleNamespace(alloc=lambda size: torch.arange(8192, 8192 + size))
        )
        coordinator.spec_swap = HiSparseSpecSwapManager(coordinator, num_draft_tokens=2)
        req.kv.set_row_len(4097)
        req.hisparse_retraction_pending = True
        coordinator.resume_after_retraction(req)
        self.assertTrue(torch.all(coordinator.spec_swap._cache_index == -1))
        self.assertTrue(
            torch.all(coordinator.req_device_buffer_tokens[:, 0, :4096] == -1)
        )
        self.assertIn(0, coordinator.spec_swap._scratch_reqs)
        self.assertFalse(req.hisparse_retraction_pending)

    def test_dsv4_retraction_preserves_every_non_offloaded_native_region(self):
        """Reused pages must not erase C128 KV or page-owned compressor rings."""
        with get_context().override_server_args(page_size=256):
            pool = DeepSeekV4TokenToKVPool(
                max_num_reqs=2,
                swa_size=2048,
                c4_size=512,
                c128_size=16,
                c4_state_pool_size=64,
                c128_state_pool_size=2,
                page_size=256,
                swa_page_size=256,
                dtype=torch.float8_e4m3fn,
                c4_state_dtype=torch.float32,
                c128_state_dtype=torch.float32,
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                indexer_head_dim=128,
                layer_num=3,
                device="cpu",
                enable_memory_saver=False,
                compression_ratios=[4, 128, 4],
                full_size=2048,
                enable_hisparse=True,
            )
        coordinator, req, _, _ = self._retraction_fixture()
        coordinator.is_dsv4_hisparse = True
        coordinator.compress_ratio = 4
        coordinator.sliding_window_size = 128
        coordinator.device_buffer_size = 2
        coordinator.padded_buffer_size = 3
        coordinator._device_buffer_arange_i32 = torch.arange(2, dtype=torch.int32)
        coordinator.token_to_kv_pool_allocator.page_size = 256
        coordinator.token_to_kv_pool_allocator.get_kvcache = lambda: pool
        coordinator.token_to_kv_pool_allocator.translate_loc_from_full_to_swa = (
            lambda ids: ids + 256
        )
        coordinator.token_to_kv_pool_allocator.alloc_device_buffer = lambda ids, size: (
            torch.tensor([8, 9, 10])
        )
        coordinator.mem_pool_device.full_to_hisparse_device_index_mapping.resize_(4096)
        coordinator.req_to_token_pool.req_to_token = torch.cat(
            (torch.arange(256, 512), torch.arange(768, 1024), torch.tensor([1280]))
        ).reshape(1, -1)
        req.kv.set_row_len(513)
        req.extend_end = 513
        # C128 compression state is request-owned, so retaining the row preserves it.
        request_state = pool.compress_state_pools[1].kv_score_buffer.kv_score
        request_state.fill_(42)
        expected, all_buffers = [], []
        for region in pool.get_device_pool_infos():
            info = region.buffer_info
            buffers = (
                info.buffers
                if isinstance(info, DSV4PagedBufferInfo)
                else tuple(state.pages for state in info.states)
            )
            for buffer in buffers:
                buffer.view(torch.uint8).fill_(len(all_buffers) + 1)
                all_buffers.append(buffer)
            if region.pool_name == PoolName.DEEPSEEK_V4_C4:
                continue  # Already held in the persistent HiSparse host pool.
            # Every KV region spans FULL pages. Window regions span only the
            # last two pages, including the partial page ending at position 513.
            old_rows = torch.tensor(
                [4, 6] if region.indices_from_pool == PoolName.SWA else [1, 3, 5]
            )
            new_rows = old_rows + 1
            expected.extend(
                (region.pool_name, buffer, new_rows.clone(), buffer[old_rows].clone())
                for buffer in buffers
            )
        event = SimpleNamespace(record=lambda stream: None, synchronize=lambda: None)
        with (
            patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
            patch(
                "sglang.srt.managers.hisparse_coordinator.device_module",
                SimpleNamespace(Event=lambda: event, current_stream=lambda: None),
            ),
        ):
            for _ in range(2):
                coordinator._skip_first_backup[0] = True
                coordinator.retract_req_state_preserving(req)
                for buffer in all_buffers:
                    buffer.zero_()  # Another request may reuse every freed page.
                coordinator.req_to_token_pool.req_to_token += 256
                coordinator.resume_after_retraction(req)
                for name, buffer, rows, saved in expected:
                    with self.subTest(region=name):
                        torch.testing.assert_close(buffer[rows], saved)
                    rows += 1
                self.assertTrue((request_state == 42).all())
                self.assertIsNone(req.hisparse_retraction_snapshot)
                self.assertTrue(all((b == 0).all() for b in pool.c4_kv_pool.kv_buffer))

    def test_retraction_snapshot_uses_native_rows_and_validates_before_writes(self):
        """Smaller native pages survive, and a later invalid region cannot partly restore."""
        swa = torch.full((80, 3), 7, dtype=torch.uint8)
        scale = torch.full((40, 5), 9, dtype=torch.uint8)

        def region(name, source, buffer):
            return DevicePoolInfo(
                pool_name=name,
                indices_from_pool=source,
                layer_ids=(9,),
                buffer_info=DSV4PagedBufferInfo.from_buffers(
                    [buffer],
                    page_slots=8,
                    layout="index_fp4",
                ),
            )

        inventory = [
            region(PoolName.SWA, PoolName.SWA, swa),
            region(PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE, PoolName.KV, scale),
        ]
        allocator = SimpleNamespace(
            page_size=256,
            get_kvcache=lambda: SimpleNamespace(
                get_device_pool_infos=lambda: inventory
            ),
            translate_loc_from_full_to_swa=lambda ids: ids,
        )
        old_locs = torch.arange(256, 512)
        snapshot = DSV4RetractionSnapshot.capture(allocator, old_locs, 128)
        swa.zero_()
        scale.zero_()
        inventory[1] = region(
            PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
            PoolName.KV,
            scale[:, :4].contiguous(),
        )
        with self.assertRaisesRegex(ValueError, "layout changed"):
            snapshot.restore(allocator, old_locs - 256)
        self.assertEqual(swa.count_nonzero(), 0)
        inventory[1] = region(PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE, PoolName.KV, scale)
        snapshot.restore(allocator, old_locs - 256)
        self.assertTrue((swa[:32] == 7).all())
        self.assertTrue((scale[:8] == 9).all())
        self.assertEqual(swa[32:].count_nonzero(), 0)
        self.assertEqual(scale[8:].count_nonzero(), 0)

    def test_unknown_retraction_region_fails_before_releasing_device_state(self):
        """A new FULL region needs an explicit location mapping before it can be freed."""
        coordinator, req, freed_device, freed_host = self._retraction_fixture()
        coordinator.is_dsv4_hisparse = True
        coordinator._skip_first_backup[0] = True
        unknown = DevicePoolInfo(
            pool_name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            layer_ids=(0,),
            buffer_info=DSV4PagedBufferInfo.from_buffers(
                [torch.ones((8, 4), dtype=torch.uint8)],
                page_slots=1,
                layout="v4",
            ),
        )
        coordinator.token_to_kv_pool_allocator.get_kvcache = lambda: SimpleNamespace(
            get_device_pool_infos=lambda: (unknown,),
        )
        with self.assertRaisesRegex(ValueError, "Unknown HiSparse FULL region"):
            coordinator.retract_req_state_preserving(req)
        self.assertFalse(freed_device)
        self.assertFalse(freed_host)
        self.assertEqual(int(coordinator.req_device_buffer_size[0]), 3)
        self.assertFalse(req.hisparse_retraction_pending)

    def test_retirement_discards_retained_host_state_and_row(self):
        """An ended retracted request releases its backup through normal retirement."""
        coordinator, original, _, freed_host = self._retraction_fixture()
        req = FakeReq(**vars(original), finished_reason=None)
        coordinator._skip_first_backup[0] = True
        coordinator.retract_req_state_preserving(req)
        req.kv.mark_kv_released()
        req.hisparse_retraction_snapshot = object()
        freed_rows = []

        def free_row(request):
            freed_rows.append(request.kv.req_pool_idx)
            request.kv.req_pool_idx = None

        coordinator.req_to_token_pool.free = free_row
        retirer = object.__new__(RequestRetirer)
        retirer.tree_cache = SimpleNamespace()
        retirer.hisparse_coordinator = coordinator
        retirer.release(req, checkpoint=False)
        self.assertEqual(freed_host, {20, 21, 22})
        self.assertEqual(freed_rows, [0])
        self.assertFalse(req.kv.holds_kv)
        self.assertFalse(req.hisparse_retraction_pending)
        self.assertIsNone(req.hisparse_retraction_snapshot)
        self.assertTrue(torch.all(coordinator.req_to_host_pool == -1))

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
        req = SimpleNamespace(kv=ReqKvInfo(kv_allocated_len=2))
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
            page_size=1,
            translate_loc_from_full_to_swa=lambda ids: ids,
            alloc_device_buffer=MagicMock(return_value=buffer_indices),
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
            page_size=1,
            translate_loc_from_full_to_swa=lambda ids: ids,
            full_to_hisparse_device_index_mapping=full_to_device_mapping,
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
                if func == torch.ops.aten.index_put_.default:
                    value = args[2]
                    if value.device.type == "cpu" and value.ndim == 0:
                        raise AssertionError(
                            "Blocking CPU scalar copy in CUDA index_put"
                        )
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
