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
import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.hisparse_spec_coordinator import HiSparseSpecCoordinator
from sglang.srt.mem_cache.hisparse_spec_state import VerifyRow


class Event:
    def __init__(self):
        self.done = False

    def record(self, stream):
        pass

    def query(self):
        return self.done


class TestCoordinator(unittest.TestCase):
    def test_resident_draft_span_includes_high_logical_ids(self):
        allocator = HiSparseTokenToKVPoolAllocator(
            128, 64, torch.float32, "cpu", MagicMock(), False, 4
        )
        self.assertEqual(allocator.draft_virtual_id_space, 512)
        # Pools add one sentinel page: the highest valid logical ID must fit.
        highest_id = allocator.full_to_hisparse_device_index_mapping.numel() - 2
        self.assertEqual(highest_id, allocator.draft_virtual_id_space + 64 - 1)
        self.assertGreater(highest_id, allocator._size_hisparse + 64 - 1)

    def setUp(self):
        self.events = []

        def event():
            e = Event()
            self.events.append(e)
            return e

        self.dm = SimpleNamespace(
            Event=event,
            current_stream=lambda: MagicMock(),
            stream=lambda _: contextlib.nullcontext(),
            synchronize=self.complete,
        )
        self.allocator = HiSparseTokenToKVPoolAllocator(
            1024, 64, torch.float32, "cpu", MagicMock(), False, 4
        )
        # Allocate the independent hot page before the provisional arena.
        hot = self.allocator.hisparse_attn_allocator.alloc(64)
        self.tags = {}
        self.host = MagicMock(page_size=64)
        self.c = SimpleNamespace(
            is_dsv4_hisparse=False,
            is_m3_hisparse=False,
            compress_ratio=1,
            mem_pool_device=SimpleNamespace(page_size=64),
            mem_pool_host=self.host,
            token_to_kv_pool_allocator=self.allocator,
            device="cpu",
            _copy_speculative_union=MagicMock(side_effect=self.copy),
            _prepare_speculative_stream=MagicMock(),
            device_buffer_size=4,
            req_device_buffer_size=torch.tensor([64]),
            req_device_buffer_tokens=torch.full((2, 1, 4), -1),
            req_device_buffer_token_locs=hot[:4].view(1, 1, 4).repeat(2, 1, 1),
            lru_slots=torch.arange(4).view(1, 1, 4).repeat(2, 1, 1),
            req_to_host_pool=torch.arange(128, 384).view(1, 256),
            req_to_host_pool_allocated_len=torch.tensor([64]),
            decode_backup_stream=MagicMock(),
            decode_producer_stream=None,
        )
        self.req = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=0), hisparse_staging=False
        )
        self.m = HiSparseSpecCoordinator(self.c, self.dm)
        self.key = self.m.prepare(self.req, 63, (127, 128), 65)

    def complete(self):
        for e in self.events:
            e.done = True

    def copy(self, layer, src, dst, count, real):
        self.assertEqual(src.dtype, torch.int64)
        self.assertEqual(dst.dtype, torch.int32)
        self.assertEqual(count.dtype, torch.int32)
        self.assertEqual(real.tolist(), [1])
        self.assertEqual(src.shape, dst.shape)
        self.assertEqual(src.shape, (1, int(count[0])))
        self.assertEqual(src.stride(0), dst.stride(0))
        for h, d in zip(src[0].tolist(), dst[0].tolist()):
            self.tags[layer, d] = (self.key.request_generation, layer, h - 128)

    def test_union_layers_and_reader_lifetime(self):
        for layer, selections in enumerate(
            (((0, 1, -1), (2, 3, -1)), ((3, 4, -1), (5, 6, -1)))
        ):
            rows = [VerifyRow(self.key, 63 + i, r) for i, r in enumerate(selections)]
            table = self.m.stage_layer(self.req, self.key, layer, rows)
            for i, row in enumerate(selections):
                for j, position in enumerate(row):
                    if position >= 0:
                        self.assertEqual(
                            self.tags[layer, int(table[i, j])], (0, layer, position)
                        )
                    else:
                        self.assertEqual(int(table[i, j]), -1)
        reader = self.m.add_reader(self.req, self.key)
        self.complete()
        self.assertFalse(self.m.retire(self.req, self.key, cancel=True))
        self.assertEqual(len(self.m.owners[0].tables), 2)
        reader.record(None)
        self.complete()
        self.assertTrue(self.m.retire(self.req, self.key, cancel=True))
        # Re-arm the identical logical reservation without allocator growth.
        key = self.m.prepare(self.req, 63, (127, 128), 65)
        self.assertGreater(key.iteration_id, self.key.iteration_id)
        self.m.teardown(self.req)

    def test_noncontiguous_physical_token_rows_and_wrapper(self):
        from unittest.mock import patch
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        self.c.req_device_buffer_token_locs[0, 0] = torch.tensor([65, 79, 97, 126])
        table = self.m.stage_layer(
            self.req, self.key, 0, [VerifyRow(self.key, 63, (7, 9, 11, 13))]
        )
        self.assertEqual(table.tolist(), [[65, 79, 97, 126]])
        args = self.c._copy_speculative_union.call_args.args
        self.assertEqual(args[1].tolist(), [[135, 137, 139, 141]])
        self.assertEqual(args[2].tolist(), [[65, 79, 97, 126]])
        c = HiSparseCoordinator.__new__(HiSparseCoordinator)
        c.mem_pool_host = SimpleNamespace(kv_buffer=[object()])
        c.mem_pool_device = SimpleNamespace(kv_buffer=[object()])
        c.item_size_bytes = 1152
        with patch(
            "sglang.srt.managers.hisparse_coordinator.copy_cache_planned_mla"
        ) as copy:
            c._copy_speculative_union(*args)
        kw = copy.call_args.kwargs
        self.assertIs(kw["host_cache"], c.mem_pool_host.kv_buffer[0])
        self.assertIs(kw["device_buffer"], c.mem_pool_device.kv_buffer[0])
        self.assertIs(kw["miss_src"], args[1])
        self.assertIs(kw["miss_dst"], args[2])
        self.assertFalse(kw["is_dsv4_layout"])
        self.assertFalse(kw["skip_io"])
        self.assertEqual(kw["item_size_bytes"], 1152)
        self.c._prepare_speculative_stream.assert_called_once()

    def test_prepare_orders_prior_streams(self):
        from unittest.mock import patch
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        c = HiSparseCoordinator.__new__(HiSparseCoordinator)
        c.write_staging_stream = object()
        c.decode_backup_stream = object()
        c.decode_producer_stream = object()
        c.prefetch_stream = object()
        c.enable_prefetch = True
        c.wait_for_pending_backup = MagicMock()
        stream = MagicMock()
        with patch("sglang.srt.managers.hisparse_coordinator.device_module") as dm:
            dm.current_stream.return_value = stream
            c._prepare_speculative_stream()
        self.assertEqual(
            [call.args[0] for call in stream.wait_stream.call_args_list],
            [
                c.write_staging_stream,
                c.decode_backup_stream,
                c.decode_producer_stream,
                c.prefetch_stream,
            ],
        )
        stream.synchronize.assert_called_once()
        c.wait_for_pending_backup.assert_called_once()

    def test_pin_hit_before_misses(self):
        slot = int(self.c.req_device_buffer_token_locs[0, 0, 0])
        self.c.req_device_buffer_tokens[0, 0, 0] = 3
        self.tags[0, slot] = (0, 0, 3)
        rows = [VerifyRow(self.key, 63, (3,)), VerifyRow(self.key, 64, (4,))]
        table = self.m.stage_layer(self.req, self.key, 0, rows)
        self.assertEqual(int(table[0, 0]), slot)
        self.assertEqual(self.tags[0, slot], (0, 0, 3))

    def test_slot_reuse_and_stale_callback(self):
        self.m.teardown(self.req)
        other = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=0), hisparse_staging=False
        )
        key = self.m.prepare(other, 63, (127, 128), 65)
        self.assertGreater(key.request_generation, self.key.request_generation)
        with self.assertRaises(ValueError):
            self.m.commit(self.req, self.key, 1)
        with self.assertRaises(ValueError):
            self.m.teardown(self.req)
        self.m.teardown(other)

    def test_accept_root_then_retire(self):
        self.m.verified(self.req, self.key)
        plan = self.m.commit(self.req, self.key, 1)
        self.assertEqual(plan.positions, (63,))
        self.assertFalse(self.m.retire(self.req, self.key))
        self.complete()
        self.assertTrue(self.m.retire(self.req, self.key))
        self.assertEqual(self.m.backend.valid_lengths[0], (0, 64))

    def test_partial_union_copy_invalidates_hot_metadata(self):
        self.c.req_device_buffer_tokens[0, 0] = torch.arange(4)
        self.c._copy_speculative_union.side_effect = RuntimeError("partial")
        with self.assertRaises(RuntimeError):
            self.m.stage_layer(self.req, self.key, 0, [VerifyRow(self.key, 63, (4,))])
        self.assertTrue((self.c.req_device_buffer_tokens[0, 0] == -1).all())
        with self.assertRaises(ValueError):
            self.m.verified(self.req, self.key)
        self.m.teardown(self.req)

    def test_actual_coordinator_teardown_and_inactive_guards(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        c = HiSparseCoordinator.__new__(HiSparseCoordinator)
        # Inactive path never reads/synchronizes request index tensors.
        indices = MagicMock()
        c._speculative_check_idle(indices)
        indices.tolist.assert_not_called()
        c._speculative_verifier = self.m
        with self.assertRaises(ValueError):
            c._speculative_check_idle(torch.tensor([0]))
        c._speculative_teardown(self.req)
        self.assertEqual(self.m.owners, {})
        self.m.admit(self.req)
        c._speculative_check_idle(indices)
        indices.tolist.assert_not_called()

    def test_copy_failure_reconciles_registry_and_stale_query(self):
        self.m.verified(self.req, self.key)
        self.host.backup_from_device_all_layer.side_effect = RuntimeError("copy")
        with self.assertRaises(RuntimeError):
            self.m.commit(self.req, self.key, 1)
        self.assertIsNone(self.m.owners[0].key)
        self.assertFalse(self.m.lifecycle.is_active(self.key))
        key = self.m.prepare(self.req, 63, (127, 128), 65)
        self.assertTrue(self.m.lifecycle.is_active(key))
        self.assertFalse(self.m.lifecycle.is_active(self.key))

    def test_host_allocation_failure_reconciles_registry(self):
        self.m.teardown(self.req)
        key = self.m.prepare(self.req, 64, (127, 128), 65)
        self.host.alloc_page.return_value = None
        self.m.verified(self.req, key)
        with self.assertRaises(MemoryError):
            self.m.commit(self.req, key, 1)
        self.assertIsNone(self.m.owners[0].key)

    def test_teardown_refuses_unrecorded_reader(self):
        reader = self.m.add_reader(self.req, self.key)
        with self.assertRaises(RuntimeError):
            self.m.teardown(self.req)
        self.assertIsNotNone(self.m.owners[0].arena)
        reader.record(None)
        self.m.teardown(self.req)


if __name__ == "__main__":
    unittest.main()
