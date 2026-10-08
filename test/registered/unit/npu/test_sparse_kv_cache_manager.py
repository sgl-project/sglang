"""CPU coverage of cache policies and PD lifecycle; NPU kernels are mocked."""

import importlib.util
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import config
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _copy_rows(src, dst, src_index, dst_index, mask, src_ndims, dst_ndims, **kwargs):
    src_rows = src.flatten(0, src_ndims - 1)
    dst_rows = dst.flatten(0, dst_ndims - 1)
    dst_rows[dst_index[mask]] = src_rows[src_index[mask]]


def _lookup(slot_map, reqs, topk, pos_mask_size=None):
    valid = (topk >= 0) & (topk < slot_map.shape[1])
    positions = slot_map[reqs.long().unsqueeze(1), topk.clamp(0).long()]
    positions = torch.where(valid, positions, -1)
    hits = positions >= 0
    if pos_mask_size is None:
        return hits.int(), positions
    hit_mask = torch.zeros((len(reqs), pos_mask_size), dtype=torch.int32)
    for row in range(len(reqs)):
        hit_mask[row, positions[row, hits[row]].long()] = 1
    return hits.int(), positions, hit_mask


def _host_buffer(shape, dtype, **kwargs):
    tensor = torch.zeros(shape, dtype=dtype)
    return tensor, tensor.data_ptr(), tensor.data_ptr()


class _RequestKV:
    req_pool_idx = None
    kv_allocated_len = 1

    @property
    def holds_kv(self):
        return self.req_pool_idx is not None


class TestSparseKVCacheManager(unittest.TestCase):
    def setUp(self):
        # Load a private module instance so the fake kernels do not leak into
        # other tests or require an installed sgl-kernel-npu on CPU workers.
        self.kernels = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")
        self.kernels.create_shm_tensor = _host_buffer
        self.kernels.slot_map_lookup = _lookup
        self.kernels.unidex_copy_inplace = _copy_rows
        self.kernel_modules = {
            "sgl_kernel_npu": ModuleType("sgl_kernel_npu"),
            "sgl_kernel_npu.sparsity_driven_kv_offload": self.kernels,
        }
        spec = importlib.util.spec_from_file_location(
            "_sparse_kv_manager_under_test",
            Path(config.__file__).with_name("manager.py"),
        )
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, self.kernel_modules):
            spec.loader.exec_module(self.module)
        self.npu = SimpleNamespace(
            Stream=Mock(side_effect=lambda: Mock()),
            Event=Mock(side_effect=lambda: Mock()),
            stream=lambda stream: nullcontext(),
            current_device=lambda: 0,
        )
        patcher = patch.object(torch, "npu", self.npu, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_manager(self, topk=1536, enable_lru=False, factor=2):
        pool = ReqToTokenPool(2, 4096, "cpu", False)
        kv = MLATokenToKVPool.__new__(MLATokenToKVPool)
        kv.start_layer = 0
        kv.layer_num = 2
        kv.kv_lora_rank = 2
        kv.qk_rope_head_dim = 2
        kv.store_dtype = torch.float32
        kv.page_size = 128
        allocator = SimpleNamespace(get_kvcache=lambda: kv)
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_NPU_SPARSE_KV_ENABLE_LRU": str(int(enable_lru)),
                    "SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR": str(factor),
                },
            ),
            patch.dict(sys.modules, self.kernel_modules),
        ):
            manager = self.module.SparseKVCacheManager(pool, allocator, topk)
        for buf in manager.host_kv_buffer:
            buf.copy_(torch.arange(buf.numel()).reshape(buf.shape))
        return manager, pool

    def materialize(self, manager, tokens, reqs=(2, 0), lengths=(32, 0)):
        topk = torch.full(
            (len(reqs), manager.sparse_context_len), -1, dtype=torch.int32
        )
        topk[0, : len(tokens)] = torch.tensor(tokens)
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor(reqs), seq_lens=torch.tensor(lengths)
        )
        selected = torch.zeros((len(reqs), manager.sparse_context_len, 1, 4))
        stream = Mock()
        ready = Mock()
        plan = manager.materialize_selected_kv(
            SimpleNamespace(layer_id=0),
            batch,
            topk,
            selected,
            stream,
            host_kv_ready_event=ready,
        )
        manager.refill_selected_kv(
            SimpleNamespace(layer_id=0), selected, *plan[:4], stream
        )
        manager._materialize_h2d_miss_stream.wait_event.assert_called_with(ready)
        return selected, plan

    def test_dynamic_window_reorders_hits_and_refills_misses(self):
        for topk in (3, 1536, 3072):
            with self.subTest(topk=topk):
                manager, _ = self.make_manager(topk)
                # The mock module deliberately has no LRU exports.
                self.assertFalse(hasattr(manager, "device_lru_slots"))
                self.assertEqual(manager.device_cache_capacity, topk)
                self.materialize(manager, [10, 11, 12])
                selected, plan = self.materialize(manager, [12, 10, 13])
                torch.testing.assert_close(
                    selected[0, :3], manager.host_kv_buffer[0][2, [12, 10, 13]]
                )
                torch.testing.assert_close(
                    manager.device_kv_buffer[0][2, :3], selected[0, :3]
                )
                self.assertEqual(manager.device_slot_map[0][2, 11].item(), -1)
                self.assertEqual(manager.device_slot_map[0][2, 12].item(), 0)
                self.assertEqual(plan[-1].tolist(), [3, 0])
                self.assertEqual(selected[1].count_nonzero().item(), 0)

    def test_lru_keeps_hit_slots_and_only_refills_misses(self):
        self.kernels.fused_timestamp_lru_metadata_update_with_probation = Mock()
        self.kernels.parallel_lru_metadata_write = Mock()
        manager, _ = self.make_manager(2048, enable_lru=True)
        manager.device_slot_map[0][2, 10] = 3500
        manager.device_kv_buffer[0][2, 3500] = manager.host_kv_buffer[0][2, 10]
        victims = torch.full((2, 2048), -1, dtype=torch.int32)
        victims[0, 1] = 777
        manager._lru_metadata_update.return_value = (
            victims,
            torch.tensor([1, 0], dtype=torch.int32),
        )
        selected, plan = self.materialize(manager, [10, 11])
        torch.testing.assert_close(
            selected[0, :2], manager.host_kv_buffer[0][2, [10, 11]]
        )
        torch.testing.assert_close(manager.device_kv_buffer[0][2, 777], selected[0, 1])
        torch.testing.assert_close(manager.device_kv_buffer[0][2, 3500], selected[0, 0])
        self.assertEqual(plan[3].sum().item(), 1)
        manager._lru_metadata_update.assert_called_once()
        manager._lru_metadata_write.assert_called_once()

    def test_lru_rejects_dynamic_topk_before_allocating_buffers(self):
        with self.assertRaisesRegex(ValueError, "index_topk=2048"):
            self.make_manager(1536, enable_lru=True)
        self.npu.Stream.assert_not_called()

    def test_pd_request_release_resets_lru_and_removes_room(self):
        self.kernels.fused_timestamp_lru_metadata_update_with_probation = Mock()
        self.kernels.parallel_lru_metadata_write = Mock()
        manager, pool = self.make_manager(2048, enable_lru=True)
        req = SimpleNamespace(
            kv=_RequestKV(),
            bootstrap_room=42,
            origin_input_ids=[1, 2, 3],
        )
        row = pool.alloc([req])[0]
        self.assertEqual(manager.get_pd_copy_metadata(42), (row, 3))
        manager.device_slot_map[0][row, 1] = 7
        manager.device_slot_tokens[0][row, 7] = 1
        manager.device_lru_slot_stamps[0][row].fill_(9)
        pool.free(req)
        self.assertEqual(pool.available_size(), 2)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertTrue((manager.device_slot_map[0][row] == -1).all())
        self.assertTrue((manager.device_slot_tokens[0][row] == -1).all())
        self.assertEqual(manager.device_lru_slot_stamps[0][row].sum().item(), 0)
        with self.assertRaisesRegex(RuntimeError, "not recorded"):
            manager.get_pd_copy_metadata(42)
        pool.alloc([req])
        pool.clear()
        self.assertEqual(manager._pd_room_to_req_pool_idx, {})

    def test_reset_failure_still_frees_request_and_pd_metadata(self):
        manager, pool = self.make_manager()
        req = SimpleNamespace(
            kv=_RequestKV(),
            bootstrap_room=42,
            origin_input_ids=[1],
        )
        pool.alloc([req])
        manager.reset_requests = Mock(side_effect=RuntimeError("reset failed"))
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            pool.free(req)
        self.assertEqual(pool.available_size(), 2)
        self.assertEqual(manager._pd_room_to_req_pool_idx, {})

    def test_pd_staging_resets_cache_before_copying_new_request(self):
        manager, _ = self.make_manager()
        manager.ensure_pd_decode_staging_buffers()
        manager.pd_decode_k_staging[0][0, :3].fill_(7)
        manager.pd_decode_v_staging[0][0, :3].fill_(8)
        manager.device_slot_map[0][2, 0] = 1
        manager.offload_pd_decode_staging_to_host(0, 2, 3)
        self.assertTrue((manager.device_slot_map[0][2] == -1).all())
        self.assertTrue((manager.host_kv_buffer[0][2, :3, :, :2] == 7).all())
        self.assertTrue((manager.host_kv_buffer[0][2, :3, :, 2:] == 8).all())


if __name__ == "__main__":
    unittest.main()
