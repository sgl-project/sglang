"""NPU PD decode retraction configuration and paged KV round trips."""

import unittest
from array import array
from contextlib import ExitStack
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.hardware_backend.npu.utils import set_default_server_args
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import backup_kv_cache, restore_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="stage-a-unit-test-npu")


class TestNPURetractionBackupArgs(unittest.TestCase):
    def _args(self, **kwargs):
        args = ServerArgs(model_path="dummy", device="npu", **kwargs)
        parse_cuda_graph_config(args)
        with (
            patch(
                "sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity",
                return_value=64 * 1024,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.utils.use_mla_backend",
                return_value=False,
            ),
        ):
            set_default_server_args(args)
        return args

    def test_explicit_and_inferred_host_pool_use_ascend_io(self):
        for backend in (None, "host_pool"):
            with self.subTest(backend=backend):
                args = self._args(
                    disaggregation_mode="decode",
                    disaggregation_decode_retraction_backup=backend,
                )
                self.assertEqual(
                    resolution_result(args, "hicache_io_backend"), "kernel_ascend"
                )
                self.assertEqual(
                    resolution_result(args, "hicache_mem_layout"), "page_first_direct"
                )

    def test_cpu_tensor_and_none_need_no_host_pool_io(self):
        for backend in ("cpu_tensor", "none"):
            with self.subTest(backend=backend):
                args = self._args(
                    disaggregation_mode="decode",
                    disaggregation_decode_retraction_backup=backend,
                )
                self.assertEqual(
                    resolution_result(args, "hicache_io_backend"), "kernel"
                )
                self.assertEqual(
                    resolution_result(args, "hicache_mem_layout"), "page_first"
                )

    def test_prefill_and_colocated_defaults_need_no_host_pool_io(self):
        for mode in ("null", "prefill"):
            with self.subTest(mode=mode):
                args = self._args(disaggregation_mode=mode)
                self.assertEqual(
                    resolution_result(args, "hicache_io_backend"), "kernel"
                )

    def test_hicache_and_decode_offload_still_use_ascend_io(self):
        for kwargs in (
            {"enable_hierarchical_cache": True},
            {"disaggregation_decode_enable_offload_kvcache": True},
        ):
            with self.subTest(kwargs=kwargs):
                args = self._args(**kwargs)
                self.assertEqual(
                    resolution_result(args, "hicache_io_backend"), "kernel_ascend"
                )


@unittest.skipUnless(is_npu(), "requires an NPU")
class TestNPURetractionBackup(unittest.TestCase):
    page_size = 128
    pool_size = 512
    num_tokens = 135

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.scope = ExitStack()
        self.addCleanup(self.scope.close)

    def _cache(self, *, fia, ratio=1.0):
        args = ServerArgs(
            model_path="dummy",
            device="npu",
            page_size=self.page_size,
            disaggregation_mode="decode",
            disaggregation_decode_retraction_backup="host_pool",
            hicache_io_backend="kernel_ascend",
            hicache_mem_layout="page_first_direct",
            hicache_ratio=ratio,
            disable_radix_cache=True,
            hicache_write_policy="write_back",
        )
        set_global_server_args_for_scheduler(args)
        self.scope.enter_context(get_parallel().override(dcp_enabled=False))
        with patch.dict("os.environ", {"ASCEND_USE_FIA": str(int(fia))}):
            pool = NPUMHATokenToKVPool(
                size=self.pool_size,
                page_size=self.page_size,
                head_num=2,
                head_dim=64,
                dtype=torch.bfloat16,
                layer_num=2,
                device="npu",
                enable_memory_saver=False,
            )
        pool.cpu_offloading_chunk_size = 7
        req_pool = ReqToTokenPool(
            size=2,
            max_context_len=self.pool_size,
            device="npu",
            enable_memory_saver=False,
        )
        allocator = PagedTokenToKVPoolAllocator(
            size=self.pool_size,
            page_size=self.page_size,
            dtype=pool.dtype,
            device="npu",
            kvcache=pool,
            need_sort=False,
        )
        params = CacheInitParams(
            disable=True,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=self.page_size,
            tree_components=(ComponentType.FULL,),
        )
        cache = UnifiedRadixCache(params)
        cache.init_hicache(args, params)
        self.addCleanup(cache.release_host_resources)
        cache.validate_retraction_host_capacity()
        return pool, req_pool, allocator, cache

    def _request(self, req_pool, allocator):
        req = Req(
            rid="retraction-test",
            origin_input_text="",
            origin_input_ids=array("q", [1] * self.num_tokens),
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        req.output_ids.append(2)
        self.assertIsNotNone(req_pool.alloc([req]))
        indices = allocator.alloc(2 * self.page_size)
        self.assertIsNotNone(indices)
        req_pool.write(
            (req.kv.req_pool_idx, slice(0, self.num_tokens)), indices[: self.num_tokens]
        )
        return req, indices

    def _buffers(self, pool):
        return [
            buffer.view(pool.layer_num, -1, pool.head_num, pool.head_dim)
            for buffer in pool.get_hicache_transfer_buffers()
        ]

    def _roundtrip(self, backend, fia):
        pool, req_pool, allocator, cache = self._cache(fia=fia)
        req, old_indices = self._request(req_pool, allocator)
        buffers = self._buffers(pool)
        expected = []
        for side, buffer in enumerate(buffers):
            pattern = torch.arange(
                buffer[:, old_indices].numel(), device="npu", dtype=torch.float32
            ).reshape_as(buffer[:, old_indices])
            buffer[:, old_indices] = (pattern % 251 + side * 1000).to(pool.dtype)
            expected.append(buffer[:, old_indices].clone())
        host_free = cache.host_pool_group.available_size()
        self.assertTrue(backup_kv_cache(req, cache, req_pool, allocator, backend))
        self.assertIsNotNone(req.kv.retraction_backup)
        for buffer in buffers:
            buffer.zero_()
        allocator.free(old_indices)
        # Another request consumes the freed pages before this request resumes.
        reused_indices = allocator.alloc(len(old_indices))
        new_indices = allocator.alloc(len(old_indices))
        self.assertIsNotNone(new_indices)
        self.assertFalse(torch.equal(old_indices, new_indices))
        req_pool.write(
            (req.kv.req_pool_idx, slice(0, self.num_tokens)),
            new_indices[: self.num_tokens],
        )
        restore_kv_cache(req, cache, req_pool, allocator, backend)
        self.assertIsNone(req.kv.retraction_backup)
        checked_tokens = len(old_indices) if backend == "host_pool" else self.num_tokens
        for buffer, saved in zip(buffers, expected, strict=True):
            self.assertTrue(
                torch.equal(
                    buffer[:, new_indices[:checked_tokens]], saved[:, :checked_tokens]
                )
            )
        self.assertEqual(cache.host_pool_group.available_size(), host_free)
        allocator.free(new_indices)
        allocator.free(reused_indices)
        req_pool.free(req)

    def test_cpu_tensor_restores_reallocated_paged_slots(self):
        for fia in (False, True):
            with self.subTest(fia=fia):
                self._roundtrip("cpu_tensor", fia)

    def test_host_pool_restores_reallocated_paged_slots(self):
        for fia in (False, True):
            with self.subTest(fia=fia):
                self._roundtrip("host_pool", fia)

    def test_exhausted_host_pool_does_not_leak_slots(self):
        _, req_pool, allocator, cache = self._cache(fia=False, ratio=0.1)
        req, indices = self._request(req_pool, allocator)
        host_free = cache.host_pool_group.available_size()
        self.assertFalse(backup_kv_cache(req, cache, req_pool, allocator, "host_pool"))
        self.assertIsNone(req.kv.retraction_backup)
        self.assertEqual(cache.host_pool_group.available_size(), host_free)
        allocator.free(indices)
        req_pool.free(req)

    def test_none_declines_backup(self):
        _, req_pool, allocator, cache = self._cache(fia=False)
        req, indices = self._request(req_pool, allocator)
        host_free = cache.host_pool_group.available_size()
        self.assertFalse(backup_kv_cache(req, cache, req_pool, allocator, "none"))
        self.assertIsNone(req.kv.retraction_backup)
        self.assertEqual(cache.host_pool_group.available_size(), host_free)
        allocator.free(indices)
        req_pool.free(req)


if __name__ == "__main__":
    unittest.main()
