"""Unquantized MHA direct-linker views: byte mapping and fail-closed boundaries."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    UnquantizedKVCacheMethod,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    _build_kv_device_pool_group,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ASSEMBLER = "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler"


class TestMHALinkerPoolAssembler(unittest.TestCase):
    def setUp(self):
        self.parallel_patch = patch(
            f"{_ASSEMBLER}.get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1),
        )
        self.memory_patch = patch(
            f"{_ASSEMBLER}.get_memory",
            return_value=SimpleNamespace(enable_unified_memory=False),
        )
        self.parallel = self.parallel_patch.start().return_value
        self.memory = self.memory_patch.start().return_value
        self.addCleanup(self.parallel_patch.stop)
        self.addCleanup(self.memory_patch.stop)
        self.params = SimpleNamespace(
            pp_size=1,
            attn_cp_size=1,
            mtp_draft_device_pools=(),
            mtp_draft_mamba_pools=(),
            sidecar_draft_device_pools=(),
            is_eagle=False,
            token_to_kv_pool_allocator=SimpleNamespace(
                state=None, hands_out_virtual_ids=False
            ),
        )

    def pool(self, dtype=torch.float32):
        # Actual native class and quantization property; no engine or GPU allocation.
        pool = object.__new__(MHATokenToKVPool)
        pool.size, pool.page_size, pool.layer_num = 8, 2, 3
        pool.start_layer, pool.end_layer = 0, 2
        pool.dtype = pool.store_dtype = dtype
        pool.head_num, pool.head_dim, pool.v_head_dim = 2, 3, 2
        pool.kv_cache_layout = "nhd"
        pool.post_capture_active = False
        pool.quant_method = UnquantizedKVCacheMethod()
        pool.k_buffer = [
            (torch.arange(60).reshape(10, 2, 3) + 100 * layer).to(dtype)
            for layer in range(3)
        ]
        # V can have a different head dimension; export every byte of both.
        pool.v_buffer = [
            (torch.arange(40).reshape(10, 2, 2) + 1000 + 100 * layer).to(dtype)
            for layer in range(3)
        ]
        for name in (
            "k_scale_buffer",
            "v_scale_buffer",
            "native_k_scale_buffer",
            "native_v_scale_buffer",
            "dq_k_buffer",
            "dq_v_buffer",
        ):
            setattr(pool, name, None)
        return pool

    def build(self, pool):
        return _build_kv_device_pool_group(pool, 2, params=self.params)

    def test_all_selected_k_and_v_bytes_round_trip_without_touching_other_pages(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                source, target = self.pool(dtype), self.pool(dtype)
                for tensor in [*target.k_buffer, *target.v_buffer]:
                    tensor.fill_(-1)
                group = self.build(source)
                self.assertEqual(set(group.entry_map), {PoolName.KV})
                source_entry = group.entry_map[PoolName.KV]
                target_entry = self.build(target).entry_map[PoolName.KV]
                slots = torch.tensor([0, 1, 6, 7])
                source_ptrs, source_sizes = source_entry.get_page_buffer_meta(slots)
                target_ptrs, target_sizes = target_entry.get_page_buffer_meta(slots)
                self.assertEqual(source_sizes, target_sizes)
                for src, dst, size in zip(source_ptrs, target_ptrs, source_sizes):
                    ctypes.memmove(dst, src, size)
                for src, dst in zip(
                    [*source.k_buffer, *source.v_buffer],
                    [*target.k_buffer, *target.v_buffer],
                ):
                    expected = torch.full_like(src, -1)
                    expected[slots] = src[slots]
                    self.assertTrue(
                        torch.equal(dst.view(torch.uint8), expected.view(torch.uint8))
                    )

    def test_rejects_topology_and_auxiliary_state_before_pointer_export(self):
        for name in (
            "pp_size",
            "attn_cp_size",
            "mtp_draft_device_pools",
            "mtp_draft_mamba_pools",
            "sidecar_draft_device_pools",
        ):
            with self.subTest(name=name):
                old = getattr(self.params, name)
                setattr(self.params, name, 2 if name.endswith("size") else (object(),))
                with self.assertRaisesRegex(ValueError, "PP=CP=DCP=1"):
                    self.build(self.pool())
                setattr(self.params, name, old)
        self.params.token_to_kv_pool_allocator.state = object()
        with self.assertRaisesRegex(ValueError, "Mamba state"):
            self.build(self.pool())

    def test_rejects_dcp_unified_memory_and_nonstandard_subclasses(self):
        self.parallel.attn_dcp_size = 2
        with self.assertRaisesRegex(ValueError, "DCP=1"):
            self.build(self.pool())
        self.parallel.attn_dcp_size = 1
        self.memory.enable_unified_memory = True
        with self.assertRaisesRegex(ValueError, "unified memory"):
            self.build(self.pool())
        self.memory.enable_unified_memory = False

        class ExtraStatePool(MHATokenToKVPool):
            pass

        with self.assertRaisesRegex(ValueError, "ExtraStatePool"):
            self.build(object.__new__(ExtraStatePool))

    def test_rejects_virtual_indices_and_eagle_without_declared_draft_buffers(self):
        self.params.token_to_kv_pool_allocator.hands_out_virtual_ids = True
        with self.assertRaisesRegex(ValueError, "virtual indices"):
            self.build(self.pool())
        self.params.token_to_kv_pool_allocator.hands_out_virtual_ids = False
        self.params.is_eagle = True
        with self.assertRaisesRegex(ValueError, "draft KV"):
            self.build(self.pool())

    def test_rejects_every_scale_and_workspace_even_v_only_state(self):
        for name in (
            "k_scale_buffer",
            "v_scale_buffer",
            "native_k_scale_buffer",
            "native_v_scale_buffer",
            "dq_k_buffer",
            "dq_v_buffer",
        ):
            with self.subTest(name=name):
                pool = self.pool()
                setattr(pool, name, object())
                with self.assertRaisesRegex(ValueError, "scale state"):
                    self.build(pool)

    def test_rejects_page_layout_quantization_and_unstable_backing(self):
        for name, value in (
            ("page_size", 4),
            ("kv_cache_layout", "hnd"),
            ("post_capture_active", True),
            ("quant_method", object()),
        ):
            with self.subTest(name=name):
                pool = self.pool()
                setattr(pool, name, value)
                with self.assertRaisesRegex(ValueError, "stable backing"):
                    self.build(pool)

    def test_rejects_missing_layers_noncontiguous_and_nonfloating_buffers(self):
        for change in (
            "missing_v",
            "layer_range",
            "row_count",
            "noncontiguous",
            "uint8",
            "head_shape",
            "store_dtype",
        ):
            with self.subTest(change=change):
                pool = self.pool()
                if change == "missing_v":
                    pool.v_buffer.pop()
                elif change == "layer_range":
                    pool.start_layer = 1
                elif change == "row_count":
                    pool.k_buffer[0] = pool.k_buffer[0][:-1]
                elif change == "noncontiguous":
                    pool.k_buffer[0] = pool.k_buffer[0].transpose(1, 2)
                elif change == "head_shape":
                    pool.k_buffer[0] = torch.zeros(10, 3, 2)
                elif change == "store_dtype":
                    pool.store_dtype = torch.uint8
                else:
                    pool.k_buffer[0] = pool.k_buffer[0].to(torch.uint8)
                with self.assertRaisesRegex(ValueError, "complete NHD layers"):
                    self.build(pool)


if __name__ == "__main__":
    unittest.main()
