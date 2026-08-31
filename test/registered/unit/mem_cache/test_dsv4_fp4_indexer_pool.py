import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

import sglang.srt.utils as srt_utils

fake_aiter = ModuleType("aiter")
fake_aiter.__path__ = []
fake_aiter_ops = ModuleType("aiter.ops")
fake_aiter_ops.__path__ = []
fake_aiter_triton = ModuleType("aiter.ops.triton")
fake_aiter_triton.__path__ = []
fake_aiter_quant = ModuleType("aiter.ops.triton.quant")
fake_aiter_quant.dynamic_mxfp4_quant = Mock()

modules_before_import = set(sys.modules)
with (
    patch.object(srt_utils, "is_hip", return_value=False),
    patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(gcnArchName="gfx950", major=9, minor=5),
    ),
    patch.dict(
        sys.modules,
        {
            "aiter": fake_aiter,
            "aiter.ops": fake_aiter_ops,
            "aiter.ops.triton": fake_aiter_triton,
            "aiter.ops.triton.quant": fake_aiter_quant,
        },
    ),
):
    import sglang.srt.mem_cache.deepseek_v4_memory_pool as dsv4_pool
    import sglang.srt.model_executor.pool_configurator as pool_configurator
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4IndexerPool,
        DeepSeekV4LayerItem,
        DeepSeekV4TokenToKVPool,
    )
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        build_deepseek_v4_hicache_stack,
    )
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

for module_name in set(sys.modules) - modules_before_import:
    if module_name.startswith("sglang."):
        sys.modules.pop(module_name, None)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _init_kv_cache(
    pool,
    size,
    page_size,
    dtype,
    layer_num,
    device,
    enable_memory_saver,
    start_layer=None,
    end_layer=None,
):
    pool.size = size
    pool.page_size = page_size
    pool.dtype = dtype
    pool.layer_num = layer_num
    pool.device = device
    pool.start_layer = start_layer or 0
    pool.end_layer = end_layer or layer_num - 1
    pool.custom_mem_pool = None
    pool.memory_saver_adapter = SimpleNamespace(region=lambda _: nullcontext())


class TestDeepSeekV4FP4IndexerPool(CustomTestCase):
    def _make_pool(self, *, hip, enabled):
        def allocate_meta(*shape, dtype, device):
            if len(shape) == 1 and isinstance(shape[0], tuple):
                shape = shape[0]
            return torch.empty(shape, dtype=dtype, device="meta")

        with (
            patch.object(KVCache, "__init__", new=_init_kv_cache),
            patch.object(dsv4_pool, "is_hip", return_value=hip),
            patch.object(
                dsv4_pool,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(enable_deepseek_v4_fp4_indexer=enabled)
                ),
            ),
            patch.object(dsv4_pool.torch, "zeros", side_effect=allocate_meta) as zeros,
        ):
            pool = DeepSeekV4IndexerPool(
                size=128,
                page_size=64,
                dtype=torch.bfloat16,
                index_head_dim=128,
                layer_num=2,
                device="cuda",
                enable_memory_saver=False,
            )
        return pool, zeros

    def test_hip_fp4_allocates_split_aiter_layout(self):
        pool, zeros = self._make_pool(hip=True, enabled=True)
        payload_dtype = torch.float4_e2m1fn_x2

        self.assertTrue(pool.uses_aiter_fp4_layout)
        self.assertEqual(pool.get_bytes_per_token(), 68)
        self.assertEqual(pool.page_size, 64)
        self.assertEqual(len(pool.index_k_payload_buffer), 2)
        self.assertEqual(len(pool.index_k_scale_buffer), 2)
        zeros.assert_has_calls(
            [
                call((3, 1, 4, 64, 16), dtype=torch.uint8, device="cuda"),
                call((3, 1, 4, 64, 16), dtype=torch.uint8, device="cuda"),
                call((3, 1, 4, 64), dtype=torch.uint8, device="cuda"),
                call((3, 1, 4, 64), dtype=torch.uint8, device="cuda"),
            ]
        )

        payload = pool.get_index_k_fp4_payload_buffer(1)
        scale = pool.get_index_k_fp4_scale_buffer(1)
        self.assertEqual(payload.shape, (3, 1, 4, 64, 16))
        self.assertEqual(payload.dtype, payload_dtype)
        self.assertEqual(payload.element_size(), 1)
        self.assertEqual(payload[0].nbytes, 4096)
        self.assertEqual(scale.shape, (3, 1, 4, 64))
        self.assertEqual(scale.dtype, torch.uint8)
        self.assertEqual(scale.element_size(), 1)
        self.assertEqual(scale[0].nbytes, 256)
        self.assertEqual(payload[0].nbytes + scale[0].nbytes, 4352)
        with self.assertRaisesRegex(RuntimeError, "split payload and scale"):
            pool.get_index_k_with_scale_buffer(0)

    def test_cuda_fp4_keeps_combined_planar_layout(self):
        pool, zeros = self._make_pool(hip=False, enabled=True)

        self.assertFalse(pool.uses_aiter_fp4_layout)
        self.assertEqual(pool.get_bytes_per_token(), 68)
        zeros.assert_has_calls(
            [
                call(3, 4352, dtype=torch.uint8, device="cuda"),
                call(3, 4352, dtype=torch.uint8, device="cuda"),
            ]
        )
        combined = pool.get_index_k_with_scale_buffer(1)
        self.assertEqual(combined.shape, (3, 4352))
        self.assertEqual(combined[0].nbytes, 4352)
        with self.assertRaisesRegex(RuntimeError, "does not use"):
            pool.get_index_k_fp4_payload_buffer(0)

    def test_default_layout_keeps_combined_public_behavior(self):
        pool, zeros = self._make_pool(hip=True, enabled=False)

        self.assertFalse(pool.uses_aiter_fp4_layout)
        self.assertEqual(pool.get_bytes_per_token(), 132)
        zeros.assert_has_calls(
            [
                call(3, 8448, dtype=torch.uint8, device="cuda"),
                call(3, 8448, dtype=torch.uint8, device="cuda"),
            ]
        )
        self.assertEqual(pool.get_index_k_with_scale_buffer(0).shape, (3, 8448))

    def test_token_pool_forwards_split_accessors_and_rejects_pd_export(self):
        indexer_pool, _ = self._make_pool(hip=True, enabled=True)
        token_pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        token_pool.c4_indexer_kv_pool = indexer_pool
        token_pool.layer_mapping = {
            7: DeepSeekV4LayerItem(compress_ratio=4, compress_layer_id=1)
        }
        token_pool.wait_layer_transfer = Mock()

        self.assertTrue(token_pool.uses_aiter_fp4_layout)
        self.assertEqual(token_pool.get_index_k_page_size(), 64)
        self.assertIs(
            token_pool.get_index_k_fp4_payload_buffer(7),
            indexer_pool.index_k_payload_buffer[1],
        )
        self.assertIs(
            token_pool.get_index_k_fp4_scale_buffer(7),
            indexer_pool.index_k_scale_buffer[1],
        )
        with self.assertRaisesRegex(RuntimeError, "split payload and scale"):
            token_pool.get_index_k_with_scale_buffer(7)
        with self.assertRaisesRegex(RuntimeError, "disaggregation/PD"):
            token_pool.get_contiguous_buf_infos()

    def test_hicache_builder_rejects_split_layout_before_host_allocation(self):
        with self.assertRaisesRegex(RuntimeError, "HiCache/hybrid host-pool"):
            build_deepseek_v4_hicache_stack(
                params=None,
                kvcache=SimpleNamespace(uses_aiter_fp4_layout=True),
                load_cache_event=None,
                storage_backend=None,
            )

    def test_pool_budget_uses_68_bytes_per_fp4_indexer_token(self):
        model_config = SimpleNamespace(
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            index_head_dim=128,
            context_len=8192,
            compress_ratios=[4],
            window_size=128,
        )
        schedule = SimpleNamespace(
            swa_full_tokens_ratio=0.5,
            max_running_requests=None,
        )
        spec = SimpleNamespace(speculative_algorithm=None)
        disagg = SimpleNamespace(
            disaggregation_mode="null",
            disaggregation_decode_extra_slots=0,
        )

        def make_configurator(enabled):
            kvc = SimpleNamespace(
                kv_cache_dtype_str="auto",
                model_config=model_config,
                layer_info=SimpleNamespace(start_layer=0, end_layer=1),
                ps=SimpleNamespace(pp_size=1, attn_dp_size=1),
                server_args=SimpleNamespace(
                    enable_deepseek_v4_fp4_indexer=not enabled,
                    max_speculative_num_draft_tokens=None,
                    enable_hisparse=False,
                ),
            )
            exec_config = SimpleNamespace(
                kernel=SimpleNamespace(enable_deepseek_v4_fp4_indexer=enabled)
            )
            with patch.object(pool_configurator, "get_exec", return_value=exec_config):
                return DSV4PoolConfigurator(kvc)

        with (
            patch.object(pool_configurator, "get_schedule", return_value=schedule),
            patch.object(pool_configurator, "get_spec", return_value=spec),
            patch.object(pool_configurator, "get_disagg", return_value=disagg),
            patch.object(
                pool_configurator,
                "max_speculative_num_draft_tokens",
                return_value=None,
            ),
            patch.object(
                pool_configurator,
                "get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
        ):
            fp4_configurator = make_configurator(True)
            legacy_configurator = make_configurator(False)

        self.assertEqual(fp4_configurator.indexer_bytes_per_token, 68)
        self.assertEqual(legacy_configurator.indexer_bytes_per_token, 132)
        self.assertEqual(
            legacy_configurator.bytes_per_full_token
            - fp4_configurator.bytes_per_full_token,
            16,
        )


if __name__ == "__main__":
    unittest.main()
