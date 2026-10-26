"""CPU regressions for NPU DSA indexer pool layer ownership."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _quantize_npu_indexer_activation,
)
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestNPUIndexerLayerLayout(CustomTestCase):
    def test_non_arch35_dsa_pool_uses_only_active_indexer_layers(self):
        """Shared layers must not consume indexer slots on non-arch35 NPU."""
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.device = "cpu"
        configurator.is_draft_worker = False
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.layer_info = SimpleNamespace(
            start_layer=10,
            end_layer=14,
            num_effective_layers=4,
        )
        configurator.model_config = SimpleNamespace(
            hf_config=object(),
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            index_head_dim=32,
        )
        pool = object()

        with (
            patch.object(
                kcc,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=2),
            ),
            patch.object(
                kcc,
                "get_schedule",
                return_value=SimpleNamespace(page_size=16),
            ),
            patch.object(
                kcc,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch.object(kcc, "_should_elide_dsa_index_k", return_value=True),
            patch.object(
                kcc,
                "dsa_layer_skips_topk",
                side_effect=lambda _config, layer_id: layer_id in (11, 13),
            ),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.memory_pool_npu.NPUMLATokenToKVPool",
                return_value=pool,
            ) as pool_class,
        ):
            result = configurator._build_ascend_mla_kv_pool(
                max_total_num_tokens=64,
                is_dsa_model=True,
            )

        self.assertIs(result, pool)
        self.assertEqual(pool_class.call_args.args, (64,))
        self.assertEqual(pool_class.call_args.kwargs["index_size"], 128)
        self.assertEqual(pool_class.call_args.kwargs["indexer_layer_ids"], (10, 12))


class TestNPUIndexerInt8Cache(CustomTestCase):
    """SGLANG_NPU_INDEXER_INT8 may replace only the BF16 indexer cache."""

    def _build_pool(self, dtype, kv_cache_dim=None):
        with patch.object(
            memory_pool_npu,
            "get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
        ):
            return memory_pool_npu.NPUMLATokenToKVPool(
                256,
                page_size=128,
                dtype=dtype,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
                index_head_dim=128,
                kv_cache_dim=kv_cache_dim,
            )

    def test_flag_swaps_only_the_bf16_indexer_cache(self):
        fp8_kv_cache_dim = 512 + 512 // 128 * 4 + 64 * 2
        cases = [
            # flag, kv dtype, kv_cache_dim, index K dtype, index K scale dtype
            (False, torch.bfloat16, None, torch.bfloat16, None),
            (True, torch.bfloat16, None, torch.int8, torch.float16),
            # The FP8 KV cache already has an FP8 indexer; the flag must keep it.
            (
                True,
                torch.float8_e4m3fn,
                fp8_kv_cache_dim,
                torch.float8_e4m3fn,
                torch.float32,
            ),
        ]
        for flag, dtype, kv_cache_dim, key_dtype, scale_dtype in cases:
            with (
                self.subTest(flag=flag, dtype=dtype),
                envs.SGLANG_NPU_INDEXER_INT8.override(flag),
            ):
                pool = self._build_pool(dtype, kv_cache_dim)
                scale = pool.index_k_scale_buffer
                self.assertEqual(pool.index_k_buffer.dtype, key_dtype)
                self.assertEqual(None if scale is None else scale.dtype, scale_dtype)

    def test_empty_input_scale_matches_int8_cache(self):
        """A batch with no tokens still writes its (empty) scale to the pool,
        whose INT8 scale buffer rejects anything but FP16."""
        x = torch.empty(0, 128, dtype=torch.bfloat16)
        k, scale = _quantize_npu_indexer_activation(x, None, torch.int8, torch.float16)
        self.assertEqual((k.dtype, scale.dtype), (torch.int8, torch.float16))


if __name__ == "__main__":
    import unittest

    unittest.main()
