"""CPU regressions for NPU DSA indexer pool layer ownership."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _quantize_npu_indexer_activation,
)
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.kv_cache_dtype import (
    QUANT_MODE_TOKEN_INT8,
    resolve_indexer_quant_mode,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestNPUIndexerLayerLayout(CustomTestCase):
    def test_non_arch35_dsa_pool_uses_only_active_indexer_layers(self):
        """Shared layers must not consume indexer slots on non-arch35 NPU."""
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.modelslim_cache_layout = None
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
        configurator.server_args = SimpleNamespace(indexer_kv_cache_dtype="int8")
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
        self.assertEqual(
            pool_class.call_args.kwargs["indexer_quant_mode"], QUANT_MODE_TOKEN_INT8
        )


class TestNPUIndexerMemorySizing(CustomTestCase):
    def test_npu_indexer_cell_size_matches_storage(self):
        from sglang.srt.model_executor import pool_configurator as pc

        configurator = pc.DefaultPoolConfigurator.__new__(pc.DefaultPoolConfigurator)
        cases = [
            (None, torch.bfloat16, 256),
            ("int8", torch.bfloat16, 130),
            (None, torch.float8_e4m3fn, 132),
        ]
        for indexer_dtype, kv_dtype, expected in cases:
            with self.subTest(indexer_dtype=indexer_dtype, kv_dtype=kv_dtype):
                kvc = SimpleNamespace(
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(
                            architectures=["DeepseekV32ForCausalLM"],
                            index_topk=2048,
                            index_head_dim=128,
                        )
                    ),
                    kv_cache_dtype=kv_dtype,
                    server_args=SimpleNamespace(
                        indexer_kv_cache_dtype=indexer_dtype,
                        enable_hisparse=True,
                    ),
                    is_draft_worker=False,
                )
                with (
                    patch.object(pc, "_is_npu", True),
                    patch.object(
                        pc,
                        "get_exec",
                        return_value=SimpleNamespace(
                            kernel=SimpleNamespace(enable_fp4_indexer=False)
                        ),
                    ),
                    patch.object(
                        pc,
                        "get_memory",
                        return_value=SimpleNamespace(enable_hisparse=False),
                    ),
                ):
                    actual = configurator._compute_dsa_indexer_cell_size(
                        kvc=kvc, num_layers=1
                    )
                self.assertEqual(actual, expected)


class TestNPUIndexerInt8Cache(CustomTestCase):
    """--indexer-kv-cache-dtype=int8 may replace only the BF16 indexer cache."""

    def _build_pool(
        self, dtype, kv_cache_dim=None, indexer_quant_mode=None, hierarchical=False
    ):
        with (
            get_context().override_server_args(enable_hierarchical_cache=hierarchical),
            patch.object(
                memory_pool_npu,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
            ),
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
                indexer_quant_mode=indexer_quant_mode,
            )

    def test_int8_mode_rejects_hierarchical_cache_in_server_args(self):
        from sglang.srt.arg_groups import kv_cache_hook

        cfg = SimpleNamespace(
            indexer_kv_cache_dtype="int8", enable_hierarchical_cache=True
        )
        with (
            patch.object(kv_cache_hook, "resolving_view", return_value=cfg),
            self.assertRaisesRegex(ValueError, "enable-hierarchical-cache"),
        ):
            kv_cache_hook.handle_indexer_kv_cache_compatibility(object())

    def test_int8_mode_rejects_hierarchical_cache_in_pool(self):
        with self.assertRaisesRegex(ValueError, "enable-hierarchical-cache"):
            self._build_pool(
                torch.bfloat16,
                indexer_quant_mode=QUANT_MODE_TOKEN_INT8,
                hierarchical=True,
            )

    def test_int8_mode_swaps_only_the_bf16_indexer_cache(self):
        fp8_kv_cache_dim = 512 + 512 // 128 * 4 + 64 * 2
        cases = [
            # quant mode, kv dtype, kv_cache_dim, index K dtype, scale dtype
            (None, torch.bfloat16, None, torch.bfloat16, None),
            (QUANT_MODE_TOKEN_INT8, torch.bfloat16, None, torch.int8, torch.float16),
            # The FP8 KV cache already has an FP8 indexer; resolve rejects
            # int8 on top of it, and the pool must keep the FP8 indexer even
            # if the mode is passed in directly.
            (
                QUANT_MODE_TOKEN_INT8,
                torch.float8_e4m3fn,
                fp8_kv_cache_dim,
                torch.float8_e4m3fn,
                torch.float32,
            ),
        ]
        for quant_mode, dtype, kv_cache_dim, key_dtype, scale_dtype in cases:
            with self.subTest(quant_mode=quant_mode, dtype=dtype):
                pool = self._build_pool(dtype, kv_cache_dim, quant_mode)
                scale = pool.index_k_scale_buffer
                self.assertEqual(pool.index_k_buffer.dtype, key_dtype)
                self.assertEqual(None if scale is None else scale.dtype, scale_dtype)

    def test_resolve_indexer_quant_mode(self):
        self.assertIsNone(resolve_indexer_quant_mode(None, torch.bfloat16))
        self.assertIsNone(resolve_indexer_quant_mode(None, torch.float8_e4m3fn))
        self.assertEqual(
            resolve_indexer_quant_mode("int8", torch.bfloat16), QUANT_MODE_TOKEN_INT8
        )
        with self.assertRaises(ValueError):
            resolve_indexer_quant_mode("int8", torch.float8_e4m3fn)
        with self.assertRaises(ValueError):
            resolve_indexer_quant_mode("fp8_e4m3", torch.bfloat16)

    def test_empty_input_scale_matches_int8_cache(self):
        """A batch with no tokens still writes its (empty) scale to the pool,
        whose INT8 scale buffer rejects anything but FP16."""
        x = torch.empty(0, 128, dtype=torch.bfloat16)
        k, scale = _quantize_npu_indexer_activation(x, None, torch.int8, torch.float16)
        self.assertEqual((k.dtype, scale.dtype), (torch.int8, torch.float16))


if __name__ == "__main__":
    import unittest

    unittest.main()
