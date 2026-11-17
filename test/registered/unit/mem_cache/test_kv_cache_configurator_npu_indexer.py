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
    resolve_indexer_kv_cache_dtype,
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
                max_running_requests=4,
                is_dsa_model=True,
            )

        self.assertIs(result, pool)
        self.assertEqual(pool_class.call_args.args, (64,))
        self.assertEqual(pool_class.call_args.kwargs["index_size"], 128)
        self.assertEqual(pool_class.call_args.kwargs["indexer_layer_ids"], (10, 12))
        self.assertEqual(pool_class.call_args.kwargs["indexer_dtype"], torch.int8)


class TestNPUIndexerMemorySizing(CustomTestCase):
    def test_npu_indexer_cell_size_matches_storage(self):
        from sglang.srt.model_executor import pool_configurator as pc

        configurator = pc.DefaultPoolConfigurator.__new__(pc.DefaultPoolConfigurator)
        cases = [
            (None, torch.bfloat16, 1, 256),
            (None, torch.bfloat16, 4, 64),
            ("int8", torch.bfloat16, 1, 130),
            (None, torch.int8, 1, 130),
            ("bf16", torch.int8, 1, 256),
            ("bfloat16", torch.int8, 1, 256),
            ("bf16", torch.float8_e4m3fn, 1, 256),
            ("int8", torch.float8_e4m3fn, 1, 130),
            (None, torch.float8_e4m3fn, 1, 132),
        ]
        for indexer_dtype, kv_dtype, kpool, expected in cases:
            with self.subTest(
                indexer_dtype=indexer_dtype, kv_dtype=kv_dtype, kpool=kpool
            ):
                kvc = SimpleNamespace(
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(
                            architectures=["DeepseekV32ForCausalLM"],
                            index_topk=2048,
                            index_head_dim=128,
                            index_kpool=kpool,
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
    """Explicit Indexer dtypes must not change the main KV cache."""

    def _build_pool(
        self, dtype, kv_cache_dim=None, indexer_dtype=None, hierarchical=False
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
                indexer_dtype=indexer_dtype,
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
                indexer_dtype=torch.int8,
                hierarchical=True,
            )

    def test_indexer_dtype_is_independent_of_main_cache(self):
        fp8_kv_cache_dim = 512 + 512 // 128 * 4 + 64 * 2
        cases = [
            # Indexer override, main dtype, packed width, Indexer dtype, scale dtype.
            (None, torch.bfloat16, None, torch.bfloat16, None),
            (torch.int8, torch.bfloat16, None, torch.int8, torch.float16),
            (None, torch.int8, fp8_kv_cache_dim, torch.int8, torch.float16),
            (torch.bfloat16, torch.int8, fp8_kv_cache_dim, torch.bfloat16, None),
            (
                None,
                torch.float8_e4m3fn,
                fp8_kv_cache_dim,
                torch.float8_e4m3fn,
                torch.float32,
            ),
            (
                torch.bfloat16,
                torch.float8_e4m3fn,
                fp8_kv_cache_dim,
                torch.bfloat16,
                None,
            ),
            (
                torch.int8,
                torch.float8_e4m3fn,
                fp8_kv_cache_dim,
                torch.int8,
                torch.float16,
            ),
        ]
        for indexer_dtype, dtype, kv_cache_dim, key_dtype, scale_dtype in cases:
            with self.subTest(indexer_dtype=indexer_dtype, dtype=dtype):
                pool = self._build_pool(dtype, kv_cache_dim, indexer_dtype)
                scale = pool.index_k_scale_buffer
                self.assertEqual(pool.k_buffer.dtype, dtype)
                self.assertEqual(pool.index_k_buffer.dtype, key_dtype)
                self.assertEqual(pool.get_index_k_buffer(0).dtype, key_dtype)
                self.assertEqual(None if scale is None else scale.dtype, scale_dtype)
                loc = torch.tensor([7, 129])
                values = torch.arange(256).reshape(2, 1, 128).to(key_dtype)

                def scatter(cache, indices, updates):
                    cache[indices.flatten().long()] = updates

                with patch.object(
                    memory_pool_npu,
                    "torch_npu",
                    SimpleNamespace(npu_scatter_nd_update_=scatter),
                    create=True,
                ):
                    pool.set_index_k_buffer(0, loc, values)
                # Compare bytes so this also works for FP8 on CPU.
                actual = pool.get_index_k_buffer(0).flatten(0, 1).view(torch.uint8)
                torch.testing.assert_close(actual[loc], values.view(torch.uint8))

    def test_resolve_indexer_kv_cache_dtype(self):
        self.assertIsNone(resolve_indexer_kv_cache_dtype(None))
        for name, expected in (
            ("bf16", torch.bfloat16),
            ("bfloat16", torch.bfloat16),
            ("int8", torch.int8),
        ):
            self.assertEqual(resolve_indexer_kv_cache_dtype(name), expected)
        with self.assertRaises(ValueError):
            resolve_indexer_kv_cache_dtype("fp8_e4m3")

    def test_empty_input_scale_matches_int8_cache(self):
        """A batch with no tokens still writes its (empty) scale to the pool,
        whose INT8 scale buffer rejects anything but FP16."""
        x = torch.empty(0, 128, dtype=torch.bfloat16)
        k, scale = _quantize_npu_indexer_activation(x, None, torch.int8, torch.float16)
        self.assertEqual((k.dtype, scale.dtype), (torch.int8, torch.float16))


if __name__ == "__main__":
    import unittest

    unittest.main()
