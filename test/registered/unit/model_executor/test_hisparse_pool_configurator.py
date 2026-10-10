import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestHiSparsePoolConfigurator(CustomTestCase):
    def test_hybrid_dsa_pool_uses_dense_layer_ids(self):
        from sglang.srt.mem_cache import kv_cache_configurator as module

        kvc = SimpleNamespace()
        kvc.layer_info = SimpleNamespace(
            start_layer=4, end_layer=12, num_effective_layers=8
        )
        kvc.model_config = SimpleNamespace(
            kv_lora_rank=512, qk_rope_head_dim=0, hf_config=SimpleNamespace()
        )
        kvc.pool_page_size = 64
        kvc.kv_cache_dtype = torch.bfloat16
        kvc.device = "cpu"
        kvc.is_draft_worker = False
        with (
            patch.object(
                module, "get_memory", return_value=SimpleNamespace(enable_hisparse=True)
            ),
            patch.object(
                module,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch.object(module, "reject_out_of_tree_path"),
            patch(
                "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                return_value=(None, None),
            ),
            patch(
                "sglang.srt.mem_cache.sparsity.parse_hisparse_config",
                return_value=SimpleNamespace(host_to_device_ratio=2),
            ),
            patch.object(module, "_should_elide_dsa_index_k", return_value=True),
            patch.object(
                module,
                "dsa_layer_skips_topk",
                side_effect=lambda config, layer: layer == 11,
            ) as skips,
            patch.object(module, "calculate_mla_kv_cache_dim", return_value=512),
            patch.object(module, "get_dsa_index_head_dim", return_value=128),
            patch.object(module, "get_dsa_index_kpool", return_value=4),
            patch.object(module, "get_dsa_index_kpool_compress", return_value=False),
            patch.object(module, "max_speculative_num_draft_tokens", return_value=None),
            patch.object(module, "HiSparseDSATokenToKVPool") as pool,
        ):
            result = module.KVCacheConfigurator._build_dsa_kv_pool(
                kvc,
                max_total_num_tokens=1024,
                max_running_requests=8,
                dsa_pool_class=object,
                full_attention_layer_ids=[7, 11],
            )
        self.assertIs(result, pool.return_value)
        kwargs = pool.call_args.kwargs
        self.assertEqual(
            (kwargs["start_layer"], kwargs["end_layer"], kwargs["layer_num"]), (0, 2, 2)
        )
        self.assertEqual(kwargs["skip_topk_layers"], [False, True])
        self.assertEqual([call.args[1] for call in skips.call_args_list], [7, 11])
        self.assertEqual(kwargs["host_to_device_ratio"], 2)

    def _compute_cell_size(
        self,
        kv_cache_dtype: torch.dtype,
        *,
        enable_hisparse: bool,
        host_to_device_ratio: int = 1,
    ) -> int:
        num_layers = 2
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
        )
        hf_config.get_text_config = lambda: hf_config

        override = get_context().override_server_args(
            enable_hisparse=enable_hisparse,
            hisparse_config=f'{{"host_to_device_ratio": {host_to_device_ratio}}}',
            enable_hierarchical_cache=False,
            disaggregation_mode="null",
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_sparse",
        )
        server_args = override.install()
        self.addCleanup(override.restore)

        kvc = MagicMock(
            use_mla_backend=True,
            kv_cache_dtype=kv_cache_dtype,
            is_draft_worker=False,
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=hf_config,
            ),
            layer_info=SimpleNamespace(start_layer=0, end_layer=num_layers),
            server_args=server_args,
        )

        with get_parallel().override(attn_tp_size=1):
            configurator = object.__new__(DefaultPoolConfigurator)
            return configurator._compute_cell_size(kvc, num_layers=num_layers)

    def test_mla_layout_without_hisparse(self):
        for kv_cache_dtype, expected_cell_size in (
            (torch.bfloat16, 2568),
            (torch.float8_e4m3fn, 1576),
        ):
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
                cell_size = self._compute_cell_size(
                    kv_cache_dtype,
                    enable_hisparse=False,
                )
                self.assertEqual(cell_size, expected_cell_size)

    def test_hisparse_indexer_scales_with_ratio(self):
        for host_to_device_ratio, expected_cell_size in (
            (2, 1840),
            (4, 2368),
        ):
            with self.subTest(host_to_device_ratio=host_to_device_ratio):
                cell_size = self._compute_cell_size(
                    torch.float8_e4m3fn,
                    enable_hisparse=True,
                    host_to_device_ratio=host_to_device_ratio,
                )
                self.assertEqual(cell_size, expected_cell_size)


if __name__ == "__main__":
    unittest.main()
