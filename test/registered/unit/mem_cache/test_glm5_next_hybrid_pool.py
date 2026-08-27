"""Regression coverage for GLM-5.3's hybrid DSA/KDA KV pool."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGlm5NextHybridPool(CustomTestCase):
    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.HybridLinearKVPool",
        autospec=True,
    )
    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
        autospec=True,
        return_value=576,
    )
    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.get_dsa_index_kpool_compress",
        return_value=True,
    )
    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.get_dsa_index_kpool",
        return_value=1,
    )
    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.get_dsa_index_head_dim",
        return_value=128,
    )
    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.dsa_layer_skips_topk",
        return_value=False,
    )
    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.is_deepseek_dsa",
        return_value=True,
    )
    def test_dsa_kv_dimension_uses_current_calculator_contract(
        self,
        _mock_is_deepseek_dsa,
        _mock_dsa_layer_skips_topk,
        _mock_index_head_dim,
        _mock_index_kpool,
        _mock_index_kpool_compress,
        mock_calculate_mla_kv_cache_dim,
        mock_hybrid_pool,
    ):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        configurator = object.__new__(KVCacheConfigurator)
        configurator.is_draft_worker = False
        configurator.mambaish_config = SimpleNamespace(full_attention_layer_ids=[3])
        configurator.layer_info = SimpleNamespace(start_layer=0, end_layer=4)
        configurator.use_mla_backend = True
        configurator.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(),
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            head_dim=128,
            get_num_kv_heads=lambda _tp_size, _dcp_size: 1,
        )
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.kv_cache_dtype_str = "bfloat16"
        configurator.device = "cuda"
        configurator.post_capture_kv_active = False

        req_to_token_pool = SimpleNamespace(mamba_pool=object())
        fake_cp_utils = ModuleType("sglang.srt.layers.cp.utils")
        fake_cp_utils.get_glm_dsa_cp_layer_shard_info = lambda _configurator: (0, 1)
        with (
            patch.dict(
                sys.modules,
                {"sglang.srt.layers.cp.utils": fake_cp_utils},
            ),
            patch.object(
                KVCacheConfigurator,
                "_build_fp4_quant_method",
                return_value=None,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=8, attn_dcp_size=1),
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_schedule",
                return_value=SimpleNamespace(page_size=64),
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
        ):
            configurator._build_hybrid_linear_kv_pool(
                max_total_num_tokens=1024,
                req_to_token_pool=req_to_token_pool,
                mha_pool_class=object,
            )

        mock_calculate_mla_kv_cache_dim.assert_called_once_with(
            model_config=configurator.model_config,
            kv_cache_dtype=torch.bfloat16,
        )
        self.assertEqual(mock_hybrid_pool.call_args.kwargs["kv_cache_dim"], 576)


if __name__ == "__main__":
    unittest.main()
