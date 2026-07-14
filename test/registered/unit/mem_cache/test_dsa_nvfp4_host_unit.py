import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import server_args as server_args_module
from sglang.srt.layers.attention.dsa.nvfp4_k_cache import NVFP4_BYTES_PER_TOKEN
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor import pool_configurator as pool_configurator_module
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def _glm_config():
    return SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        index_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )


class TestDSANVFP4HostUnit(unittest.TestCase):
    def test_mla_cache_dim_uses_mixed_nvfp4_layout(self):
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=_glm_config(),
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            ),
            kv_cache_dtype=torch.float4_e2m1fn_x2,
            server_args=SimpleNamespace(
                dsa_prefill_backend="flashmla_sparse",
                dsa_decode_backend="flashmla_kv",
            ),
        )
        self.assertEqual(
            ModelRunnerKVCacheMixin.calculate_mla_kv_cache_dim(runner),
            NVFP4_BYTES_PER_TOKEN,
        )

    def test_capacity_accounts_for_rope_bf16_and_indexer(self):
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=_glm_config(),
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            ),
            kv_cache_dtype=torch.float4_e2m1fn_x2,
            use_mla_backend=True,
            is_draft_worker=False,
            server_args=SimpleNamespace(enable_dsa_cache_layer_split=False),
        )
        with patch.object(
            pool_configurator_module,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=8),
        ):
            cell_size = DefaultPoolConfigurator._compute_cell_size(
                object(), runner, num_layers=2
            )
        # 416-byte latent/RoPE row + 128-byte index key + 4-byte index scale.
        self.assertEqual(cell_size, 2 * (416 + 132))

    def test_native_decode_dispatches_raw_cache_and_scale(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.real_page_size = 64
        backend.kv_cache_dim = NVFP4_BYTES_PER_TOKEN
        backend.dsa_kv_cache_store_nvfp4 = True
        backend.dsa_kv_cache_store_fp8 = False
        backend.dsa_index_topk = 64
        global_scale = torch.tensor([1.25], dtype=torch.float32)
        backend.token_to_kv_pool = SimpleNamespace(
            get_mla_kv_global_scale=lambda _layer_id: global_scale
        )

        q = torch.zeros((2, 8, 576), dtype=torch.bfloat16)
        cache = torch.zeros((128, 1, NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8)
        page_table = torch.zeros((2, 64), dtype=torch.int32)
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=8, head_dim=576, v_head_dim=512
        )
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.full((2,), 64, dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(
                flashmla_metadata=torch.zeros((1, 8), dtype=torch.int32),
                num_splits=torch.zeros(3, dtype=torch.int32),
            ),
        )
        expected = torch.zeros((2, 1, 8, 512), dtype=torch.bfloat16)

        with patch(
            "sgl_kernel.flash_mla.flash_mla_with_kvcache_nvfp4",
            return_value=(expected, torch.empty(0)),
            create=True,
        ) as native_fwd:
            actual = backend._forward_flashmla_kv(
                q_all=q,
                kv_cache=cache,
                v_head_dim=512,
                sm_scale=0.125,
                layer=layer,
                metadata=metadata,
                page_table_1=page_table,
            )

        self.assertIs(actual, expected)
        kwargs = native_fwd.call_args.kwargs
        self.assertIs(kwargs["kv_global_scale"], global_scale)
        self.assertEqual(tuple(kwargs["k_cache"].shape), (2, 64, 1, 416))
        self.assertEqual(tuple(kwargs["indices"].shape), (2, 1, 64))

    def test_server_gate_accepts_sm90_native_decode(self):
        args = SimpleNamespace(
            kv_cache_dtype="fp4_e2m1",
            fp4_kv_cache_recipe="nvfp4",
            enable_hisparse=False,
            enable_dsa_cache_layer_split=False,
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_kv",
            page_size=64,
            get_model_config=lambda: SimpleNamespace(hf_config=_glm_config()),
        )
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            ServerArgs._handle_kv4_compatibility(args)

    def test_server_gate_rejects_non_sm90(self):
        args = SimpleNamespace(
            kv_cache_dtype="fp4_e2m1",
            fp4_kv_cache_recipe="nvfp4",
            enable_hisparse=False,
            enable_dsa_cache_layer_split=False,
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_kv",
            page_size=64,
            get_model_config=lambda: SimpleNamespace(hf_config=_glm_config()),
        )
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            self.assertRaisesRegex(ValueError, "requires SM90"),
        ):
            ServerArgs._handle_kv4_compatibility(args)


if __name__ == "__main__":
    unittest.main()
