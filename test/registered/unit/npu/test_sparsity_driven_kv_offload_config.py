import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    get_sparse_kv_budget,
    get_sparsity_driven_kv_offload_cell_size,
    get_sparsity_driven_kv_offload_sparse_context_len,
    is_sparsity_driven_kv_offload_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_glm51_model_config():
    hf_config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_head_dim=128,
        index_topk=1536,
    )
    hf_config.get_text_config = lambda: hf_config
    return SimpleNamespace(
        hf_config=hf_config,
        index_head_dim=128,
    )


class TestSparsityDrivenKVOffloadConfig(CustomTestCase):
    def test_copy_sfa_heads_follow_model_and_attention_tp(self):
        from sglang.srt.environ import envs

        model = _make_glm51_model_config()
        model.kv_lora_rank, model.qk_rope_head_dim = 512, 64
        model.num_attention_heads = 80
        model.hf_config.index_n_heads = 32
        model.hf_config.index_topk = 2048
        prefix = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config."
        with (
            envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True),
            envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.override(0),
            envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.override(0),
            patch(
                prefix + "get_parallel", return_value=SimpleNamespace(attn_tp_size=8)
            ),
            patch(prefix + "get_schedule", return_value=SimpleNamespace(page_size=128)),
            patch(prefix + "get_eager_max_batch_size", return_value=4),
            patch(prefix + "get_sparse_kv_max_query_width", return_value=1),
        ):
            plan = get_sparse_kv_budget(
                model_config=model,
                request_rows=3,
                context_len=4097,
                num_layers=2,
                top_k=2048,
                kv_bytes_per_token=1152,
                graph_batch_capacity=4,
            )
        self.assertEqual(plan.copy_sfa_query_heads, 16)
        self.assertEqual(plan.manager_buffers["fused_attn_out"].shape, (4, 16, 512))
        self.assertEqual(plan.adapter_buffers["query_scale"].shape, (4, 32))

    def test_graph_alignment_is_reserved_before_buckets_are_resolved(self):
        from sglang.srt.environ import envs

        prefix = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config."
        with (
            envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True),
            envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.override(0),
            envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.override(0),
            patch(
                prefix + "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(
                        disable_cuda_graph=False, cuda_graph_config=None
                    )
                ),
            ),
            patch(prefix + "get_cuda_graph_max_batch_size", return_value=32),
            patch(prefix + "get_eager_max_batch_size", return_value=24),
            patch(
                prefix + "get_parallel", return_value=SimpleNamespace(attn_tp_size=1)
            ),
            patch(prefix + "get_schedule", return_value=SimpleNamespace(page_size=128)),
            patch(
                prefix + "get_spec",
                return_value=SimpleNamespace(speculative_num_draft_tokens=1),
            ),
        ):
            plan = get_sparse_kv_budget(
                model_config=SimpleNamespace(
                    num_attention_heads=128,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    index_head_dim=128,
                    hf_config=SimpleNamespace(
                        architectures=["GlmMoeDsaForCausalLM"],
                        index_head_dim=128,
                        index_n_heads=64,
                        index_topk=2048,
                    ),
                ),
                request_rows=18,
                context_len=4097,
                num_layers=2,
                top_k=2048,
                kv_bytes_per_token=1152,
            )
        self.assertEqual(plan.batch_capacity, 32)

    def test_glm_dsa_model_enables_sparse_kv_offload(self):
        with (
            patch.dict(
                os.environ,
                {"SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1"},
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.is_npu",
                return_value=True,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.attention_backends",
                return_value=("ascend", "ascend"),
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.get_schedule",
                return_value=SimpleNamespace(max_running_requests=8),
            ),
        ):
            model_config = _make_glm51_model_config()

            self.assertTrue(
                is_sparsity_driven_kv_offload_enabled(
                    model_config=model_config,
                    use_mla_backend=True,
                )
            )
            self.assertEqual(
                get_sparsity_driven_kv_offload_sparse_context_len(
                    model_config=model_config
                ),
                1536,
            )
            self.assertEqual(
                get_sparsity_driven_kv_offload_cell_size(
                    model_config=model_config,
                    use_mla_backend=True,
                    num_layers=2,
                    element_size=2,
                ),
                512,
            )

    def test_split_attention_backend_rejects_sparse_kv_offload(self):
        with (
            patch.dict(
                os.environ,
                {"SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1"},
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.is_npu",
                return_value=True,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.attention_backends",
                return_value=("ascend", "torch_native"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Ascend MLA attention backend"):
                is_sparsity_driven_kv_offload_enabled(
                    model_config=_make_glm51_model_config(),
                    use_mla_backend=True,
                )

    def test_missing_request_capacity_rejects_sparse_kv_offload(self):
        with (
            patch.dict(
                os.environ,
                {"SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1"},
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.is_npu",
                return_value=True,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.attention_backends",
                return_value=("ascend", "ascend"),
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.get_schedule",
                return_value=SimpleNamespace(max_running_requests=None),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "max_running_requests"):
                is_sparsity_driven_kv_offload_enabled(
                    model_config=_make_glm51_model_config(),
                    use_mla_backend=True,
                )


if __name__ == "__main__":
    unittest.main()
