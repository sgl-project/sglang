import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    get_sparse_kv_budget,
    get_sparsity_driven_kv_offload_cell_size,
    get_sparsity_driven_kv_offload_sparse_context_len,
    is_sparsity_driven_kv_offload_enabled,
    use_pd_split_kv,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


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


def _glm_indexer_types():
    full_layers = {0, 1, *range(2, 78, 4)}
    return ["full" if layer_id in full_layers else "shared" for layer_id in range(78)]


class TestSparsityDrivenKVOffloadConfig(CustomTestCase):
    def test_pd_split_kv_transport_and_data_parallelism(self):
        from sglang.srt.environ import envs

        prefix = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config."
        parallel = SimpleNamespace(
            nnodes=1,
            pp_size=1,
            attn_cp_size=1,
            dcp_enabled=False,
            dp_size=1,
            attn_dp_size=1,
        )
        disagg = SimpleNamespace(
            disaggregation_mode="decode",
            enable_pd_role_switch=False,
            disaggregation_transfer_backend="ascend",
            disaggregation_decode_host_receive_threshold=1,
        )
        with (
            envs.SGLANG_NPU_ENABLE_PD_SPLIT_KV.override(True),
            envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True),
            envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
            patch(prefix + "is_npu", return_value=True),
            patch(prefix + "attention_backends", return_value=("ascend", "ascend")),
            patch(prefix + "get_disagg", return_value=disagg),
            patch(prefix + "get_parallel", return_value=parallel),
            patch(prefix + "get_schedule", return_value=SimpleNamespace(page_size=128)),
            patch(
                prefix + "get_memory",
                return_value=SimpleNamespace(
                    enable_hisparse=False,
                    enable_hierarchical_cache=False,
                ),
            ),
        ):
            for backend in ("ascend", "mooncake"):
                disagg.disaggregation_transfer_backend = backend
                for mode in ("prefill", "decode"):
                    disagg.disaggregation_mode = mode
                    for dp_size, attn_dp_size in ((1, 1), (2, 1), (2, 2)):
                        parallel.dp_size, parallel.attn_dp_size = dp_size, attn_dp_size
                        with self.subTest(
                            backend=backend, mode=mode, dp=dp_size, attn_dp=attn_dp_size
                        ):
                            self.assertTrue(use_pd_split_kv())

            disagg.disaggregation_decode_host_receive_threshold = 0
            self.assertTrue(
                use_pd_split_kv()
            )  # Mooncake registers host buffers directly.
            disagg.disaggregation_transfer_backend = "ascend"
            with self.assertRaisesRegex(ValueError, "host receive threshold"):
                use_pd_split_kv()
            disagg.disaggregation_decode_host_receive_threshold = 1
            for obj, attribute, value, message in (
                (disagg, "disaggregation_transfer_backend", "fake", "transfer backend"),
                (parallel, "nnodes", 2, "single-node"),
                (parallel, "pp_size", 2, "PP=1"),
                (parallel, "attn_cp_size", 2, "context parallelism"),
                (parallel, "dcp_enabled", True, "context parallelism"),
                (disagg, "enable_pd_role_switch", True, "role switching"),
            ):
                previous = getattr(obj, attribute)
                with (
                    self.subTest(attribute=attribute),
                    patch.object(obj, attribute, value),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        use_pd_split_kv()
                self.assertEqual(getattr(obj, attribute), previous)

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
            patch(
                prefix + "get_schedule",
                return_value=SimpleNamespace(page_size=128, prefill_max_requests=None),
            ),
            patch(
                prefix + "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_mode="null", enable_pd_role_switch=False
                ),
            ),
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
            patch(
                prefix + "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_mode="null", enable_pd_role_switch=False
                ),
            ),
            patch(
                prefix + "get_schedule",
                return_value=SimpleNamespace(page_size=128, prefill_max_requests=None),
            ),
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
        self.assertEqual(plan.prefill_workspace_tokens, 17 * 4224)

    def test_prefill_workspace_role_and_request_cap(self):
        from sglang.srt.environ import envs

        prefix = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config."
        for mode, switch, cap, rows in (
            ("null", False, None, 4),
            ("prefill", False, 2, 2),
            ("decode", False, None, 0),
            ("decode", True, 8, 4),
        ):
            with (
                self.subTest(mode=mode, switch=switch, cap=cap),
                envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True),
                envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.override(0),
                envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.override(0),
                patch(
                    prefix + "get_disagg",
                    return_value=SimpleNamespace(
                        disaggregation_mode=mode, enable_pd_role_switch=switch
                    ),
                ),
                patch(
                    prefix + "get_schedule",
                    return_value=SimpleNamespace(
                        page_size=128, prefill_max_requests=cap
                    ),
                ),
                patch(prefix + "get_sparse_kv_max_query_width", return_value=1),
                patch(prefix + "get_eager_max_batch_size", return_value=4),
            ):
                plan = get_sparse_kv_budget(
                    model_config=SimpleNamespace(
                        kv_lora_rank=512,
                        qk_rope_head_dim=64,
                        index_head_dim=128,
                        hf_config=SimpleNamespace(index_n_heads=64, index_topk=2048),
                    ),
                    copy_sfa_query_heads=8,
                    request_rows=5,
                    context_len=129,
                    num_layers=1,
                    top_k=2048,
                    kv_bytes_per_token=1152,
                    graph_batch_capacity=4,
                )
                self.assertEqual(plan.prefill_workspace_tokens, rows * 256)

    def test_sparse_offload_cell_size_matches_compact_owner_count(self):
        model_config = _make_glm51_model_config()
        model_config.hf_config.indexer_types = _glm_indexer_types()
        with (
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.is_sparsity_driven_kv_offload_enabled",
                return_value=True,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator._should_elide_dsa_index_k",
                return_value=True,
            ),
        ):
            cases = (
                (0, 78, 21),
                (20, 20, 5),
                (3, 3, 0),
            )
            for start_layer, num_layers, owner_layers in cases:
                with self.subTest(start_layer=start_layer, num_layers=num_layers):
                    self.assertEqual(
                        get_sparsity_driven_kv_offload_cell_size(
                            model_config=model_config,
                            use_mla_backend=True,
                            num_layers=num_layers,
                            element_size=2,
                            start_layer=start_layer,
                        ),
                        128 * owner_layers * 2,
                    )

        with (
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.is_sparsity_driven_kv_offload_enabled",
                return_value=True,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator._should_elide_dsa_index_k",
                return_value=False,
            ) as should_elide,
        ):
            self.assertEqual(
                get_sparsity_driven_kv_offload_cell_size(
                    model_config=model_config,
                    use_mla_backend=True,
                    num_layers=78,
                    element_size=2,
                    is_draft_worker=True,
                ),
                128 * 78 * 2,
            )
            should_elide.assert_called_once_with(is_draft_worker=True)

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
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_memory",
                return_value=SimpleNamespace(
                    enable_hierarchical_cache=False,
                    enable_hisparse=False,
                    enable_unified_cache_external_linker=False,
                ),
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
        ):
            model_config = _make_glm51_model_config()
            model_config.hf_config.indexer_types = ["full", "full"]

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
