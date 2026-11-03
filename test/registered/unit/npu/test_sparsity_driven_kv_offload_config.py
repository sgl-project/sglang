import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    get_sparsity_driven_kv_offload_cell_size,
    get_sparsity_driven_kv_offload_sparse_context_len,
    is_sparsity_driven_kv_offload_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci

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


class TestSparsityDrivenKVOffloadConfig(unittest.TestCase):
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
