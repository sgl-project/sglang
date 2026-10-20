import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import get_dsa_indexer_layer_ids
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    get_sparsity_driven_kv_offload_cell_size,
    get_sparsity_driven_kv_offload_sparse_context_len,
    is_sparsity_driven_kv_offload_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci

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


def _glm_indexer_types(num_layers=78, full_layers=None):
    if full_layers is None:
        full_layers = (0, 1, *range(2, num_layers, 4))
    full_layers = set(full_layers)
    return [
        "full" if layer_id in full_layers else "shared"
        for layer_id in range(num_layers)
    ]


class TestSparsityDrivenKVOffloadConfig(unittest.TestCase):
    def test_glm52_indexer_owner_ids_use_model_layer_ids(self):
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            indexer_types=_glm_indexer_types(),
        )

        self.assertEqual(
            get_dsa_indexer_layer_ids(hf_config, 0, 78),
            (0, 1, *range(2, 78, 4)),
        )
        self.assertEqual(
            get_dsa_indexer_layer_ids(hf_config, 20, 40),
            (22, 26, 30, 34, 38),
        )
        hf_config.indexer_types = ["full"] * 78
        self.assertEqual(get_dsa_indexer_layer_ids(hf_config, 0, 78), tuple(range(78)))

    def test_sparse_offload_cell_size_uses_compact_owner_count(self):
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
            self.assertEqual(
                get_sparsity_driven_kv_offload_cell_size(
                    model_config=model_config,
                    use_mla_backend=True,
                    num_layers=78,
                    element_size=2,
                ),
                128 * 21 * 2,
            )

    def test_sparse_offload_cell_size_counts_nonzero_pp_layer_range(self):
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
            self.assertEqual(
                get_sparsity_driven_kv_offload_cell_size(
                    model_config=model_config,
                    use_mla_backend=True,
                    num_layers=20,
                    element_size=2,
                    start_layer=20,
                ),
                128 * 5 * 2,
            )

    def test_sparse_offload_cell_size_falls_back_to_dense_for_excluded_pool(self):
        model_config = _make_glm51_model_config()
        model_config.hf_config.indexer_types = _glm_indexer_types(
            full_layers=range(0, 78, 4)
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
                    start_layer=20,
                    is_draft_worker=True,
                ),
                128 * 78 * 2,
            )
            should_elide.assert_called_once_with(is_draft_worker=True)

    def test_sparse_offload_budget_preserves_real_pool_exclusion_guards(self):
        model_config = _make_glm51_model_config()
        model_config.hf_config.indexer_types = _glm_indexer_types()
        excluded_pools = (
            ("draft", True, "null", False, False, False),
            ("PD", False, "decode", False, False, False),
            ("HiCache", False, "null", True, False, False),
            ("HiSparse", False, "null", False, True, False),
            ("external linker", False, "null", False, False, True),
        )
        for name, is_draft, disagg_mode, hicache, hisparse, external in excluded_pools:
            with self.subTest(name=name):
                memory_config = SimpleNamespace(
                    enable_hierarchical_cache=hicache,
                    enable_hisparse=hisparse,
                    enable_unified_cache_external_linker=external,
                )
                with (
                    patch(
                        "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.is_sparsity_driven_kv_offload_enabled",
                        return_value=True,
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator._is_npu",
                        True,
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator.get_memory",
                        return_value=memory_config,
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator.get_disagg",
                        return_value=SimpleNamespace(disaggregation_mode=disagg_mode),
                    ),
                ):
                    self.assertEqual(
                        get_sparsity_driven_kv_offload_cell_size(
                            model_config=model_config,
                            use_mla_backend=True,
                            num_layers=78,
                            element_size=2,
                            is_draft_worker=is_draft,
                        ),
                        128 * 78 * 2,
                    )

    def test_sparse_offload_cell_size_stays_dense_for_all_full_layers(self):
        model_config = _make_glm51_model_config()
        model_config.hf_config.indexer_types = ["full"] * 78
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
            self.assertEqual(
                get_sparsity_driven_kv_offload_cell_size(
                    model_config=model_config,
                    use_mla_backend=True,
                    num_layers=78,
                    element_size=2,
                ),
                128 * 78 * 2,
            )

    def test_sparse_offload_cell_size_falls_back_for_owner_empty_pp_range(self):
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
            self.assertEqual(
                get_sparsity_driven_kv_offload_cell_size(
                    model_config=model_config,
                    use_mla_backend=True,
                    num_layers=3,
                    element_size=2,
                    start_layer=3,
                ),
                128 * 3 * 2,
            )

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
