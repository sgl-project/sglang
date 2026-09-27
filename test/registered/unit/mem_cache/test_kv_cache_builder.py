"""CPU-only regression tests for KV cache construction."""

from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.mem_cache import kv_cache_builder
from sglang.test.test_utils import CustomTestCase


class TestBuildKVCache(CustomTestCase):
    def test_mlx_skips_transformers_architecture_resolution(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(),
            is_multimodal=False,
        )
        req_to_token_pool = object()
        token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        tp_worker = SimpleNamespace(
            is_hybrid_swa=False,
            model_runner=SimpleNamespace(
                model_config=model_config,
                mtp_draft_device_pools=None,
            ),
            get_memory_pool=lambda: (
                req_to_token_pool,
                token_to_kv_pool_allocator,
            ),
        )
        server_args = SimpleNamespace(
            enable_mamba_extra_buffer=lambda: False,
            enable_mamba_extra_buffer_lazy=lambda: False,
        )
        tree_cache = object()

        with (
            mock.patch.object(kv_cache_builder, "use_mlx", return_value=True),
            mock.patch.object(
                kv_cache_builder,
                "get_resolved_model_impl",
                side_effect=ValueError("cannot resolve remote architecture"),
            ),
            mock.patch.object(
                kv_cache_builder, "linear_attn_model_spec", return_value=None
            ),
            mock.patch.object(kv_cache_builder, "hybrid_gdn_config", return_value=None),
            mock.patch.object(
                kv_cache_builder, "hybrid_lightning_config", return_value=None
            ),
            mock.patch.object(
                kv_cache_builder, "kimi_linear_config", return_value=None
            ),
            mock.patch.object(kv_cache_builder, "mamba2_config", return_value=None),
            mock.patch.object(kv_cache_builder, "is_deepseek_dsa", return_value=False),
            mock.patch.object(
                kv_cache_builder,
                "get_parallel",
                return_value=SimpleNamespace(
                    dcp_enabled=False,
                    enable_dp_attention=False,
                ),
            ),
            mock.patch.object(
                kv_cache_builder,
                "get_memory",
                return_value=SimpleNamespace(
                    disable_radix_cache=False,
                    enable_session_radix_cache=False,
                    radix_eviction_policy="lru",
                ),
            ),
            mock.patch.object(
                kv_cache_builder,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_enable_radix_cache=False,
                ),
            ),
            mock.patch.object(
                kv_cache_builder,
                "get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=None),
            ),
            mock.patch.object(
                kv_cache_builder,
                "resolve_decode_retraction_backup",
                return_value="cpu_tensor",
            ),
            mock.patch.object(
                kv_cache_builder, "create_tree_cache", return_value=tree_cache
            ),
            mock.patch.object(kv_cache_builder, "init_mm_embedding_cache"),
        ):
            result = kv_cache_builder.build_kv_cache(
                server_args=server_args,
                model_config=model_config,
                tp_worker=tp_worker,
                page_size=1,
                spec_algorithm=SimpleNamespace(is_eagle=lambda: False),
                attn_tp_cpu_group=None,
                tp_cpu_group=None,
                attn_cp_cpu_group=None,
                enable_metrics=False,
                enable_kv_cache_events=False,
                ps=SimpleNamespace(pp_rank=0, pp_size=1, tp_rank=0, tp_size=1),
                tp_group=None,
                pp_group=SimpleNamespace(cpu_group=None),
                enable_hierarchical_cache=False,
            )

        self.assertIs(result.tree_cache, tree_cache)


if __name__ == "__main__":
    import unittest

    unittest.main()
