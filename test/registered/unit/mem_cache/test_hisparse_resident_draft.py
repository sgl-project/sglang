"""Actual configurator calls with CPU allocation and GPU pool constructors mocked."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator as config
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator


class TestHiSparseResidentDraft(unittest.TestCase):
    def test_shared_logical_span_reaches_pool_builder(self):
        allocator = HiSparseTokenToKVPoolAllocator(
            128, 64, torch.float32, "cpu", MagicMock(), False, 4
        )
        sizes = config._PoolSizes(
            max_total_num_tokens=128,
            max_running_requests=1,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
            c4_max_total_num_tokens=0,
            c128_max_total_num_tokens=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            c4_state_dtype=None,
            c128_state_dtype=None,
        )
        pool = SimpleNamespace(size=512)
        owner = SimpleNamespace(
            is_draft_worker=True,
            is_hybrid_swa=False,
            is_hybrid_swa_mtp_draft=False,
            pool_page_size=64,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["GlmMoeDsaForCausalLMNextN"], index_topk=2048
                )
            ),
            _validate_prefill_only_disable_kv_cache_pool_family=MagicMock(),
            _build_token_to_kv_pool=MagicMock(return_value=pool),
            _build_token_to_kv_pool_allocator=MagicMock(return_value=allocator),
        )
        with (
            patch.object(
                config,
                "get_memory",
                return_value=SimpleNamespace(enable_unified_memory=False),
            ),
            patch.object(
                config,
                "get_schedule",
                return_value=SimpleNamespace(prefill_only_disable_kv_cache=False),
            ),
        ):
            result = config.KVCacheConfigurator._init_pools(
                owner,
                sizes=sizes,
                req_to_token_pool=object(),
                token_to_kv_pool_allocator=allocator,
            )
        used_sizes = owner._build_token_to_kv_pool.call_args.kwargs["sizes"]
        self.assertEqual(used_sizes.max_total_num_tokens, 512)
        self.assertEqual(sizes.max_total_num_tokens, 128)
        self.assertIs(result.token_to_kv_pool, pool)
        self.assertIs(result.token_to_kv_pool_allocator, allocator)

    def test_draft_dsa_builder_selects_resident_pool(self):
        owner = SimpleNamespace(
            is_draft_worker=True,
            pool_page_size=64,
            kv_cache_dtype=torch.bfloat16,
            device="cpu",
            model_config=SimpleNamespace(
                kv_lora_rank=512, qk_rope_head_dim=64, hf_config=object()
            ),
            layer_info=SimpleNamespace(
                num_effective_layers=1, start_layer=0, end_layer=1
            ),
        )
        with (
            patch(
                "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                return_value=(None, None),
            ),
            patch.object(
                config, "get_memory", return_value=SimpleNamespace(enable_hisparse=True)
            ),
            patch.object(config, "_should_elide_dsa_index_k", return_value=False),
            patch.object(config, "calculate_mla_kv_cache_dim", return_value=576),
            patch.object(
                config,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch.object(config, "get_dsa_index_head_dim", return_value=128),
            patch.object(config, "get_dsa_index_kpool", return_value=1),
            patch.object(config, "get_dsa_index_kpool_compress", return_value=False),
            patch.object(config, "max_speculative_num_draft_tokens", return_value=4),
            patch.object(config, "DSATokenToKVPool") as resident,
            patch.object(config, "HiSparseDSATokenToKVPool") as sparse,
        ):
            result = config.KVCacheConfigurator._build_dsa_kv_pool(
                owner, max_total_num_tokens=512, max_running_requests=1
            )
        self.assertIs(result, resident.return_value)
        sparse.assert_not_called()
        self.assertEqual(resident.call_args.args, (512,))
        self.assertNotIn("host_to_device_ratio", resident.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
