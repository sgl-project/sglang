import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestHiSparsePoolConfigurator(CustomTestCase):
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


class TestHiSparseDraftPool(CustomTestCase):
    def test_draft_uses_resident_logical_capacity(self):
        # Exercise the real pool selection with allocation replaced by a spy.
        for is_draft in (False, True):
            for ratio in (1, 2, 4):
                with self.subTest(is_draft=is_draft, ratio=ratio):
                    kvc = SimpleNamespace(
                        is_draft_worker=is_draft,
                        pool_page_size=64,
                        kv_cache_dtype=torch.bfloat16,
                        model_config=SimpleNamespace(
                            kv_lora_rank=512, qk_rope_head_dim=64, hf_config=object()
                        ),
                        layer_info=SimpleNamespace(
                            num_effective_layers=1, start_layer=0, end_layer=1
                        ),
                        device="cpu",
                    )
                    module = "sglang.srt.mem_cache.kv_cache_configurator"
                    with (
                        patch(
                            f"{module}.get_memory",
                            return_value=SimpleNamespace(enable_hisparse=True),
                        ),
                        patch(
                            f"{module}.get_exec",
                            return_value=SimpleNamespace(
                                features=SimpleNamespace(enable_memory_saver=False)
                            ),
                        ),
                        patch(
                            "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                            return_value=(None, None),
                        ),
                        patch(
                            "sglang.srt.mem_cache.sparsity.parse_hisparse_config",
                            return_value=SimpleNamespace(host_to_device_ratio=ratio),
                        ),
                        patch(
                            f"{module}._should_elide_dsa_index_k", return_value=False
                        ),
                        patch(f"{module}.calculate_mla_kv_cache_dim", return_value=576),
                        patch(f"{module}.get_dsa_index_head_dim", return_value=128),
                        patch(f"{module}.get_dsa_index_kpool", return_value=1),
                        patch(
                            f"{module}.get_dsa_index_kpool_compress", return_value=False
                        ),
                        patch(
                            f"{module}.max_speculative_num_draft_tokens", return_value=6
                        ),
                        patch(f"{module}.DSATokenToKVPool") as resident,
                        patch(f"{module}.HiSparseDSATokenToKVPool") as sparse,
                    ):
                        KVCacheConfigurator._build_dsa_kv_pool(
                            kvc, max_total_num_tokens=128, max_running_requests=4
                        )
                    selected, unused = (
                        (resident, sparse) if is_draft else (sparse, resident)
                    )
                    unused.assert_not_called()
                    self.assertEqual(
                        selected.call_args.args[0], 128 * ratio if is_draft else 128
                    )
                    if is_draft:
                        self.assertNotIn(
                            "host_to_device_ratio", selected.call_args.kwargs
                        )
                    else:
                        self.assertEqual(
                            selected.call_args.kwargs["host_to_device_ratio"], ratio
                        )
                    self.assertEqual(selected.call_args.kwargs["tail_extra_slots"], 6)

    def test_resident_draft_does_not_create_target_coordinator(self):
        runner = SimpleNamespace(
            enable_hisparse=True,
            is_draft_worker=True,
            token_to_kv_pool=object.__new__(DSATokenToKVPool),
            hisparse_coordinator=None,
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(index_topk=2048)
            ),
            device="cpu",
            tp_group=SimpleNamespace(cpu_group=object()),
            spec_algorithm=SimpleNamespace(is_speculative=lambda: True),
        )
        with (
            patch(
                "sglang.srt.managers.hisparse_coordinator.HiSparseCoordinator"
            ) as coordinator,
            patch(
                "sglang.srt.managers.hisparse_coordinator.resolve_shared_index_layers",
                return_value=None,
            ),
            patch(
                "sglang.srt.model_executor.model_runner.get_parallel",
                return_value=SimpleNamespace(enable_dp_attention=False, pp_size=1),
            ),
            patch(
                "sglang.srt.mem_cache.sparsity.parse_hisparse_config",
                return_value=SimpleNamespace(
                    top_k=2048,
                    device_buffer_size=4096,
                    host_to_device_ratio=2,
                    swap_in_block_size=960,
                ),
            ),
        ):
            ModelRunner.maybe_init_hisparse_coordinator(runner)
            coordinator.assert_not_called()
            self.assertIsNone(runner.hisparse_coordinator)
            # The target still owns and initializes its sparse coordinator.
            runner.is_draft_worker = False
            ModelRunner.maybe_init_hisparse_coordinator(runner)
            coordinator.assert_called_once()
            self.assertIs(runner.hisparse_coordinator, coordinator.return_value)

    def test_target_budget_includes_full_resident_draft_kv(self):
        for ratio in (1, 2, 4):
            with self.subTest(ratio=ratio):
                kvc = SimpleNamespace(
                    kv_cache_dtype_str="bfloat16",
                    model_config=SimpleNamespace(hf_config=object()),
                    layer_info=SimpleNamespace(num_effective_layers=2),
                    is_draft_worker=False,
                    spec_algorithm=SimpleNamespace(
                        is_eagle=lambda: True, is_dflash_family=lambda: False
                    ),
                    spec_aux_config=SimpleNamespace(eagle_draft_num_layers=1),
                )
                module = "sglang.srt.model_executor.pool_configurator"
                with (
                    patch(f"{module}.mambaish_config", return_value=None),
                    patch(
                        f"{module}.get_schedule",
                        return_value=SimpleNamespace(max_total_tokens=1024),
                    ),
                    patch(
                        f"{module}.get_memory",
                        return_value=SimpleNamespace(enable_hisparse=True),
                    ),
                    patch(f"{module}.is_deepseek_dsa", return_value=True),
                    patch(
                        "sglang.srt.layers.cp.utils.get_glm_dsa_layer_split_effective_num_layers",
                        return_value=2,
                    ),
                    patch(
                        "sglang.srt.mem_cache.sparsity.parse_hisparse_config",
                        return_value=SimpleNamespace(host_to_device_ratio=ratio),
                    ),
                    patch.object(
                        DefaultPoolConfigurator,
                        "_compute_cell_size",
                        return_value=200 + 20 * ratio,
                    ),
                    patch.object(
                        DefaultPoolConfigurator,
                        "_compute_dsa_indexer_cell_size",
                        side_effect=[20 * ratio, 10 * ratio],
                    ),
                ):
                    config = DefaultPoolConfigurator(kvc)
                # Two target KV layers + logical target index + full draft KV/index.
                self.assertEqual(config._cell_size, 200 + 20 * ratio + 110 * ratio)


if __name__ == "__main__":
    unittest.main()
