import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHiSparsePoolConfigurator(CustomTestCase):
    def _compute_cell_size(
        self,
        kv_cache_dtype: torch.dtype,
        *,
        enable_hisparse: bool,
        host_to_device_ratio: int = 1,
        num_layers: int = 2,
        disaggregation_mode: str = "null",
        index_topk_pattern: list[str] | None = None,
    ) -> int:
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
        )
        if index_topk_pattern is not None:
            hf_config.index_topk_pattern = index_topk_pattern
        hf_config.get_text_config = lambda: hf_config

        override = get_context().override_server_args(
            enable_hisparse=enable_hisparse,
            hisparse_config=f'{{"host_to_device_ratio": {host_to_device_ratio}}}',
            enable_hierarchical_cache=False,
            disaggregation_mode=disaggregation_mode,
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

    def test_hisparse_decode_prices_only_indexer_owning_layers(self):
        # One anchor layer plus three shared-topk layers. On a HiSparse decode
        # engine the shared layers own no index-K, so the cell must price one
        # indexer layer, not four. The sizing path and the allocation path read
        # the same `_should_elide_dsa_index_k`; if sizing priced all four,
        # max_total_num_tokens would hide the memory the pool never allocates.
        #
        # fp8 cell = 656 B/layer latent + 132 B/token/indexer-layer * ratio:
        #   decode  (1 priced layer)  = 4 * 656 + 132 * 1 * 2 = 2888
        #   prefill (4 priced layers) = 4 * 656 + 132 * 4 * 2 = 3680
        kwargs = dict(
            enable_hisparse=True,
            host_to_device_ratio=2,
            num_layers=4,
            index_topk_pattern=["A", "S", "S", "S"],
        )
        decode_cell = self._compute_cell_size(
            torch.float8_e4m3fn, disaggregation_mode="decode", **kwargs
        )
        prefill_cell = self._compute_cell_size(
            torch.float8_e4m3fn, disaggregation_mode="prefill", **kwargs
        )
        self.assertEqual(decode_cell, 2888)
        self.assertEqual(prefill_cell, 3680)

    def test_hisparse_single_engine_prices_every_layer(self):
        # Without PD there is no prefill peer to own the shared layers' index-K,
        # so a single HiSparse engine keeps all four indexer layers.
        cell_size = self._compute_cell_size(
            torch.float8_e4m3fn,
            enable_hisparse=True,
            host_to_device_ratio=2,
            num_layers=4,
            index_topk_pattern=["A", "S", "S", "S"],
        )
        self.assertEqual(cell_size, 3680)


if __name__ == "__main__":
    unittest.main()
