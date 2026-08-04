import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparsePoolConfigurator(CustomTestCase):
    def _compute_cell_size(
        self,
        kv_cache_dtype: torch.dtype,
        *,
        enable_hisparse: bool,
        host_to_device_ratio: int = 1,
    ) -> int:
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
        )
        hf_config.get_text_config = lambda: hf_config
        kvc = MagicMock(
            use_mla_backend=True,
            kv_cache_dtype=kv_cache_dtype,
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=hf_config,
            ),
            server_args=SimpleNamespace(
                enable_hisparse=enable_hisparse,
                hisparse_config=(f'{{"host_to_device_ratio": {host_to_device_ratio}}}'),
                dsa_prefill_backend="flashmla_sparse",
                dsa_decode_backend="flashmla_sparse",
            ),
        )

        with get_parallel().override(attn_tp_size=1):
            configurator = object.__new__(DefaultPoolConfigurator)
            return configurator._compute_cell_size(kvc, num_layers=2)

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


if __name__ == "__main__":
    unittest.main()
