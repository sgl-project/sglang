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
    def test_cell_size_matches_allocated_kv_layout(self):
        for kv_cache_dtype, expected_cell_size in (
            (torch.bfloat16, 3360),
            (torch.float8_e4m3fn, 2368),
        ):
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
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
                        enable_hisparse=True,
                        hisparse_config='{"host_to_device_ratio": 4}',
                        dsa_prefill_backend="flashmla_sparse",
                        dsa_decode_backend="flashmla_sparse",
                    ),
                )

                with get_parallel().override(attn_tp_size=1):
                    configurator = object.__new__(DefaultPoolConfigurator)
                    cell_size = configurator._compute_cell_size(kvc, num_layers=2)

                self.assertEqual(cell_size, expected_cell_size)


if __name__ == "__main__":
    unittest.main()
