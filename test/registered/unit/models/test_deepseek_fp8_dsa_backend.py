"""CPU regression for FP8 DSA metadata ownership in hybrid backends."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mha,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tbo_backend(primary):
    backend = object.__new__(TboAttnBackend)
    backend.primary = primary
    return backend


class TestResolveAttnBackend(unittest.TestCase):
    def test_fp8_dsa_reads_metadata_from_hybrid_full_backend(self):
        full_backend = SimpleNamespace(name="full")
        hybrid_backend = SimpleNamespace(full_attn_backend=full_backend)
        tbo_backend = _tbo_backend(hybrid_backend)
        page_table = torch.tensor([3, 5], dtype=torch.int32)
        latent_cache = torch.zeros((2, 1, 4), dtype=torch.float32)
        dequantized = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
        full_backend.forward_metadata = SimpleNamespace(
            page_table_1_flattened=page_table
        )
        fake_self = SimpleNamespace(
            attn_mha=SimpleNamespace(layer_id=0),
            kv_lora_rank=2,
        )
        token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda _: latent_cache)

        with (
            mock.patch.object(
                forward_mha, "get_attn_backend", return_value=tbo_backend
            ),
            mock.patch.object(
                forward_mha,
                "get_token_to_kv_pool",
                return_value=token_to_kv_pool,
            ),
            mock.patch.object(
                forward_mha,
                "dequantize_k_cache_paged",
                return_value=dequantized,
            ) as dequantize,
        ):
            kv_a, k_pe = DeepseekMHAForwardMixin._get_mla_kv_buffer_from_fp8_for_dsa(
                fake_self, SimpleNamespace()
            )

        dequantize.assert_called_once_with(latent_cache, page_table)
        torch.testing.assert_close(kv_a, dequantized[:, :, :2].squeeze(1))
        torch.testing.assert_close(k_pe, dequantized[:, :, 2:])

    def test_keeps_regular_backend(self):
        backend = SimpleNamespace(name="regular")

        with mock.patch.object(forward_mha, "get_attn_backend", return_value=backend):
            resolved = forward_mha.resolve_attn_backend(
                SimpleNamespace(forward_mode=SimpleNamespace())
            )

        self.assertIs(resolved, backend)


if __name__ == "__main__":
    unittest.main()
