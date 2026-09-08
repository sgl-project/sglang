"""CPU regression for FP8 DSA metadata resolution in hybrid models."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TokenToKVPool:
    def __init__(self, key_buffer: torch.Tensor):
        self.key_buffer = key_buffer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.key_buffer


class _Backend(AttentionBackend):
    needs_cpu_seq_lens = False

    def __init__(self, token_to_kv_pool: _TokenToKVPool):
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token_pool = None
        self.kv_index_translator = None
        self.max_context_len = 8


class _ForwardMetadata:
    def __init__(self, page_table: torch.Tensor):
        self.page_table_1_flattened = page_table


class _AttentionLayer:
    layer_id = 0


class _MHALayer:
    attn_mha = _AttentionLayer()
    kv_lora_rank = 2


class TestDeepseekFP8DSABackend(CustomTestCase):
    def test_cached_prefix_uses_full_attention_metadata(self):
        """A hybrid FP8 DSA cached-prefix read uses the full backend page table."""
        key_buffer = torch.arange(24, dtype=torch.bfloat16).reshape(6, 1, 4)
        page_table = torch.tensor([3, 5], dtype=torch.int32)
        token_to_kv_pool = _TokenToKVPool(key_buffer)
        full_backend = _Backend(token_to_kv_pool)
        full_backend.forward_metadata = _ForwardMetadata(page_table)
        hybrid_backend = HybridLinearAttnBackend(
            full_attn_backend=full_backend,
            linear_attn_backend=_Backend(token_to_kv_pool),
            full_attn_layers=[0],
        )

        with (
            forward_context(ForwardContext(attn_backend=hybrid_backend)),
            mock.patch.object(
                forward_mha,
                "dequantize_k_cache_paged",
                side_effect=lambda cache, indices: cache[indices],
            ),
        ):
            kv_a, k_pe = DeepseekMHAForwardMixin._get_mla_kv_buffer_from_fp8_for_dsa(
                _MHALayer(), object()
            )

        expected = key_buffer[page_table]
        torch.testing.assert_close(kv_a, expected[:, :, :2].squeeze(1))
        torch.testing.assert_close(k_pe, expected[:, :, 2:])


if __name__ == "__main__":
    unittest.main()
