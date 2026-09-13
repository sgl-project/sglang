"""Unit tests for Gemma3n attention output normalization."""

import unittest
from unittest.mock import MagicMock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch
from torch import nn

from sglang.srt.models.gemma3n_causal import Gemma3nAttention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGemma3nAttention(CustomTestCase):
    def test_flattens_head_shaped_output_before_o_proj(self):
        num_tokens = 3
        num_heads = 2
        num_kv_heads = 1
        head_dim = 4

        attention = object.__new__(Gemma3nAttention)
        nn.Module.__init__(attention)
        attention.num_heads = num_heads
        attention.num_kv_heads = num_kv_heads
        attention.head_dim = head_dim
        attention.q_size = num_heads * head_dim
        attention.kv_size = num_kv_heads * head_dim
        attention.is_kv_shared_layer = False
        attention.kv_shared_layer_index = None
        attention.qkv_proj = MagicMock(
            return_value=(
                torch.zeros((num_tokens, attention.q_size + 2 * attention.kv_size)),
                None,
            )
        )
        attention.q_norm = nn.Identity()
        attention.k_norm = nn.Identity()
        attention.v_norm = nn.Identity()
        attention.rotary_emb = MagicMock(side_effect=lambda _, q, k: (q, k))
        attention.attn = MagicMock(
            return_value=torch.zeros((num_tokens, num_heads, head_dim))
        )
        attention.o_proj = MagicMock(
            return_value=(torch.zeros((num_tokens, num_heads * head_dim)), None)
        )

        output = attention(
            torch.zeros((num_tokens, 1)),
            torch.arange(num_tokens),
            object(),
        )

        self.assertEqual(output.shape, (num_tokens, num_heads * head_dim))
        self.assertEqual(
            attention.o_proj.call_args.args[0].shape,
            (num_tokens, num_heads * head_dim),
        )


if __name__ == "__main__":
    unittest.main()
