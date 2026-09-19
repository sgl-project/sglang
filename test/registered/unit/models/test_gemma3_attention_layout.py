import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.rotary_embedding.utils import (
    apply_rotary_pos_emb_native_eager,
)
from sglang.srt.models.gemma3_causal import Gemma3Attention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Projection:
    def __init__(self, output):
        self.output = output
        self.input_shape = None

    def __call__(self, inputs):
        self.input_shape = tuple(inputs.shape)
        return inputs if self.output is None else self.output, None


class _Attention:
    def __init__(self):
        self.input_shapes = None

    def __call__(self, q, k, v, forward_batch):
        self.input_shapes = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
        return q


class _RotaryEmbedding:
    def __init__(self):
        self.input_shapes = None

    def __call__(self, positions, q, k):
        self.input_shapes = (tuple(positions.shape), tuple(q.shape), tuple(k.shape))
        return q, k


class TestGemma3AttentionLayout(CustomTestCase):
    tokens = 8
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    def _attention(self):
        qkv = torch.arange(
            self.tokens * (self.q_size + 2 * self.kv_size), dtype=torch.float32
        ).reshape(self.tokens, -1)
        attn = _Attention()
        o_proj = _Projection(None)
        rotary_emb = _RotaryEmbedding()
        layer = SimpleNamespace(
            qkv_proj=_Projection(qkv),
            q_size=self.q_size,
            kv_size=self.kv_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            q_norm=nn.Identity(),
            k_norm=nn.Identity(),
            rotary_emb=rotary_emb,
            attn=attn,
            o_proj=o_proj,
        )
        return layer, attn, o_proj, rotary_emb

    def _assert_token_major_contract(self, attn, o_proj, output):
        self.assertEqual(
            attn.input_shapes,
            (
                (self.tokens, self.num_heads, self.head_dim),
                (self.tokens, self.num_kv_heads, self.head_dim),
                (self.tokens, self.kv_size),
            ),
        )
        self.assertEqual(o_proj.input_shape, (self.tokens, self.q_size))
        self.assertEqual(tuple(output.shape), (self.tokens, self.q_size))

    def test_forward_native_passes_token_major_q_and_k_to_radix_attention(self):
        layer, attn, o_proj, _ = self._attention()
        hidden_states = torch.empty(self.tokens, self.q_size)
        position_embeddings = (
            torch.empty(1, self.tokens, self.head_dim),
            torch.empty(1, self.tokens, self.head_dim),
        )

        with patch(
            "sglang.srt.models.gemma3_causal.apply_rotary_pos_emb",
            side_effect=lambda q, k, cos, sin: (q, k),
        ) as rotary_emb:
            output = Gemma3Attention.forward_native(
                layer,
                positions=torch.arange(self.tokens),
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                forward_batch=object(),
            )

        q, k, cos, sin = rotary_emb.call_args.args
        self.assertEqual(tuple(q.shape), (self.tokens, self.num_heads, self.head_dim))
        self.assertEqual(
            tuple(k.shape), (self.tokens, self.num_kv_heads, self.head_dim)
        )
        self.assertEqual(tuple(cos.shape), (self.tokens, self.head_dim))
        self.assertEqual(tuple(sin.shape), (self.tokens, self.head_dim))
        self._assert_token_major_contract(attn, o_proj, output)

    def test_forward_cpu_passes_token_major_q_and_k_to_radix_attention(self):
        layer, attn, o_proj, rotary_emb = self._attention()

        output = Gemma3Attention.forward_cpu(
            layer,
            positions=torch.arange(self.tokens),
            hidden_states=torch.empty(self.tokens, self.q_size),
            position_embeddings=(torch.empty(0), torch.empty(0)),
            forward_batch=object(),
        )

        self.assertEqual(
            rotary_emb.input_shapes,
            (
                (self.tokens,),
                (self.tokens, self.num_heads, self.head_dim),
                (self.tokens, self.num_kv_heads, self.head_dim),
            ),
        )
        self._assert_token_major_contract(attn, o_proj, output)

    def test_token_major_rope_matches_the_previous_batch_major_layout(self):
        q = torch.arange(
            self.tokens * self.num_heads * self.head_dim, dtype=torch.float32
        ).reshape(self.tokens, self.num_heads, self.head_dim)
        k = torch.arange(
            self.tokens * self.num_kv_heads * self.head_dim, dtype=torch.float32
        ).reshape(self.tokens, self.num_kv_heads, self.head_dim)
        angles = torch.arange(self.tokens * self.head_dim, dtype=torch.float32).reshape(
            1, self.tokens, self.head_dim
        )
        cos = angles.cos()
        sin = angles.sin()

        old_q, old_k = apply_rotary_pos_emb_native_eager(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            cos,
            sin,
        )
        old_q = old_q.permute(0, 2, 1, 3).squeeze(0)
        old_k = old_k.permute(0, 2, 1, 3).squeeze(0)

        new_q, new_k = apply_rotary_pos_emb_native_eager(
            q,
            k,
            cos.reshape(-1, cos.shape[-1]),
            sin.reshape(-1, sin.shape[-1]),
        )

        torch.testing.assert_close(new_q, old_q)
        torch.testing.assert_close(new_k, old_k)


if __name__ == "__main__":
    unittest.main()
