"""CPU-only routing tests for the Qwen3.5 GDN MXFP4 handoff."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _OutProj:
    def __init__(self, scheme):
        self.scheme = scheme
        self.last_input = None

    def __call__(self, value):
        self.last_input = value
        return value, None


def _make_norm(head_dim: int):
    return SimpleNamespace(
        group_size=None,
        norm_before_gate=True,
        bias=None,
        weight=torch.empty(head_dim, dtype=torch.bfloat16),
        eps=1e-6,
        activation="swish",
    )


class TestQwen35GDNMXFP4Routing(unittest.TestCase):
    def test_prequantized_tuple_bypasses_dense_norm_output(self):
        num_tokens, num_heads, head_dim = 2, 3, 128
        x = torch.empty((num_tokens * num_heads, head_dim), dtype=torch.bfloat16)
        z_storage = torch.empty(
            (num_tokens, num_heads, head_dim * 2), dtype=torch.bfloat16
        )
        z = z_storage[..., head_dim:]
        packed = torch.empty((num_tokens, num_heads * head_dim // 2), dtype=torch.uint8)
        scales = torch.empty((256, num_heads * head_dim // 32), dtype=torch.uint8)
        prepare_input = Mock(return_value=(packed, scales))
        out_proj = _OutProj(
            SimpleNamespace(prepare_fused_rmsnorm_gated_input=prepare_input)
        )
        norm = _make_norm(head_dim)
        owner = SimpleNamespace(
            norm=norm,
            out_proj=out_proj,
            num_v_heads=num_heads,
            attn_tp_size=1,
        )

        actual = Qwen3_5GatedDeltaNet._norm_and_out_proj(owner, x, z, z.shape)

        self.assertIs(actual[0], packed)
        self.assertIs(actual[1], scales)
        self.assertIs(out_proj.last_input[0], packed)
        self.assertIs(out_proj.last_input[1], scales)
        prepare_input.assert_called_once_with(
            out_proj,
            x,
            z,
            norm.weight,
            norm.eps,
            num_heads=num_heads,
            activation="swish",
        )

    def test_unquantized_scheme_preserves_dense_fallback(self):
        num_tokens, num_heads, head_dim = 2, 3, 128
        x = torch.empty((num_tokens * num_heads, head_dim), dtype=torch.bfloat16)
        z = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.bfloat16)
        dense = torch.empty_like(x)
        norm = Mock(return_value=dense)
        norm.group_size = None
        norm.norm_before_gate = True
        norm.bias = None
        norm.weight = torch.empty(head_dim, dtype=torch.bfloat16)
        norm.eps = 1e-6
        norm.activation = "swish"
        out_proj = _OutProj(scheme=None)
        owner = SimpleNamespace(
            norm=norm,
            out_proj=out_proj,
            num_v_heads=num_heads,
            attn_tp_size=1,
        )

        actual = Qwen3_5GatedDeltaNet._norm_and_out_proj(owner, x, z, z.shape)

        norm.assert_called_once_with(x, z)
        self.assertEqual(actual.shape, (num_tokens, num_heads * head_dim))
        self.assertIs(actual, out_proj.last_input)

    def test_declined_fusion_preserves_dense_fallback(self):
        num_tokens, num_heads, head_dim = 2, 3, 128
        x = torch.empty((num_tokens * num_heads, head_dim), dtype=torch.bfloat16)
        z = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.bfloat16)
        dense = torch.empty_like(x)
        prepare_input = Mock(return_value=None)
        out_proj = _OutProj(
            SimpleNamespace(prepare_fused_rmsnorm_gated_input=prepare_input)
        )
        norm = Mock(return_value=dense)
        norm.group_size = None
        norm.norm_before_gate = True
        norm.bias = None
        norm.weight = torch.empty(head_dim, dtype=torch.bfloat16)
        norm.eps = 1e-6
        norm.activation = "swish"
        owner = SimpleNamespace(
            norm=norm,
            out_proj=out_proj,
            num_v_heads=num_heads,
            attn_tp_size=1,
        )

        actual = Qwen3_5GatedDeltaNet._norm_and_out_proj(owner, x, z, z.shape)

        prepare_input.assert_called_once()
        norm.assert_called_once_with(x, z)
        self.assertIs(actual, out_proj.last_input)

    def test_incompatible_norm_policy_does_not_call_quant_scheme(self):
        num_tokens, num_heads, head_dim = 2, 3, 128
        x = torch.empty((num_tokens * num_heads, head_dim), dtype=torch.bfloat16)
        z = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.bfloat16)
        dense = torch.empty_like(x)
        prepare_input = Mock()
        out_proj = _OutProj(
            SimpleNamespace(prepare_fused_rmsnorm_gated_input=prepare_input)
        )
        norm = Mock(return_value=dense)
        norm.group_size = None
        norm.norm_before_gate = False
        norm.bias = None
        norm.weight = torch.empty(head_dim, dtype=torch.bfloat16)
        norm.eps = 1e-6
        norm.activation = "swish"
        owner = SimpleNamespace(
            norm=norm,
            out_proj=out_proj,
            num_v_heads=num_heads,
            attn_tp_size=1,
        )

        Qwen3_5GatedDeltaNet._norm_and_out_proj(owner, x, z, z.shape)

        prepare_input.assert_not_called()
        norm.assert_called_once_with(x, z)


if __name__ == "__main__":
    unittest.main()
