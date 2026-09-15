"""Existing DFlash call contracts and CPU numerics.

Real projection, norm, RoPE and worker code run with CPU tensors. NPU/table
kernels and KV storage are mocked at their boundaries. These tests cover
caller compatibility, not real accelerator execution or NPU 192 support.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding.base import (
    LinearScalingRotaryEmbedding,
    RotaryEmbedding,
)
from sglang.srt.models import dflash
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CountingLinear(nn.Linear):
    def __init__(self, in_features, out_features, dtype):
        super().__init__(in_features, out_features, bias=False, dtype=dtype)
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return super().forward(hidden_states), None


class _CaptureAttention(nn.Module):
    layer_id = 0

    def forward(self, q, k, v, forward_batch):
        self.qkv = tuple(x.detach().clone() for x in (q, k, v))
        return q


def _make_attention(head_dim, *, dtype=torch.float32, neox=True, scaling=1.0):
    # Avoid distributed group initialization while exercising the production
    # forward with real projection, norm and rotary modules.
    attn = dflash.DFlashAttention.__new__(dflash.DFlashAttention)
    nn.Module.__init__(attn)
    attn.head_dim = head_dim
    attn.num_heads = 4
    attn.num_kv_heads = 2
    attn.q_size = 4 * head_dim
    attn.kv_size = 2 * head_dim
    attn.qkv_proj = _CountingLinear(16, attn.q_size + 2 * attn.kv_size, dtype)
    attn.o_proj = _CountingLinear(attn.q_size, 16, dtype)
    attn.q_norm = RMSNorm(head_dim, eps=1e-6).to(dtype=dtype)
    attn.k_norm = RMSNorm(head_dim, eps=1e-6).to(dtype=dtype)
    with torch.no_grad():
        attn.q_norm.weight.copy_(torch.linspace(0.5, 1.5, head_dim))
        attn.k_norm.weight.copy_(torch.linspace(1.7, 0.7, head_dim))
    rope_kwargs = dict(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=neox,
        dtype=dtype,
    )
    if scaling == 1.0:
        attn.rotary_emb = RotaryEmbedding(**rope_kwargs)
    else:
        attn.rotary_emb = LinearScalingRotaryEmbedding(
            **rope_kwargs, scaling_factors=[scaling]
        )
    attn.attn = _CaptureAttention()
    attn.attention_sink_bias = None
    attn.use_table_qk_norm_rope = False
    return attn


def _reference_qkv(attn, hidden_states, positions, *, neox, scaling):
    q, k, v = F.linear(hidden_states, attn.qkv_proj.weight).split(
        [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
    )

    def norm_and_rotate(x, weight):
        by_head = x.float().reshape(x.shape[0], -1, attn.head_dim)
        normalized = by_head * torch.rsqrt(
            by_head.square().mean(-1, keepdim=True) + 1e-6
        )
        normalized = (normalized.to(x.dtype) * weight).to(x.dtype)
        inv_freq = 10000.0 ** (
            -torch.arange(0, attn.head_dim, 2, dtype=torch.float32) / attn.head_dim
        )
        angles = positions.float()[:, None] / scaling * inv_freq[None, :]
        cos = angles.cos().to(x.dtype)[:, None, :]
        sin = angles.sin().to(x.dtype)[:, None, :]
        if neox:
            first, second = normalized.chunk(2, dim=-1)
            rotated = torch.cat(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            )
        else:
            first, second = normalized[..., ::2], normalized[..., 1::2]
            rotated = torch.stack(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            ).flatten(-2)
        return rotated.reshape_as(x)

    return (
        norm_and_rotate(q, attn.q_norm.weight),
        norm_and_rotate(k, attn.k_norm.weight),
        v,
    )


class TestDFlashNpuQkNormRope(CustomTestCase):
    def test_other_kv_rope_paths_keep_module_dispatch(self):
        torch.manual_seed(2026)
        for is_npu, head_dim in ((False, 192), (True, 128), (True, 256)):
            with self.subTest(is_npu=is_npu, head_dim=head_dim):
                attn = _make_attention(head_dim)
                rope_forward = Mock(wraps=attn.rotary_emb.forward_native)
                attn.rotary_emb._forward_method = rope_forward
                hidden_states = torch.randn(3, 16)
                positions = torch.tensor([0, 7, 31], dtype=torch.int64)
                expected_qkv = _reference_qkv(
                    attn, hidden_states, positions, neox=True, scaling=1.0
                )
                with patch.object(dflash, "_is_npu", is_npu):
                    k, _ = attn.kv_proj_only(hidden_states)
                    actual_k = attn.apply_k_rope(positions, attn.apply_k_norm(k))
                rope_forward.assert_called_once()
                torch.testing.assert_close(
                    actual_k, expected_qkv[1], rtol=1e-5, atol=1e-5
                )

    def test_non_npu_forward_keeps_separate_ops(self):
        torch.manual_seed(2026)
        for head_dim in (128, 192, 256):
            with self.subTest(head_dim=head_dim):
                attn = _make_attention(head_dim)
                hidden_states = torch.randn(3, 16)
                positions = torch.tensor([0, 7, 31], dtype=torch.int64)
                expected_qkv = _reference_qkv(
                    attn, hidden_states, positions, neox=True, scaling=1.0
                )
                fused = Mock(
                    side_effect=AssertionError("NPU kernel must not run on CPU")
                )
                with (
                    patch.object(dflash, "_is_npu", False),
                    patch.object(dflash, "split_qkv_rmsnorm_rope", fused, create=True),
                ):
                    actual = attn(positions, hidden_states, forward_batch=None)
                fused.assert_not_called()
                self.assertEqual(attn.qkv_proj.calls, 1)
                for got, expected in zip(attn.attn.qkv, expected_qkv):
                    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(
                    actual,
                    F.linear(expected_qkv[0], attn.o_proj.weight),
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_supported_head_dims_keep_fused_dispatch(self):
        torch.manual_seed(2026)
        for head_dim in (128, 256):
            for neox in (True, False):
                with self.subTest(head_dim=head_dim, neox=neox):
                    attn = _make_attention(head_dim, neox=neox)
                    hidden_states = torch.randn(3, 16)
                    positions = torch.tensor([0, 7, 31], dtype=torch.int64)
                    projected = F.linear(hidden_states, attn.qkv_proj.weight)
                    fused_outputs = tuple(
                        x.clone()
                        for x in projected.split(
                            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
                        )
                    )
                    fused = Mock(return_value=fused_outputs)
                    with (
                        patch.object(dflash, "_is_npu", True),
                        patch.object(
                            dflash, "split_qkv_rmsnorm_rope", fused, create=True
                        ),
                        patch.object(
                            dflash,
                            "apply_qk_norm",
                            side_effect=AssertionError(
                                "separate norm must not run for supported fused sizes"
                            ),
                        ),
                    ):
                        actual = attn(positions, hidden_states, forward_batch=None)

                    fused.assert_called_once()
                    self.assertEqual(fused.call_args.args[5], head_dim)
                    torch.testing.assert_close(fused.call_args.args[0], projected)
                    for got, expected in zip(attn.attn.qkv, fused_outputs):
                        torch.testing.assert_close(got, expected)
                    torch.testing.assert_close(
                        actual, F.linear(fused_outputs[0], attn.o_proj.weight)
                    )

    def test_npu_worker_projects_hidden_before_writing_layer_kv(self):
        from sglang.srt.speculative import dflash_worker_v2

        torch.manual_seed(2026)
        for head_dim in (128, 256):
            with self.subTest(head_dim=head_dim):
                attentions = [_make_attention(head_dim) for _ in range(2)]
                layers = [
                    SimpleNamespace(self_attn=attn, offset=index + 1)
                    for index, attn in enumerate(attentions)
                ]
                for attn in attentions:
                    attn.attn.k_scale = None
                    attn.attn.v_scale = None
                pool = SimpleNamespace(set_kv_buffer=Mock())
                worker = SimpleNamespace(
                    draft_model=SimpleNamespace(
                        layers=layers,
                        prepare_context_hidden_for_kv=lambda layer, hidden: (
                            hidden + layer.offset
                        ),
                    ),
                    draft_model_runner=SimpleNamespace(token_to_kv_pool=pool),
                )
                hidden = torch.randn(3, 16)
                positions = torch.tensor([0, 7, 31], dtype=torch.int64)
                cache_loc = torch.tensor([2, 4, 6], dtype=torch.int64)

                def fused_split(qkv, sin, cos, q_size, kv_size, dim, **kwargs):
                    self.assertEqual(qkv.shape, (3, q_size + 2 * kv_size))
                    self.assertEqual(dim, head_dim)
                    return qkv.split([q_size, kv_size, kv_size], dim=-1)

                with (
                    patch.object(dflash_worker_v2, "_is_npu", True),
                    patch.object(
                        dflash,
                        "split_qkv_rmsnorm_rope",
                        side_effect=fused_split,
                        create=True,
                    ) as fused,
                ):
                    dflash_worker_v2.DFlashWorkerV2._append_target_hidden_sequential(
                        worker, hidden, positions, cache_loc
                    )

                self.assertEqual(fused.call_count, len(layers))
                self.assertEqual(pool.set_kv_buffer.call_count, len(layers))
                for layer, call in zip(layers, pool.set_kv_buffer.call_args_list):
                    attn = layer.self_attn
                    expected = F.linear(hidden + layer.offset, attn.qkv_proj.weight)
                    _, expected_k, expected_v = expected.split(
                        [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
                    )
                    self.assertEqual(attn.qkv_proj.calls, 1)
                    self.assertIs(call.args[0], attn.attn)
                    torch.testing.assert_close(call.args[1], cache_loc)
                    torch.testing.assert_close(
                        call.args[2], expected_k.reshape(3, attn.num_kv_heads, head_dim)
                    )
                    torch.testing.assert_close(
                        call.args[3], expected_v.reshape(3, attn.num_kv_heads, head_dim)
                    )

    def test_non_npu_bf16_keeps_table_dispatch(self):
        from sglang.srt.speculative import dflash_utils

        torch.manual_seed(2026)
        attn = _make_attention(128, dtype=torch.bfloat16)
        attn.use_table_qk_norm_rope = True
        hidden = torch.randn(3, 16, dtype=torch.bfloat16)
        positions = torch.tensor([0, 7, 31], dtype=torch.int64)
        projected = F.linear(hidden, attn.qkv_proj.weight)
        with (
            patch.object(dflash, "_is_npu", False),
            patch.object(dflash_utils, "table_qk_norm_rope_") as table,
            patch.object(
                dflash,
                "apply_qk_norm",
                side_effect=AssertionError("table path must not use separate norm"),
            ),
            patch.object(
                dflash,
                "split_qkv_rmsnorm_rope",
                side_effect=AssertionError("non-NPU path must not use NPU split"),
                create=True,
            ),
        ):
            attn(positions, hidden, forward_batch=None)
        table.assert_called_once()
        torch.testing.assert_close(table.call_args.args[0], projected)
        torch.testing.assert_close(table.call_args.args[1], positions)
        self.assertIs(table.call_args.args[2], attn.q_norm.weight)
        self.assertIs(table.call_args.args[3], attn.k_norm.weight)
        self.assertEqual(attn.qkv_proj.calls, 1)
        expected_qkv = projected.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
        )
        for got, expected in zip(attn.attn.qkv, expected_qkv):
            torch.testing.assert_close(got, expected)


if __name__ == "__main__":
    unittest.main()
