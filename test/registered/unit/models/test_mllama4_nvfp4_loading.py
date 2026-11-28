"""Llama 4 ModelOpt NVFP4 checkpoints load expert block scales and permuted q/k rows.

NVIDIA's ModelOpt NVFP4 Llama 4 checkpoints store each layer's experts fused and
transposed, block scales included: experts.down_proj_weight_scale is
[experts, in/16, out], like experts.down_proj. Scalar scales stay broadcast.
"""

import unittest

import torch

from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFIX = "language_model.model.layers.0.feed_forward.experts"


class RecordingParam:
    def __init__(self, shape):
        self.data = torch.zeros(shape)
        self.calls = []

    def weight_loader(self, param, weight, name, shard_id, expert_id):
        self.calls.append((shard_id, expert_id, weight.clone()))


def load(name, tensor, params, num_experts):
    model = object.__new__(Llama4ForConditionalGeneration)
    loaded = set()
    handled = model._handle_expert_weights(
        name, tensor, [], params, num_experts, loaded
    )
    return handled, loaded


class TestFusedExpertScales(CustomTestCase):
    def test_down_proj_block_scales_load_per_expert_transposed(self):
        experts, in_blocks, out = 3, 4, 6
        scales = torch.randn(experts, in_blocks, out)
        params = {
            f"{PREFIX}.w2_weight_scale": RecordingParam((experts, out, in_blocks))
        }

        handled, loaded = load(
            f"{PREFIX}.down_proj_weight_scale", scales, params, experts
        )

        self.assertTrue(handled)
        self.assertEqual(loaded, {f"{PREFIX}.w2_weight_scale"})
        calls = params[f"{PREFIX}.w2_weight_scale"].calls
        self.assertEqual(
            [(s, e) for s, e, _ in calls], [("w2", e) for e in range(experts)]
        )
        for _, e, w in calls:
            torch.testing.assert_close(w, scales[e].T)

    def test_gate_up_block_scales_split_into_w1_and_w3(self):
        experts, in_blocks, inter = 2, 5, 4
        scales = torch.randn(experts, in_blocks, 2 * inter)
        params = {
            f"{PREFIX}.w13_weight_scale": RecordingParam(
                (experts, 2 * inter, in_blocks)
            )
        }

        load(f"{PREFIX}.gate_up_proj_weight_scale", scales, params, experts)

        calls = params[f"{PREFIX}.w13_weight_scale"].calls
        got = {(s, e): w for s, e, w in calls}
        self.assertEqual(len(calls), 2 * experts)
        for e in range(experts):
            torch.testing.assert_close(got[("w1", e)], scales[e, :, :inter].T)
            torch.testing.assert_close(got[("w3", e)], scales[e, :, inter:].T)

    def test_scalar_scales_still_broadcast_to_every_expert(self):
        experts = 3
        params = {f"{PREFIX}.w2_weight_scale_2": RecordingParam((experts,))}

        load(f"{PREFIX}.down_proj_weight_scale_2", torch.tensor(0.5), params, experts)

        torch.testing.assert_close(
            params[f"{PREFIX}.w2_weight_scale_2"].data, torch.full((experts,), 0.5)
        )
        self.assertEqual(params[f"{PREFIX}.w2_weight_scale_2"].calls, [])


def rotary_model(heads, kv_heads, head_dim, hidden):
    from types import SimpleNamespace

    config = SimpleNamespace(
        head_dim=head_dim,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        original_num_attention_heads=heads,
        original_total_num_kv_heads=kv_heads,
    )
    model = object.__new__(Llama4ForConditionalGeneration)
    object.__setattr__(model, "_modules", {})
    object.__setattr__(model, "language_model", SimpleNamespace(config=config))
    return model


def rotary_rows(rows, heads):
    # Row order the rotary permutation produces, for any column count.
    order = torch.arange(rows)
    return order.view(heads, rows // heads // 2, 2).transpose(1, 2).reshape(rows)


class TestRotaryPermuteQuantized(CustomTestCase):
    HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 4, 2, 8, 64

    def permute(self, name, tensor):
        model = rotary_model(self.HEADS, self.KV_HEADS, self.HEAD_DIM, self.HIDDEN)
        return model.permute_qk_weight_for_rotary(name, tensor)[1]

    def test_bf16_weight_order_is_unchanged(self):
        rows = self.KV_HEADS * self.HEAD_DIM
        w = torch.randn(rows, self.HIDDEN)
        got = self.permute("language_model.model.layers.0.self_attn.k_proj.weight", w)
        torch.testing.assert_close(got, w[rotary_rows(rows, self.KV_HEADS)])

    def test_packed_fp4_weight_and_block_scales_take_the_same_rows(self):
        rows = self.HEADS * self.HEAD_DIM
        packed = torch.randint(0, 255, (rows, self.HIDDEN // 2), dtype=torch.uint8)
        scales = torch.randn(rows, self.HIDDEN // 16)
        order = rotary_rows(rows, self.HEADS)
        prefix = "language_model.model.layers.0.self_attn.q_proj"

        torch.testing.assert_close(
            self.permute(f"{prefix}.weight", packed), packed[order]
        )
        torch.testing.assert_close(
            self.permute(f"{prefix}.weight_scale", scales), scales[order]
        )

    def test_scalar_scales_are_not_permuted(self):
        s = torch.tensor(0.25)
        prefix = "language_model.model.layers.0.self_attn.k_proj"
        torch.testing.assert_close(self.permute(f"{prefix}.weight_scale_2", s), s)
        torch.testing.assert_close(self.permute(f"{prefix}.input_scale", s), s)


if __name__ == "__main__":
    unittest.main()
