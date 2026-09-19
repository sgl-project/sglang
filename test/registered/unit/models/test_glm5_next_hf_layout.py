"""GLM-5.3-Flash checkpoints written by transformers load under released names.

transformers renames GLM-5.3-Flash tensors while reading the released
checkpoint and does not reverse that on save, so `save_pretrained` output --
a fine-tune, a merged adapter, a bf16 re-export -- reaches the loader under
transformers module names. Those names miss `params_dict`, so the MoE, mHC and
KDA tensors used to be dropped with no error and the server came up on a
partly-initialised model. These tests pin the mapping in both directions: the
transformers layout has to arrive as the released one, and the released layout
has to pass through untouched.
"""

import unittest

import torch

from sglang.srt.models.glm5_next import convert_hf_native_weights
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LAYER = "model.language_model.layers.0"


class TestGlm5NextHfLayout(CustomTestCase):
    def test_mhc_parameters_arrive_under_released_names(self):
        weights = [
            (f"{LAYER}.attn_hc.fn", torch.zeros(2, 3)),
            (f"{LAYER}.attn_hc.base", torch.zeros(2)),
            (f"{LAYER}.attn_hc.scale", torch.zeros(3)),
            (f"{LAYER}.ffn_hc.fn", torch.zeros(2, 3)),
            (f"{LAYER}.ffn_hc.base", torch.zeros(2)),
            (f"{LAYER}.ffn_hc.scale", torch.zeros(3)),
        ]

        got = [name for name, _ in convert_hf_native_weights(weights)]

        self.assertEqual(
            got,
            [
                f"{LAYER}.hc_attn_fn",
                f"{LAYER}.hc_attn_base",
                f"{LAYER}.hc_attn_scale",
                f"{LAYER}.hc_ffn_fn",
                f"{LAYER}.hc_ffn_base",
                f"{LAYER}.hc_ffn_scale",
            ],
        )

    def test_forget_gate_submodule_is_flattened_onto_self_attn(self):
        weights = [
            (f"{LAYER}.self_attn.forget_gate.f_a_proj.weight", torch.zeros(4, 8)),
            (f"{LAYER}.self_attn.forget_gate.f_b_proj.weight", torch.zeros(6, 4)),
            (f"{LAYER}.self_attn.forget_gate.dt_bias", torch.zeros(6)),
            (f"{LAYER}.self_attn.forget_gate.A_log", torch.zeros(6)),
        ]

        got = [name for name, _ in convert_hf_native_weights(weights)]

        self.assertEqual(
            got,
            [
                f"{LAYER}.self_attn.f_a_proj.weight",
                f"{LAYER}.self_attn.f_b_proj.weight",
                f"{LAYER}.self_attn.dt_bias",
                f"{LAYER}.self_attn.A_log",
            ],
        )

    def test_packed_experts_split_gate_before_up(self):
        n_experts, inter, hidden = 3, 4, 5
        gate = torch.randn(n_experts, inter, hidden)
        up = torch.randn(n_experts, inter, hidden)
        packed = torch.cat([gate, up], dim=1)

        got = dict(
            convert_hf_native_weights([(f"{LAYER}.mlp.experts.gate_up_proj", packed)])
        )

        self.assertEqual(len(got), 2 * n_experts)
        for expert in range(n_experts):
            torch.testing.assert_close(
                got[f"{LAYER}.mlp.experts.{expert}.gate_proj.weight"], gate[expert]
            )
            torch.testing.assert_close(
                got[f"{LAYER}.mlp.experts.{expert}.up_proj.weight"], up[expert]
            )

    def test_packed_down_proj_splits_per_expert(self):
        packed = torch.randn(3, 5, 4)

        got = dict(
            convert_hf_native_weights([(f"{LAYER}.mlp.experts.down_proj", packed)])
        )

        self.assertEqual(len(got), 3)
        for expert in range(3):
            torch.testing.assert_close(
                got[f"{LAYER}.mlp.experts.{expert}.down_proj.weight"], packed[expert]
            )

    def test_packed_conv1d_splits_into_qkv_keeping_the_singleton_dim(self):
        projection, kernel = 4, 3
        parts = [torch.randn(projection, 1, kernel) for _ in range(3)]
        packed = torch.cat(parts, dim=0)

        got = list(
            convert_hf_native_weights([(f"{LAYER}.self_attn.conv1d.weight", packed)])
        )

        self.assertEqual(
            [name for name, _ in got],
            [
                f"{LAYER}.self_attn.q_conv1d.weight",
                f"{LAYER}.self_attn.k_conv1d.weight",
                f"{LAYER}.self_attn.v_conv1d.weight",
            ],
        )
        for (_, actual), expected in zip(got, parts):
            self.assertEqual(tuple(actual.shape), (projection, 1, kernel))
            torch.testing.assert_close(actual, expected)

    def test_released_layout_passes_through_untouched(self):
        weights = [
            (f"{LAYER}.hc_attn_fn", torch.zeros(2, 3)),
            (f"{LAYER}.hc_ffn_scale", torch.zeros(3)),
            (f"{LAYER}.self_attn.f_a_proj.weight", torch.zeros(4, 8)),
            (f"{LAYER}.self_attn.dt_bias", torch.zeros(6)),
            (f"{LAYER}.self_attn.q_conv1d.weight", torch.zeros(4, 1, 3)),
            (f"{LAYER}.mlp.experts.0.gate_proj.weight", torch.zeros(4, 5)),
            (f"{LAYER}.mlp.experts.0.gate_proj.weight_scale_inv", torch.zeros(1)),
            (f"{LAYER}.mlp.experts.0.down_proj.weight", torch.zeros(5, 4)),
            (f"{LAYER}.mlp.shared_experts.gate_proj.weight", torch.zeros(4, 5)),
            (f"{LAYER}.mlp.shared_experts.down_proj.weight", torch.zeros(5, 4)),
            (f"{LAYER}.self_attn.o_proj.weight", torch.zeros(5, 5)),
        ]

        got = list(convert_hf_native_weights(weights))

        self.assertEqual([name for name, _ in got], [name for name, _ in weights])
        for (_, actual), (_, expected) in zip(got, weights):
            self.assertIs(actual, expected)

    def test_malformed_packed_tensors_raise_instead_of_loading_silently(self):
        for name, weight in (
            (f"{LAYER}.mlp.experts.gate_up_proj", torch.zeros(3, 5, 4)),
            (f"{LAYER}.mlp.experts.gate_up_proj", torch.zeros(8, 4)),
            (f"{LAYER}.mlp.experts.down_proj", torch.zeros(5, 4)),
            (f"{LAYER}.self_attn.conv1d.weight", torch.zeros(8, 1, 3)),
        ):
            with self.assertRaises(ValueError):
                list(convert_hf_native_weights([(name, weight)]))


if __name__ == "__main__":
    unittest.main()
