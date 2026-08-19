# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.gguf import (
    GGUFLinearMethod,
    _ordered_gguf_shard_ids,
)
from sglang.srt.model_loader.kimi_k3_gguf import (
    _kda_a_log_target_value,
    _residual_target_value,
    kimi_k3_checkpoint_targets,
    routed_expert_tensor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKimiK3GGUFMapping(unittest.TestCase):
    def test_maps_dense_kda_mla_moe_and_residual_tensors(self) -> None:
        cases = {
            "token_embd.weight": ("model.embed_tokens.weight",),
            "blk.0.ffn_gate.weight": ("model.layers.0.mlp.gate_proj.weight",),
            "blk.1.ssm_g.weight": ("model.layers.1.self_attn.g_proj.weight",),
            "blk.3.attn_q_b.weight": ("model.layers.3.self_attn.q_b_proj.weight",),
            "blk.3.attn_k_b.weight": ("model.layers.3.self_attn.k_b_qweight",),
            "blk.3.attn_v_b.weight": ("model.layers.3.self_attn.v_b_qweight",),
            "blk.1.ffn_routed_down.weight": (
                "model.layers.1.mlp.routed_expert_down_proj.weight",
            ),
            "blk.1.ffn_gate_shexp.weight": (
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
            ),
            "blk.2.attn_res_score.weight": (
                "model.layers.2.self_attention_res_proj.weight",
                "model.layers.2.self_attention_res_norm.weight",
            ),
            "output_res_score.weight": (
                "model.output_attn_res_proj.weight",
                "model.output_attn_res_norm.weight",
            ),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(kimi_k3_checkpoint_targets(source), expected)

    def test_only_routed_aggregate_tensors_are_skipped(self) -> None:
        self.assertTrue(routed_expert_tensor("blk.92.ffn_up_exps.weight"))
        self.assertTrue(routed_expert_tensor("blk.1.ffn_down_exps.weight"))
        self.assertFalse(routed_expert_tensor("blk.1.ffn_up_shexp.weight"))
        self.assertFalse(routed_expert_tensor("blk.1.ffn_routed_up.weight"))

    def test_unknown_tensor_fails_closed(self) -> None:
        with self.assertRaisesRegex(KeyError, "unsupported Kimi-K3"):
            kimi_k3_checkpoint_targets("blk.7.unexpected.weight")

    def test_residual_score_preserves_exact_combined_weight(self) -> None:
        source = torch.tensor([0.5, -1.25, 3.0], dtype=torch.float32)
        projection = _residual_target_value(source, 0)
        norm = _residual_target_value(source, 1)
        self.assertEqual(tuple(projection.shape), (1, 3))
        self.assertEqual(tuple(norm.shape), (3,))
        torch.testing.assert_close(norm * projection.squeeze(0), source)

    def test_residual_score_rejects_non_vector_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a vector"):
            _residual_target_value(torch.ones(1, 3), 0)

    def test_restores_kda_a_log_from_gguf_transform(self) -> None:
        original = torch.tensor([-0.75, 0.0, 1.5], dtype=torch.float32)
        stored = -torch.exp(original)
        torch.testing.assert_close(_kda_a_log_target_value(stored), original)
        with self.assertRaisesRegex(ValueError, "only -exp"):
            _kda_a_log_target_value(torch.tensor([-1.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            _kda_a_log_target_value(torch.tensor([-1.0, float("nan")]))

    def test_merged_gguf_output_uses_logical_shard_order(self) -> None:
        qweight = torch.tensor([[30], [0], [20], [10]], dtype=torch.uint8)
        qweight.shard_id = [3, 0, 2, 1]
        qweight.shard_offset_map = {
            3: (0, 1, 1),
            0: (1, 2, 1),
            2: (2, 3, 1),
            1: (3, 4, 1),
        }
        qweight.gguf_prefix = ""
        layer = SimpleNamespace(
            qweight=qweight,
            qweight_type=SimpleNamespace(shard_weight_type={0: 0, 1: 0, 2: 0, 3: 0}),
        )
        method = object.__new__(GGUFLinearMethod)

        def fake_matmul(_x, weight, _weight_type):
            return weight[:, 0].float().unsqueeze(0)

        with patch(
            "sglang.srt.layers.quantization.gguf.fused_mul_mat_gguf",
            side_effect=fake_matmul,
        ):
            output = method.apply(layer, torch.zeros(1, 1))
        torch.testing.assert_close(output, torch.tensor([[0.0, 10.0, 20.0, 30.0]]))

    def test_unknown_gguf_shard_layouts_preserve_checkpoint_order(self) -> None:
        self.assertEqual(_ordered_gguf_shard_ids(["q", "k"]), ["q", "k"])
        self.assertEqual(_ordered_gguf_shard_ids([4, 2]), [4, 2])


if __name__ == "__main__":
    unittest.main()
