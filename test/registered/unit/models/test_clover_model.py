"""Clover configuration, coreset, and weight-loader unit tests."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.clover_moe import CloverMoeConfig
from sglang.srt.models.clover_moe import block_coreset
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_config_preserves_irregular_layers_and_rope():
    c = CloverMoeConfig(
        num_hidden_layers=3,
        layer_types=["full_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 12345},
    )
    assert c.full_attention_layer_ids == [0, 2]
    assert c.linear_layer_ids == [1]
    from sglang.srt.utils.hf_transformers.common import get_rope_config

    assert get_rope_config(c) == (12345, c.rope_parameters)
    assert c.model_type == "clover_moe"


def test_coreset_is_per_request_and_stable_on_ties():
    logits = torch.tensor([[9.0, 8.0, 0.0, 0.0]] * 4 + [[0.0, 0.0, 8.0, 9.0]] * 4)
    together = block_coreset(logits, 4, 2, 0.5)
    separate = torch.cat([block_coreset(row, 4, 2, 0.5) for row in logits.split(4)])
    torch.testing.assert_close(together, separate)
    assert together[:4].argmax(-1).tolist() == [0] * 4
    assert together[4:].argmax(-1).tolist() == [3] * 4
    tied = block_coreset(torch.zeros(4, 4), 4, 2, 0.5)
    assert torch.isfinite(tied[:, 0]).all()
    assert torch.isneginf(tied[:, 1:]).all()


def test_tp2_right_convolution_loader_shards_q_k_v_independently():
    from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader

    full = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 1, 3)
    for rank in (0, 1):
        local = torch.empty(8, 1, 3)
        mamba_v2_sharded_weight_loader(
            [(4, 0, False), (4, 0, False), (8, 0, False)], 2, rank
        )(local, full)
        expected = torch.cat([part.chunk(2)[rank] for part in full.split([4, 4, 8])])
        torch.testing.assert_close(local, expected)


class TestSharedExpertLoader(CustomTestCase):
    def test_shared_slot_and_missing_shard(self):
        """Shared weights must occupy the extra slot for both HF expert layouts.

        A packed routed tensor still contains only routed experts; marking it
        loaded must not hide a missing shared-expert projection.
        """
        from sglang.srt.models.clover_moe import CloverMoeModelLM

        for packed in (False, True):
            with self.subTest(packed=packed):
                w13 = torch.nn.Parameter(torch.zeros(3, 2, 2, 2), requires_grad=False)
                w2 = torch.nn.Parameter(torch.zeros(3, 2, 2), requires_grad=False)

                def loader(param, weight, name, *, shard_id, expert_id):
                    target = param[expert_id]
                    if shard_id != "w2":
                        target = target[0 if shard_id == "w1" else 1]
                    target.copy_(weight)

                w13.weight_loader = loader
                w2.weight_loader = loader
                params = {
                    "model.layers.0.mlp.experts.w13_weight": w13,
                    "model.layers.0.mlp.experts.w2_weight": w2,
                }
                model = SimpleNamespace(
                    config=SimpleNamespace(num_experts=2),
                    num_fused_shared_experts=1,
                    enable_shared_expert_fusion=True,
                    named_parameters=lambda: params.items(),
                    modules=lambda: (),
                )
                prefix = "model.layers.0.mlp."
                gate = torch.arange(12.0).reshape(3, 2, 2)
                up, down = gate + 20, gate + 40
                if packed:
                    weights = [
                        (
                            prefix + "experts.gate_up_proj",
                            torch.cat((gate[:2], up[:2]), dim=1),
                        ),
                        (prefix + "experts.down_proj", down[:2]),
                    ]
                else:
                    weights = [
                        (prefix + f"experts.{i}.{proj}.weight", values[i])
                        for i in range(2)
                        for proj, values in (
                            ("gate_proj", gate),
                            ("up_proj", up),
                            ("down_proj", down),
                        )
                    ]
                shared = [
                    (prefix + f"shared_expert.{proj}.weight", values[2])
                    for proj, values in (
                        ("gate_proj", gate),
                        ("up_proj", up),
                        ("down_proj", down),
                    )
                ]
                CloverMoeModelLM.load_weights(model, weights + shared)
                torch.testing.assert_close(w13[:, 0], gate)
                torch.testing.assert_close(w13[:, 1], up)
                torch.testing.assert_close(w2, down)
                with self.assertRaisesRegex(ValueError, "missing shards"):
                    CloverMoeModelLM.load_weights(
                        model, weights + shared[:1] + shared[2:]
                    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
