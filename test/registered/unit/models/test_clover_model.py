"""Clover configuration, coreset, and weight-loader unit tests."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.clover_moe import CloverMoeConfig
from sglang.srt.environ import envs
from sglang.srt.models.clover_moe import block_coreset
from sglang.srt.runtime_context import get_context
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
    for bad, message in (
        ({"num_shared_experts": 0}, "shared expert"),
        ({"mamba_ssm_dtype": "bfloat16"}, "float32 recurrent state"),
        ({"hidden_act": "gelu"}, "SiLU"),
        ({"attn_output_gate": False}, "SiLU"),
        ({"mask_token_id": c.vocab_size}, "outside the vocabulary"),
        ({"layer_types": ["full_attention"] * 2}, "layer_types"),
        (
            {"layer_types": ["full_attention", "sliding_attention", "full_attention"]},
            "layer_types",
        ),
    ):
        fields = {"num_hidden_layers": 3, "layer_types": c.layer_types, **bad}
        with pytest.raises(ValueError, match=message):
            CloverMoeConfig(**fields)
    with get_context().override_server_args():
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            with pytest.raises(ValueError, match="float32 recurrent state"):
                c.mamba2_cache_params


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


class TestSharedExpertLoader(CustomTestCase):
    def test_shared_slot_and_missing_shard(self):
        """The shared expert takes the slot after the packed routed experts."""
        from sglang.srt.models.clover_moe import CloverMoeModelLM

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
        weights = [
            (
                prefix + "experts.gate_up_proj",
                torch.cat((gate[:2], up[:2]), dim=1),
            ),
            (prefix + "experts.down_proj", down[:2]),
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
            CloverMoeModelLM.load_weights(model, weights + shared[:1] + shared[2:])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
