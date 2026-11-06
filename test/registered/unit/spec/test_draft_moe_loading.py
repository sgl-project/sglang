"""Customized DFlash drafts build DraftMoE from the config and load SpecForge's MoE checkpoints."""

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from sglang.srt.layers.draft.ffn import DraftMoE
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.models.dflash import (
    CustomizedDFlashDraftModel,
    DFlashDraftModel,
    DFlashMLP,
)
from sglang.srt.models.dspark import CustomizedDSparkDraftModel, DSparkDraftModel
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

E, K, H, I, S, LAYERS = 4, 2, 16, 8, 12, 2

BASE = dict(
    hidden_size=H,
    intermediate_size=32,
    num_hidden_layers=LAYERS,
    num_target_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    vocab_size=64,
    rms_norm_eps=1e-6,
    rope_theta=1e4,
    max_position_embeddings=4096,
    dflash_config={"block_size": 4, "target_layer_ids": [0, 1]},
)


def _moe_fields(**overrides):
    """SpecForge's qwen3_5_moe export: softmax top-k, gated shared expert, trained bias."""
    fields = dict(
        BASE,
        num_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=S,
        norm_topk_prob=False,
        dflash_config=dict(BASE["dflash_config"], moe_router_enable_expert_bias=True),
    )
    fields.update(overrides)
    return fields


def _layer_checkpoint(layer, *, per_expert=False, prefix="", expert_bias=True):
    generator = torch.Generator().manual_seed(layer)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    base = f"{prefix}layers.{layer}.mlp."
    gate_up, down = rand(E, 2 * I, H), rand(E, H, I)
    weights = {
        base + "gate.weight": rand(E, H),
        base + "shared_expert.gate_proj.weight": rand(S, H),
        base + "shared_expert.up_proj.weight": rand(S, H),
        base + "shared_expert.down_proj.weight": rand(H, S),
        base + "shared_expert_gate.weight": rand(1, H),
    }
    if expert_bias:
        weights[base + "gate.expert_bias"] = torch.linspace(-0.3, 0.3, E)
    if per_expert:
        for expert in range(E):
            gate, up = gate_up[expert].chunk(2, dim=0)
            weights[base + f"experts.{expert}.gate_proj.weight"] = gate
            weights[base + f"experts.{expert}.up_proj.weight"] = up
            weights[base + f"experts.{expert}.down_proj.weight"] = down[expert]
    else:
        weights[base + "experts.gate_up_proj"] = gate_up
        weights[base + "experts.down_proj"] = down
    return weights, gate_up, down


def _checkpoint(**kwargs):
    weights, experts = {}, {}
    for layer in range(LAYERS):
        layer_weights, gate_up, down = _layer_checkpoint(layer, **kwargs)
        weights.update(layer_weights)
        experts[layer] = (gate_up, down)
    return weights, experts


def _reference_experts(gate_up, down):
    """The same experts through FusedMoE's own loader, so no runner layout is assumed."""
    experts = FusedMoE(
        num_experts=E, hidden_size=H, intermediate_size=I, layer_id=0, top_k=K
    )
    for expert in range(E):
        gate, up = gate_up[expert].chunk(2, dim=0)
        for shard_id, tensor in (("w1", gate), ("w3", up), ("w2", down[expert])):
            name = "w2_weight" if shard_id == "w2" else "w13_weight"
            experts.weight_loader(
                getattr(experts, name), tensor, name, shard_id, expert
            )
    return experts


class TestCustomizedDraftLoading(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        cpu_rope = patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        cpu_rope.start()
        self.addCleanup(cpu_rope.stop)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")

    def test_only_customized_drafts_build_moe_layers(self):
        config = Qwen3Config(**_moe_fields())
        for layer in CustomizedDFlashDraftModel(config).layers:
            self.assertIsInstance(layer.mlp, DraftMoE)
            self.assertIsNotNone(layer.mlp.shared_expert)
            self.assertIsNotNone(layer.mlp.shared_expert_gate)
            self.assertIsNotNone(layer.mlp.gate.expert_bias)
        for layer in DFlashDraftModel(config).layers:  # native: dense as before
            self.assertIsInstance(layer.mlp, DFlashMLP)
        for layer in CustomizedDFlashDraftModel(Qwen3Config(**BASE)).layers:
            self.assertIsInstance(layer.mlp, DFlashMLP)  # customized, dense config

    def test_customized_dspark_builds_moe_layers(self):
        dflash = dict(_moe_fields()["dflash_config"], mask_token_id=3)
        config = Qwen3Config(
            **_moe_fields(
                dflash_config=dflash, markov_rank=4, markov_head_type="vanilla"
            )
        )
        self.assertIsInstance(
            CustomizedDSparkDraftModel(config).layers[0].mlp, DraftMoE
        )
        self.assertIsInstance(DSparkDraftModel(config).layers[0].mlp, DFlashMLP)

    def test_stacked_per_expert_and_prefixed_exports_load_the_same(self):
        for per_expert, prefix in ((False, ""), (True, ""), (False, "model.")):
            with self.subTest(per_expert=per_expert, prefix=prefix):
                model = CustomizedDFlashDraftModel(Qwen3Config(**_moe_fields()))
                weights, experts = _checkpoint(per_expert=per_expert, prefix=prefix)
                model.load_weights(list(weights.items()))
                for layer_id, layer in enumerate(model.layers):
                    reference = _reference_experts(*experts[layer_id])
                    moe = layer.mlp
                    torch.testing.assert_close(
                        moe.experts.w13_weight, reference.w13_weight
                    )
                    torch.testing.assert_close(
                        moe.experts.w2_weight, reference.w2_weight
                    )
                    base = f"{prefix}layers.{layer_id}.mlp."
                    torch.testing.assert_close(
                        moe.gate.weight, weights[base + "gate.weight"]
                    )
                    torch.testing.assert_close(
                        moe.gate.expert_bias, weights[base + "gate.expert_bias"]
                    )
                    torch.testing.assert_close(
                        moe.shared_expert_gate.weight,
                        weights[base + "shared_expert_gate.weight"],
                    )
                    gate_up = moe.shared_expert.gate_up_proj.weight
                    torch.testing.assert_close(
                        gate_up[:S], weights[base + "shared_expert.gate_proj.weight"]
                    )
                    torch.testing.assert_close(
                        gate_up[S:], weights[base + "shared_expert.up_proj.weight"]
                    )

    def test_partial_or_mismatched_checkpoints_are_refused(self):
        weights, _ = _checkpoint()
        cases = {
            "missing experts": {
                k: v
                for k, v in weights.items()
                if "down_proj" not in k or "shared" in k
            },
            "missing bias": {
                k: v for k, v in weights.items() if "expert_bias" not in k
            },
            "missing shared expert": {
                k: v for k, v in weights.items() if "shared_expert." not in k
            },
        }
        for name, partial in cases.items():
            with self.subTest(name=name):
                model = CustomizedDFlashDraftModel(Qwen3Config(**_moe_fields()))
                with self.assertRaisesRegex(ValueError, "missing"):
                    model.load_weights(list(partial.items()))
        with self.subTest(name="bias without the flag"):
            config = Qwen3Config(**_moe_fields(dflash_config=BASE["dflash_config"]))
            with self.assertRaisesRegex(ValueError, "does not enable"):
                CustomizedDFlashDraftModel(config).load_weights(list(weights.items()))
        with self.subTest(name="wrong shape"):
            bad = dict(weights)
            bad["layers.0.mlp.gate.weight"] = torch.zeros(E, H + 1)
            with self.assertRaisesRegex(ValueError, "expected"):
                CustomizedDFlashDraftModel(Qwen3Config(**_moe_fields())).load_weights(
                    list(bad.items())
                )
        with self.subTest(name="MoE tensor on a native dense draft"):
            with self.assertRaisesRegex(ValueError, "layer is dense"):
                DFlashDraftModel(Qwen3Config(**_moe_fields())).load_weights(
                    list(weights.items())
                )


if __name__ == "__main__":
    unittest.main()
