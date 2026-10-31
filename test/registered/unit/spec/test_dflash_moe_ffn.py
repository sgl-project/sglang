"""MoE FFN drafts: the config-driven switch and routing spec, strict loading of
the Qwen-style (SpecForge) and DeepSeek-style checkpoint names, and the
checkpoint-driven router bias, on CPU."""

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.models.dflash import (
    DFlashDraftModel,
    DFlashMLP,
    DFlashMoE,
    DFlashMoEDraftModel,
    parse_dflash_moe_spec,
)
from sglang.srt.models.dspark import DSparkDraftModel, Qwen3MoeDSparkModel
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

E, K, H, I, S, LAYERS = 4, 2, 16, 8, 12, 2

_BASE = dict(
    architectures=["DFlashDraftModel"],
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


def _qwen_fields(**overrides):
    """A SpecForge MoE export (Qwen3.5-MoE conventions): softmax top-k without
    renormalization and a sigmoid-gated shared expert."""
    fields = dict(
        _BASE,
        moe_preset="qwen3_5_moe",
        num_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=S,
        norm_topk_prob=False,
    )
    fields.update(overrides)
    return fields


def _deepseek_fields(**overrides):
    """DeepSeek-V3 conventions: sigmoid scores, group-limited top-k with the
    router bias, renormalized and scaled weights, summed shared experts."""
    fields = dict(
        _BASE,
        n_routed_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=I,
        n_shared_experts=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        topk_method="noaux_tc",
        routed_scaling_factor=2.5,
    )
    fields.update(overrides)
    return fields


def _ffn_checkpoint(
    layer, *, per_expert=False, prefix="", expert_bias=True, style="qwen"
):
    """One layer's MoE FFN under the export names of one family: the stacked
    expert tensors or the per-expert names, the family's shared expert and
    router bias names."""
    generator = torch.Generator().manual_seed(layer)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    base = f"{prefix}layers.{layer}.mlp."
    gate_up, down = rand(E, 2 * I, H), rand(E, H, I)
    weights = {base + "gate.weight": rand(E, H)}
    if style == "qwen":
        weights.update(
            {
                base + "shared_expert.gate_proj.weight": rand(S, H),
                base + "shared_expert.up_proj.weight": rand(S, H),
                base + "shared_expert.down_proj.weight": rand(H, S),
                base + "shared_expert_gate.weight": rand(1, H),
            }
        )
        bias_key = "gate.expert_bias"
    else:
        weights.update(
            {
                base + "shared_experts.gate_proj.weight": rand(I, H),
                base + "shared_experts.up_proj.weight": rand(I, H),
                base + "shared_experts.down_proj.weight": rand(H, I),
            }
        )
        bias_key = "gate.e_score_correction_bias"
    if expert_bias:
        weights[base + bias_key] = torch.linspace(-0.3, 0.3, E)
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


def _reference_experts(gate_up, down):
    """The same experts through FusedMoE's own per-shard loader, so the check
    does not assume the runner's w13 row layout."""
    experts = FusedMoE(
        num_experts=E, hidden_size=H, intermediate_size=I, layer_id=0, top_k=K
    )
    for expert in range(E):
        gate, up = gate_up[expert].chunk(2, dim=0)
        for shard_id, tensor in (("w1", gate), ("w3", up), ("w2", down[expert])):
            name = "w2_weight" if shard_id == "w2" else "w13_weight"
            param = getattr(experts, name)
            experts.weight_loader(
                param, tensor, name, shard_id=shard_id, expert_id=expert
            )
    return experts


class TestDFlashMoEFFN(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")

    @staticmethod
    def _model(fields, cls=DFlashDraftModel):
        return cls(Qwen3Config(**fields))

    @staticmethod
    def _load(model, weights):
        model.load_weights(list(weights.items()))

    @staticmethod
    def _checkpoint(**kwargs):
        weights = {}
        for layer in range(LAYERS):
            weights.update(_ffn_checkpoint(layer, **kwargs)[0])
        return weights

    def test_the_config_describes_the_routing(self):
        qwen = parse_dflash_moe_spec(Qwen3Config(**_qwen_fields()))
        self.assertEqual(
            (qwen.num_experts, qwen.top_k, qwen.intermediate_size), (E, K, I)
        )
        self.assertEqual(qwen.scoring_func, "softmax")
        self.assertFalse(qwen.renormalize)
        self.assertFalse(qwen.grouped)
        self.assertIsNone(qwen.routed_scaling_factor)
        self.assertEqual((qwen.shared_intermediate_size, qwen.shared_gated), (S, True))

        deepseek = parse_dflash_moe_spec(Qwen3Config(**_deepseek_fields()))
        self.assertEqual(deepseek.num_experts, E)
        self.assertEqual(deepseek.scoring_func, "sigmoid")
        self.assertTrue(deepseek.renormalize)
        self.assertEqual((deepseek.num_expert_group, deepseek.topk_group), (2, 1))
        self.assertEqual(deepseek.routed_scaling_factor, 2.5)
        self.assertEqual(
            (deepseek.shared_intermediate_size, deepseek.shared_gated), (I, False)
        )
        # transformers' default when the config does not say.
        self.assertTrue(
            parse_dflash_moe_spec(
                Qwen3Config(**_qwen_fields(norm_topk_prob=None))
            ).renormalize
        )
        self.assertIsNone(parse_dflash_moe_spec(Qwen3Config(**_BASE)))
        for bad in (
            dict(num_experts_per_tok=E + 1),
            dict(scoring_func="tanh"),
            dict(n_group=3),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_dflash_moe_spec(Qwen3Config(**_qwen_fields(**bad)))

    def test_routed_experts_in_the_config_switch_the_ffn(self):
        model = self._model(_qwen_fields())
        for layer in model.layers:
            self.assertIsInstance(layer.mlp, DFlashMoE)
            self.assertIsNotNone(layer.mlp.shared_expert)
            self.assertIsNotNone(layer.mlp.shared_expert_gate)
            # Until a checkpoint says otherwise: no bias, SGLang's own router.
            self.assertFalse(layer.mlp.has_expert_bias)
            self.assertIsNone(layer.mlp.topk.topk_config.custom_routing_function)
            self.assertIsNone(layer.mlp.topk.topk_config.correction_bias)
        for layer in self._model(_BASE).layers:
            self.assertIsInstance(layer.mlp, DFlashMLP)
        deepseek = self._model(_deepseek_fields()).layers[0].mlp
        self.assertIsNotNone(deepseek.shared_expert)
        self.assertIsNone(deepseek.shared_expert_gate)
        self.assertTrue(deepseek.topk.topk_config.use_grouped_topk)
        self.assertEqual(deepseek.topk.topk_config.scoring_func, "sigmoid")
        plain = (
            self._model(_qwen_fields(shared_expert_intermediate_size=0)).layers[0].mlp
        )
        self.assertIsNone(plain.shared_expert)
        with self.assertRaisesRegex(ValueError, "silu"):
            self._model(_qwen_fields(hidden_act="gelu"))

    def test_stacked_and_per_expert_exports_load_the_same_ffn(self):
        for per_expert, prefix in ((False, ""), (True, ""), (False, "model.")):
            with self.subTest(per_expert=per_expert, prefix=prefix):
                model = self._model(_qwen_fields())
                weights, references = {}, {}
                for layer in range(LAYERS):
                    layer_weights, gate_up, down = _ffn_checkpoint(
                        layer, per_expert=per_expert, prefix=prefix
                    )
                    weights.update(layer_weights)
                    references[layer] = (gate_up, down)
                self._load(model, weights)
                for layer in range(LAYERS):
                    ffn = model.layers[layer].mlp
                    expected = _reference_experts(*references[layer])
                    base = f"{prefix}layers.{layer}.mlp."
                    checks = [
                        (ffn.experts.w13_weight, expected.w13_weight),
                        (ffn.experts.w2_weight, expected.w2_weight),
                        (ffn.gate.weight, weights[base + "gate.weight"]),
                        (ffn.gate_expert_bias, weights[base + "gate.expert_bias"]),
                        (
                            ffn.shared_expert.gate_up_proj.weight,
                            torch.cat(
                                (
                                    weights[base + "shared_expert.gate_proj.weight"],
                                    weights[base + "shared_expert.up_proj.weight"],
                                )
                            ),
                        ),
                        (
                            ffn.shared_expert.down_proj.weight,
                            weights[base + "shared_expert.down_proj.weight"],
                        ),
                        (
                            ffn.shared_expert_gate.weight,
                            weights[base + "shared_expert_gate.weight"],
                        ),
                    ]
                    for got, want in checks:
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
                    self.assertEqual(ffn.gate_expert_bias.dtype, torch.float32)
                    # A bias with softmax scores installs the torch router.
                    self.assertTrue(ffn.has_expert_bias)
                    self.assertIsNotNone(ffn.topk.topk_config.custom_routing_function)

    def test_an_export_without_a_bias_keeps_the_native_router(self):
        model = self._model(_qwen_fields())
        self._load(model, self._checkpoint(expert_bias=False))
        for layer in model.layers:
            self.assertFalse(layer.mlp.has_expert_bias)
            self.assertTrue(torch.all(layer.mlp.gate_expert_bias == 0))
            self.assertIsNone(layer.mlp.topk.topk_config.custom_routing_function)
            self.assertEqual(len(layer.mlp._loaded_experts), 3 * E)

    def test_deepseek_style_export_loads_with_its_names(self):
        model = self._model(_deepseek_fields())
        weights, references = {}, {}
        for layer in range(LAYERS):
            layer_weights, gate_up, down = _ffn_checkpoint(
                layer, per_expert=True, style="deepseek"
            )
            weights.update(layer_weights)
            references[layer] = (gate_up, down)
        self._load(model, weights)
        for layer in range(LAYERS):
            ffn = model.layers[layer].mlp
            expected = _reference_experts(*references[layer])
            base = f"layers.{layer}.mlp."
            torch.testing.assert_close(
                ffn.experts.w13_weight, expected.w13_weight, rtol=0, atol=0
            )
            torch.testing.assert_close(
                ffn.experts.w2_weight, expected.w2_weight, rtol=0, atol=0
            )
            torch.testing.assert_close(
                ffn.gate_expert_bias,
                weights[base + "gate.e_score_correction_bias"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                ffn.shared_expert.gate_up_proj.weight,
                torch.cat(
                    (
                        weights[base + "shared_experts.gate_proj.weight"],
                        weights[base + "shared_experts.up_proj.weight"],
                    )
                ),
                rtol=0,
                atol=0,
            )
            # Sigmoid scores: the bias rides SGLang's own biased top-k.
            self.assertTrue(ffn.has_expert_bias)
            self.assertIsNone(ffn.topk.topk_config.custom_routing_function)
            self.assertIs(ffn.topk.topk_config.correction_bias, ffn.gate_expert_bias)

    def test_older_specforge_names_still_load(self):
        """Exports normalized for SpecForge's interim serving package carry
        gate.selection_bias and architectures=["Qwen3MoeDSparkModel"]."""
        self.assertTrue(issubclass(Qwen3MoeDSparkModel, DSparkDraftModel))
        self.assertTrue(issubclass(DFlashMoEDraftModel, DFlashDraftModel))
        model = self._model(
            _qwen_fields(markov_rank=4, markov_head_type="vanilla"), Qwen3MoeDSparkModel
        )
        weights = {
            key.replace("gate.expert_bias", "gate.selection_bias"): value
            for key, value in self._checkpoint().items()
        }
        self._load(model, weights)
        ffn = model.layers[0].mlp
        self.assertIsInstance(ffn, DFlashMoE)
        self.assertTrue(ffn.has_expert_bias)
        torch.testing.assert_close(
            ffn.gate_expert_bias,
            weights["layers.0.mlp.gate.selection_bias"],
            rtol=0,
            atol=0,
        )

    def test_dspark_draft_serves_the_moe_ffn(self):
        model = self._model(
            _qwen_fields(markov_rank=4, markov_head_type="vanilla"), DSparkDraftModel
        )
        self.assertTrue(all(isinstance(layer.mlp, DFlashMoE) for layer in model.layers))
        weights = self._checkpoint()
        self._load(model, weights)
        ffn = model.layers[1].mlp
        torch.testing.assert_close(
            ffn.gate_expert_bias,
            weights["layers.1.mlp.gate.expert_bias"],
            rtol=0,
            atol=0,
        )
        self.assertTrue(ffn.has_expert_bias)
        self.assertEqual(len(ffn._loaded_experts), 3 * E)

    def test_refuses_incomplete_or_mismatched_ffn_checkpoints(self):
        weights = self._checkpoint(per_expert=True)
        with self.subTest(case="a missing expert"):
            partial = dict(weights)
            del partial["layers.1.mlp.experts.3.down_proj.weight"]
            with self.assertRaisesRegex(ValueError, r"layers\.1\.mlp.*experts\.3\.w2"):
                self._load(self._model(_qwen_fields()), partial)
        with self.subTest(case="a missing router"):
            partial = dict(weights)
            del partial["layers.0.mlp.gate.weight"]
            with self.assertRaisesRegex(ValueError, r"layers\.0\.mlp.*gate\.weight"):
                self._load(self._model(_qwen_fields()), partial)
        with self.subTest(case="a shared expert the config does not size"):
            model = self._model(_qwen_fields(shared_expert_intermediate_size=0))
            with self.assertRaisesRegex(ValueError, "shared expert"):
                self._load(model, weights)
        with self.subTest(case="a shared expert gate the config does not describe"):
            with self.assertRaisesRegex(ValueError, "gated shared expert"):
                self._load(
                    self._model(_deepseek_fields()),
                    {"layers.0.mlp.shared_expert_gate.weight": torch.zeros(1, H)},
                )
        with self.subTest(case="MoE tensors for a dense draft"):
            with self.assertRaisesRegex(ValueError, "routed experts"):
                self._load(self._model(_BASE), weights)
        with self.subTest(case="a stacked tensor of the wrong shape"):
            stacked = _ffn_checkpoint(0)[0]
            key = "layers.0.mlp.experts.gate_up_proj"
            stacked[key] = stacked[key][:, :, :-1]
            with self.assertRaisesRegex(ValueError, "experts.gate_up_proj"):
                self._load(self._model(_qwen_fields()), stacked)
        with self.subTest(case="a bias of the wrong width"):
            with self.assertRaisesRegex(ValueError, "gate.expert_bias"):
                self._load(
                    self._model(_qwen_fields()),
                    {"layers.0.mlp.gate.expert_bias": torch.zeros(E + 1)},
                )
        with self.subTest(case="an expert the config does not have"):
            with self.assertRaisesRegex(ValueError, f"expert {E}"):
                self._load(
                    self._model(_qwen_fields()),
                    {f"layers.0.mlp.experts.{E}.down_proj.weight": torch.zeros(H, I)},
                )
        with self.subTest(case="an unknown FFN tensor"):
            with self.assertRaisesRegex(ValueError, "experts.w13_weight"):
                self._load(
                    self._model(_qwen_fields()),
                    {"layers.0.mlp.experts.w13_weight": torch.zeros(1)},
                )

    def test_biased_softmax_routing_follows_specforge(self):
        model = self._model(_qwen_fields())
        self._load(model, self._checkpoint())
        ffn = model.layers[0].mlp
        torch.manual_seed(0)
        x = torch.randn(10, H)
        logits = x @ ffn.gate.weight.T
        scores = torch.softmax(logits, dim=-1)
        expected_ids = torch.topk(scores + ffn.gate_expert_bias, k=K, dim=-1).indices
        for renormalize in (False, True):
            with self.subTest(renormalize=renormalize):
                weights, ids = ffn._route_with_expert_bias(
                    hidden_states=x,
                    gating_output=logits,
                    topk=K,
                    renormalize=renormalize,
                )
                self.assertEqual(ids.dtype, torch.int32)
                self.assertTrue(torch.equal(ids.long(), expected_ids))
                # The weights are the unbiased scores of the picked experts.
                expected = scores.gather(-1, expected_ids)
                if renormalize:
                    expected = expected / expected.sum(-1, keepdim=True)
                torch.testing.assert_close(weights, expected)
        self.assertIs(
            ffn.topk.topk_config.custom_routing_function.__func__,
            DFlashMoE._route_with_expert_bias,
        )
        # Without a bias the picks are the plain softmax top-k.
        ffn.gate_expert_bias.zero_()
        _, ids = ffn._route_with_expert_bias(
            hidden_states=x, gating_output=logits, topk=K, renormalize=False
        )
        self.assertTrue(
            torch.equal(ids.long(), torch.topk(scores, k=K, dim=-1).indices)
        )


if __name__ == "__main__":
    unittest.main()
