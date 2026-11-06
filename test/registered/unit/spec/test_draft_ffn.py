"""The draft FFN: its spec from the config, and the MoE layers a draft builds and loads from it."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from sglang.srt.layers.draft.ffn import DraftMoE
from sglang.srt.layers.draft.ffn_spec import (
    FFN_KIND_DENSE,
    FFN_KIND_MOE,
    DraftDenseFFNSpec,
    DraftMixFFNSpec,
    DraftMoeFFNSpec,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.models.dflash import DFlashDraftModel, DFlashMLP
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

DFLASH = {"target_layer_ids": [1, 3, 5], "mask_token_id": 7}


def _dense(**overrides):
    cfg = {
        "num_hidden_layers": 3,
        "hidden_size": 16,
        "intermediate_size": 48,
        "hidden_act": "silu",
        "dflash_config": dict(DFLASH),
    }
    cfg.update(overrides)
    return cfg


def _qwen_moe(**overrides):
    """SpecForge's Qwen3-8B MoE DSpark config, reduced to the FFN fields."""
    cfg = {
        "num_hidden_layers": 3,
        "hidden_size": 16,
        "intermediate_size": 48,
        "hidden_act": "silu",
        "moe_preset": "qwen3_5_moe",
        "num_experts": 8,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 12,
        "norm_topk_prob": False,
        "shared_expert_intermediate_size": 0,
        "dflash_config": dict(DFLASH, moe_router_enable_expert_bias=True),
    }
    cfg.update(overrides)
    return cfg


def _deepseek_v3(**overrides):
    cfg = {
        "num_hidden_layers": 6,
        "hidden_act": "silu",
        "n_routed_experts": 64,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 14,
        "intermediate_size": 48,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 8,
        "topk_group": 4,
        "topk_method": "noaux_tc",
        "routed_scaling_factor": 2.5,
        "n_shared_experts": 2,
        "first_k_dense_replace": 2,
        "moe_layer_freq": 1,
    }
    cfg.update(overrides)
    return cfg


def _without(cfg, *keys):
    return {k: v for k, v in cfg.items() if k not in keys}


class TestFFNSpec(CustomTestCase):
    def test_dense_draft(self):
        spec = DraftMixFFNSpec.from_config(_dense())
        self.assertEqual(spec.layer_kinds, (FFN_KIND_DENSE,) * 3)
        self.assertFalse(spec.has_moe)
        self.assertIsNone(spec.moe)
        self.assertEqual(
            spec.dense, DraftDenseFFNSpec(intermediate_size=48, hidden_act="silu")
        )
        # Config objects and a nested text_config read the same as the dict.
        self.assertEqual(DraftMixFFNSpec.from_config(SimpleNamespace(**_dense())), spec)
        self.assertEqual(DraftMixFFNSpec.from_config({"text_config": _dense()}), spec)
        uniform = DraftMixFFNSpec.from_config(_without(_dense(), "num_hidden_layers"))
        self.assertIsNone(uniform.layer_kinds)
        self.assertEqual(uniform.kind(41), FFN_KIND_DENSE)
        with self.assertRaisesRegex(ValueError, "outside"):
            spec.kind(3)

    def test_qwen_moe_export(self):
        # An all-MoE draft needs no dense intermediate_size.
        spec = DraftMixFFNSpec.from_config(_without(_qwen_moe(), "intermediate_size"))
        self.assertEqual(spec.layer_kinds, (FFN_KIND_MOE,) * 3)
        self.assertIsNone(spec.dense)
        self.assertEqual(
            spec.moe,
            DraftMoeFFNSpec(
                num_experts=8,
                top_k=1,
                intermediate_size=12,
                scoring_func="softmax",
                norm_topk_prob=False,
                expert_bias=True,
                num_expert_group=None,
                topk_group=None,
                routed_scaling_factor=1.0,
                hidden_act="silu",
                swiglu_limit=None,
                swiglu_alpha=None,
                shared_expert_intermediate_size=None,
                shared_expert_gated=False,
            ),
        )
        gated = DraftMixFFNSpec.from_config(
            _qwen_moe(shared_expert_intermediate_size=24)
        ).moe
        self.assertEqual(gated.shared_expert_intermediate_size, 24)
        self.assertTrue(gated.shared_expert_gated)
        nested_wins = _qwen_moe(
            moe_router_enable_expert_bias=False
        )  # dflash_config says True
        self.assertTrue(DraftMixFFNSpec.from_config(nested_wins).moe.expert_bias)

    def test_defaults_and_aliases(self):
        cfg = _without(
            _qwen_moe(),
            "moe_preset",
            "norm_topk_prob",
            "shared_expert_intermediate_size",
        )
        cfg["dflash_config"] = dict(DFLASH)
        moe = DraftMixFFNSpec.from_config(cfg).moe
        self.assertTrue(moe.norm_topk_prob)
        self.assertIsNone(moe.shared_expert_intermediate_size)
        self.assertFalse(moe.expert_bias)
        self.assertEqual(moe.routed_scaling_factor, 1.0)
        for alias in ("n_routed_experts", "num_local_experts"):
            cfg = _without(_qwen_moe(), "num_experts")
            cfg[alias] = 8
            self.assertEqual(DraftMixFFNSpec.from_config(cfg).moe.num_experts, 8)
        with self.assertRaisesRegex(ValueError, "declared twice"):
            DraftMixFFNSpec.from_config(_qwen_moe(n_routed_experts=4))
        mixtral = {  # experts sized by intermediate_size
            "num_hidden_layers": 2,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "intermediate_size": 64,
        }
        self.assertEqual(DraftMixFFNSpec.from_config(mixtral).moe.intermediate_size, 64)
        with self.assertRaisesRegex(ValueError, "moe_intermediate_size is required"):
            DraftMixFFNSpec.from_config(_without(mixtral, "intermediate_size"))

    def test_other_families(self):
        v3 = DraftMixFFNSpec.from_config(_deepseek_v3()).moe
        self.assertEqual(v3.scoring_func, "sigmoid")
        self.assertEqual((v3.num_expert_group, v3.topk_group), (8, 4))
        self.assertTrue(v3.expert_bias)  # noaux_tc routes with the stored bias
        self.assertEqual(v3.routed_scaling_factor, 2.5)
        self.assertEqual(v3.shared_expert_intermediate_size, 28)  # 2 summed experts
        self.assertFalse(v3.shared_expert_gated)
        cfg = _deepseek_v3(
            scoring_func="sqrtsoftplus", swiglu_limit=10.0, n_group=1, topk_group=1
        )
        v4 = DraftMixFFNSpec.from_config(_without(cfg, "topk_method")).moe
        self.assertEqual(
            (v4.scoring_func, v4.swiglu_limit, v4.num_expert_group, v4.expert_bias),
            ("sqrtsoftplus", 10.0, None, False),
        )
        gpt_oss = DraftMixFFNSpec.from_config(
            {
                "num_hidden_layers": 1,
                "num_local_experts": 32,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 64,
                "hidden_act": "swigluoai",
                "swiglu_limit": 7.0,
                "swiglu_alpha": 1.702,
            }
        ).moe
        self.assertEqual(
            (gpt_oss.hidden_act, gpt_oss.swiglu_limit, gpt_oss.swiglu_alpha),
            ("swigluoai", 7.0, 1.702),
        )

    def test_layer_schedules(self):
        cases = [
            (
                _qwen_moe(ffn_layer_types=["dense", "moe", "moe"]),
                ("dense", "moe", "moe"),
            ),
            (
                _qwen_moe(mlp_layer_types=["dense", "sparse", "sparse"]),
                ("dense", "moe", "moe"),
            ),
            (_deepseek_v3(), ("dense", "dense", "moe", "moe", "moe", "moe")),
            (
                _deepseek_v3(moe_layer_freq=2),
                ("dense", "dense", "moe", "dense", "moe", "dense"),
            ),
            (
                _qwen_moe(
                    num_hidden_layers=4, mlp_only_layers=[0], decoder_sparse_step=1
                ),
                ("dense", "moe", "moe", "moe"),
            ),
            (
                _qwen_moe(
                    num_hidden_layers=4, mlp_only_layers=[], decoder_sparse_step=2
                ),
                ("dense", "moe", "dense", "moe"),
            ),
        ]
        for cfg, kinds in cases:
            with self.subTest(kinds=kinds):
                spec = DraftMixFFNSpec.from_config(cfg)
                self.assertEqual(spec.layer_kinds, kinds)
                self.assertEqual(spec.dense.intermediate_size, 48)
                self.assertIsNotNone(spec.moe)
        minimax = {  # MiniMax's DSpark draft: target MoE fields, dense draft layers
            "num_hidden_layers": 5,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "intermediate_size": 3072,
            "moe_layer_freq": [0, 0, 0, 0, 0],
            "scoring_func": "sigmoid",
            "hidden_act": "swigluoai",
        }
        spec = DraftMixFFNSpec.from_config(minimax)
        self.assertEqual(spec.layer_kinds, ("dense",) * 5)
        self.assertIsNone(spec.moe)
        self.assertEqual(spec.dense.intermediate_size, 3072)

    def test_refusals(self):
        cases = [
            (
                _qwen_moe(ffn_layer_types=["moe", "moe"]),
                "has 2 entries but num_hidden_layers=3",
            ),
            (_qwen_moe(ffn_layer_types=["moe", "mlp", "moe"]), "not one of"),
            (_qwen_moe(ffn_layer_types=[]), "non-empty list"),
            (
                _qwen_moe(mlp_layer_types=["dense", "moe", "sparse"]),
                "mlp_layer_types\\[1\\]='moe'",
            ),
            (
                _dense(ffn_layer_types=["moe", "dense", "dense"]),
                "names MoE layers but none of",
            ),
            (_without(_deepseek_v3(), "num_hidden_layers"), "needs num_hidden_layers"),
            (_qwen_moe(scoring_func="relu"), "scoring_func='relu' is not implemented"),
            (_qwen_moe(hidden_act="gelu"), "hidden_act='gelu' is not implemented"),
            (_qwen_moe(topk_method="foo"), "topk_method='foo' is not implemented"),
            (
                _without(_qwen_moe(), "num_experts_per_tok"),
                "num_experts_per_tok is required",
            ),
            (_qwen_moe(num_experts_per_tok=9), "exceeds num_experts"),
            (_qwen_moe(num_experts=True), "positive integer"),
            (_qwen_moe(norm_topk_prob=1), "norm_topk_prob must be a bool"),
            (_qwen_moe(shared_expert_intermediate_size=-1), "non-negative"),
            (_qwen_moe(routed_scaling_factor=0), "positive number"),
            (_deepseek_v3(topk_group=None), "topk_group is required when n_group > 1"),
            (_deepseek_v3(n_group=7), "does not divide"),
            (
                _deepseek_v3(shared_expert_intermediate_size=512),
                "two shared-expert layouts",
            ),
        ]
        for cfg, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    DraftMixFFNSpec.from_config(cfg)

    def test_minimal_dense_config_parses_as_dense(self):
        # A dense draft keeps its minimal config: nothing to size, nothing to refuse.
        minimal = SimpleNamespace(**_without(_dense(), "intermediate_size"))
        spec = DraftMixFFNSpec.from_config(minimal)
        self.assertEqual(
            (spec.has_moe, spec.dense, spec.kind(2)), (False, None, "dense")
        )
        parse_dflash_draft_config(draft_hf_config=minimal)
        minimal.num_experts = 8  # now an MoE draft, and its fields are checked
        with self.assertRaisesRegex(ValueError, "num_experts_per_tok is required"):
            DraftMixFFNSpec.from_config(minimal)


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


class TestDraftMoeLoading(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        cpu_rope = patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        cpu_rope.start()
        self.addCleanup(cpu_rope.stop)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")

    def test_expert_fields_build_moe_layers(self):
        for layer in DFlashDraftModel(Qwen3Config(**_moe_fields())).layers:
            self.assertIsInstance(layer.mlp, DraftMoE)
            self.assertIsNotNone(layer.mlp.shared_expert)
            self.assertIsNotNone(layer.mlp.shared_expert_gate)
            self.assertIsNotNone(layer.mlp.gate.expert_bias)
        for layer in DFlashDraftModel(Qwen3Config(**BASE)).layers:  # no experts
            self.assertIsInstance(layer.mlp, DFlashMLP)

    def test_dspark_builds_moe_layers(self):
        dspark = dict(markov_rank=4, markov_head_type="vanilla")
        dflash = dict(BASE["dflash_config"], mask_token_id=3)
        moe = Qwen3Config(**_moe_fields(dflash_config=dflash, **dspark))
        self.assertIsInstance(DSparkDraftModel(moe).layers[0].mlp, DraftMoE)
        dense = Qwen3Config(**dict(BASE, dflash_config=dflash, **dspark))
        self.assertIsInstance(DSparkDraftModel(dense).layers[0].mlp, DFlashMLP)

    def test_stacked_per_expert_and_prefixed_exports_load_the_same(self):
        for per_expert, prefix in ((False, ""), (True, ""), (False, "model.")):
            with self.subTest(per_expert=per_expert, prefix=prefix):
                model = DFlashDraftModel(Qwen3Config(**_moe_fields()))
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
                model = DFlashDraftModel(Qwen3Config(**_moe_fields()))
                with self.assertRaisesRegex(ValueError, "missing"):
                    model.load_weights(list(partial.items()))
        with self.subTest(name="bias without the flag"):
            config = Qwen3Config(**_moe_fields(dflash_config=BASE["dflash_config"]))
            with self.assertRaisesRegex(ValueError, "does not enable"):
                DFlashDraftModel(config).load_weights(list(weights.items()))
        with self.subTest(name="wrong shape"):
            bad = dict(weights)
            bad["layers.0.mlp.gate.weight"] = torch.zeros(E, H + 1)
            with self.assertRaisesRegex(ValueError, "expected"):
                DFlashDraftModel(Qwen3Config(**_moe_fields())).load_weights(
                    list(bad.items())
                )
        with self.subTest(name="MoE tensor on a dense draft"):
            with self.assertRaisesRegex(ValueError, "layer is dense"):
                DFlashDraftModel(Qwen3Config(**BASE)).load_weights(
                    list(weights.items())
                )


if __name__ == "__main__":
    unittest.main()
