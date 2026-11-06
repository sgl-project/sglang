"""The FFN spec of a draft: dense, MoE or mixed, read from the draft config."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.draft.ffn_spec import (
    FFN_KIND_DENSE,
    FFN_KIND_MOE,
    DraftDenseFFNSpec,
    DraftMixFFNSpec,
    DraftMoeFFNSpec,
)
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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


class TestDenseDraft(CustomTestCase):
    def test_no_experts_means_every_layer_dense(self):
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

    def test_unknown_layer_count_is_uniform(self):
        spec = DraftMixFFNSpec.from_config(_without(_dense(), "num_hidden_layers"))
        self.assertIsNone(spec.layer_kinds)
        self.assertEqual(spec.kind(41), FFN_KIND_DENSE)
        with self.assertRaisesRegex(ValueError, "outside"):
            DraftMixFFNSpec.from_config(_dense()).kind(3)


class TestQwenStyleMoe(CustomTestCase):
    def test_specforge_config(self):
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

    def test_defaults_and_expert_count_aliases(self):
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

    def test_gated_shared_expert_and_bias_flag(self):
        moe = DraftMixFFNSpec.from_config(
            _qwen_moe(shared_expert_intermediate_size=24)
        ).moe
        self.assertEqual(moe.shared_expert_intermediate_size, 24)
        self.assertTrue(moe.shared_expert_gated)
        top_level = _qwen_moe(moe_router_enable_expert_bias=True)
        top_level["dflash_config"] = dict(DFLASH)
        self.assertTrue(DraftMixFFNSpec.from_config(top_level).moe.expert_bias)
        nested_wins = _qwen_moe(moe_router_enable_expert_bias=False)
        self.assertTrue(DraftMixFFNSpec.from_config(nested_wins).moe.expert_bias)

    def test_mixtral_style_sizes_experts_by_intermediate_size(self):
        cfg = {
            "num_hidden_layers": 2,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "intermediate_size": 64,
        }
        self.assertEqual(DraftMixFFNSpec.from_config(cfg).moe.intermediate_size, 64)
        with self.assertRaisesRegex(ValueError, "moe_intermediate_size is required"):
            DraftMixFFNSpec.from_config(_without(cfg, "intermediate_size"))


class TestOtherFamilies(CustomTestCase):
    def test_deepseek_v3_style(self):
        moe = DraftMixFFNSpec.from_config(_deepseek_v3()).moe
        self.assertEqual(moe.scoring_func, "sigmoid")
        self.assertEqual((moe.num_expert_group, moe.topk_group), (8, 4))
        self.assertTrue(moe.expert_bias)  # noaux_tc routes with the stored bias
        self.assertEqual(moe.routed_scaling_factor, 2.5)
        self.assertEqual(moe.shared_expert_intermediate_size, 28)  # 2 summed experts
        self.assertFalse(moe.shared_expert_gated)

    def test_deepseek_v4_style(self):
        cfg = _deepseek_v3(
            scoring_func="sqrtsoftplus", swiglu_limit=10.0, n_group=1, topk_group=1
        )
        moe = DraftMixFFNSpec.from_config(_without(cfg, "topk_method")).moe
        self.assertEqual(moe.scoring_func, "sqrtsoftplus")
        self.assertEqual(moe.swiglu_limit, 10.0)
        self.assertIsNone(moe.num_expert_group)
        self.assertFalse(moe.expert_bias)

    def test_gpt_oss_style_activation(self):
        cfg = {
            "num_hidden_layers": 1,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 64,
            "hidden_act": "swigluoai",
            "swiglu_limit": 7.0,
            "swiglu_alpha": 1.702,
        }
        moe = DraftMixFFNSpec.from_config(cfg).moe
        self.assertEqual(
            (moe.hidden_act, moe.swiglu_limit, moe.swiglu_alpha),
            ("swigluoai", 7.0, 1.702),
        )


class TestLayerSchedule(CustomTestCase):
    def test_every_schedule_key(self):
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

    def test_moe_layer_freq_list_marks_dense_layers(self):
        # MiniMax's DSpark draft: target MoE fields in the config, dense draft layers.
        cfg = {
            "num_hidden_layers": 5,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "intermediate_size": 3072,
            "moe_layer_freq": [0, 0, 0, 0, 0],
            "scoring_func": "sigmoid",
            "hidden_act": "swigluoai",
        }
        spec = DraftMixFFNSpec.from_config(cfg)
        self.assertEqual(spec.layer_kinds, ("dense",) * 5)
        self.assertIsNone(spec.moe)
        self.assertEqual(spec.dense.intermediate_size, 3072)

    def test_schedule_errors(self):
        cases = [
            (
                _qwen_moe(ffn_layer_types=["moe", "moe"]),
                "has 2 entries but num_hidden_layers=3",
            ),
            (_qwen_moe(ffn_layer_types=["moe", "mlp", "moe"]), "not one of"),
            (_qwen_moe(ffn_layer_types=[]), "non-empty list"),
            (_qwen_moe(ffn_layer_types="moe"), "non-empty list"),
            (
                _qwen_moe(mlp_layer_types=["dense", "moe", "sparse"]),
                "mlp_layer_types\\[1\\]='moe'",
            ),
            (
                _dense(ffn_layer_types=["moe", "dense", "dense"]),
                "names MoE layers but none of",
            ),
            (_without(_deepseek_v3(), "num_hidden_layers"), "needs num_hidden_layers"),
        ]
        for cfg, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    DraftMixFFNSpec.from_config(cfg)


class TestRefusals(CustomTestCase):
    def test_what_sglang_cannot_run_is_refused_by_name(self):
        cases = [
            (_qwen_moe(scoring_func="relu"), "scoring_func='relu' is not implemented"),
            (_qwen_moe(hidden_act="gelu"), "hidden_act='gelu' is not implemented"),
            (_qwen_moe(topk_method="foo"), "topk_method='foo' is not implemented"),
        ]
        for cfg, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    DraftMixFFNSpec.from_config(cfg)

    def test_inconsistent_fields(self):
        cases = [
            (_without(_dense(), "intermediate_size"), "intermediate_size is required"),
            (
                _without(_qwen_moe(), "num_experts_per_tok"),
                "num_experts_per_tok is required",
            ),
            (_qwen_moe(num_experts_per_tok=9), "exceeds num_experts"),
            (_qwen_moe(num_experts=True), "positive integer"),
            (_qwen_moe(norm_topk_prob=1), "norm_topk_prob must be a bool"),
            (_qwen_moe(shared_expert_intermediate_size=-1), "non-negative"),
            (_qwen_moe(routed_scaling_factor=0), "positive number"),
            (_qwen_moe(swiglu_limit=-1), "positive number"),
            (_deepseek_v3(topk_group=None), "topk_group is required when n_group > 1"),
            (_deepseek_v3(n_group=7), "does not divide"),
            (_deepseek_v3(topk_group=9), "exceeds n_group"),
            (
                _deepseek_v3(shared_expert_intermediate_size=512),
                "two shared-expert layouts",
            ),
        ]
        for cfg, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    DraftMixFFNSpec.from_config(cfg)
        bad_bias = _qwen_moe()
        bad_bias["dflash_config"]["moe_router_enable_expert_bias"] = "yes"
        with self.assertRaisesRegex(ValueError, "moe_router_enable_expert_bias"):
            DraftMixFFNSpec.from_config(bad_bias)


class TestDFlashDraftConfigWiring(CustomTestCase):
    def test_dflash_draft_config_carries_the_ffn_spec(self):
        dense = parse_dflash_draft_config(draft_hf_config=SimpleNamespace(**_dense()))
        self.assertIsInstance(dense.ffn, DraftMixFFNSpec)
        self.assertFalse(dense.ffn.has_moe)
        moe = parse_dflash_draft_config(draft_hf_config=SimpleNamespace(**_qwen_moe()))
        self.assertEqual(moe.ffn.kind(2), FFN_KIND_MOE)
        with self.assertRaisesRegex(ValueError, "scoring_func='relu'"):
            parse_dflash_draft_config(
                draft_hf_config=SimpleNamespace(**_qwen_moe(scoring_func="relu"))
            )


if __name__ == "__main__":
    unittest.main()
