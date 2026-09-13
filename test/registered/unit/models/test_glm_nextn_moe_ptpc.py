"""CI for SGLANG_GLM_NEXTN_MOE_PTPC=1 (GLM-5.2 NextN per-channel FP8 draft MoE).

The feature is off by default. Without a case that turns the flag on, CI never
touches the Quark scheme rewrite and cannot claim the path works. These tests
exercise that ON wiring on CPU without loading a 70B MXFP4 checkpoint.

A full serve+generate job still needs the MXFP4 weights in the runner cache;
register that separately as nightly if the checkpoint is present.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
    NextNEnabledConfig,
)
from sglang.srt.models.deepseek_common.utils import (
    enable_glm_nextn_moe_ptpc,
    should_apply_glm_nextn_moe_ptpc,
)
from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LAYER = 78
PREFIX = f"model.layers.{LAYER}"
EXPERT_LEAF = f"{PREFIX}.mlp.experts.0.w1"
ATTN_LEAF = f"{PREFIX}.self_attn.q_proj"
EXPERT_WEIGHT = f"{PREFIX}.mlp.experts.0.gate_proj.weight"
_PTPC_ENV = "sglang.srt.models.deepseek_common.utils.envs.SGLANG_GLM_NEXTN_MOE_PTPC.get"


def _quark_cfg(*, exclude=None, layer_quant=None):
    return SimpleNamespace(
        get_name=lambda: "quark",
        quant_config={"layer_quant_config": dict(layer_quant or {}), "exclude": []},
        exclude_layers=list(
            exclude if exclude is not None else [EXPERT_LEAF, ATTN_LEAF]
        ),
    )


def _nextn_conf():
    return NextNEnabledConfig(
        num_nextn_layers=1,
        nextn_layer_id=LAYER,
        nextn_layer_prefix=PREFIX,
        nextn_spec_weight_names=[],
    )


class TestEnableGlmNextnMoePtpc(CustomTestCase):
    def test_off_by_default(self):
        self.assertFalse(enable_glm_nextn_moe_ptpc(_quark_cfg()))

    def test_on_requires_quark(self):
        with patch(
            _PTPC_ENV,
            return_value=True,
        ):
            self.assertTrue(enable_glm_nextn_moe_ptpc(_quark_cfg()))
            self.assertFalse(
                enable_glm_nextn_moe_ptpc(SimpleNamespace(get_name=lambda: "fp8"))
            )
            self.assertFalse(enable_glm_nextn_moe_ptpc(None))

    def test_apply_requires_excluded_mtp_experts(self):
        with patch(
            _PTPC_ENV,
            return_value=True,
        ):
            self.assertTrue(
                should_apply_glm_nextn_moe_ptpc(_quark_cfg(), LAYER, model_type="glm4")
            )
            self.assertFalse(
                should_apply_glm_nextn_moe_ptpc(
                    _quark_cfg(exclude=[ATTN_LEAF]), LAYER, model_type="glm4"
                )
            )
            self.assertFalse(
                should_apply_glm_nextn_moe_ptpc(
                    _quark_cfg(), LAYER, model_type="deepseek_v3"
                )
            )


class TestResolveNextnQuantConfigPtpcOn(CustomTestCase):
    def _resolve(self, cfg, flag: bool):
        model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
        hf = SimpleNamespace(num_hidden_layers=LAYER)
        with patch(
            _PTPC_ENV,
            return_value=flag,
        ):
            return model._resolve_nextn_quant_config(hf, cfg)

    def test_flag_off_excludes_fused_experts(self):
        src = _quark_cfg()
        out = self._resolve(src, flag=False)
        self.assertIn("model.decoder.mlp.experts", out.exclude_layers)
        self.assertNotIn(
            "model.decoder.mlp.experts",
            out.quant_config.get("layer_quant_config", {}),
        )

    def test_flag_on_assigns_ptpc_scheme_instead_of_bf16_exclude(self):
        src = _quark_cfg()
        out = self._resolve(src, flag=True)
        self.assertNotIn("model.decoder.mlp.experts", out.exclude_layers)
        scheme = out.quant_config["layer_quant_config"]["model.decoder.mlp.experts"]
        self.assertEqual(scheme["weight"]["dtype"], "fp8_e4m3")
        self.assertEqual(scheme["weight"]["qscheme"], "per_channel")
        self.assertFalse(scheme["weight"]["is_dynamic"])
        self.assertEqual(scheme["input_tensors"]["dtype"], "fp8_e4m3")
        self.assertEqual(scheme["input_tensors"]["qscheme"], "per_channel")
        self.assertTrue(scheme["input_tensors"]["is_dynamic"])

    def test_flag_on_does_not_mutate_caller_config(self):
        src = _quark_cfg()
        orig_exclude = list(src.exclude_layers)
        orig_layer = dict(src.quant_config.get("layer_quant_config") or {})
        self._resolve(src, flag=True)
        self.assertEqual(src.exclude_layers, orig_exclude)
        self.assertEqual(src.quant_config.get("layer_quant_config") or {}, orig_layer)

    def test_flag_on_skips_ptpc_when_experts_not_excluded(self):
        src = _quark_cfg(exclude=[ATTN_LEAF])
        out = self._resolve(src, flag=True)
        self.assertNotIn("model.decoder.mlp.experts", out.exclude_layers)
        self.assertNotIn(
            "model.decoder.mlp.experts",
            out.quant_config.get("layer_quant_config", {}),
        )

    def test_flag_on_skips_ptpc_when_mtp_not_in_exclude(self):
        src = _quark_cfg(exclude=[])
        out = self._resolve(src, flag=True)
        self.assertNotIn(
            "model.decoder.mlp.experts",
            out.quant_config.get("layer_quant_config", {}),
        )


class TestMaybeQuantGlmNextnMoeToPtpc(CustomTestCase):
    def _cast(self, cfg, flag: bool, model_type: str = "glm4"):
        loader = DeepseekV2WeightLoaderMixin.__new__(DeepseekV2WeightLoaderMixin)
        loader.quant_config = cfg
        loader.config = SimpleNamespace(model_type=model_type)
        weights = [(EXPERT_WEIGHT, torch.ones(4, 8, dtype=torch.bfloat16))]
        with patch(
            _PTPC_ENV,
            return_value=flag,
        ):
            return list(
                loader._maybe_quant_glm_nextn_moe_to_ptpc(weights, _nextn_conf())
            )

    def test_flag_on_casts_excluded_bf16_experts(self):
        out = self._cast(_quark_cfg(), flag=True)
        names = [name for name, _ in out]
        self.assertIn(EXPERT_WEIGHT, names)
        self.assertIn(EXPERT_WEIGHT[: -len("weight")] + "weight_scale", names)
        weight = dict(out)[EXPERT_WEIGHT]
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)

    def test_flag_on_does_not_cast_when_experts_not_excluded(self):
        src = _quark_cfg(exclude=[ATTN_LEAF])
        out = self._cast(src, flag=True)
        self.assertEqual(len(out), 1)
        name, tensor = out[0]
        self.assertEqual(name, EXPERT_WEIGHT)
        self.assertEqual(tensor.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
