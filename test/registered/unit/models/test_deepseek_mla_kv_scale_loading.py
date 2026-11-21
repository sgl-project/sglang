"""Checkpoint KV-cache scales must not reach the MLA attention of DeepSeek models."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common import deepseek_weight_loader
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.test.test_utils import CustomTestCase


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight):
        self.loaded = loaded_weight


class _FakeDeepseekLoader(DeepseekV2WeightLoaderMixin):
    def __init__(self, params):
        self.config = SimpleNamespace(
            n_routed_experts=0,
            q_lora_rank=None,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
        )
        self.model = SimpleNamespace()
        self.quant_config = SimpleNamespace(get_name=lambda: "compressed-tensors")
        self.num_fused_shared_experts = 0
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())

    def _maybe_quant_weights_to_fp8_ue8m0(
        self, weights, attn_quant_modules, nextn_conf
    ):
        return weights

    def post_load_weights(self, is_nextn=False, weight_names=None):
        pass


ATTN = "model.layers.0.self_attn"
# compressed-tensors (LLM Compressor, AutoRound) and ModelOpt checkpoint names
CT_NAMES = [f"{ATTN}.k_scale", f"{ATTN}.v_scale"]
MODELOPT_NAMES = [f"{ATTN}.k_proj.k_scale", f"{ATTN}.v_proj.v_scale"]


class TestDeepseekMlaKvScaleLoading(CustomTestCase):
    def _load(self, names, value):
        params = {
            f"{ATTN}.{attn}.{s}_scale": _FakeParam()
            for attn in ("attn_mqa", "attn_mha")
            for s in "kv"
        }
        params[f"{ATTN}.o_proj.weight"] = _FakeParam()
        weights = [(n, torch.tensor(value)) for n in names]
        weights.append((f"{ATTN}.o_proj.weight", torch.ones(2, 2)))
        logger = deepseek_weight_loader.logger
        with (
            patch.object(logger, "warning") as warning,
            patch.object(logger, "warning_once") as warning_once,
        ):
            _FakeDeepseekLoader(params).do_load_weights(weights)
        return params, warning, warning_once

    def test_calibrated_kv_scales_are_not_loaded(self):
        for names in (CT_NAMES, MODELOPT_NAMES):
            with self.subTest(names=names):
                params, warning, warning_once = self._load(names, 0.0823)
                for attn in ("attn_mqa", "attn_mha"):
                    for s in "kv":
                        self.assertIsNone(params[f"{ATTN}.{attn}.{s}_scale"].loaded)
                self.assertIsNotNone(params[f"{ATTN}.o_proj.weight"].loaded)
                warning.assert_not_called()
                warning_once.assert_called()
                self.assertIn("KV-cache scales", warning_once.call_args.args[0])

    def test_unit_kv_scales_are_skipped_without_warning(self):
        params, warning, warning_once = self._load(MODELOPT_NAMES + CT_NAMES, 1.0)
        self.assertIsNone(params[f"{ATTN}.attn_mqa.k_scale"].loaded)
        warning.assert_not_called()
        warning_once.assert_not_called()

    def test_kv_cache_scale_names(self):
        one = torch.tensor(1.0)
        for name in (
            "model.decoder.self_attn.k_proj.k_scale",  # NextN
            "model.layers.3.self_attn.1.v_scale",  # LongCat-Flash
        ):
            self.assertTrue(deepseek_weight_loader.skip_mla_kv_cache_scale(name, one))
        for name in (
            f"{ATTN}.attn_mqa.k_scale",
            f"{ATTN}.kv_a_proj_with_mqa.weight_scale",
            "model.layers.0.mlp.experts.0.down_proj.input_scale",
        ):
            self.assertFalse(deepseek_weight_loader.skip_mla_kv_cache_scale(name, one))


if __name__ == "__main__":
    unittest.main()
