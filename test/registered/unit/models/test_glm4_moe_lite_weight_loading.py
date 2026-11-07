"""ModelOpt KV-cache scales (self_attn.{k,v}_proj.{k,v}_scale) of GLM-4.7-Flash and its
MTP draft must reach the MLA skip: attn_mqa keeps 1.0 and one warning is logged."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.models.deepseek_common import deepseek_weight_loader
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
from sglang.srt.models.glm4_moe_lite_nextn import Glm4MoeLiteForCausalLMNextN
from sglang.test.test_utils import CustomTestCase


def _make_model(model_cls, attn_prefix, attn_mqa):
    model = object.__new__(model_cls)
    model.config = SimpleNamespace(
        num_hidden_layers=47,
        num_nextn_predict_layers=1,
        n_routed_experts=4,
        q_lora_rank=768,
    )
    model.num_fused_shared_experts = 0
    params = {f"{attn_prefix}.attn_mqa.{n}": p for n, p in attn_mqa.named_parameters()}
    model.named_parameters = lambda: iter(params.items())
    model.post_load_weights = lambda is_nextn=False, weight_names=None: None
    return model


class TestGlm4MoeLiteModelOptKvScales(CustomTestCase):
    def test_modelopt_kv_scales_are_skipped_with_one_warning(self):
        cases = [
            (Glm4MoeLiteForCausalLM, "model.layers.0", "model.layers.0"),
            (Glm4MoeLiteForCausalLMNextN, "model.layers.47", "model.decoder"),
        ]
        kv_cache_method = BaseKVCacheMethod(quant_config=None)
        logger = deepseek_weight_loader.logger
        for model_cls, ckpt_prefix, module_prefix in cases:
            with self.subTest(model=model_cls.__name__):
                attn_mqa = torch.nn.Module()
                kv_cache_method.create_weights(attn_mqa)
                model = _make_model(model_cls, f"{module_prefix}.self_attn", attn_mqa)
                weights = [
                    (f"{ckpt_prefix}.self_attn.k_proj.k_scale", torch.tensor(0.5)),
                    (f"{ckpt_prefix}.self_attn.v_proj.v_scale", torch.tensor(0.25)),
                ]

                with patch.object(logger, "warning_once") as warning_once:
                    model.load_weights(weights)
                kv_cache_method.process_weights_after_loading(attn_mqa)

                self.assertEqual(attn_mqa.k_scale_float, 1.0)
                self.assertEqual(attn_mqa.v_scale_float, 1.0)
                messages = {call.args[0] for call in warning_once.call_args_list}
                self.assertEqual(len(messages), 1)
                self.assertIn("KV-cache scales", messages.pop())


if __name__ == "__main__":
    unittest.main()
