"""KimiK3LinearForCausalLM must extend, not replace, the quant config's mapping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Exclude lists name the checkpoint shards, not the fused runtime module.
EXCLUDE_MODULES = [f"model.layers.0.self_attn.{s}_conv1d" for s in "qkv"]


class TestKimiK3PackedModulesMapping(CustomTestCase):
    def test_causal_lm_keeps_loader_seeded_mapping(self):
        # The loader seeds the mapping from the registered outer model class.
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=EXCLUDE_MODULES,
            packed_modules_mapping=dict(
                KimiK3ForConditionalGeneration.packed_modules_mapping
            ),
        )

        with (
            patch.object(kimi_k3, "KimiK3LinearModel"),
            patch.object(kimi_k3, "get_parallel"),
            patch.object(kimi_k3, "ParallelLMHead"),
            patch.object(kimi_k3, "LogitsProcessor"),
        ):
            KimiK3LinearForCausalLM(
                SimpleNamespace(vocab_size=16, hidden_size=16), quant_config
            )

        self.assertEqual(
            quant_config.packed_modules_mapping,
            {
                **KimiK3ForConditionalGeneration.packed_modules_mapping,
                **KimiK3LinearForCausalLM.packed_modules_mapping,
            },
        )
        self.assertTrue(
            is_layer_skipped(
                "model.layers.0.self_attn.qkv_conv1d",
                quant_config.exclude_modules,
                quant_config.packed_modules_mapping,
            )
        )


class TestKimiK3PackedExpertLoader(CustomTestCase):
    """WNA16 packed parameters must actually be written; they must not be silently
    dropped after the name remapping."""

    def test_packed_names_follow_quantization_and_npu_layout(self):
        cases = (
            ("mxfp4", None, False, "weight"),
            ("compressed-tensors", "pack-quantized-mxfp4", False, "weight"),
            ("compressed-tensors", "pack-quantized", False, "weight_packed"),
            (None, None, False, "weight_packed"),
            ("mxfp4", None, True, "weight_packed"),
        )
        for quant_name, quant_format, npu, suffix in cases:
            with self.subTest(
                quant_name=quant_name, quant_format=quant_format, npu=npu
            ):
                param = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
                expected = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)

                def copy_expert(target, weight, name, *, expert_id, shard_id):
                    target.data.copy_(weight)

                param.weight_loader = copy_expert
                target_name = f"model.layers.0.mlp.experts.w2_{suffix}"
                quant_config = (
                    SimpleNamespace(
                        get_name=lambda: quant_name, quant_format=quant_format
                    )
                    if quant_name is not None
                    else None
                )
                # Skip the model build and GPU post-processing while still running
                # the real load_weights name resolution.
                model = SimpleNamespace(
                    config=SimpleNamespace(
                        linear_attn_config={},
                        is_moe=True,
                        num_experts=1,
                        num_hidden_layers=1,
                    ),
                    quant_config=quant_config,
                    model=SimpleNamespace(start_layer=0, end_layer=1),
                    named_parameters=lambda: [(target_name, param)],
                    post_load_weights=lambda: None,
                )
                with patch.object(kimi_k3, "_is_npu", npu):
                    loaded = KimiK3LinearForCausalLM.load_weights(
                        model,
                        [("model.layers.0.mlp.experts.0.w2.weight_packed", expected)],
                    )
                self.assertEqual(loaded, {target_name})
                torch.testing.assert_close(param, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
