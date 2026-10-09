"""KimiK3LinearForCausalLM must extend, not replace, the quant config's mapping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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


if __name__ == "__main__":
    unittest.main()
