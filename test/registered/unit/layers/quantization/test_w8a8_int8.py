"""Regression tests for W8A8 INT8 weight-name mapping."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.models.utils import WeightsMapper
from sglang.test.test_utils import CustomTestCase


class TestW8A8Int8WeightNameMapping(CustomTestCase):
    def test_weight_name_mapper_updates_shared_expert_ignore(self):
        """HF block_sparse_moe ignores must match SGLang's mlp module names."""
        old_prefix = "language_model.model.layers.3.block_sparse_moe.shared_experts"
        new_prefix = "language_model.model.layers.3.mlp.shared_experts"
        config = W8A8Int8Config(
            {
                "ignore": [
                    f"{old_prefix}.gate_proj",
                    f"{old_prefix}.up_proj",
                ],
                "packed_modules_mapping": {"gate_up_proj": ["gate_proj", "up_proj"]},
            }
        )

        config.apply_weight_name_mapper(
            WeightsMapper(orig_to_new_substr={".block_sparse_moe.": ".mlp."})
        )

        expected_ignore = [
            f"{new_prefix}.gate_proj",
            f"{new_prefix}.up_proj",
        ]
        self.assertEqual(config.ignore, expected_ignore)
        self.assertEqual(config.quant_description["ignore"], expected_ignore)

        layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
        torch.nn.Module.__init__(layer)
        layer.output_sizes = [3072, 3072]
        method = config.get_quant_method(
            layer,
            prefix=f"{new_prefix}.gate_up_proj",
        )
        self.assertIsInstance(method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
