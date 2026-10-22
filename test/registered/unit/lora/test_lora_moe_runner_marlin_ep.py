"""CPU contracts for MXFP4 Marlin LoRA quant info."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMxfp4MarlinLoRAQuantInfo(CustomTestCase):
    def test_quant_info_reads_the_prepared_marlin_experts(self):
        expert_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        layer = SimpleNamespace(
            w13_weight=torch.zeros(2, 4, 8, dtype=torch.int32),
            w2_weight=torch.zeros(2, 2, 8, dtype=torch.int32),
            w13_weight_scale=torch.zeros(2, 2, 16, dtype=torch.float8_e8m0fnu),
            w2_weight_scale=torch.zeros(2, 1, 16, dtype=torch.float8_e8m0fnu),
            dispatcher=SimpleNamespace(local_expert_mapping=expert_map, num_experts=4),
        )

        info = Mxfp4MarlinMoEMethod(None, "experts").get_marlin_quant_info(layer)

        self.assertIs(info.w13_qweight, layer.w13_weight)
        self.assertIs(info.w2_qweight, layer.w2_weight)
        self.assertIs(info.w13_scales, layer.w13_weight_scale)
        self.assertIs(info.w2_scales, layer.w2_weight_scale)
        self.assertEqual(info.weight_bits, 4)
        self.assertIs(info.expert_map, expert_map)
        self.assertEqual(info.global_num_experts, 4)


if __name__ == "__main__":
    unittest.main()
