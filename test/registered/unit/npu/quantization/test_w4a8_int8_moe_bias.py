"""Regression tests for W4A8 down-projection compensation under MoE TP."""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

# Initialize the quantization package before the NPU backend to avoid its
# base_config / linear_method_npu circular import on a cold interpreter.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A8Int8MoEMethod,
)
from sglang.test.test_utils import CustomTestCase


class TestW4A8Int8MoEBias(CustomTestCase):
    @staticmethod
    def make_layer(bias, *, tp_size, tp_rank, prefix="w2"):
        layer = torch.nn.Module()
        layer.moe_tp_size = tp_size
        layer.moe_tp_rank = tp_rank
        layer.register_parameter(
            f"{prefix}_scale_bias",
            torch.nn.Parameter(bias.clone(), requires_grad=False),
        )
        return layer

    def test_w2_compensation_follows_each_k_partition(self):
        # Nonuniform segments distinguish the correct rank-local compensation
        # from both a replicated full sum and a full sum divided by TP size.
        bias = torch.arange(2 * 3 * 16, dtype=torch.float32).square().view(2, 3, 16)
        for tp_size in (1, 2, 4, 8, 16):
            rank_biases = []
            for tp_rank, shard in enumerate(bias.chunk(tp_size, dim=-1)):
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank):
                    layer = self.make_layer(bias, tp_size=tp_size, tp_rank=tp_rank)
                    parameter = layer.w2_scale_bias
                    NPUW4A8Int8MoEMethod._update_bias(layer, "w2")
                    rank_biases.append(layer.w2_scale_bias)
                    torch.testing.assert_close(
                        layer.w2_scale_bias, shard.sum(dim=-1), rtol=0, atol=0
                    )
                    self.assertIs(layer.w2_scale_bias, parameter)
                    self.assertFalse(layer.w2_scale_bias.requires_grad)
            # Reducing all ranks must add each checkpoint segment exactly once.
            with self.subTest(tp_size=tp_size, check="reduction"):
                torch.testing.assert_close(
                    torch.stack(rank_biases).sum(dim=0),
                    bias.sum(dim=-1),
                    rtol=0,
                    atol=0,
                )

    def test_mixed_tp_ep_uses_moe_tp_rank(self):
        bias = torch.arange(16, dtype=torch.float32).view(1, 1, 16)
        layer = self.make_layer(bias, tp_size=4, tp_rank=1)
        layer.tp_size = 16
        layer.tp_rank = 13
        NPUW4A8Int8MoEMethod._update_bias(layer, "w2")
        torch.testing.assert_close(layer.w2_scale_bias, torch.tensor([[22.0]]))

    def test_ep_with_moe_tp_one_keeps_all_segments(self):
        bias = torch.arange(16, dtype=torch.float32).view(1, 1, 16)
        layer = self.make_layer(bias, tp_size=1, tp_rank=0)
        layer.tp_size = 16
        layer.tp_rank = 15
        NPUW4A8Int8MoEMethod._update_bias(layer, "w2")
        torch.testing.assert_close(layer.w2_scale_bias, torch.tensor([[120.0]]))

    def test_w13_compensation_is_already_output_sharded(self):
        bias = torch.arange(12, dtype=torch.float32).view(2, 6, 1)
        layer = self.make_layer(bias, tp_size=16, tp_rank=15, prefix="w13")
        NPUW4A8Int8MoEMethod._update_bias(layer, "w13")
        torch.testing.assert_close(layer.w13_scale_bias, bias.squeeze(-1))

    def test_missing_compensation_is_unchanged(self):
        layer = torch.nn.Module()
        for prefix in ("w13", "w2"):
            NPUW4A8Int8MoEMethod._update_bias(layer, prefix)
        self.assertEqual(list(layer.parameters()), [])

    def test_rejects_nondivisible_compensation_segments(self):
        for tp_size in (3, 32):
            with self.subTest(tp_size=tp_size):
                layer = self.make_layer(
                    torch.ones(2, 3, 16), tp_size=tp_size, tp_rank=0
                )
                with self.assertRaisesRegex(AssertionError, "divisible by MoE TP size"):
                    NPUW4A8Int8MoEMethod._update_bias(layer, "w2")


if __name__ == "__main__":
    unittest.main()
