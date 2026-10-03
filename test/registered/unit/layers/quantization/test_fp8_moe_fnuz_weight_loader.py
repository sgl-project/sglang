"""ROCm FNUZ normalization of block-FP8 MoE weights must keep the loadable Parameters."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import Module, Parameter

import sglang.srt.layers.quantization.fp8 as fp8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

EXPERTS, INTER, HIDDEN, BLOCK = 2, 128, 256, 128


def _loader(*args, **kwargs):
    raise AssertionError("not called here")


def _layer():
    layer = Module()
    shapes = {
        "w13_weight": ((EXPERTS, 2 * INTER, HIDDEN), torch.float8_e4m3fn),
        "w2_weight": ((EXPERTS, HIDDEN, INTER), torch.float8_e4m3fn),
        "w13_weight_scale_inv": (
            (EXPERTS, 2 * INTER // BLOCK, HIDDEN // BLOCK),
            torch.float32,
        ),
        "w2_weight_scale_inv": (
            (EXPERTS, HIDDEN // BLOCK, INTER // BLOCK),
            torch.float32,
        ),
    }
    for name, (shape, dtype) in shapes.items():
        data = torch.full(shape, 0.5).to(dtype)
        param = Parameter(data, requires_grad=False)
        param.weight_loader = _loader
        setattr(layer, name, param)
    # Set at weight creation for UE8M0 checkpoints; stale once the scales are block-128 FP32.
    layer.w13_weight_scale_inv.format_ue8m0 = True
    layer.w2_weight_scale_inv.format_ue8m0 = True
    return layer


def _method():
    method = fp8.Fp8MoEMethod.__new__(fp8.Fp8MoEMethod)
    method.is_fp4_expert = False
    method.dequant_fp4_to_fp8 = False
    method.convert_mxfp8_to_block = False
    method.use_mxfp8 = False
    return method


class TestFp8MoEFnuzWeightLoader(unittest.TestCase):
    def test_plain_block_fp8_keeps_parameters_and_their_weight_loader(self):
        """The path a standard DeepSeek FP8 checkpoint takes on gfx94x (use_mxfp8 is False)."""
        layer = _layer()
        before = {name: getattr(layer, name) for name in ("w13_weight", "w2_weight")}
        before |= {
            name: getattr(layer, name)
            for name in ("w13_weight_scale_inv", "w2_weight_scale_inv")
        }

        with (
            patch.object(fp8, "_is_fp8_fnuz", True),
            patch.object(fp8, "_use_aiter", False),
        ):
            _method().process_weights_after_loading_block_quant(layer)

        for name, param in before.items():
            self.assertIs(getattr(layer, name), param, name)
            self.assertIs(getattr(layer, name).weight_loader, _loader, name)
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fnuz)
        self.assertEqual(layer.w2_weight.dtype, torch.float8_e4m3fnuz)


if __name__ == "__main__":
    unittest.main()
