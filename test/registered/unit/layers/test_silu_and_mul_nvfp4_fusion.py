"""Which SiluAndMul outputs are fused with the NVFP4 activation quant of the next linear."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import activation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _linear(scheme=None):
    linear = nn.Linear(4, 4)
    if scheme is not None:
        linear.scheme = scheme
    return linear


class TestFuseOutputQuant(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())
        enter_scope(self, patch.object(activation, "_is_cuda", True))
        enter_scope(
            self,
            patch.object(
                activation, "_flashinfer_silu_and_mul_nvfp4_available", lambda: True
            ),
        )

    def test_other_devices_never_fuse(self):
        act = SiluAndMul()
        with patch.object(activation, "_is_cuda", False):
            act.fuse_output_quant(_linear(CompressedTensorsW4A4Fp4()))
        self.assertIsNone(act._quant_linear)

    def test_needs_the_flashinfer_kernel(self):
        act = SiluAndMul()
        with patch.object(
            activation, "_flashinfer_silu_and_mul_nvfp4_available", lambda: False
        ):
            act.fuse_output_quant(_linear(CompressedTensorsW4A4Fp4()))
        self.assertIsNone(act._quant_linear)

    def test_keeps_only_w4a4_nvfp4_linears(self):
        nvfp4, plain, other_scheme = SiluAndMul(), SiluAndMul(), SiluAndMul()
        down = _linear(CompressedTensorsW4A4Fp4())
        nvfp4.fuse_output_quant(down)
        plain.fuse_output_quant(_linear())
        other_scheme.fuse_output_quant(_linear(SimpleNamespace()))

        self.assertIs(nvfp4._quant_linear, down)
        self.assertIsNone(plain._quant_linear)
        self.assertIsNone(other_scheme._quant_linear)

    def test_projection_is_not_a_child_module(self):
        act = SiluAndMul()
        act.fuse_output_quant(_linear(CompressedTensorsW4A4Fp4()))
        self.assertEqual(list(act.children()), [])
        self.assertEqual(list(act.parameters()), [])

    def test_fusable_inputs(self):
        bf16 = torch.empty(256, 64, dtype=torch.bfloat16)
        self.assertTrue(activation._can_fuse_nvfp4_act_quant(bf16))
        self.assertTrue(activation._can_fuse_nvfp4_act_quant(bf16.to(torch.float16)))
        self.assertFalse(activation._can_fuse_nvfp4_act_quant(bf16.float()))
        strided = torch.empty(256, 128, dtype=torch.bfloat16)[:, :64]
        self.assertFalse(activation._can_fuse_nvfp4_act_quant(strided))
        self.assertFalse(
            activation._can_fuse_nvfp4_act_quant(
                torch.empty(256, 48, dtype=torch.bfloat16)
            )
        )

    def test_small_batches_keep_the_unfused_path(self):
        # Tokens are counted across leading dims: [2, 128, d] is 256 tokens.
        for shape, fused in (
            ((255, 64), False),
            ((256, 64), True),
            ((2, 128, 64), True),
        ):
            x = torch.empty(shape, dtype=torch.bfloat16)
            self.assertEqual(activation._can_fuse_nvfp4_act_quant(x), fused)

    def test_lora_and_sequence_parallelism_keep_the_unfused_path(self):
        x = torch.empty(256, 64, dtype=torch.bfloat16)
        with patch.object(
            activation, "get_lora", return_value=SimpleNamespace(enable_lora=True)
        ):
            self.assertFalse(activation._can_fuse_nvfp4_act_quant(x))
        with patch.object(
            activation, "get_forward", return_value=SimpleNamespace(sp_active=True)
        ):
            self.assertFalse(activation._can_fuse_nvfp4_act_quant(x))


if __name__ == "__main__":
    unittest.main()
