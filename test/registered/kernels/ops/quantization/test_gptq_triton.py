"""
Tests for the ROCm GPTQ Triton W4A16 path (GPTQTritonLinearKernel).

Run with:
    python -m unittest test_gptq_triton.py
"""

import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.gptq_triton import (
    GPTQ_TRITON_SUPPORTED_GROUP_SIZES,
)
from sglang.srt.hardware_backend.gpu.quantization.gptq_triton_kernels import (
    GPTQTritonLinearKernel,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-a-test-1-gpu-small-amd")

device = get_device()


def to_int32(words: torch.Tensor) -> torch.Tensor:
    return ((words + 2**31) % 2**32 - 2**31).to(torch.int32)


def pack_along_k(codes: torch.Tensor) -> torch.Tensor:
    """[K, N] 4-bit codes -> GPTQ qweight [K/8, N]; nibble i holds row 8r+i."""
    k, n = codes.shape
    codes = codes.view(k // 8, 8, n).to(torch.int64)
    return to_int32(sum(codes[:, i] << (4 * i) for i in range(8)))


def pack_along_n(codes: torch.Tensor) -> torch.Tensor:
    """[G, N] 4-bit codes -> GPTQ qzeros [G, N/8]; nibble j holds column 8c+j."""
    g, n = codes.shape
    codes = codes.view(g, n // 8, 8).to(torch.int64)
    return to_int32(sum(codes[:, :, j] << (4 * j) for j in range(8)))


def make_layer(k, n, group_size, sym, dtype, checkpoint_format=""):
    """Random GPTQ checkpoint loaded through the kernel, plus its fp32 weight."""
    eff_group_size = k if group_size == -1 else group_size
    num_groups = k // eff_group_size
    codes = torch.randint(0, 16, (k, n), device=device)
    if sym:
        zeros = torch.full((num_groups, n), 8, device=device)
    else:
        zeros = torch.randint(1, 16, (num_groups, n), device=device)
    scales = (torch.rand(num_groups, n, device=device) * 0.02 + 0.001).to(dtype)
    ref_weight = (codes - zeros.repeat_interleave(eff_group_size, 0)).float()
    ref_weight *= scales.float().repeat_interleave(eff_group_size, 0)

    # GPTQ v1 checkpoints store zero - 1; v2 stores the zero point itself.
    stored_zeros = zeros if checkpoint_format == "gptq_v2" else zeros - 1

    def param(t):
        return torch.nn.Parameter(t, requires_grad=False)

    layer = torch.nn.Module()
    layer.qweight = param(pack_along_k(codes))
    layer.qzeros = param(pack_along_n(stored_zeros))
    layer.scales = param(scales)
    layer.g_idx = param(
        torch.arange(k, device=device, dtype=torch.int32) // eff_group_size
    )
    config = SimpleNamespace(
        weight_bits=4,
        group_size=group_size,
        desc_act=False,
        checkpoint_format=checkpoint_format,
    )
    kernel = GPTQTritonLinearKernel(config)
    kernel.process_weights_after_loading(layer)
    return kernel, layer, ref_weight


def rel_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).norm() / ref.norm()).item()


class TestGPTQTriton(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_linear_matches_reference(self):
        shapes = [(256, 128), (1024, 512), (5120, 96)]
        for (k, n), group_size, sym, dtype in itertools.product(
            shapes,
            sorted(GPTQ_TRITON_SUPPORTED_GROUP_SIZES),
            [True, False],
            [torch.float16, torch.bfloat16],
        ):
            kernel, layer, ref_weight = make_layer(k, n, group_size, sym, dtype)
            bias = torch.randn(n, device=device, dtype=dtype)
            for m in [1, 8, 33, 257]:
                with self.subTest(
                    k=k, n=n, group_size=group_size, sym=sym, dtype=dtype, m=m
                ):
                    x = torch.randn(m, k, device=device, dtype=dtype)
                    out = kernel.apply(layer, x, bias)
                    self.assertEqual(out.dtype, dtype)
                    ref = x.float() @ ref_weight + bias.float()
                    self.assertLess(rel_error(out, ref), 1e-2)

    def test_zero_point_handling(self):
        for sym in [True, False]:
            with self.subTest(sym=sym):
                _, layer, _ = make_layer(1024, 256, 128, sym, torch.float16)
                if sym:
                    self.assertIsNone(layer.qzeros)
                else:
                    self.assertEqual(tuple(layer.qzeros.shape), (256 // 8, 1024 // 128))
                self.assertEqual(tuple(layer.qweight.shape), (256, 1024 // 8))
                self.assertEqual(tuple(layer.scales.shape), (256, 1024 // 128))
                self.assertFalse(hasattr(layer, "g_idx"))

    def test_gptq_v2_checkpoint(self):
        for sym in [True, False]:
            with self.subTest(sym=sym):
                kernel, layer, ref_weight = make_layer(
                    1024, 256, 128, sym, torch.bfloat16, checkpoint_format="gptq_v2"
                )
                self.assertEqual(layer.qzeros is None, sym)
                x = torch.randn(16, 1024, device=device, dtype=torch.bfloat16)
                self.assertLess(
                    rel_error(kernel.apply(layer, x), x.float() @ ref_weight), 1e-2
                )

    def test_3d_input(self):
        kernel, layer, ref_weight = make_layer(1024, 512, 128, True, torch.bfloat16)
        x = torch.randn(2, 5, 1024, device=device, dtype=torch.bfloat16)
        out = kernel.apply(layer, x)
        self.assertEqual(tuple(out.shape), (2, 5, 512))
        self.assertLess(rel_error(out, x.float() @ ref_weight), 1e-2)

    def test_unsupported_configs_raise(self):
        cases = [
            (dict(weight_bits=8, group_size=128, desc_act=False), NotImplementedError),
            (dict(weight_bits=4, group_size=128, desc_act=True), NotImplementedError),
            (dict(weight_bits=4, group_size=16, desc_act=False), ValueError),
        ]
        for fields, error in cases:
            with self.subTest(**fields):
                kernel = GPTQTritonLinearKernel(
                    SimpleNamespace(checkpoint_format="", **fields)
                )
                with self.assertRaises(error):
                    kernel.process_weights_after_loading(torch.nn.Module())


if __name__ == "__main__":
    unittest.main(verbosity=2)
