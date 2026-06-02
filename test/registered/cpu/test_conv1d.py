import itertools
import unittest

import torch
import torch.nn as nn
from utils import precision

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

torch.manual_seed(1234)

conv1d_cpu = torch.ops.sgl_kernel.conv1d_cpu
conv1d_weight_pack = torch.ops.sgl_kernel.conv1d_weight_pack


class TestConv1d(CustomTestCase):

    def _test_conv1d(self, N, IC, OC, L, kernel_size, stride, padding, dtype):
        """Test conv1d_cpu against torch.nn.Conv1d reference."""
        conv_ref = nn.Conv1d(
            IC, OC, kernel_size=kernel_size, stride=stride, padding=padding, bias=True
        ).to(dtype)

        x = torch.randn(N, IC, L, dtype=dtype)
        weight = conv_ref.weight.detach()
        bias = conv_ref.bias.detach()

        # reference: nn.Conv1d returns [N, OC, L_out], our kernel returns [N, L_out, OC]
        with torch.no_grad():
            ref_out = conv_ref(x).permute(0, 2, 1).contiguous()

        # sgl_kernel conv1d with on-the-fly packing
        out = conv1d_cpu(x, weight, bias, stride, padding, False)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

        # sgl_kernel conv1d with pre-packed weight
        packed_w = conv1d_weight_pack(weight)
        out2 = conv1d_cpu(x, packed_w, bias, stride, padding, True)
        torch.testing.assert_close(ref_out, out2, atol=atol, rtol=rtol)

    def _test_conv1d_no_bias(self, N, IC, OC, L, kernel_size, stride, padding, dtype):
        """Test conv1d_cpu without bias."""
        conv_ref = nn.Conv1d(
            IC, OC, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        ).to(dtype)

        x = torch.randn(N, IC, L, dtype=dtype)
        weight = conv_ref.weight.detach()

        with torch.no_grad():
            ref_out = conv_ref(x).permute(0, 2, 1).contiguous()

        out = conv1d_cpu(x, weight, None, stride, padding, False)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_conv1d_whisper(self):
        """Test conv1d in whisper"""
        for N in [1, 4]:
            for dtype in [torch.bfloat16]:
                with self.subTest(N=N, dtype=dtype):
                    self._test_conv1d(
                        N=N,
                        IC=128,
                        OC=1280,
                        L=3000,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dtype=dtype,
                    )
                    self._test_conv1d(
                        N=N,
                        IC=1280,
                        OC=1280,
                        L=3000,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        dtype=dtype,
                    )

    def test_conv1d_small_shapes(self):
        """Small shapes for correctness validation."""
        params = list(
            itertools.product(
                [1, 2],  # N
                [32, 64],  # IC
                [32, 64],  # OC
                [16, 100],  # L
                [3],  # kernel_size
                [1, 2],  # stride
                [0, 1],  # padding
                [torch.bfloat16],  # dtype
            )
        )
        for N, IC, OC, L, kernel_size, stride, padding, dtype in params:
            with self.subTest(
                N=N,
                IC=IC,
                OC=OC,
                L=L,
                ks=kernel_size,
                stride=stride,
                pad=padding,
                dtype=dtype,
            ):
                self._test_conv1d(N, IC, OC, L, kernel_size, stride, padding, dtype)

    def test_conv1d_no_bias(self):
        """Test without bias."""
        for N in [1, 2]:
            for dtype in [torch.bfloat16]:
                with self.subTest(N=N, dtype=dtype):
                    self._test_conv1d_no_bias(
                        N=N,
                        IC=128,
                        OC=1280,
                        L=3000,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dtype=dtype,
                    )

    def test_conv1d_weight_pack(self):
        """Test that weight packing produces correct layout."""
        OC, IC, kernel_size = 64, 32, 3
        weight = torch.randn(OC, IC, kernel_size, dtype=torch.bfloat16)
        packed = conv1d_weight_pack(weight)

        # verify shape is preserved element-wise
        self.assertEqual(packed.numel(), weight.numel())


if __name__ == "__main__":
    unittest.main()
