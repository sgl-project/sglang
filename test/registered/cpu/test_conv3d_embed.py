import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

conv3d_embed_weight_pack = torch.ops.sgl_kernel.conv3d_embed_weight_pack
conv3d_embed = torch.ops.sgl_kernel.conv3d_embed_cpu


class TestConv3dEmbed(CustomTestCase):
    def _check_shape(
        self,
        *,
        batch_size: int,
        out_channels: int,
        in_channels: int,
        kernel_size: tuple[int, int, int],
    ) -> None:
        dtype = torch.bfloat16
        input_tensor = torch.randn(batch_size, in_channels, *kernel_size, dtype=dtype)
        weight = torch.randn(out_channels, in_channels, *kernel_size, dtype=dtype)
        bias = torch.randn(out_channels, dtype=dtype)

        expected = F.conv3d(input_tensor, weight, bias).flatten(1)
        packed_weight = conv3d_embed_weight_pack(weight)
        actual = conv3d_embed(
            input_tensor,
            packed_weight,
            bias,
            True,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_glm_ocr_k_tail(self):
        """The GLM-OCR patch embedding requires a 24-element K tail."""
        self._check_shape(
            batch_size=4,
            out_channels=1024,
            in_channels=3,
            kernel_size=(2, 14, 14),
        )

    def test_aligned_k(self):
        self._check_shape(
            batch_size=4,
            out_channels=32,
            in_channels=1,
            kernel_size=(2, 4, 4),
        )

    def test_k_smaller_than_tile(self):
        self._check_shape(
            batch_size=4,
            out_channels=32,
            in_channels=1,
            kernel_size=(1, 2, 2),
        )


if __name__ == "__main__":
    unittest.main()
