"""Image normalization and patchification parity."""

import unittest

import torch

from sglang.kernels.ops.mm.process.image import (
    _normalize_and_patchify_torch,
    normalize_and_patchify,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestImagePatchify(CustomTestCase):
    def test_normalize_and_patchify(self):
        torch.manual_seed(5)
        image = torch.randn(2, 3, 17, 19, device="cuda")
        scale = torch.randn(1, 3, 1, 1, device="cuda")
        bias = torch.randn(1, 3, 1, 1, device="cuda")
        args = (image, scale, bias, 4, 20, 20)
        actual = normalize_and_patchify(
            args[0],
            args[1],
            args[2],
            patch_size=args[3],
            padded_height=args[4],
            padded_width=args[5],
        )
        expected = _normalize_and_patchify_torch(
            args[0],
            args[1],
            args[2],
            patch_size=args[3],
            padded_height=args[4],
            padded_width=args[5],
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
