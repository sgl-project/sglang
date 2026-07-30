import unittest

import torch

from sglang.kernels.ops.mm.process.image import (
    _normalize_and_patchify_torch,
    normalize_and_patchify,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestNormalizeAndPatchify(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_triton_matches_torch_with_padding(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                image = torch.randn(
                    (2, 3, 17, 19),
                    generator=generator,
                    device="cuda",
                    dtype=dtype,
                )
                scale = torch.randn(
                    (1, 3, 1, 1),
                    generator=generator,
                    device="cuda",
                    dtype=dtype,
                )
                bias = torch.randn(
                    (1, 3, 1, 1),
                    generator=generator,
                    device="cuda",
                    dtype=dtype,
                )
                image_before = image.clone()

                actual = normalize_and_patchify(
                    image, scale, bias, patch_size=4, padded_height=20, padded_width=20
                )
                expected = _normalize_and_patchify_torch(
                    image, scale, bias, patch_size=4, padded_height=20, padded_width=20
                )

                self.assertEqual(tuple(actual.shape), (2, 25, 3, 4, 4))
                torch.testing.assert_close(
                    actual.float(), expected.float(), rtol=1e-2, atol=1e-2
                )
                self.assertTrue(torch.equal(image, image_before))

    def test_noncontiguous_input_uses_equivalent_fallback(self):
        image = torch.randn((1, 3, 8, 10), device="cuda")[:, :, :, ::2]
        scale = torch.ones((1, 3, 1, 1), device="cuda")
        bias = torch.zeros((1, 3, 1, 1), device="cuda")

        actual = normalize_and_patchify(
            image, scale, bias, patch_size=2, padded_height=8, padded_width=6
        )
        expected = _normalize_and_patchify_torch(
            image, scale, bias, patch_size=2, padded_height=8, padded_width=6
        )

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
