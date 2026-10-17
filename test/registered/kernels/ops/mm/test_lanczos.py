# SPDX-License-Identifier: Apache-2.0
"""Numerical and storage-lifetime checks for GPU RGB Lanczos preprocessing."""

import math
import unittest

import numpy as np
import torch
from PIL import Image

from sglang.kernels.ops.mm.process.lanczos import resize_crop_normalize_rgb
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def pillow_reference(array, height, width):
    image = Image.fromarray(array)
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (math.ceil(scale * image.width), math.ceil(scale * image.height)),
        Image.Resampling.LANCZOS,
    )
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    cropped = resized.crop((left, top, left + width, top + height))
    return (
        torch.from_numpy(np.asarray(cropped, dtype=np.float32) / 127.5 - 1.0)
        .permute(2, 0, 1)
        .contiguous()
    )


class TestLanczosPreprocess(CustomTestCase):
    def test_matches_pillow_geometry_and_rounding(self):
        rng = np.random.default_rng(9843)
        shapes = [
            ((704, 1280), (480, 832)),
            ((480, 832), (704, 1280)),
            ((101, 73), (64, 96)),
            ((17, 31), (37, 21)),
            ((32, 32), (32, 32)),
            ((7, 9), (1, 1)),
            ((1, 1), (24, 40)),
            ((1, 43), (9, 17)),
            ((43, 1), (17, 9)),
            ((21, 17), (16, 16)),
            ((73, 101), (55, 88)),
            ((100, 101), (93, 93)),
        ]
        for source, target in shapes:
            for pattern in ("random", "checkerboard", "impulse"):
                with self.subTest(source=source, target=target, pattern=pattern):
                    array = rng.integers(0, 256, (*source, 3), dtype=np.uint8)
                    if pattern == "checkerboard":
                        checkerboard = (
                            (np.indices(source).sum(axis=0) % 2) * 255
                        ).astype(np.uint8)
                        array = np.repeat(checkerboard[..., None], 3, axis=2)
                    elif pattern == "impulse":
                        array[:] = 0
                        array[source[0] // 2, source[1] // 2] = 255
                    output = resize_crop_normalize_rgb(
                        torch.from_numpy(array).cuda(), *target
                    )
                    expected = pillow_reference(array, *target)
                    self.assertTrue(torch.equal(output.cpu(), expected))
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertTrue(output.is_contiguous())

    def test_padded_rows_and_borrowed_input_lifetime(self):
        rng = np.random.default_rng(72)
        array = rng.integers(0, 256, (61, 93, 3), dtype=np.uint8)
        storage = torch.empty((61, 110, 3), device="cuda", dtype=torch.uint8)
        borrowed = storage[:, :93]
        borrowed.copy_(torch.from_numpy(array))
        self.assertFalse(borrowed.is_contiguous())
        output = resize_crop_normalize_rgb(borrowed, 40, 56)
        # The output must own its pixels when a producer reuses its allocation.
        torch.cuda.current_stream().synchronize()
        borrowed.zero_()
        self.assertTrue(torch.equal(output.cpu(), pillow_reference(array, 40, 56)))

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            resize_crop_normalize_rgb(torch.zeros((8, 8, 3), dtype=torch.uint8), 8, 8)
        with self.assertRaises(ValueError):
            resize_crop_normalize_rgb(
                torch.zeros((8, 8, 4), device="cuda", dtype=torch.uint8), 8, 8
            )
        with self.assertRaises(ValueError):
            resize_crop_normalize_rgb(torch.zeros((8, 8, 3), device="cuda"), 8, 8)
        with self.assertRaises(ValueError):
            resize_crop_normalize_rgb(
                torch.zeros((8, 8, 3), device="cuda", dtype=torch.uint8), 0, 8
            )


if __name__ == "__main__":
    unittest.main()
