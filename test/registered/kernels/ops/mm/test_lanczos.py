# SPDX-License-Identifier: Apache-2.0
"""Numerical and storage-lifetime checks for GPU RGB Lanczos preprocessing."""

import unittest

import numpy as np
import torch
from PIL import Image

from sglang.kernels.ops.mm.process.lanczos import lanczos_resize
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def pillow_reference(array, height, width, *, crop=None, mean=None, std=None):
    image = Image.fromarray(array)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    if crop is not None:
        top, left, crop_h, crop_w = crop
        image = image.crop((left, top, left + crop_w, top + crop_h))
    array = np.asarray(image).copy()
    if mean is not None:
        array = (
            array.astype(np.float32) / 255.0 - np.asarray(mean, dtype=np.float32)
        ) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(array)


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
                    output = lanczos_resize(torch.from_numpy(array).cuda(), *target)
                    expected = pillow_reference(array, *target)
                    self.assertTrue(torch.equal(output.cpu(), expected))
                    self.assertEqual(output.shape, (*target, 3))
                    self.assertEqual(output.dtype, torch.uint8)
                    self.assertTrue(output.is_contiguous())

    def test_optional_fusions_match_resize_then_crop_and_normalize(self):
        """Crop uses full-resize coordinates; normalization is channel-specific."""
        rng = np.random.default_rng(139)
        array = rng.integers(0, 256, (61, 93, 3), dtype=np.uint8)
        image = torch.from_numpy(array).cuda()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        for crop in (None, (3, 5, 37, 71)):
            for normalize in (False, True):
                with self.subTest(crop=crop, normalize=normalize):
                    options = dict(
                        crop=crop,
                        mean=mean if normalize else None,
                        std=std if normalize else None,
                    )
                    output = lanczos_resize(image, 47, 101, **options)
                    expected = pillow_reference(array, 47, 101, **options)
                    self.assertTrue(torch.equal(output.cpu(), expected))
                    self.assertEqual(
                        output.dtype, torch.float32 if normalize else torch.uint8
                    )
                    self.assertTrue(output.is_contiguous())

    def test_padded_rows_and_borrowed_input_lifetime(self):
        rng = np.random.default_rng(72)
        array = rng.integers(0, 256, (61, 93, 3), dtype=np.uint8)
        storage = torch.empty((61, 110, 3), device="cuda", dtype=torch.uint8)
        borrowed = storage[:, :93]
        borrowed.copy_(torch.from_numpy(array))
        self.assertFalse(borrowed.is_contiguous())
        output = lanczos_resize(borrowed, 40, 56)
        # The output must own its pixels when a producer reuses its allocation.
        torch.cuda.current_stream().synchronize()
        borrowed.zero_()
        self.assertTrue(torch.equal(output.cpu(), pillow_reference(array, 40, 56)))

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            lanczos_resize(torch.zeros((8, 8, 3), dtype=torch.uint8), 8, 8)
        with self.assertRaises(ValueError):
            lanczos_resize(
                torch.zeros((8, 8, 4), device="cuda", dtype=torch.uint8), 8, 8
            )
        with self.assertRaises(ValueError):
            lanczos_resize(torch.zeros((8, 8, 3), device="cuda"), 8, 8)
        image = torch.zeros((8, 8, 3), device="cuda", dtype=torch.uint8)
        with self.assertRaises(ValueError):
            lanczos_resize(image.unsqueeze(0)[:0], 8, 8)
        with self.assertRaises(ValueError):
            lanczos_resize(image, 0, 8)
        for crop in ((0, 0, 9, 8), (-1, 0, 8, 8), (0, 0, 0, 8), (0, 0, 8)):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                lanczos_resize(image, 8, 8, crop=crop)
        for options in (
            {"mean": (0.5, 0.5, 0.5)},
            {"std": (0.5, 0.5, 0.5)},
            {"mean": (0.5,), "std": (0.5,)},
            {"mean": (0.5, 0.5, 0.5), "std": (1.0, 0.0, 1.0)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                lanczos_resize(image, 8, 8, **options)

    def test_batch_matches_independent_frames_with_padded_strides(self):
        """Batch offsets must not mix frames, including gaps between rows/images."""
        rng = np.random.default_rng(458)
        for batch_size in (1, 2, 5, 8):
            array = rng.integers(0, 256, (batch_size, 31, 47, 3), dtype=np.uint8)
            storage = torch.empty(
                (batch_size * 2, 35, 53, 3), device="cuda", dtype=torch.uint8
            )
            images = storage[::2, :31, :47]
            images.copy_(torch.from_numpy(array))
            for crop in (None, (2, 3, 17, 29)):
                for normalize in (False, True):
                    with self.subTest(
                        batch_size=batch_size, crop=crop, normalize=normalize
                    ):
                        options = dict(
                            crop=crop,
                            mean=(0.485, 0.456, 0.406) if normalize else None,
                            std=(0.229, 0.224, 0.225) if normalize else None,
                        )
                        output = lanczos_resize(images, 23, 39, **options)
                        expected = torch.stack(
                            [
                                pillow_reference(frame, 23, 39, **options)
                                for frame in array
                            ]
                        )
                        self.assertTrue(torch.equal(output.cpu(), expected))
                        self.assertTrue(output.is_contiguous())
                        self.assertEqual(
                            output.dtype, torch.float32 if normalize else torch.uint8
                        )


if __name__ == "__main__":
    unittest.main()
