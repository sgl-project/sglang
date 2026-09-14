"""Unit tests for parallel image preprocessing.

The parallel processor must produce the stock ``Qwen2VLImageProcessorPil`` output
bit for bit, stay off unless ``SGLANG_MM_IMAGE_PREPROCESS_THREADS`` enables it,
and fall back to the stock path for anything it does not plan for.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import copy
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.multimodal.processors import parallel_image_preprocess as parallel
from sglang.test.test_utils import CustomTestCase

Qwen2VLImageProcessorPil = parallel.Qwen2VLImageProcessorPil

# Mixed sizes (below min, above max, portrait, landscape) and modes, so the
# per-image row plan and RGB conversion are both exercised.
_SIZES_AND_MODES = [
    ((700, 500), "RGB"),
    ((300, 900), "L"),
    ((20, 30), "RGBA"),
    ((1500, 1200), "RGB"),
    ((512, 512), "P"),
    ((33, 77), "RGB"),
    ((1280, 1804), "RGB"),
    ((640, 480), "L"),
]


def _images():
    rng = np.random.default_rng(0)
    images = []
    for (width, height), mode in _SIZES_AND_MODES:
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        images.append(Image.fromarray(pixels).convert(mode))
    return images


def _processor():
    return Qwen2VLImageProcessorPil(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        size={"shortest_edge": 64 * 64, "longest_edge": 768 * 768},
    )


def _enabled_processor(threads=4):
    holder = types.SimpleNamespace(image_processor=_processor())
    with envs.SGLANG_MM_IMAGE_PREPROCESS_THREADS.override(threads):
        enabled = parallel.maybe_enable_parallel_image_preprocess(holder)
    return holder.image_processor, enabled


@unittest.skipIf(Qwen2VLImageProcessorPil is None, "transformers lacks the PIL backend")
class TestParallelImagePreprocess(CustomTestCase):
    def _assert_same(self, expected, actual):
        self.assertEqual(expected["pixel_values"].dtype, actual["pixel_values"].dtype)
        self.assertTrue(torch.equal(expected["pixel_values"], actual["pixel_values"]))
        self.assertTrue(
            torch.equal(expected["image_grid_thw"], actual["image_grid_thw"])
        )

    def test_output_is_bit_identical_to_stock(self):
        images = _images()
        processor, enabled = _enabled_processor()
        self.assertTrue(enabled)
        self.assertIsInstance(processor, Qwen2VLImageProcessorPil)
        expected = _processor()(images, return_tensors="pt")
        with envs.SGLANG_MM_IMAGE_PREPROCESS_THREADS.override(4):
            with patch.object(
                parallel, "_preprocess_parallel", wraps=parallel._preprocess_parallel
            ) as spy:
                actual = processor(images, return_tensors="pt")
        spy.assert_called_once()
        self._assert_same(expected, actual)

    def test_disabled_by_default(self):
        holder = types.SimpleNamespace(image_processor=_processor())
        with envs.SGLANG_MM_IMAGE_PREPROCESS_THREADS.override(0):
            self.assertFalse(parallel.maybe_enable_parallel_image_preprocess(holder))
        self.assertIs(type(holder.image_processor), Qwen2VLImageProcessorPil)

    def test_deepcopy_keeps_parallel_class(self):
        processor, _ = _enabled_processor()
        self.assertIs(
            type(copy.deepcopy(processor)), parallel.ParallelQwen2VLImageProcessorPil
        )

    def test_unsupported_inputs_use_stock_path(self):
        images = _images()
        processor, _ = _enabled_processor()
        arrays = [np.asarray(image.convert("RGB")) for image in images]
        with envs.SGLANG_MM_IMAGE_PREPROCESS_THREADS.override(4):
            with patch.object(
                parallel,
                "_preprocess_parallel",
                side_effect=AssertionError("parallel path used"),
            ):
                self._assert_same(
                    _processor()(arrays, return_tensors="pt"),
                    processor(arrays, return_tensors="pt"),
                )
                single = images[:1]
                self._assert_same(
                    _processor()(single, return_tensors="pt"),
                    processor(single, return_tensors="pt"),
                )

    def test_wrong_row_plan_falls_back_to_stock(self):
        images = _images()
        processor, _ = _enabled_processor()
        expected = _processor()(images, return_tensors="pt")

        def wrong_plan(*args, **kwargs):
            rows = parallel_plan(*args, **kwargs)
            return [rows[0]] + [count + 1 for count in rows[1:]]

        parallel_plan = parallel._plan_rows
        with envs.SGLANG_MM_IMAGE_PREPROCESS_THREADS.override(4):
            with patch.object(parallel, "_plan_rows", side_effect=wrong_plan):
                with self.assertLogs(parallel.logger, level="WARNING"):
                    actual = processor(images, return_tensors="pt")
        self._assert_same(expected, actual)


if __name__ == "__main__":
    unittest.main()
