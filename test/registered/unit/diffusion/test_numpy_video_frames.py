# SPDX-License-Identifier: Apache-2.0
"""Parallel video materialization preserves frame order, clipping and dtype casts."""

import unittest
from unittest.mock import patch

import numpy as np

from sglang.multimodal_gen.runtime.entrypoints import utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# This CPU-only logic imports diffusion modules, whose dependencies are installed
# in the diffusion unit-test job.
register_cuda_ci(est_time=5, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


class TestNumpyVideoFrames(CustomTestCase):
    def test_parallel_conversion_matches_sequential_for_strided_frames(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                sample = np.linspace(-0.5, 1.5, 4 * 17 * 29 * 3, dtype=dtype).reshape(
                    4, 17, 29, 3
                )
                sample = sample[:, ::2, ::-1]
                before = sample.copy()
                expected = (np.clip(sample, 0, 1) * 255).astype(np.uint8)
                with patch.object(utils, "_PARALLEL_NUMPY_VIDEO_MIN_BYTES", 0):
                    frames = utils._sample_to_uint8_frames(sample)
                np.testing.assert_array_equal(np.stack(frames), expected)
                np.testing.assert_array_equal(sample, before)
                self.assertTrue(all(frame.dtype == np.uint8 for frame in frames))

    def test_small_images_and_uint8_videos_avoid_thread_pool(self):
        for sample in (
            np.zeros((17, 29, 3), dtype=np.float32),
            np.zeros((2, 17, 29, 3), dtype=np.float32),
            np.full((4, 17, 29, 3), 123, dtype=np.uint8),
        ):
            with (
                self.subTest(shape=sample.shape, dtype=sample.dtype),
                patch.object(
                    utils,
                    "ThreadPoolExecutor",
                    side_effect=AssertionError("unneeded threads"),
                ),
            ):
                frames = utils._sample_to_uint8_frames(sample)
                expected = sample[None] if sample.ndim == 3 else sample
                np.testing.assert_array_equal(
                    np.stack(frames), expected.astype(np.uint8)
                )


if __name__ == "__main__":
    unittest.main()
