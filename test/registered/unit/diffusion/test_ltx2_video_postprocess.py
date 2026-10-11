# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from diffusers.video_processor import VideoProcessor

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.decoding_av import (
    LTX2AVDecodingStage,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


class TestLTX2VideoPostprocess(CustomTestCase):
    def test_preserves_float_samples_and_byte_frames(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for normalize in (False, True):
                for channels_last in (False, True):
                    with self.subTest(
                        dtype=dtype, normalize=normalize, channels_last=channels_last
                    ):
                        video = torch.linspace(
                            -2, 2, 2 * 3 * 7 * 16 * 24, device="cuda", dtype=dtype
                        ).reshape(2, 3, 7, 16, 24)
                        if channels_last:
                            video = video.contiguous(
                                memory_format=torch.channels_last_3d
                            )
                        before = video.clone()
                        processor = VideoProcessor(
                            vae_scale_factor=32, do_normalize=normalize
                        )
                        stage = SimpleNamespace(video_processor=processor)
                        expected = processor.postprocess_video(video, output_type="np")
                        actual = LTX2AVDecodingStage._postprocess_video(stage, video)
                        self.assertEqual(actual.dtype, np.float32)
                        self.assertEqual(actual.shape, (2, 7, 16, 24, 3))
                        self.assertTrue(actual.flags.c_contiguous)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(
                            (np.clip(actual, 0, 1) * 255).astype(np.uint8),
                            (np.clip(expected, 0, 1) * 255).astype(np.uint8),
                        )
                        torch.testing.assert_close(video, before, rtol=0, atol=0)

    def test_pinned_outputs_remain_owned_by_their_numpy_arrays(self):
        stage = SimpleNamespace(video_processor=VideoProcessor(vae_scale_factor=32))
        first = torch.zeros(1, 3, 5, 8, 8, device="cuda", dtype=torch.bfloat16)
        result = LTX2AVDecodingStage._postprocess_video(stage, first)
        expected = result.copy()
        for value in (1.0, -1.0):
            first.fill_(value)
            second = LTX2AVDecodingStage._postprocess_video(stage, first)
            np.testing.assert_array_equal(result, expected)
            self.assertFalse(np.shares_memory(result, second))

    def test_pin_allocation_failure_falls_back_to_cpu_copy(self):
        stage = SimpleNamespace(video_processor=VideoProcessor(vae_scale_factor=32))
        video = torch.randn(1, 3, 5, 8, 8, device="cuda", dtype=torch.bfloat16)
        expected = stage.video_processor.postprocess_video(video, output_type="np")
        with patch(
            "torch.empty", side_effect=RuntimeError("pin allocation unavailable")
        ):
            actual = LTX2AVDecodingStage._postprocess_video(stage, video)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
