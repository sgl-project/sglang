"""Real HF spatial preprocessing must survive the pre-sampled input boundary."""

import asyncio
import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.multimodal.processors.glm4v import glm_sample_and_decode_sync
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.utils import load_video
from sglang.srt.utils.pre_sampled_video import PreSampledVideo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestPreSampledVideoProcessors(CustomTestCase):
    def test_spatial_processing_and_encoder_parity(self):
        for model_type, processor_cls, patch_size in (
            ("qwen3_vl", Qwen3VLVideoProcessor, 16),
            ("glm4v", Glm4vVideoProcessor, 14),
        ):
            for count in (3, 4):
                with (
                    self.subTest(model=model_type, count=count),
                    ThreadPoolExecutor(max_workers=1) as executor,
                ):
                    factor = patch_size * 2
                    frames = np.random.default_rng(19).integers(
                        0,
                        256,
                        (count, factor * 2 + 5, factor * 3 + 7, 3),
                        dtype=np.uint8,
                    )
                    indices = [0, 3, 17, 29][:count]
                    video = load_video(PreSampledVideo(frames, 30.0, 60, indices))
                    hf = processor_cls(
                        size={
                            "shortest_edge": factor**2,
                            "longest_edge": (factor * 4) ** 2,
                        }
                    )
                    _, metadata = video.to_processor_inputs()
                    # Sampling is forbidden, but actual resize/normalize/patchify runs.
                    with patch.object(
                        hf,
                        "sample_frames",
                        side_effect=AssertionError("HF sampled supplied frames"),
                    ):
                        reference = hf(
                            videos=[frames],
                            video_metadata=[copy.deepcopy(metadata)],
                            do_sample_frames=False,
                            return_tensors="pt",
                            device="cpu",
                        )
                        if model_type == "qwen3_vl":
                            data, metadata = asyncio.run(preprocess_video(video))
                        else:
                            data, metadata = glm_sample_and_decode_sync(video)
                        actual = hf(
                            videos=[data],
                            video_metadata=[metadata],
                            do_sample_frames=False,
                            return_tensors="pt",
                            device="cpu",
                        )
                        for key in ("pixel_values_videos", "video_grid_thw"):
                            torch.testing.assert_close(
                                actual[key], reference[key], rtol=0, atol=0
                            )
                        self.assertEqual(
                            int(actual["video_grid_thw"][0, 0]), (count + 1) // 2
                        )
                        self.assertTrue(
                            actual["pixel_values_videos"].is_floating_point()
                        )

                        encoder = EncoderPreprocessor.__new__(EncoderPreprocessor)
                        encoder.model_type = model_type
                        encoder.io_executor = executor
                        encoder.preproc_executor = executor
                        encoder.vision_config = {}
                        encoder.video_processor = hf
                        encoder.server_args = SimpleNamespace(
                            mm_enable_dp_encoder=False
                        )
                        encoder.model_config = SimpleNamespace(
                            hf_config=SimpleNamespace(
                                vision_config=SimpleNamespace(spatial_merge_size=2)
                            )
                        )
                        encoded = asyncio.run(
                            encoder._process_video_items([video.to_wire()], None)
                        )
                        for key in ("pixel_values_videos", "video_grid_thw"):
                            torch.testing.assert_close(
                                torch.as_tensor(encoded[key]),
                                reference[key],
                                rtol=0,
                                atol=0,
                            )
                        if model_type == "qwen3_vl":
                            self.assertAlmostEqual(
                                encoded["video_timestamps"][0][0], 0.05
                            )
                            last = (17 + (29 if count == 4 else 17)) / 60
                            self.assertAlmostEqual(
                                encoded["video_timestamps"][0][1], last
                            )

    def test_glm_encoder_mixed_pre_sampled_and_legacy_frames(self):
        frames = np.random.default_rng(23).integers(
            0, 256, (4, 56, 84, 3), dtype=np.uint8
        )
        video = PreSampledVideo(frames, 30.0, 60, [0, 3, 17, 29])
        legacy = [
            {"frame_image": Image.fromarray(frame), "timestamp": index / 30}
            for frame, index in zip(frames, video.frame_indices)
        ]
        hf = Glm4vVideoProcessor()
        reference_data, reference_metadata = zip(
            glm_sample_and_decode_sync(video), glm_sample_and_decode_sync(legacy)
        )
        reference = hf(
            videos=list(reference_data),
            video_metadata=list(reference_metadata),
            do_sample_frames=False,
            return_tensors="pt",
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            encoder = EncoderPreprocessor.__new__(EncoderPreprocessor)
            encoder.model_type = "glm4v"
            encoder.io_executor = encoder.preproc_executor = executor
            encoder.vision_config = {}
            encoder.video_processor = hf
            encoder.server_args = SimpleNamespace(mm_enable_dp_encoder=False)
            with patch.object(
                hf, "sample_frames", side_effect=AssertionError("Unexpected sampling")
            ):
                actual = asyncio.run(
                    encoder._process_video_items([video.to_wire(), legacy], None)
                )
        for key in ("pixel_values_videos", "video_grid_thw"):
            torch.testing.assert_close(
                torch.as_tensor(actual[key]), reference[key], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
