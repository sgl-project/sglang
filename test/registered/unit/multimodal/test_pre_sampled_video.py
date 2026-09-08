"""Preserve externally selected video frames across native and encoder inputs."""

import asyncio
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.disaggregation.encoder.receiver import _encoder_media_item
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.glm4v import glm_sample_and_decode_sync
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.utils import load_video
from sglang.srt.utils.pre_sampled_video import PreSampledVideo, load_pre_sampled_video
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def sample(**overrides):
    args = dict(
        frames=np.random.default_rng(7).integers(
            0, 256, (4, 32, 48, 3), dtype=np.uint8
        ),
        source_fps=30.0,
        total_num_frames=60,
        frame_indices=[0, 3, 17, 29],
        timestamps=[0, 3 / 30, 17 / 30, 29 / 30],
    )
    args.update(overrides)
    return PreSampledVideo(**args)


class TestPreSampledVideo(CustomTestCase):
    def test_lossless_json_round_trip_preserves_timeline(self):
        original = sample()
        with patch(
            "sglang.srt.utils.common.VideoDecoderWrapper",
            side_effect=AssertionError("codec called"),
        ):
            loaded = load_video(json.loads(json.dumps(original.to_wire())))
        np.testing.assert_array_equal(loaded.frames, original.frames)
        self.assertEqual(loaded.frame_indices, original.frame_indices)
        self.assertEqual(loaded.timestamps, original.timestamps)
        _, metadata = loaded.to_processor_inputs()
        self.assertEqual(metadata["fps"], 30.0)
        self.assertEqual(metadata["duration"], 2.0)
        self.assertEqual(metadata["total_num_frames"], 60)
        self.assertEqual(metadata["frames_indices"], [0, 3, 17, 29])

    def test_in_process_representations(self):
        original = sample()
        for frames in (
            original.frames,
            torch.from_numpy(original.frames),
            [Image.fromarray(frame) for frame in original.frames],
        ):
            with self.subTest(type=type(frames)):
                loaded = load_video(sample(frames=frames))
                np.testing.assert_array_equal(loaded.frames, original.frames)

    def test_repeated_indices_are_not_deduplicated(self):
        video = sample(frame_indices=[0, 0, 3, 3], timestamps=None)
        loaded = load_video(video)
        self.assertEqual(loaded.frame_indices, [0, 0, 3, 3])
        self.assertEqual(len(loaded.frames), 4)

    def test_invalid_metadata_is_rejected(self):
        invalid = [
            {"source_fps": 0},
            {"source_fps": float("nan")},
            {"source_fps": True},
            {"total_num_frames": 1.5},
            {"total_num_frames": 0},
            {"frame_indices": [0, 1, 2]},
            {"frame_indices": [0, 3, 2, 4]},
            {"frame_indices": [0, 3, 17, 60]},
            {"frame_indices": [False, 3, 17, 29]},
            {"frame_indices": np.array(1)},
            {"frame_indices": np.zeros((4, 1))},
            {"timestamps": [0, 1, 2, 3]},
            {"timestamps": [0, float("inf"), 0, 0]},
            {"do_sample_frames": True},
            {"do_sample_frames": 0},
            {"frames": []},
            {"frames": np.zeros((4, 3))},
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                load_pre_sampled_video(sample(**args))

    def test_invalid_rgb_frames_are_rejected(self):
        invalid = [
            np.zeros((4, 32, 48, 3), dtype=np.float32),
            np.zeros((4, 3, 32, 48), dtype=np.uint8),
            np.zeros((4, 0, 48, 3), dtype=np.uint8),
            [Image.new("RGBA", (48, 32))] * 4,
            [Image.new("RGB", (48, 32)), Image.new("RGB", (49, 32))] * 2,
        ]
        for frames in invalid:
            with (
                self.subTest(shape=getattr(frames, "shape", None)),
                self.assertRaises(ValueError),
            ):
                load_pre_sampled_video(sample(frames=frames))

    def test_wire_rejects_external_frame_urls_and_unknown_fields(self):
        for update in (
            {"frames": ["https://example.com/frame.png"] * 4},
            {"frames": ["/etc/passwd"] * 4},
            {"unknown": 1},
            {"frames": [[0]] * 4},
        ):
            wire = sample().to_wire()
            wire.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                load_pre_sampled_video(wire)

    def test_native_request_schema_and_batching(self):
        app = FastAPI()

        @app.post("/generate")
        def receive(request: GenerateReqInput):
            request.normalize_batch_and_arguments()
            return {"is_single": request.is_single, "video_data": request.video_data}

        wire = sample().to_wire()
        with TestClient(app) as client:
            response = client.post(
                "/generate", json={"text": "video", "video_data": wire}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["video_data"], wire)
            self.assertEqual(client.get("/openapi.json").status_code, 200)
        request = GenerateReqInput(text=["first", "second"], video_data=[wire, wire])
        request.normalize_batch_and_arguments()
        self.assertEqual(request[0].video_data, wire)
        self.assertEqual(request[1].video_data, wire)

    def test_unsupported_processor_rejects_before_loading(self):
        processor = SimpleNamespace(supports_pre_sampled_video=False)
        with self.assertRaisesRegex(ValueError, "does not support"):
            asyncio.run(
                BaseMultimodalProcessor.load_mm_data(
                    processor,
                    prompt="video",
                    multimodal_tokens=None,
                    video_data=[sample().to_wire()],
                )
            )

    def test_qwen_and_glm_skip_temporal_sampling(self):
        video = load_video(sample())
        with patch(
            "sglang.srt.multimodal.processors.qwen_vl.smart_nframes",
            side_effect=AssertionError("sampled again"),
        ):
            frames, metadata = asyncio.run(
                preprocess_video(video, video_config={"nframes": 2})
            )
        torch.testing.assert_close(
            frames, torch.from_numpy(video.frames).permute(0, 3, 1, 2)
        )
        self.assertEqual(metadata["frames_indices"], [0, 3, 17, 29])
        with patch(
            "sglang.srt.multimodal.processors.glm4v.glm_sample_frame_indices",
            side_effect=AssertionError("sampled again"),
        ):
            frames, metadata = glm_sample_and_decode_sync(video, {"max_frames": 2})
        np.testing.assert_array_equal(frames, video.frames)
        self.assertEqual(metadata["frames_indices"], [0, 3, 17, 29])

    def test_http_encoder_payload_and_preprocessing(self):
        for model_type in ("qwen3_vl", "glm4v"):
            with (
                self.subTest(model_type=model_type),
                ThreadPoolExecutor(max_workers=1) as executor,
            ):
                processor = EncoderPreprocessor.__new__(EncoderPreprocessor)
                processor.model_type = model_type
                processor.io_executor = executor
                processor.vision_config = {}
                processor.video_processor = None
                processor.server_args = SimpleNamespace(mm_enable_dp_encoder=False)
                wire = _encoder_media_item(
                    {"url": sample(), "modality": Modality.VIDEO}
                )
                wire = json.loads(json.dumps(wire))
                videos, kwargs = asyncio.run(processor._flatten_and_load_videos([wire]))
                self.assertEqual(len(videos[0]), 4)
                self.assertFalse(kwargs["do_sample_frames"])
                self.assertEqual(
                    kwargs["video_metadata"][0]["frames_indices"], [0, 3, 17, 29]
                )
                self.assertEqual(kwargs["video_metadata"][0]["fps"], 30.0)


if __name__ == "__main__":
    unittest.main()
