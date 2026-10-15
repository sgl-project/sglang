"""Muse Spark video preprocessing follows the checkpoint's processor_config.json.

Regression: the frame cap, per-frame token budget, and timestamp template were
hardcoded (96, 144, {s:.1f}) instead of read from the video_processor block.
"""

import asyncio
import json
import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.muse_spark import MuseSpark1xOmniConfig
from sglang.srt.layers.muse_spark_navit import _patch_grid_positions
from sglang.srt.multimodal.processors.muse_spark_omni import (
    MuseSpark1xOmniMultimodalProcessor,
    PreprocessConfig,
    _video_frames,
    find_resize_target,
    preprocess_image,
)
from sglang.srt.runtime_context import get_context, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The video_processor values a released Muse Spark checkpoint ships.
CHECKPOINT_PROCESSOR_CONFIG = {
    "video_processor": {
        "fps": 2.0,
        "frame_timestamp_format": "Time: {timestamp:.2f}s",
        "max_video_frame_tokens": 256,
        "num_frames": 600,
    }
}
SPECIAL_TOKENS = (
    "<|image_start|>",
    "<|image_end|>",
    "<|patch|>",
    "<|vid_start|>",
    "<|vid_end|>",
    "<|vid_frame_separator|>",
    "<|video|>",
    "<|audio|>",
    "<|soft_audio_start|>",
    "<|soft_audio_end|>",
)
SPECIAL_PATTERN = "(" + "|".join(map(re.escape, SPECIAL_TOKENS)) + ")"
# Above every Unicode code point, so special ids never collide with characters.
SPECIAL_ID_BASE = 0x110000


class _CharTokenizer:
    """Stands in for the checkpoint tokenizer: one id per character or special."""

    unk_token_id = None

    def convert_tokens_to_ids(self, token):
        return SPECIAL_ID_BASE + SPECIAL_TOKENS.index(token)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in re.split(SPECIAL_PATTERN, text):
            if part in SPECIAL_TOKENS:
                ids.append(self.convert_tokens_to_ids(part))
            else:
                ids.extend(map(ord, part))
        return ids

    def decode(self, ids):
        return "".join(
            SPECIAL_TOKENS[i - SPECIAL_ID_BASE] if i >= SPECIAL_ID_BASE else chr(i)
            for i in ids
        )


class TestMuseSparkVideoProcessorConfig(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _make_processor(
        self, processor_config_text=None, disable_fast_image_processor=True
    ):
        model_dir = tempfile.TemporaryDirectory()
        self.addCleanup(model_dir.cleanup)
        if processor_config_text is not None:
            path = os.path.join(model_dir.name, "processor_config.json")
            with open(path, "w") as f:
                f.write(processor_config_text)
        # The PIL backend keeps preprocessing on CPU without the optional Rust path.
        override = get_context().override_server_args(
            model_path=model_dir.name,
            disable_fast_image_processor=disable_fast_image_processor,
            mm_process_config={},
        )
        server_args = override.install()
        self.addCleanup(override.restore)
        processor = MuseSpark1xOmniMultimodalProcessor(
            MuseSpark1xOmniConfig(),
            server_args,
            _CharTokenizer(),
            "default",
            skip_mm_pool=True,
        )
        self.addCleanup(processor.shutdown)
        return processor

    def _process(self, processor, pretokenized):
        # A frame tensor is read at 1 fps, so 200 frames request 200 * sampling fps
        # samples (400 at 2 fps, where only the frame cap binds); 448px frames form
        # a 16x16 token grid.
        videos = [
            torch.zeros(200, 3, 28, 28, dtype=torch.uint8),
            torch.zeros(2, 3, 448, 448, dtype=torch.uint8),
        ]
        if pretokenized:
            request = SimpleNamespace(input_ids=_CharTokenizer().encode("<video>" * 2))
            coro = processor.process_mm_data_async(
                video_data=videos, request_obj=request
            )
        else:
            coro = processor.process_mm_data_async(
                input_text="<video>" * 2, video_data=videos
            )
        output = asyncio.run(coro)
        frames = [item.feature.shape[0] for item in output.mm_items]
        return frames, _CharTokenizer().decode(output.input_ids)

    def test_checkpoint_video_config_drives_sampling_budget_and_timestamps(self):
        processor = self._make_processor(json.dumps(CHECKPOINT_PROCESSOR_CONFIG))
        long_video = (
            "<|vid_start|>"
            + "<|vid_frame_separator|>".join(
                f"Time: {second}.00s<|video|>" for second in range(0, 200, 2)
            )
            + "<|vid_end|>"
        )
        large_frame_video = (
            "<|vid_start|>Time: 0.00s" + "<|video|>" * 256 + "<|vid_end|>"
        )
        for pretokenized in (False, True):
            with self.subTest(pretokenized=pretokenized):
                frames, prompt = self._process(processor, pretokenized)
                self.assertEqual(frames, [200, 2])
                self.assertEqual(prompt, long_video + large_frame_video)

    def test_checkpoint_fps_sets_the_sampling_rate(self):
        video_processor = {**CHECKPOINT_PROCESSOR_CONFIG["video_processor"], "fps": 0.5}
        processor = self._make_processor(
            json.dumps({"video_processor": video_processor})
        )
        for pretokenized in (False, True):
            with self.subTest(pretokenized=pretokenized):
                frames, _ = self._process(processor, pretokenized)
                self.assertEqual(frames, [100, 2])

    def test_checkpoint_without_video_config_keeps_legacy_behavior(self):
        for name, processor_config_text in (
            ("no processor_config.json", None),
            ("no video_processor block", json.dumps({"image_processor": {}})),
        ):
            processor = self._make_processor(processor_config_text)
            for pretokenized in (False, True):
                with self.subTest(name, pretokenized=pretokenized):
                    frames, prompt = self._process(processor, pretokenized)
                    self.assertEqual(frames, [96, 2])
                    self.assertTrue(
                        prompt.startswith("<|vid_start|>Time: 0.0s<|video|><|vid_")
                    )
                    self.assertTrue(
                        prompt.endswith(
                            "<|vid_start|>Time: 0.0s"
                            + "<|video|>" * 144
                            + "<|vid_end|>"
                        )
                    )

    def test_fast_backend_keeps_preprocessing_on_cpu(self):
        """Regression: on CUDA hosts the fast backend preprocessed media on the serving
        GPU inside the tokenizer process, so concurrent long videos ran out of memory."""
        module = "sglang.srt.multimodal.processors.muse_spark_omni"
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch(f"{module}.is_cuda", return_value=True, create=True),
        ):
            processor = self._make_processor(disable_fast_image_processor=False)
        self.assertFalse(processor.preprocess_config.use_gpu_preprocess)
        self.assertFalse(processor._preprocessing_competes_with_the_scheduler())

    def test_malformed_processor_config_is_an_error(self):
        with self.assertRaises(json.JSONDecodeError):
            self._make_processor('{"video_processor": ')


class _IndexDecoder:
    avg_fps = 30.0

    def __init__(self, total: int) -> None:
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> int:
        return index


class TestMuseSparkVideoFrameSampling(CustomTestCase):
    """Regression: a frame pick fell one frame short of the checkpoint processor's when
    its evenly spaced position landed just below an integer (114.99999 became 114)."""

    def test_frame_indices_match_checkpoint_processor(self):
        expected = [0, 16, 32, 49, 65, 82, 98, 115, 131, 147, 164]
        expected += [180, 197, 213, 230, 246, 262, 279, 295, 312, 328, 345]
        config = PreprocessConfig(num_frames_per_video=600, sampling_fps=2.0)
        videos = {
            "decoder": _IndexDecoder(346),
            "decoded": (list(range(346)), {"fps": 30.0}),
        }
        for source, video in videos.items():
            with self.subTest(source=source):
                frames, timestamps = _video_frames(video, config)
                self.assertEqual(frames, expected)
                self.assertEqual([round(t * 30) for t in timestamps], expected)


class TestMuseSparkImagePreprocess(CustomTestCase):
    """Regression: equal-distortion grids broke ties unlike the checkpoint processor,
    and narrow images lost their pixels when the CHW layout was re-guessed."""

    def test_equal_distortion_tie_matches_checkpoint_processor(self):
        self.assertEqual(
            find_resize_target(
                image_h=2121,
                image_w=3360,
                patch_size=28,
                minimum_tokens=1,
                maximum_tokens=4096,
            ),
            (1428, 2240, 4080),
        )

    def test_narrow_images_keep_pixels(self):
        rgb = torch.tensor([10, 120, 240], dtype=torch.uint8)
        expected = (rgb.float() / 255 - 0.5) / 0.5
        images = {
            "hwc": rgb.expand(60, 3, 3).numpy().copy(),
            "chw": rgb[:, None, None].expand(3, 60, 3).clone(),
        }
        for layout, image in images.items():
            with self.subTest(layout=layout):
                feature, _ = preprocess_image(
                    image, PreprocessConfig(use_rust_resize=False)
                )
                self.assertGreater(feature.shape[1], feature.shape[2])
                torch.testing.assert_close(
                    feature.flatten(1),
                    expected[:, None].expand(3, feature[0].numel()),
                )


POSITION_TABLE_SIDE = 32


def _reference_position_cells(grid_h: int, grid_w: int) -> torch.Tensor:
    patch = torch.arange(grid_h * grid_w, dtype=torch.float64)
    row = torch.div(patch, grid_h, rounding_mode="floor") + 0.5
    col = patch % grid_h + 0.5
    return torch.stack(
        (
            row * POSITION_TABLE_SIDE / grid_w - 0.5,
            col * POSITION_TABLE_SIDE / grid_h - 0.5,
        ),
        dim=-1,
    )


class TestMuseSparkNaViTPositionGrid(CustomTestCase):
    """Regression: non-square grids read the wrong position-table cells (ij instead of
    the checkpoint's xy meshgrid), and bf16 sampling coordinates drifted."""

    def test_sampling_coordinates_follow_checkpoint_layout(self):
        for grid_h, grid_w in ((24, 40), (40, 24), (46, 46), (58, 90)):
            with self.subTest(grid_h=grid_h, grid_w=grid_w):
                positions = _patch_grid_positions(
                    grid_h=grid_h, grid_w=grid_w, device=torch.device("cpu")
                )
                self.assertEqual(positions.dtype, torch.float32)
                cells = ((positions.double() + 1) * POSITION_TABLE_SIDE - 1) / 2
                torch.testing.assert_close(
                    cells.flip(-1),
                    _reference_position_cells(grid_h, grid_w),
                    atol=1e-5,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
