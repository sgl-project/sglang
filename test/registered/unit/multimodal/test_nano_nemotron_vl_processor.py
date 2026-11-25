"""Unit tests for the Nano Nemotron VL processor."""

import asyncio
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from transformers import ProcessorMixin

from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from sglang.srt.multimodal.evs import EVSConfig, EVSProcessor
from sglang.srt.multimodal.internvl_utils import get_video_target_size_and_feature_size
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.nano_nemotron_vl import (
    NanoNemotronVLImageProcessor,
    _create_visual_items,
    _resolve_video_sizing,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestNanoNemotronVLProcessor(CustomTestCase):
    def _tokenized_processor(self):
        processor = object.__new__(NanoNemotronVLImageProcessor)
        tokenizer = Mock()
        tokenizer.convert_ids_to_tokens.side_effect = {
            0: "<unk>",
            18: "<image>",
            19: "<img>",
            20: "</img>",
        }.get
        tokenizer.decode.return_value = "<img><image><image></img>"
        tokenizer.encode.return_value = [19, 18, 18, 20]
        processor.__dict__.update(
            tokenizer=tokenizer,
            IMG_CONTEXT_TOKEN="<image>",
            IMG_START_TOKEN="<img>",
            IMG_END_TOKEN="</img>",
            VIDEO_CONTEXT_TOKEN="<video>",
            AUDIO_CONTEXT_TOKEN="<audio>",
            img_start_token_id=19,
            img_end_token_id=20,
            PLACEHOLDER_ID=0,
            audio_extractor=None,
            video_temporal_patch_size=1,
            num_image_token=4,
            dynamic_resolution=True,
            max_model_len=128,
            patch_size=16,
            downsample_ratio=0.5,
            min_num_patches=1,
            max_num_patches=4,
            norm_mean=(0.5, 0.5, 0.5),
            norm_std=(0.5, 0.5, 0.5),
            mm_tokens=SimpleNamespace(
                image_token_id=18, video_token_id=0, audio_token_id=None
            ),
            evs=SimpleNamespace(
                static_size_data_items=lambda **kwargs: (lambda **kwargs: [], [])
            ),
            load_mm_data=AsyncMock(
                return_value=SimpleNamespace(
                    images=[SimpleNamespace(width=32, height=32)] * 2,
                    videos=[],
                    audios=[],
                )
            ),
        )
        return processor

    def test_tokenized_images_preserve_text_ids_and_budget(self):
        processor = self._tokenized_processor()
        ids = [0, 101, 18, 7, 18, 102, 103]
        module = "sglang.srt.multimodal.processors.nano_nemotron_vl"
        with (
            patch(
                f"{module}.compute_budgeted_image_sizes", return_value=[(32, 32, 2)] * 2
            ) as budget,
            patch(
                f"{module}.resize_image_to_pixels", return_value=torch.zeros(3, 32, 32)
            ),
        ):
            result = asyncio.run(
                processor.process_mm_data_async(
                    [object(), object()], [], ids, SimpleNamespace(video_data=[])
                )
            )
        self.assertEqual(
            result.input_ids, [0, 101, 19, 18, 18, 20, 7, 19, 18, 18, 20, 102, 103]
        )
        self.assertEqual(budget.call_args.args[1], 128 - (len(ids) - 2))
        self.assertEqual(
            [item.offsets for item in result.mm_items], [[(3, 4)], [(8, 9)]]
        )
        self.assertEqual(
            processor.load_mm_data.call_args.kwargs["prompt"], "<image><image>"
        )

    def test_video_marker_cleanup_preserves_text_unknown_ids(self):
        processor = self._tokenized_processor()
        processor.PLACEHOLDER = "<unk>"
        processor.video_target_num_patches = 0
        processor.mm_tokens.video_token_id = 21
        processor.tokenizer.convert_ids_to_tokens.side_effect = {
            0: "<unk>",
            18: "<image>",
            21: "<video>",
        }.get
        processor.tokenizer.decode.return_value = "<unk><video>text"
        processor.tokenizer.encode.return_value = [0, 18, 18, 18, 18, 20]
        processor.load_mm_data.return_value = SimpleNamespace(
            images=[], videos=[object()], audios=[]
        )
        processor.parse_video = Mock(
            return_value=(torch.zeros(1, 2, 2, 3, dtype=torch.uint8).numpy(), [0.0])
        )
        processor.preprocess_image = Mock(return_value=torch.zeros(1, 3, 2, 2))
        processor.evs = object.__new__(EVSProcessor)
        processor.evs.evs_config = None

        result = asyncio.run(
            processor.process_mm_data_async(
                [], [], [0, 21, 7], SimpleNamespace(video_data=[object()])
            )
        )
        self.assertEqual(result.input_ids, [0, 19, 18, 18, 18, 18, 20, 7])
        self.assertEqual(result.mm_items[0].offsets, [(2, 5)])

    def test_tokenized_image_count_checked_before_loading(self):
        processor = self._tokenized_processor()
        with self.assertRaisesRegex(ValueError, "image placeholder count"):
            asyncio.run(
                processor.process_mm_data_async(
                    [object(), object()], [], [18], SimpleNamespace(video_data=[])
                )
            )
        processor.load_mm_data.assert_not_awaited()

    def test_visual_offsets_ignore_literal_boundaries_and_preserve_audio(self):
        model = object.__new__(NemotronH_Nano_VL_V2)
        for text, start in (
            ([5, 19, 18, 18, 20, 7, 20], 2),
            ([5, 19, 9, 19, 18, 18, 20, 7], 4),
            ([19, 4, 4, 20, 5, 19, 18, 18, 20], 6),
        ):
            with self.subTest(text=text):
                mm = SimpleNamespace(
                    mm_items=[
                        MultimodalDataItem(
                            modality=Modality.IMAGE,
                            offsets=[(start, start + 1)],
                            pad_value=1_000_001,
                        ),
                        MultimodalDataItem(
                            modality=Modality.AUDIO,
                            offsets=[(len(text) + 1, len(text) + 2)],
                            pad_value=1_000_002,
                        ),
                    ],
                    data_offsets=[],
                    im_start_id=19,
                    im_end_id=20,
                    im_token_id=18,
                    video_token_id=18,
                    audio_token_id=32,
                    audio_start_id=30,
                    audio_end_id=31,
                )
                padded = model.pad_input_ids(array("q", text + [30, 32, 32, 31]), mm)
                expected = text[:]
                expected[start : start + 2] = [1_000_001] * 2
                self.assertEqual(
                    list(padded), expected + [30, 1_000_002, 1_000_002, 31]
                )
                self.assertEqual(mm.data_offsets, [start - 1, len(text)])

    def test_video_size_falls_back_to_preprocessor_config(self):
        """A video patch target set only in preprocessor_config.json must be used."""
        hf_processor = Mock(spec=ProcessorMixin, image_processor=Mock())
        hf_processor.image_processor.to_dict.return_value = {
            "video_target_num_patches": 1024,
            "video_maintain_aspect_ratio": True,
        }
        tokenizer = Mock()
        for config_target, processor, expected in (
            (0, hf_processor, (1024, True)),
            (512, hf_processor, (512, False)),
            (0, tokenizer, (0, False)),
        ):
            with self.subTest(config_target=config_target, processor=processor):
                hf_config = SimpleNamespace(
                    video_target_num_patches=config_target,
                    video_maintain_aspect_ratio=False,
                )
                self.assertEqual(
                    _resolve_video_sizing(hf_config=hf_config, hf_processor=processor),
                    expected,
                )

    def test_video_frame_size_matches_checkpoint_processor(self):
        """Frame sizes from the checkpoints' processors; none exceeds the patch target."""
        for (width, height, target, keep_aspect), expected in (
            ((720, 480, 1024, True), (608, 416, 247)),
            ((4096, 2160, 1024, True), (704, 352, 242)),
            ((640, 360, 1000, False), (480, 480, 225)),
        ):
            with self.subTest(size=(width, height), target=target):
                self.assertEqual(
                    get_video_target_size_and_feature_size(
                        orig_w=width,
                        orig_h=height,
                        target_num_patches=target,
                        maintain_aspect_ratio=keep_aspect,
                        patch_size=16,
                        downsample_ratio=0.5,
                    ),
                    expected,
                )

    def test_videos_with_different_frame_sizes_are_rejected(self):
        """Two videos that resize to different frame sizes must fail as a bad request."""
        processor = self._tokenized_processor()
        processor.__dict__.update(
            video_temporal_patch_size=2,
            video_target_num_patches=1024,
            video_maintain_aspect_ratio=True,
        )
        processor.mm_tokens.video_token_id = 21
        processor.tokenizer.convert_ids_to_tokens.side_effect = {
            18: "<image>",
            21: "<video>",
        }.get
        processor.load_mm_data.return_value = SimpleNamespace(
            images=[], videos=[object(), object()], audios=[]
        )
        processor.parse_video = Mock(
            side_effect=[
                (torch.zeros(2, 360, 640, 3, dtype=torch.uint8).numpy(), [0.0, 0.5]),
                (torch.zeros(2, 480, 720, 3, dtype=torch.uint8).numpy(), [0.0, 0.5]),
            ]
        )
        with self.assertRaisesRegex(ValueError, "same frame size"):
            asyncio.run(
                processor.process_mm_data_async(
                    [], [], [21, 21], SimpleNamespace(video_data=[object(), object()])
                )
            )

    def test_tubelet_label_matches_hf_reference(self):
        """Multi-frame tubelet labels must match the HF processor's prompt format."""
        processor = object.__new__(NanoNemotronVLImageProcessor)
        processor.PLACEHOLDER = "<unk>"
        processor.IMG_CONTEXT_TOKEN = "<image>"
        processor.IMG_END_TOKEN = "</img>"

        rendered = processor.render_tubelet(
            1, frame_indices=[2, 3], timestamps=[1.0, 1.5], num_tokens=2
        )

        self.assertEqual(
            rendered,
            "Frame 3 sampled at 1.00 seconds and frame 4 sampled at 1.50 seconds: "
            "<unk><image><image></img>",
        )

    def test_evs_video_next_to_dynamic_images_keeps_evs_metadata(self):
        """A video sent with dynamic-resolution images must still carry EVS metadata."""
        evs = EVSProcessor.__new__(EVSProcessor)
        evs.evs_config = EVSConfig(video_pruning_rate=0.5)
        create_data_items, _ = evs.static_size_data_items(
            frames_per_video=[2], num_images=1, rows=2, cols=2
        )
        input_ids = [1, 2, 3]

        image, video = _create_visual_items(
            create_data_items=create_data_items,
            image_feature=[torch.zeros(3, 32, 48)],
            image_offsets=[(0, 0)],
            num_tokens_per_image=[6],
            video_feature=torch.zeros(3, 3, 32, 32),
            video_offsets=[(1, 1), (2, 2)],
            frames_per_video=[3],
            input_ids_list=input_ids,
        )

        self.assertTrue(image.is_dynamic)
        self.assertEqual(image.num_tokens, 6)
        self.assertEqual(video.thw_grids, [(2, 2, 2)])
        self.assertEqual(video.pre_chunked_input_ids, input_ids)
        self.assertEqual(video.frames_per_video, [3])


if __name__ == "__main__":
    unittest.main()
