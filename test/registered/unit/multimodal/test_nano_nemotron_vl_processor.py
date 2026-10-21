"""Unit tests for the Nano Nemotron VL processor."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

from sglang.srt.multimodal.evs import EVSConfig, EVSProcessor
from sglang.srt.multimodal.processors.nano_nemotron_vl import (
    NanoNemotronVLImageProcessor,
    _create_visual_items,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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
