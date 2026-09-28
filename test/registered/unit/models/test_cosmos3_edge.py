# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# CPU coverage for Cosmos3-Edge checkpoint mapping and video prompt preparation.

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import asyncio
import unittest

import numpy as np
import torch

from sglang.srt.configs.cosmos3 import Cosmos3EdgeConfig
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
)
from sglang.srt.models.cosmos3_edge import Cosmos3EdgeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.cosmos3_edge import (
    Cosmos3EdgeProcessor,
    _smart_resize,
)
from sglang.test.test_utils import CustomTestCase


class TestCosmos3EdgeConfig(CustomTestCase):
    def test_checkpoint_without_pad_token_has_explicit_none(self):
        config = Cosmos3EdgeConfig(text_config={"eos_token_id": 11})

        self.assertTrue(hasattr(config.text_config, "pad_token_id"))
        self.assertIsNone(config.text_config.pad_token_id)
        self.assertEqual(config.text_config.eos_token_id, 11)
        self.assertFalse(config.text_config.tie_word_embeddings)
        self.assertTrue(hasattr(config, "pad_token_id"))
        self.assertIsNone(config.pad_token_id)
        self.assertEqual(config.eos_token_id, 11)


class TestCosmos3EdgeWeightsMapper(CustomTestCase):
    def setUp(self):
        self.mapper = Cosmos3EdgeForConditionalGeneration.hf_to_sglang_mapper

    def test_text_tower_is_renamed_for_arcee(self):
        inputs = [
            "embed_tokens.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.self_attn.to_q.weight",
            "layers.0.self_attn.to_k.weight",
            "layers.0.self_attn.to_v.weight",
            "layers.0.self_attn.to_out.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "norm.weight",
            "lm_head.weight",
        ]
        expected = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        self.assertEqual(self.mapper.apply_list(inputs), expected)

    def test_generation_and_routed_vision_weights_are_dropped(self):
        dropped = [
            "layers.0.self_attn.k_norm_und_for_gen.weight",
            "layers.0.self_attn.add_q_proj.weight",
            "layers.0.self_attn.to_add_out.weight",
            "layers.0.mlp_moe_gen.up_proj.weight",
            "proj_in.weight",
            "time_embedder.linear_1.weight",
            "model.visual.encoder.layers.0.self_attn.q_proj.weight",
            "model.projector.linear_fc1.weight",
        ]
        self.assertEqual(self.mapper.apply_list(dropped), [])


class TestCosmos3EdgeVideoSampling(CustomTestCase):
    def _processor(self, video_config=None):
        processor = object.__new__(Cosmos3EdgeProcessor)
        processor.video_config = video_config or {}
        return processor

    def test_default_sampling_matches_qwen3_vl(self):
        processor = self._processor()
        indices = processor._select_frame_indices(total_frames=300, video_fps=30.0)
        expected = np.linspace(0, 299, num=20).round().astype(np.int64).tolist()
        self.assertEqual(indices, expected)

    def test_default_sampling_clamps_to_frame_limits(self):
        processor = self._processor()
        self.assertEqual(
            len(processor._select_frame_indices(total_frames=30, video_fps=30.0)),
            4,
        )
        self.assertEqual(
            len(processor._select_frame_indices(total_frames=15_000, video_fps=10.0)),
            768,
        )

    def test_num_frames_and_legacy_nframes(self):
        num_frames = self._processor({"num_frames": 5})
        nframes = self._processor({"nframes": 5})
        expected = [0, 25, 50, 74, 99]
        self.assertEqual(num_frames._select_frame_indices(100, 30.0), expected)
        self.assertEqual(nframes._select_frame_indices(100, 30.0), expected)

    def test_explicit_frame_count_and_fps_are_mutually_exclusive(self):
        processor = self._processor({"num_frames": 5, "fps": 2.0})
        with self.assertRaisesRegex(ValueError, "Specify only one"):
            processor._select_frame_indices(100, 30.0)

    def test_missing_source_fps_uses_24_fps(self):
        processor = self._processor()
        self.assertEqual(
            len(processor._select_frame_indices(total_frames=240, video_fps=0.0)),
            20,
        )


class TestCosmos3EdgeResize(CustomTestCase):
    def test_video_max_pixels_is_a_total_temporal_budget(self):
        height, width = _smart_resize(
            1024,
            1024,
            factor=32,
            min_pixels=4096,
            max_pixels=4 * 1024 * 1024,
            num_frames=16,
        )
        self.assertEqual((height, width), (512, 512))
        self.assertLessEqual(16 * height * width, 4 * 1024 * 1024)

    def test_image_resize_keeps_single_frame_semantics(self):
        height, width = _smart_resize(
            1024,
            1024,
            factor=32,
            min_pixels=4096,
            max_pixels=512 * 512,
        )
        self.assertEqual((height, width), (512, 512))


class _TimestampTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        timestamp = float(text.removeprefix("<").split()[0])
        return [1000 + int(timestamp * 10)]


class TestCosmos3EdgePromptExpansion(CustomTestCase):
    def _processor(self):
        processor = object.__new__(Cosmos3EdgeProcessor)
        processor.IMAGE_TOKEN_ID = 19
        processor.VIDEO_TOKEN_ID = 18
        processor.IM_START_TOKEN_ID = 20
        processor.IM_END_TOKEN_ID = 21
        processor._spatial_merge_size = 2
        processor._tokenizer = _TimestampTokenizer()
        return processor

    def test_video_placeholder_expands_once_per_frame(self):
        processor = self._processor()
        prompt = [7, 20, 18, 21, 8]
        video_grid_thw = torch.tensor([[3, 4, 4]], dtype=torch.long)
        timestamps = [[0.0, 0.5, 1.0]]

        input_ids, offsets, modalities = processor._build_input_ids(
            prompt,
            img_grid_thw=None,
            video_grid_thw=video_grid_thw,
            video_timestamps=timestamps,
        )

        expected = [7]
        expected_offsets = []
        for timestamp_id in (1000, 1005, 1010):
            expected.extend([timestamp_id, 20])
            offset_start = len(expected)
            expected.extend([18] * 4)
            expected_offsets.append((offset_start, len(expected) - 1))
            expected.append(21)
        expected.append(8)

        self.assertEqual(input_ids, expected)
        self.assertEqual(offsets, [expected_offsets])
        self.assertEqual(modalities, [Modality.VIDEO])

    def test_expanded_video_mrope_matches_qwen3_vl(self):
        processor = self._processor()
        video_grid_thw = torch.tensor([[3, 4, 4]], dtype=torch.long)
        input_ids, _, _ = processor._build_input_ids(
            [7, 20, 18, 21, 8],
            img_grid_thw=None,
            video_grid_thw=video_grid_thw,
            video_timestamps=[[0.0, 0.5, 1.0]],
        )
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        kwargs = dict(
            spatial_merge_size=2,
            image_token_id=19,
            video_token_id=18,
            vision_start_token_id=20,
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
        )
        edge_positions, edge_delta = MRotaryEmbedding.get_rope_index(
            model_type="cosmos3_edge", **kwargs
        )
        qwen_positions, qwen_delta = MRotaryEmbedding.get_rope_index(
            model_type="qwen3_vl", **kwargs
        )

        self.assertEqual(edge_positions.shape, (3, 1, len(input_ids[0])))
        self.assertTrue(torch.equal(edge_positions, qwen_positions))
        self.assertTrue(torch.equal(edge_delta, qwen_delta))


def _processed_input_processor():
    processor = object.__new__(Cosmos3EdgeProcessor)
    processor.IMAGE_TOKEN_ID = 19
    processor.VIDEO_TOKEN_ID = 18
    processor.IM_START_TOKEN_ID = 20
    processor.IM_END_TOKEN_ID = 21
    processor.vision_start_token_id = 20
    processor.model_type = "cosmos3_edge"
    processor._spatial_merge_size = 2
    processor._tokenizer = _TimestampTokenizer()
    processor._processor = processor._tokenizer
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token="<image>",
        video_token="<video>",
        image_token_id=19,
        video_token_id=18,
    )
    processor.mm_processor_executor = None
    processor.use_cuda_ipc = False
    processor.precompute_hash_before_cpu_transfer = False
    processor.ATTR_NAME_TO_MODALITY = {
        "pixel_values": Modality.IMAGE,
        "image_grid_thw": Modality.IMAGE,
        "pixel_values_videos": Modality.VIDEO,
        "video_grid_thw": Modality.VIDEO,
    }
    processor.FEATURE_NAMES = ["pixel_values", "pixel_values_videos"]
    return processor


class TestCosmos3EdgeProcessedInputs(CustomTestCase):
    def test_image_processor_output_keeps_pixels_for_vision_tower(self):
        processor = _processed_input_processor()
        input_ids = [7, 20, 19, 19, 19, 19, 21, 8]
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        pixels = torch.randn(16, 768)
        processor_data = {
            "format": "processor_output",
            "input_ids": input_ids,
            "pixel_values": pixels,
            "image_grid_thw": grid,
        }
        base_output = BaseMultiModalProcessorOutput(
            input_text="",
            input_ids=input_ids,
            images=[processor_data],
        )

        output = asyncio.run(processor._process_preprocessed_mm_data(base_output))

        self.assertEqual(output.input_ids, input_ids)
        self.assertEqual(len(output.mm_items), 1)
        item = output.mm_items[0]
        self.assertEqual(item.format, MultimodalInputFormat.PROCESSOR_OUTPUT)
        self.assertIs(item.feature, pixels)
        self.assertEqual(item.offsets, [(2, 5)])
        self.assertEqual(output.mrope_positions.shape, (3, len(input_ids)))

    def test_image_precomputed_embedding_is_preserved(self):
        processor = _processed_input_processor()
        input_ids = [7, 20, 19, 19, 19, 19, 21, 8]
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        embeddings = torch.randn(4, 2048)
        processor_data = {
            "format": "precomputed_embedding",
            "input_ids": input_ids,
            "feature": embeddings,
            "image_grid_thw": grid,
        }
        base_output = BaseMultiModalProcessorOutput(
            input_text="",
            input_ids=input_ids,
            images=[processor_data],
        )

        output = asyncio.run(processor._process_preprocessed_mm_data(base_output))

        item = output.mm_items[0]
        self.assertEqual(item.format, MultimodalInputFormat.PRECOMPUTED_EMBEDDING)
        self.assertIs(item.feature, embeddings)
        self.assertEqual(item.offsets, [(2, 5)])
        self.assertEqual(output.mrope_positions.shape, (3, len(input_ids)))

    def test_video_processor_output_retains_per_frame_offsets(self):
        processor = _processed_input_processor()
        grid = torch.tensor([[3, 4, 4]], dtype=torch.long)
        input_ids, offsets, _ = processor._build_input_ids(
            [7, 20, 18, 21, 8],
            img_grid_thw=None,
            video_grid_thw=grid,
            video_timestamps=[[0.0, 0.5, 1.0]],
        )
        pixels = torch.randn(48, 768)
        processor_data = {
            "format": "processor_output",
            "input_ids": input_ids,
            "pixel_values_videos": pixels,
            "video_grid_thw": grid,
        }
        base_output = BaseMultiModalProcessorOutput(
            input_text="",
            input_ids=input_ids,
            videos=[processor_data],
        )

        output = asyncio.run(processor._process_preprocessed_mm_data(base_output))

        self.assertEqual(len(output.mm_items), 1)
        item = output.mm_items[0]
        self.assertEqual(item.format, MultimodalInputFormat.PROCESSOR_OUTPUT)
        self.assertTrue(torch.equal(item.feature, pixels))
        self.assertEqual(item.offsets, offsets[0])
        self.assertEqual(output.mrope_positions.shape, (3, len(input_ids)))

    def test_video_precomputed_embedding_is_preserved(self):
        processor = _processed_input_processor()
        grid = torch.tensor([[3, 4, 4]], dtype=torch.long)
        input_ids, offsets, _ = processor._build_input_ids(
            [7, 20, 18, 21, 8],
            img_grid_thw=None,
            video_grid_thw=grid,
            video_timestamps=[[0.0, 0.5, 1.0]],
        )
        embeddings = torch.randn(12, 2048)
        processor_data = {
            "format": "precomputed_embedding",
            "input_ids": input_ids,
            "feature": embeddings,
            "video_grid_thw": grid,
        }
        base_output = BaseMultiModalProcessorOutput(
            input_text="",
            input_ids=input_ids,
            videos=[processor_data],
        )

        output = asyncio.run(processor._process_preprocessed_mm_data(base_output))

        self.assertEqual(len(output.mm_items), 1)
        item = output.mm_items[0]
        self.assertEqual(item.format, MultimodalInputFormat.PRECOMPUTED_EMBEDDING)
        self.assertIs(item.feature, embeddings)
        self.assertEqual(item.offsets, offsets[0])
        self.assertEqual(output.mrope_positions.shape, (3, len(input_ids)))


class TestCosmos3EdgePrecomputedVisionFeatures(CustomTestCase):
    def _model(self):
        model = Cosmos3EdgeForConditionalGeneration.__new__(
            Cosmos3EdgeForConditionalGeneration
        )
        torch.nn.Module.__init__(model)
        model.language_model_only = False
        return model

    def test_image_and_video_embeddings_bypass_vision_tower(self):
        model = self._model()
        image_embeddings = torch.randn(4, 2048)
        video_embeddings = torch.randn(12, 2048)
        image_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=image_embeddings,
            format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
        )
        video_item = MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=video_embeddings,
            format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
        )

        self.assertTrue(
            torch.equal(model.get_image_feature([image_item]), image_embeddings)
        )
        self.assertTrue(
            torch.equal(model.get_video_feature([video_item]), video_embeddings)
        )

    def test_mixed_features_are_rejected(self):
        model = self._model()
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.randn(4, 2048),
                format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
            ),
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.randn(16, 768),
            ),
        ]

        with self.assertRaisesRegex(ValueError, "cannot mix"):
            model.get_image_feature(items)


if __name__ == "__main__":
    unittest.main()
