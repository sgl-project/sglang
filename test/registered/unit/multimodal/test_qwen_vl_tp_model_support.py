"""CPU contracts for the model families sharing rank-local image preprocessing."""

import asyncio
import base64
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig
from sglang.srt.configs.qwen3_vl import Qwen3VLConfig, Qwen3VLMoeConfig
from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.environ import envs
from sglang.srt.managers.mm_schedule import _uses_qwen_vl_tp_image_preprocessing
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
from sglang.srt.models import qwen3_vl
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors import qwen_vl
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.multimodal.qwen_vl_tp_preprocess import (
    DEFERRED_IMAGE_PATCH_WIDTH_KEY,
    _decode_owner_image,
)
from sglang.srt.runtime_context import get_model, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


MODEL_FAMILIES = (
    ("qwen3_vl", Qwen3VLForConditionalGeneration, Qwen3VLConfig),
    ("qwen3_vl_moe", Qwen3VLMoeForConditionalGeneration, Qwen3VLMoeConfig),
    ("qwen3_5", Qwen3_5ForConditionalGeneration, Qwen3_5Config),
    ("qwen3_5_moe", Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeConfig),
    ("qwen4_exp", Qwen4ExpForConditionalGeneration, Qwen4ExpConfig),
)


def _image_bytes(size, image_format):
    image = Image.new("RGB", size, (17, 31, 254))
    ImageDraw.Draw(image).rectangle(
        (1, 7, size[0] // 2, size[1] // 3), fill=(230, 85, 6)
    )
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


class TestQwenVLTPModelSupport(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.args = ServerArgs(
            model_path="dummy",
            model_impl="sglang",
            tp_size=2,
            mm_feature_transport="cpu",
            mm_process_config={},
            mm_preprocess_cache_size_mb=0,
            mm_processor_worker_num=1,
            mm_io_worker_num=1,
            cuda_graph_config=CudaGraphConfig(),
        )
        publish(self.args, role="test")
        self.enterContext(
            get_parallel().override(
                pp_group=SimpleNamespace(world_size=1, is_last_rank=True),
                tp_group=SimpleNamespace(world_size=2),
                attn_dp_size=1,
                attn_cp_size=1,
            )
        )

    @staticmethod
    def _hf_processor():
        vocab = {
            "<unk>": 0,
            "<|vision_start|>": 1,
            "<|image_pad|>": 2,
            "<|vision_end|>": 3,
            "hello": 4,
            "<|video_pad|>": 5,
            "<pad>": 6,
        }
        backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        backend.decoder = decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<pad>",
            additional_special_tokens=list(vocab)[1:4] + ["<|video_pad|>"],
        )
        return Qwen2VLProcessor(
            image_processor=Qwen2VLImageProcessor(
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                size={"shortest_edge": 32 * 32, "longest_edge": 256 * 256},
            ),
            video_processor=Qwen2VLVideoProcessor(),
            tokenizer=tokenizer,
        )

    def _processor(self, model_type, enabled, model_cls=None):
        # A real config is needed by the processor fingerprint. Only image
        # metadata is used here; no model weights or device allocations are made.
        if model_cls is None:
            model_cls, config_cls = next(
                (m, c) for t, m, c in MODEL_FAMILIES if t == model_type
            )
        else:
            config_cls = Qwen3_5MoeConfig
        hf_config = config_cls(
            architectures=[model_cls.__name__],
            vision_start_token_id=1,
            image_token_id=2,
            vision_end_token_id=3,
            video_token_id=5,
            vision_config={"spatial_merge_size": 2, "patch_size": 16},
        )
        hf_config.model_type = model_type
        hf_processor = self._hf_processor()
        with (
            envs.SGLANG_QWEN_VL_TP_IMAGE_PREPROCESS.override(enabled),
            patch.object(qwen_vl, "is_cuda", return_value=True),
        ):
            processor = QwenVLImageProcessor(
                hf_config, self.args, hf_processor, None, skip_mm_pool=True
            )
        self.addCleanup(processor.shutdown)
        return processor, hf_processor

    def test_future_shared_vision_models_inherit_support_and_can_opt_out(self):
        class FutureQwenWithSharedVision(Qwen3_5MoeForConditionalGeneration):
            pass

        class FutureQwenWithDifferentVision(FutureQwenWithSharedVision):
            supports_tp_image_preprocessing = False

        for model_cls, expected in (
            (FutureQwenWithSharedVision, True),
            (FutureQwenWithDifferentVision, False),
        ):
            with (
                self.subTest(model_cls=model_cls.__name__),
                patch.object(QwenVLImageProcessor, "models", [model_cls]),
            ):
                processor, _ = self._processor(
                    "future_qwen", enabled=True, model_cls=model_cls
                )
                self.assertEqual(processor._tp_image_preprocess, expected)
                self.assertEqual(processor.uses_media_artifacts_without_cache, expected)

    def test_new_families_keep_compressed_images_without_a_preprocess_cache(self):
        for model_type, _, _ in MODEL_FAMILIES:
            with self.subTest(model_type=model_type):
                processor, hf_processor = self._processor(model_type, enabled=True)
                self.assertTrue(processor._tp_image_preprocess)
                self.assertFalse(processor.mm_preprocess_cache.enabled)
                self.assertIsNotNone(processor.processor_fingerprint)
                images = [
                    _image_bytes((64, 96), "JPEG"),
                    _image_bytes((160, 96), "PNG"),
                ]
                request = SimpleNamespace(
                    video_data=None, audio_data=None, mm_content_hashes=None
                )
                with patch.object(
                    processor,
                    "_fast_image_processor_device",
                    return_value="cuda:0",
                ):
                    result = asyncio.run(
                        processor.process_mm_data_async(
                            image_data=[
                                "data:image/png;base64," + base64.b64encode(b).decode()
                                for b in images
                            ],
                            input_text="hello <image> hello <image>",
                            request_obj=request,
                        )
                    )
                self.assertEqual([item.feature for item in result.mm_items], images)
                processed_bytes = 0
                for item in result.mm_items:
                    self.assertIn(
                        DEFERRED_IMAGE_PATCH_WIDTH_KEY, item.model_specific_data
                    )
                    self.assertEqual(
                        item.model_specific_data[DEFERRED_IMAGE_PATCH_WIDTH_KEY], 1536
                    )
                    decoded = _decode_owner_image(item.feature, torch.device("cpu"))
                    native = hf_processor.image_processor(
                        images=[decoded], return_tensors="pt", device="cpu"
                    )
                    processed_bytes += (
                        native.pixel_values.numel() * native.pixel_values.element_size()
                    )
                    torch.testing.assert_close(
                        item.image_grid_thw, native.image_grid_thw
                    )
                    self.assertEqual(
                        item.offsets[0][1] - item.offsets[0][0] + 1,
                        int(native.image_grid_thw.prod()) // 4,
                    )
                self.assertLess(
                    sum(len(item.feature) for item in result.mm_items), processed_bytes
                )
                expected_positions = processor.compute_mrope_positions(
                    result.input_ids, result.mm_items
                )
                torch.testing.assert_close(
                    result.mrope_positions, expected_positions[0]
                )
                torch.testing.assert_close(
                    result.mrope_position_delta, expected_positions[1]
                )

    def test_opt_out_and_unsupported_families_keep_the_standard_path(self):
        for model_type in ("qwen3_5", "qwen3_5_moe", "qwen4_exp"):
            with self.subTest(model_type=model_type):
                processor, _ = self._processor(model_type, enabled=False)
                self.assertFalse(processor._tp_image_preprocess)
                self.assertFalse(processor.uses_media_artifacts_without_cache)
        processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        processor._supports_tp_image_preprocess = False
        for model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_omni_moe", "unknown"):
            processor.model_type = model_type
            with envs.SGLANG_QWEN_VL_TP_IMAGE_PREPROCESS.override(True):
                self.assertFalse(processor._enable_tp_image_preprocessing(None, None))

    def test_inherited_capability_keeps_device_and_topology_guards(self):
        processor, hf_processor = self._processor("qwen3_5_moe", enabled=True)
        with (
            envs.SGLANG_QWEN_VL_TP_IMAGE_PREPROCESS.override(True),
            patch.object(qwen_vl, "is_cuda", return_value=False),
            self.assertRaisesRegex(ValueError, "single-node pure CUDA TP"),
        ):
            processor._enable_tp_image_preprocessing(self.args, hf_processor)
        with (
            envs.SGLANG_QWEN_VL_TP_IMAGE_PREPROCESS.override(True),
            patch.object(qwen_vl, "is_cuda", return_value=True),
            self.assertRaisesRegex(ValueError, "single-node pure CUDA TP"),
        ):
            processor._enable_tp_image_preprocessing(
                SimpleNamespace(nnodes=2), hf_processor
            )
        with (
            envs.SGLANG_QWEN_VL_TP_IMAGE_PREPROCESS.override(True),
            patch.object(qwen_vl, "is_cuda", return_value=True),
            get_model().override(model_impl="transformers"),
            self.assertRaisesRegex(ValueError, "native SGLang"),
        ):
            processor._enable_tp_image_preprocessing(self.args, hf_processor)

    def test_inherited_encoder_and_graph_policy_follow_the_outer_model(self):
        for model_type, model_cls, _ in MODEL_FAMILIES:
            for enabled in (False, True):
                with self.subTest(model_type=model_type, enabled=enabled):
                    model = model_cls.__new__(model_cls)
                    config = SimpleNamespace(
                        model_type=model_type,
                        vision_config=SimpleNamespace(deepstack_visual_indexes=[]),
                        text_config=SimpleNamespace(rope_scaling={}),
                        encoder_only=True,
                    )
                    with (
                        envs.SGLANG_QWEN_VL_TP_IMAGE_PREPROCESS.override(enabled),
                        patch.object(qwen3_vl, "Qwen3VLMoeVisionModel") as vision_cls,
                        patch.object(qwen3_vl, "LogitsProcessor"),
                        patch.object(qwen3_vl, "Pooler"),
                    ):
                        Qwen3VLForConditionalGeneration.__init__(
                            model, config, language_model_cls=object
                        )
                        self.assertEqual(
                            vision_cls.call_args.kwargs["use_bounded_cuda_graph"],
                            enabled,
                        )
                        self.assertEqual(
                            _uses_qwen_vl_tp_image_preprocessing(
                                model.get_image_feature
                            ),
                            enabled,
                        )
                        self.assertFalse(
                            _uses_qwen_vl_tp_image_preprocessing(
                                model.get_video_feature
                            )
                        )
                        self.assertIs(model.config, config.text_config)

    def test_shared_image_entry_dispatches_compressed_inputs_for_derived_models(self):
        from sglang.srt.managers.req_types import MultimodalDataItem

        for _, model_cls, _ in MODEL_FAMILIES:
            with self.subTest(model_cls=model_cls.__name__):
                model = model_cls.__new__(model_cls)
                torch.nn.Module.__init__(model)
                model.tp_image_preprocessing_enabled = True
                model.visual = SimpleNamespace(
                    device=torch.device("cpu"), dtype=torch.bfloat16
                )
                item = MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=_image_bytes((64, 96), "PNG"),
                    model_specific_data={DEFERRED_IMAGE_PATCH_WIDTH_KEY: 1536},
                )
                expected = torch.ones(6, 8, dtype=torch.bfloat16)
                with (
                    envs.SGLANG_QWEN_VL_TP_IMAGE_PREPROCESS.override(True),
                    patch(
                        "sglang.srt.multimodal.qwen_vl_tp_preprocess.encode_tp_images",
                        return_value=[expected],
                    ) as encode,
                ):
                    actual = model.get_image_feature([item])
                self.assertIs(actual[0], expected)
                self.assertEqual(encode.call_args.args[0], [item])
                self.assertEqual(
                    encode.call_args.args[4:6], (torch.device("cpu"), torch.bfloat16)
                )


if __name__ == "__main__":
    unittest.main()
