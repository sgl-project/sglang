"""MiMo-V2 images are normalized with the checkpoint's image_mean/image_std, as HF does."""

import asyncio
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor

from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.environ import envs
from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.mimo_v2 import (
    ImageInput,
    MiMoProcessor,
    MiMoV2Processor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# Stats from XiaomiMiMo/MiMo-V2.6-* and MiMo-V2.5 preprocessor_config.json.
IMAGENET = dict(image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
CLIP = dict(
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
)

# Subset of a MiMo-V2.6 config.json processor_config the constructor requires.
_PROCESSOR_CONFIG = dict(
    vision_start_token_id=1,
    vision_end_token_id=2,
    image_token_id=3,
    video_token_id=4,
    audio_token_id=5,
    audio_start_token_id=6,
    audio_end_token_id=7,
    video_start_token_id=8,
    video_end_token_id=9,
    audio_sampling_rate=24000,
    image_min_pixels=8192,
    image_max_pixels=8388608,
    use_video_timestamps=True,
)

COLORS = [(255, 0, 0), (128, 128, 128), (255, 255, 255), (12, 200, 77)]


def _base_init(self, hf_config, server_args, _processor, *args, **kwargs):
    self.hf_config = hf_config
    self.server_args = server_args
    self._processor = _processor


def _hf_config():
    return SimpleNamespace(
        vision_config={"patch_size": 16, "spatial_merge_size": 2},
        processor_config=dict(_PROCESSOR_CONFIG),
        audio_config={},
        rope_scaling=None,
    )


def _make_processor(image_processor):
    hf_config = _hf_config()
    tokenizer = SimpleNamespace(
        pad_token_id=0, convert_ids_to_tokens=lambda ids: [f"<{i}>" for i in ids]
    )
    hf_processor = SimpleNamespace(image_processor=image_processor, tokenizer=tokenizer)
    # The base init wires serving infrastructure (runtime context, caches,
    # transport) that has no bearing on preprocessing; GPU preprocessing would
    # read the device from that unpublished runtime context.
    with (
        patch.object(BaseMultimodalProcessor, "__init__", _base_init),
        envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.override(False),
    ):
        return MiMoV2Processor(hf_config, None, hf_processor, None).mimo_processor


def _make_encoder_processor(image_processor):
    """The MiMo processor an encode server builds when a video request arrives first."""
    model = SimpleNamespace(_encoder_processor=None, config=_hf_config())
    hook = types.MethodType(MiMoV2ForCausalLM.preprocess_mm_for_encoder, model)
    preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
    preprocessor.image_processor = image_processor
    preprocessor.vision_config = {}
    # Media decoding is not under test; only the processor the hook builds is.
    with patch.object(MiMoProcessor, "preprocess_for_encoder", return_value=None):
        asyncio.run(preprocessor._process_video_items([], hook))
    return model._encoder_processor


def _channel_values(processor, hf_image_processor, color):
    """Per-channel normalized value of a solid-color image, from sglang and from HF."""
    image = Image.new("RGB", (128, 128), color)
    ours = processor.process_image(ImageInput(image=image))
    pixels = hf_image_processor(images=image, return_tensors="pt")["pixel_values"]
    # HF rows are flattened (channel, temporal, patch_h, patch_w) patches.
    reference = pixels.view(pixels.shape[0], 3, -1).mean(dim=(0, 2))
    return ours.mean(dim=(1, 2)), reference


class TestMiMoV2ImageNormalization(CustomTestCase):
    def _assert_matches_hf(self, processor, hf_image_processor):
        for color in COLORS:
            with self.subTest(color=color):
                ours, reference = _channel_values(processor, hf_image_processor, color)
                torch.testing.assert_close(ours, reference, atol=1e-4, rtol=0)

    def test_each_checkpoint_uses_its_own_stats(self):
        # CLIP first, so a cache keyed only by device would serve CLIP to ImageNet.
        clip_hf = Qwen2VLImageProcessor(**CLIP)
        imagenet_hf = Qwen2VLImageProcessor(**IMAGENET)
        clip, imagenet = _make_processor(clip_hf), _make_processor(imagenet_hf)
        self._assert_matches_hf(clip, clip_hf)
        self._assert_matches_hf(imagenet, imagenet_hf)
        self._assert_matches_hf(clip, clip_hf)

    def test_encode_server_uses_checkpoint_stats(self):
        imagenet_hf = Qwen2VLImageProcessor(**IMAGENET)
        self._assert_matches_hf(_make_encoder_processor(imagenet_hf), imagenet_hf)


if __name__ == "__main__":
    unittest.main()
