# SPDX-License-Identifier: Apache-2.0
"""Guard input rounding, crop geometry and strides across CPU/CUDA preprocessing."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.flux import Flux2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPlusPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3ImagePreprocessStage,
    _pil_to_normalized_tensor,
    _resize_crop_pil,
)
from sglang.multimodal_gen.runtime.utils.vision import preprocess_image
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestImagePreprocessing(CustomTestCase):
    def test_cpu_cuda_values_and_strides(self):
        rng = np.random.default_rng(991)
        for shape, size, crop in (
            ((256, 256), None, None),
            ((128, 128), (1024, 1024), None),
            ((1081, 1921), (832, 480), (0, 0, 480, 832)),
            ((731, 517), (640, 904), (200, 0, 480, 640)),
            ((31, 47), (96, 64), None),
        ):
            image = Image.fromarray(rng.integers(0, 256, (*shape, 3), dtype=np.uint8))
            for mode in ("RGB", "RGBA", "L"):
                with self.subTest(shape=shape, mode=mode):
                    source = image.convert(mode)
                    with envs.SGLANG_FORCE_CPU_IMAGE_PREPROCESSING.override(True):
                        expected = preprocess_image(
                            source, device=torch.device("cuda"), size=size, crop=crop
                        )
                    with envs.SGLANG_FORCE_CPU_IMAGE_PREPROCESSING.override(False):
                        actual = preprocess_image(
                            source, device=torch.device("cuda"), size=size, crop=crop
                        )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.stride(), expected.stride())

    def test_flux2_and_qwen_edit_vae_processor_parity(self):
        rng = np.random.default_rng(71)
        flux = Flux2PipelineConfig()
        flux.vae_config.arch_config.vae_scale_factor = 8
        processor = Flux2ImageProcessor(vae_scale_factor=16)
        qwen = QwenImageEditPlusPipelineConfig()
        qwen_processor = VaeImageProcessor(vae_scale_factor=8)
        for height, width in ((1024, 1024), (1081, 1921), (731, 517), (2160, 3840)):
            image = Image.fromarray(
                rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            )
            with self.subTest(height=height, width=width):
                source = (
                    processor._resize_to_target_area(image)
                    if width * height > 1024**2
                    else image
                )
                out_w, out_h = source.width // 16 * 16, source.height // 16 * 16
                expected = processor.preprocess(
                    source, height=out_h, width=out_w, resize_mode="crop"
                ).cuda()
                actual, size = flux.preprocess_condition_image(
                    image, width, height, processor
                )
                self.assertEqual(size, (out_w, out_h))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.stride(), expected.stride())
                out_w, out_h = qwen.calculate_vae_image_size(image, width, height)
                expected = qwen_processor.preprocess(image, out_h, out_w).cuda()
                batch = Req(
                    sampling_params=SamplingParams(prompt="test", seed=42),
                    condition_image=[image],
                )
                qwen.preprocess_vae_image(batch, qwen_processor)
                torch.testing.assert_close(batch.vae_image[0], expected, rtol=0, atol=0)
                self.assertEqual(batch.vae_image[0].stride(), expected.stride())

    def test_cosmos_image_stage_and_shared_vae_fallback(self):
        rng = np.random.default_rng(19)
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
            return_value=SimpleNamespace(comfyui_mode=False),
        ):
            cosmos = Cosmos3ImagePreprocessStage()
            stage = ImageVAEEncodingStage(vae=None)
        with TemporaryDirectory() as directory:
            source = Path(directory) / "input.png"
            for shape in ((1081, 1921), (731, 517)):
                image = Image.fromarray(
                    rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
                )
                image.save(source)
                batch = Req(
                    sampling_params=SamplingParams(
                        prompt="test",
                        seed=42,
                        image_path=str(source),
                        width=832,
                        height=480,
                    )
                )
                cosmos.forward(batch, None)
                expected = (
                    _pil_to_normalized_tensor(_resize_crop_pil(image, 832, 480))
                    .unsqueeze(0)
                    .cuda()
                )
                torch.testing.assert_close(
                    batch.preprocessed_image, expected, rtol=0, atol=0
                )
                self.assertEqual(batch.preprocessed_image.stride(), expected.stride())

        for values in (np.linspace(-255, 255, 256), np.arange(256)):
            image = Image.fromarray(values.astype(np.float32).reshape(16, 16))
            expected = torch.from_numpy(np.array(image)[None, None] / 255.0)
            if expected.min() >= 0:
                expected = 2 * expected - 1
            torch.testing.assert_close(
                stage.preprocess(image), expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
