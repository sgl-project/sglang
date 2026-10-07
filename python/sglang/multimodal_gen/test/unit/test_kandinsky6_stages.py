# SPDX-License-Identifier: Apache-2.0
"""Image batching and audio/video alignment through real Kandinsky 6 stages."""

import math
from types import SimpleNamespace

import numpy as np
import PIL.Image
import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _build_video_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import select_output_audio
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6 import (
    decoding,
    image_encoding,
    latent_preparation,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6.decoding import (
    Kandinsky6AudioDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6.image_encoding import (
    TAIL_COND_ACTIVE_EXTRA_KEY,
    VISUAL_TOKEN_TYPE_IDS_EXTRA_KEY,
    Kandinsky6ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6.latent_preparation import (
    Kandinsky6LatentPreparationStage,
)


@pytest.fixture
def server_args(monkeypatch):
    args = SimpleNamespace(
        pipeline_config=Kandinsky6TI2VAPipelineConfig(
            vae_precision="fp32", audio_vae_precision="fp32"
        ),
        component_precisions={},
        disable_autocast=True,
        pipeline_class_name="Kandinsky6TI2VAPipeline",
        backend="auto",
        model_id=None,
        model_path="kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers",
        served_model_name="kandinsky6-ti2va-test",
        output_path=None,
        comfyui_mode=False,
        num_gpus=1,
    )
    # residency hooks capture the global args when each stage is constructed
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
        args,
    )
    for module in (decoding, image_encoding, latent_preparation):
        monkeypatch.setattr(
            module, "get_local_torch_device", lambda: torch.device("cpu")
        )
    return args


class ImageVAE:
    def __init__(self, latent):
        self.use_tiling = False
        self.scaling_factor = 1.0
        self.shift_factor = None
        self.latent = latent

    def to(self, *args, **kwargs):
        return self

    def encode(self, image):
        return SimpleNamespace(sample=lambda generator=None: self.latent)


@pytest.mark.parametrize("source_size", [(1920, 1080), (1080, 1920), (288, 192)])
@pytest.mark.parametrize("tensor_input", [False, True], ids=["pil", "tensor"])
def test_image_preprocessing_preserves_aspect_and_center_crops(
    source_size, tensor_input
):
    src_w, src_h = source_size
    height, width = 64, 96
    pixels = np.full((src_h, src_w, 3), 128, dtype=np.uint8)
    pixels[..., 0] = np.linspace(0, 255, src_w, dtype=np.uint8)[None, :]
    pixels[..., 1] = np.linspace(0, 255, src_h, dtype=np.uint8)[:, None]
    image = PIL.Image.fromarray(pixels)

    # independent PIL oracle for the reference resize + center-crop geometry
    scale = min(src_h / height, src_w / width)
    new_h, new_w = int(src_h / scale), int(src_w / scale)
    top, left = (new_h - height) // 2, (new_w - width) // 2
    expected = image.resize((new_w, new_h), PIL.Image.Resampling.BILINEAR).crop(
        (left, top, left + width, top + height)
    )
    expected = torch.from_numpy(np.array(expected)).permute(2, 0, 1).float() / 127.5 - 1
    source = (
        torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1
        if tensor_input
        else image
    )
    actual = Kandinsky6ImageEncodingStage._preprocess(source, height, width)
    torch.testing.assert_close(actual, expected.unsqueeze(0), atol=2 / 255, rtol=0)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_image_request_broadcasts_one_reference_to_each_output(server_args, batch_size):
    request = VideoGenerationsRequest(
        prompt="a cat playing piano",
        input_reference="assets/girl.png",
        num_outputs_per_prompt=batch_size,
    )
    params = _build_video_sampling_params("k6-ti2va-multi-output", request)
    assert params.num_outputs_per_prompt == batch_size
    assert params.image_path == "assets/girl.png"
    image_latent = torch.randn(1, 4, 1, 4, 6)
    batch = Req(
        sampling_params=params,
        condition_image=PIL.Image.new("RGB", (96, 64)),
        latents=torch.randn(batch_size, 3, 4, 6, 9),
        height=64,
        width=96,
        generator=None,
    )
    result = Kandinsky6ImageEncodingStage(ImageVAE(image_latent)).forward(
        batch, server_args
    )
    assert result.latents.shape == (batch_size, 4, 4, 6, 9)
    assert result.image_latent.shape[0] == batch_size
    for reference in result.image_latent:
        torch.testing.assert_close(reference, result.image_latent[0], rtol=0, atol=0)
    assert result.extra[TAIL_COND_ACTIVE_EXTRA_KEY] is True
    assert result.extra[VISUAL_TOKEN_TYPE_IDS_EXTRA_KEY].shape == (batch_size, 4)


class AudioVAE(torch.nn.Module):
    scaling_factor = 1.0
    mean_value = 0.0

    def __init__(self, waveform):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.waveform = waveform

    def wrapped_decode(self, latents):
        return self.waveform


def test_audio_decoding_preserves_each_output_track(server_args):
    values = [-0.5, 0.0, 0.5]
    waveform = torch.stack([torch.full((128,), value) for value in values]).unsqueeze(1)
    batch = Req(sampling_params=SamplingParams(), audio_latents=torch.zeros(3, 5, 2))
    result = Kandinsky6AudioDecodingStage(AudioVAE(waveform)).forward(
        batch, server_args
    )
    assert result.audio.shape == (3, 128)
    for index, value in enumerate(values):
        torch.testing.assert_close(
            select_output_audio(result.audio, index),
            torch.full((128,), value),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("num_frames,fps", [(1, 24), (61, 24), (121, 24), (121, 30)])
def test_latent_preparation_aligns_audio_to_requested_video(
    server_args, num_frames, fps
):
    config = server_args.pipeline_config
    batch = Req(
        prompt="a cat playing piano",
        height=16,
        width=16,
        num_frames=num_frames,
        fps=fps,
    )
    result = Kandinsky6LatentPreparationStage().forward(batch, server_args)
    assert result.latents.shape[1] == (num_frames - 1) // 4 + 1
    expected = math.ceil(
        num_frames / fps * config.audio_sample_rate / config.audio_downsample_factor
    )
    assert result.audio_latents.shape[1] == expected
