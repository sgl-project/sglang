# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Kandinsky6 TI2VA SGLang PR review comments:

1. ``Kandinsky6ImageEncodingStage`` must broadcast its single conditioning
   image to the full ``num_outputs_per_prompt`` sample batch instead of
   letting the ``torch.cat`` onto ``batch.latents`` fail with a batch-size
   mismatch (also covered through the real HTTP request-building path).
2. ``Kandinsky6AudioDecodingStage`` must keep the audio batch dimension so
   the generic ``select_output_audio`` save-path helper can select one
   waveform per output instead of reusing the first track for every video.
3. ``Kandinsky6LatentPreparationStage`` must size the audio latents from the
   request's own ``fps``, not the pipeline's fixed default, so a
   non-default-fps request doesn't desync audio/video duration.

No GPU, no real model weights: VAEs are stand-ins returning tensors the test
controls directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import PIL.Image
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
    decoding as decoding_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6 import (
    image_encoding as image_encoding_module,
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
    audio_latent_duration,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args


def _use_as_global_server_args(monkeypatch, server_args) -> None:
    """Make ``server_args`` the ambient global server args for this test.

    ``PipelineStage.__init__`` snapshots ``get_global_server_args()`` into
    ``self.server_args`` at construction time, and component-residency hooks
    (``component_uses`` via ``use_declared_component``) read off that
    snapshot rather than the ``server_args`` a test passes to ``.forward()``.
    Construct the stage under test only *after* calling this, so both agree.
    """
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
        server_args,
    )
    assert get_global_server_args() is server_args


# --------------------------------------------------------------------------- #
# 1a. Kandinsky6ImageEncodingStage: num_outputs_per_prompt batch expansion
# --------------------------------------------------------------------------- #


class _StubVAE:
    """Duck-types just what ``Kandinsky6ImageEncodingStage.forward`` touches."""

    def __init__(self, image_latent: torch.Tensor) -> None:
        self.use_tiling = False
        self.scaling_factor = 1.0
        self.shift_factor = None
        self._image_latent = image_latent

    def to(self, *_args, **_kwargs):
        return self

    def encode(self, _image):
        return SimpleNamespace(sample=lambda generator=None: self._image_latent)


def _gradient_image(width: int, height: int) -> PIL.Image.Image:
    xs = np.linspace(0, 255, width, dtype=np.uint8)
    ys = np.linspace(0, 255, height, dtype=np.uint8)
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    grid[..., 0] = xs[None, :]
    grid[..., 1] = ys[:, None]
    grid[..., 2] = 128
    return PIL.Image.fromarray(grid, mode="RGB")


def test_image_encoding_expands_single_reference_to_multi_output_batch(monkeypatch):
    """num_outputs_per_prompt=2: the video batch is 2, one conditioning image
    produces a reference batch of 1 -- the stage must broadcast it rather
    than let ``torch.cat`` fail with a batch-size mismatch (SGLang PR #2,
    image_encoding.py line 209)."""
    monkeypatch.setattr(
        image_encoding_module, "get_local_torch_device", lambda: torch.device("cpu")
    )

    num_channels = 4
    latent_h, latent_w = 4, 6
    sample_batch_size = 2  # num_outputs_per_prompt=2, one distinct prompt

    server_args = SimpleNamespace(
        pipeline_config=Kandinsky6TI2VAPipelineConfig(vae_precision="fp32"),
        component_precisions={},
        disable_autocast=True,
    )
    _use_as_global_server_args(monkeypatch, server_args)
    image_latent_sample = torch.randn(1, num_channels, 1, latent_h, latent_w)
    stage = Kandinsky6ImageEncodingStage(vae=_StubVAE(image_latent_sample))

    num_video_frames = 3
    visual_cond_channels = 2 * num_channels + 1  # [real, cond, mask]
    latents = torch.randn(
        sample_batch_size, num_video_frames, latent_h, latent_w, visual_cond_channels
    )
    batch = Req(
        sampling_params=SamplingParams(),
        condition_image=_gradient_image(96, 64),
        latents=latents,
        height=64,
        width=96,
        generator=None,
    )

    result = stage.forward(batch, server_args)

    # The reference frame (and the stored `image_latent`, read again every
    # denoising step to re-pin the tail-cond frame) must carry the full
    # per-sample batch, not the single-image batch it was encoded at.
    assert result.image_latent.shape[0] == sample_batch_size
    assert tuple(result.latents.shape) == (
        sample_batch_size,
        num_video_frames + 1,
        latent_h,
        latent_w,
        visual_cond_channels,
    )
    # The two reference-frame copies (one per output) came from the same
    # single encoded image, not two independent encodes.
    assert torch.equal(result.image_latent[0], result.image_latent[1])
    assert result.extra[TAIL_COND_ACTIVE_EXTRA_KEY] is True
    assert tuple(result.extra[VISUAL_TOKEN_TYPE_IDS_EXTRA_KEY].shape) == (
        sample_batch_size,
        num_video_frames + 1,
    )


def test_image_encoding_is_a_noop_for_a_single_output_request(monkeypatch):
    """Sanity check: the expansion path is skipped (no behavior change) when
    the reference batch already matches the video batch (single-output,
    T2VA-without-expansion case)."""
    monkeypatch.setattr(
        image_encoding_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    num_channels = 4
    latent_h, latent_w = 4, 6
    server_args = SimpleNamespace(
        pipeline_config=Kandinsky6TI2VAPipelineConfig(vae_precision="fp32"),
        component_precisions={},
        disable_autocast=True,
    )
    _use_as_global_server_args(monkeypatch, server_args)
    image_latent_sample = torch.randn(1, num_channels, 1, latent_h, latent_w)
    stage = Kandinsky6ImageEncodingStage(vae=_StubVAE(image_latent_sample))

    latents = torch.randn(1, 3, latent_h, latent_w, 2 * num_channels + 1)
    batch = Req(
        sampling_params=SamplingParams(),
        condition_image=_gradient_image(96, 64),
        latents=latents,
        height=64,
        width=96,
        generator=None,
    )

    result = stage.forward(batch, server_args)
    assert result.image_latent.shape[0] == 1
    assert result.latents.shape[0] == 1


# --------------------------------------------------------------------------- #
# 1b. The HTTP request-building path: num_outputs_per_prompt + an image
#     reach SamplingParams together (SGLang PR #2, image_encoding.py line 209
#     review comment: "...cover it through the HTTP path").
# --------------------------------------------------------------------------- #


def _fake_server_args(**overrides) -> SimpleNamespace:
    base = dict(
        pipeline_config=Kandinsky6TI2VAPipelineConfig(),
        pipeline_class_name="Kandinsky6TI2VAPipeline",
        backend="auto",
        model_id=None,
        model_path="kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers",
        served_model_name="kandinsky6-ti2va-test",
        output_path=None,
        comfyui_mode=False,
        num_gpus=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _patched_global_server_args(server_args):
    return patch(
        "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
        server_args,
    )


def test_http_request_carries_image_and_num_outputs_per_prompt_together():
    """The real ``_build_video_sampling_params`` (used by both the multipart
    and JSON ``/v1/videos`` branches of ``video_api.create_video``), given an
    image reference and ``num_outputs_per_prompt=2``, must build a
    ``Req``/``SamplingParams`` pair that carries both values -- the
    combination ``Kandinsky6ImageEncodingStage`` has to handle."""
    server_args = _fake_server_args()
    req = VideoGenerationsRequest(
        prompt="a cat playing piano",
        input_reference="assets/girl.png",
        num_outputs_per_prompt=2,
    )

    with _patched_global_server_args(server_args):
        sampling_params = _build_video_sampling_params("k6-ti2va-multi-output", req)

    assert sampling_params.num_outputs_per_prompt == 2
    assert sampling_params.image_path == "assets/girl.png"


# --------------------------------------------------------------------------- #
# 2. Kandinsky6AudioDecodingStage: preserve the audio batch dimension
# --------------------------------------------------------------------------- #


class _StubAudioVAE:
    def __init__(self, waveform: torch.Tensor) -> None:
        self.scaling_factor = 1.0
        self.mean_value = 0.0
        self._waveform = waveform
        self._params = [torch.zeros(1)]

    def to(self, *_args, **_kwargs):
        return self

    def parameters(self):
        return iter(self._params)

    def wrapped_decode(self, _latents):
        return self._waveform


def test_audio_decoding_preserves_batch_dimension_for_select_output_audio(monkeypatch):
    """SGLang PR #2 review comment (decoding.py line 117): collapsing to
    ``waveform[0, 0]`` drops every audio track but the first, so a
    multi-output request's save path reused one waveform for every video.
    ``select_output_audio`` (entrypoints/utils.py) already knows how to pick
    one track per output from a batched [B, samples] tensor -- the stage just
    has to stop throwing the batch dimension away before handing it off."""
    monkeypatch.setattr(
        decoding_module, "get_local_torch_device", lambda: torch.device("cpu")
    )

    batch_size, samples = 3, 128
    # Distinct per-output values, kept inside the stage's own [-1, 1] clamp.
    per_output_values = [-0.5, 0.0, 0.5]
    waveform = torch.stack(
        [torch.full((samples,), v) for v in per_output_values]
    ).unsqueeze(1)  # [B, 1, samples], matching the vocoder's own output shape
    server_args = SimpleNamespace(
        pipeline_config=Kandinsky6TI2VAPipelineConfig(audio_vae_precision="fp32"),
        component_precisions={},
    )
    _use_as_global_server_args(monkeypatch, server_args)
    stage = Kandinsky6AudioDecodingStage(audio_vae=_StubAudioVAE(waveform))

    audio_latents = torch.zeros(batch_size, 5, 2)
    batch = Req(sampling_params=SamplingParams(), audio_latents=audio_latents)

    result = stage.forward(batch, server_args)

    assert result.audio.shape == (batch_size, samples)
    for idx, expected_value in enumerate(per_output_values):
        per_output = select_output_audio(result.audio, idx)
        assert per_output is not None
        assert torch.equal(per_output, torch.full((samples,), expected_value))


# --------------------------------------------------------------------------- #
# 3. Kandinsky6LatentPreparationStage: audio duration follows request fps
# --------------------------------------------------------------------------- #


def test_latent_preparation_audio_duration_follows_request_fps(monkeypatch):
    """SGLang PR #2 review comment (latent_preparation.py line 120): audio
    duration must come from the request's own fps, not the pipeline's fixed
    default -- otherwise a non-default-fps request produces audio of the
    wrong length relative to the video actually saved at that fps."""
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6.latent_preparation.get_local_torch_device",
        lambda: torch.device("cpu"),
    )

    pipeline_config = Kandinsky6TI2VAPipelineConfig()
    assert pipeline_config.sample_fps == 24.0  # the (wrong, pre-fix) constant

    requested_fps = 30
    num_frames = 121
    height, width = (
        pipeline_config.vae_config.arch_config.spatial_compression_ratio
        * (pipeline_config.dit_config.arch_config.patch_size[1]),
        pipeline_config.vae_config.arch_config.spatial_compression_ratio
        * (pipeline_config.dit_config.arch_config.patch_size[2]),
    )

    batch = Req(
        prompt="a cat playing piano",
        height=height,
        width=width,
        num_frames=num_frames,
        fps=requested_fps,
    )
    server_args = SimpleNamespace(pipeline_config=pipeline_config)

    result = Kandinsky6LatentPreparationStage().forward(batch, server_args)

    num_latent_frames = result.latents.shape[1]
    expected_at_requested_fps = audio_latent_duration(
        num_latent_frames,
        fps=float(requested_fps),
        audio_sample_rate=pipeline_config.audio_sample_rate,
        audio_downsample_factor=pipeline_config.audio_downsample_factor,
    )
    expected_at_wrong_default_fps = audio_latent_duration(
        num_latent_frames,
        fps=pipeline_config.sample_fps,
        audio_sample_rate=pipeline_config.audio_sample_rate,
        audio_downsample_factor=pipeline_config.audio_downsample_factor,
    )
    assert expected_at_requested_fps != expected_at_wrong_default_fps
    assert result.audio_latents.shape[1] == expected_at_requested_fps
