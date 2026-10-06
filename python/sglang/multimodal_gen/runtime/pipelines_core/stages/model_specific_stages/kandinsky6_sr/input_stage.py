# SPDX-License-Identifier: Apache-2.0
"""Input stage of Kandinsky 6 video SR: source video -> aligned clip (+ source audio)."""

import os

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_REQUESTED_HW_KEY,
    SR_TILING_SCALE_KEY,
    SR_VIDEO_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    VAE_SPATIAL_FACTOR,
    pad_to_spatial_factor,
    pre_upscale_video,
    resolve_scale_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.video_io import (
    SOURCE_AUDIO_SAMPLE_RATE,
    DecodedClip,
    decode_clip,
    extract_source_audio,
    synthetic_clip,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# Warmup requests carry no (usable) video: run the real path on a small synthetic clip.
WARMUP_CLIP_FRAMES = 9
WARMUP_CLIP_HW = (256, 384)


class Kandinsky6SRInputStage(PipelineStage):
    """Streams the source video into a ``[T, 3, H, W]`` uint8 clip.

    Follows the reference CLI: fps resampling toward 24 fps, the first 121 frames floored
    to ``1 + 8k``, an optional x1.125 pre-upscale for the 2.25 scale, then edge-replicate
    padding to the VAE spatial factor for every scale (``tiling.pad_to_spatial_factor``) so
    whole-video KVAE encoding never runs on a misaligned frame.  Fills ``batch.fps`` (an
    int), ``num_frames``, ``height`` and ``width`` (the *requested* SR result size, before
    any delivery resize -- i.e. with the alignment padding already excluded), and puts the
    source audio (mono float, trimmed to the processed duration) into ``batch.audio``.  No
    prompt is required.
    """

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        tiling_scale, pre_upscale = resolve_scale_request(batch.sr_resolution_scale)
        clip = self._load_clip(batch)
        video = clip.frames
        if pre_upscale != 1.0:
            video = pre_upscale_video(video, pre_upscale, VAE_SPATIAL_FACTOR)
        frames = video.shape[0]
        # Align to the VAE spatial factor for every scale, not only 2.25 (whose pixel
        # pre-upscale happens to already land on an aligned size): without this, a source
        # whose dims are not already a multiple of 16 (e.g. a plain 1080p source at x2/x4,
        # which runs no pixel resize at all) fails partway through the whole-video KVAE
        # encode's four halvings instead of at a clear boundary. Padding (not resizing) keeps
        # every source pixel; batch.height / batch.width below, and the output stage's final
        # crop, restore the originally requested dimensions.
        video, (requested_h, requested_w) = pad_to_spatial_factor(
            video, VAE_SPATIAL_FACTOR
        )

        batch.extra[SR_VIDEO_KEY] = video
        batch.extra[SR_TILING_SCALE_KEY] = tiling_scale
        batch.extra[SR_REQUESTED_HW_KEY] = (requested_h, requested_w)
        batch.fps = clip.fps
        batch.num_frames = frames
        batch.height = requested_h * tiling_scale
        batch.width = requested_w * tiling_scale
        self._attach_source_audio(batch, frames=frames, fps=clip.fps)
        return batch

    def _load_clip(self, batch: Req) -> DecodedClip:
        if batch.is_warmup:
            height, width = WARMUP_CLIP_HW
            frames = synthetic_clip(
                height=height, width=width, frames=WARMUP_CLIP_FRAMES
            )
            return DecodedClip(frames=frames, fps=24, source_fps=24.0)
        path = batch.sampling_params.source_video_path()
        if path is None or not os.path.isfile(path):
            raise ValueError(f"Kandinsky6 SR input video not found: {path!r}")
        self.log_info("Decoding %s", path)
        return decode_clip(path)

    @staticmethod
    def _attach_source_audio(batch: Req, *, frames: int, fps: int) -> None:
        if batch.is_warmup:
            return
        path = batch.sampling_params.source_video_path()
        audio = extract_source_audio(
            path, sample_rate=SOURCE_AUDIO_SAMPLE_RATE, max_seconds=frames / fps
        )
        if audio is None:
            batch.audio = None
            batch.audio_sample_rate = None
            return
        batch.audio = torch.from_numpy(audio)
        batch.audio_sample_rate = SOURCE_AUDIO_SAMPLE_RATE
