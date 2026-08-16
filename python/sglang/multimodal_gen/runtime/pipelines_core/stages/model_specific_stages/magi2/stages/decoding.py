# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.sample.magi2 import (
    MAGI2_PREVIEW_FPS,
    MAGI2_REFINER_FPS,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Magi2DecodingStage(PipelineStage):
    """Three egress requirements fail silently: numpy THWC uint8 frames, audio inside the sample tuple, and fps on the request."""

    def __init__(
        self,
        *,
        video_vae,
        audio_vae,
        turbo_vae=None,
        latents_mean: tuple[float, ...],
        latents_std: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.turbo_vae = turbo_vae
        self.latents_mean = torch.tensor(latents_mean).view(1, -1, 1, 1, 1)
        self.latents_std = torch.tensor(latents_std).view(1, -1, 1, 1, 1)

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    def _decode_video(self, batch: Req, *, use_turbo: bool) -> torch.Tensor:
        _, channels, frames, height, width = batch.raw_latent_shape
        grid = (
            batch.latents.reshape(frames, height, width, channels)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
        )
        # Un-normalize: skipping this decodes washed-out, colour-shifted video.
        std = self.latents_std.to(device=grid.device, dtype=grid.dtype)
        mean = self.latents_mean.to(device=grid.device, dtype=grid.dtype)
        grid = grid * std + mean
        decoder = (
            self.turbo_vae
            if use_turbo and self.turbo_vae is not None
            else self.video_vae
        )
        weight = next(decoder.parameters())
        with torch.no_grad():
            return decoder.decode(grid.to(device=weight.device, dtype=weight.dtype))

    def _decode_audio(self, latents: torch.Tensor | None) -> torch.Tensor | None:
        if latents is None or latents.numel() == 0:
            return None
        with torch.no_grad():
            return self.audio_vae.decode(latents.t().unsqueeze(0))

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        config = server_args.pipeline_config
        refined = batch.extra["magi2_enable_refiner"]

        batch.latents = batch.latents.contiguous()
        torch.cuda.empty_cache()

        pixels = self._decode_video(batch, use_turbo=config.use_turbo_vae)
        # The refiner gets zero audio tokens, so the latent stays where handoff put it.
        audio_latents = (
            batch.extra["magi2_preview_audio_latents"]
            if refined
            else batch.audio_latents
        )
        audio = self._decode_audio(audio_latents)

        frames = _to_thwc_uint8(pixels)

        fps = MAGI2_REFINER_FPS if refined else MAGI2_PREVIEW_FPS
        if batch.sampling_params is not None:
            batch.sampling_params.fps = fps

        audio_np = None
        sample_rate = None
        if audio is not None:
            audio_np = audio.squeeze(0).float().cpu().numpy()
            sample_rate = config.output_audio_sample_rate

        # Audio rides in the tuple: the OutputBatch field alone is a silent track.
        sample = (frames, audio_np) if audio_np is not None else frames

        return OutputBatch(
            output=[sample],
            audio=audio_np,
            audio_sample_rate=sample_rate,
            metrics=batch.metrics,
        )


def _to_thwc_uint8(pixels: torch.Tensor):
    """A torch tensor would instead be read as (C, T, H, W) float and PIL would reject the frame count as a channel count."""
    clip = pixels[0].float().clamp(-1, 1)
    clip = ((clip + 1) * 127.5).round().to(torch.uint8)
    return clip.permute(1, 2, 3, 0).cpu().numpy()
