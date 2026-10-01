# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time

import numpy as np
import soundfile as sf
import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Yue2VAEDecodeStage(PipelineStage):
    """Decode YuE2 NAR latents into 48 kHz stereo audio."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        vae = self.vae
        latents = batch.extra["yue2_latents"]
        if latents.ndim == 2:
            latents = latents.unsqueeze(0).transpose(1, 2)
        latents = latents.to(
            device=next(vae.parameters()).device, dtype=torch.float32
        )
        started = time.perf_counter()
        audio = vae.decode_tiled(
            latents,
            core_frames=int(getattr(batch.sampling_params, "vae_core_frames", 1024)),
            halo_frames=int(getattr(batch.sampling_params, "vae_halo_frames", 16)),
            output_device="cpu",
        )
        sample_rate = int(
            getattr(batch.sampling_params, "output_sample_rate", 48000) or 48000
        )
        output_file_name = batch.output_file_name or (
            f"{batch.extra['yue2_request_id']}.wav"
        )
        # The generic diffusion request layer appends an image extension to
        # output_file_name; strip it before forcing the audio extension.
        for image_ext in (".png", ".jpg", ".jpeg", ".webp"):
            if output_file_name.endswith(image_ext):
                output_file_name = output_file_name[: -len(image_ext)]
                break
        if not output_file_name.endswith(".wav"):
            output_file_name = f"{output_file_name}.wav"
        output_path = os.path.join(batch.output_path or "outputs/", output_file_name)
        if batch.save_output:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(
                output_path,
                audio.squeeze(0).cpu().numpy().transpose(1, 0),
                sample_rate,
            )
        batch.extra.update(
            {
                "yue2_vae_seconds": time.perf_counter() - started,
                "yue2_audio_seconds": audio.shape[-1] / sample_rate,
            }
        )
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

        return OutputBatch(
            output=None,
            audio=audio,
            audio_sample_rate=sample_rate,
            output_file_paths=[output_path] if batch.save_output else None,
            metrics=batch.metrics,
        )
