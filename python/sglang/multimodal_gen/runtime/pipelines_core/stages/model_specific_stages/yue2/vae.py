# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time

import soundfile as sf
import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Yue2VAEDecodeStage(PipelineStage):
    """Decode YuE2 NAR latents into 48 kHz stereo audio."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def _decode_params(self, batch: Req) -> tuple[int, int, int]:
        params = batch.sampling_params
        return (
            int(getattr(params, "vae_core_frames", 1024)),
            int(getattr(params, "vae_halo_frames", 16)),
            int(getattr(params, "output_sample_rate", 48000) or 48000),
        )

    def _latent(self, batch: Req) -> torch.Tensor:
        """NAR latents as ``[1, 64, T]`` on CPU fp32."""
        latents = batch.extra["yue2_latents"]
        if latents.ndim == 2:
            latents = latents.unsqueeze(0).transpose(1, 2)
        return latents.float()

    def _output_path(self, batch: Req) -> str:
        output_file_name = batch.output_file_name or f"{batch.extra['yue2_request_id']}.wav"
        # The generic diffusion request layer appends an image extension to
        # output_file_name; strip it before forcing the audio extension.
        for image_ext in (".png", ".jpg", ".jpeg", ".webp"):
            if output_file_name.endswith(image_ext):
                output_file_name = output_file_name[: -len(image_ext)]
                break
        if not output_file_name.endswith(".wav"):
            output_file_name = f"{output_file_name}.wav"
        return os.path.join(batch.output_path or "outputs/", output_file_name)

    def _finish(self, batch: Req, audio: torch.Tensor, sample_rate: int,
                started: float) -> OutputBatch:
        output_path = self._output_path(batch)
        if batch.save_output:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, audio.squeeze(0).cpu().numpy().transpose(1, 0), sample_rate)
        batch.extra.update(
            {
                "yue2_vae_seconds": time.perf_counter() - started,
                "yue2_audio_seconds": audio.shape[-1] / sample_rate,
            }
        )
        return OutputBatch(
            output=None,
            audio=audio,
            audio_sample_rate=sample_rate,
            output_file_paths=[output_path] if batch.save_output else None,
            metrics=batch.metrics,
        )

    def _decode(self, latents: torch.Tensor, core: int, halo: int) -> torch.Tensor:
        return self.vae.decode_tiled(
            latents.to(device=next(self.vae.parameters()).device, dtype=torch.float32),
            core_frames=core,
            halo_frames=halo,
            output_device="cpu",
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        core, halo, sample_rate = self._decode_params(batch)
        started = time.perf_counter()
        audio = self._decode(self._latent(batch), core, halo)
        return self._finish(batch, audio, sample_rate, started)

    def run_grouped_requests(self, batches: list[Req], server_args: ServerArgs):
        """Decode batched requests that share a latent frame count.

        The decoder's convolutions are non-causal, so unequal-length latents
        cannot be padded safely.

        When the optional async-synthesis patch is active it defers NAR to a
        background thread.
        """
        if len(batches) < 2:
            return [self(batch, server_args) for batch in batches]

        for batch in batches:
            future = batch.extra.pop("yue2_synth_future", None)
            if future is not None:
                # NAR ends in a CPU copy, so future.result() is already synced.
                batch.extra["yue2_latents"] = future.result()

        latents = [self._latent(batch) for batch in batches]
        params = [self._decode_params(batch) for batch in batches]
        buckets: dict[tuple, list[int]] = {}
        for index, (latent, (core, halo, _)) in enumerate(zip(latents, params)):
            buckets.setdefault((latent.shape[-1], core, halo), []).append(index)

        results: list = [None] * len(batches)
        started = time.perf_counter()
        for (_, core, halo), indices in buckets.items():
            if len(indices) == 1:
                index = indices[0]
                audio = self._decode(latents[index], core, halo)
                results[index] = self._finish(
                    batches[index], audio, params[index][2], started)
                continue
            stacked = torch.cat([latents[index] for index in indices], dim=0)
            decode_started = time.perf_counter()
            audio = self._decode(stacked, core, halo)
            logger.info("Batched VAE: %d rows (%d frames) in %.2fs", len(indices),
                        stacked.shape[-1], time.perf_counter() - decode_started)
            for row, index in enumerate(indices):
                results[index] = self._finish(
                    batches[index], audio[row:row + 1], params[index][2], started)
        return results
