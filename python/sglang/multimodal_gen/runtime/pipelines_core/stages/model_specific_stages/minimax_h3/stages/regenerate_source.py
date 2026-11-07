# SPDX-License-Identifier: Apache-2.0
"""Open-H3-Regenerate-2K source stage: draft video -> upscaled, re-noised latent.

Follows NVlabs/Sana ``models/Open-H3-Regenerate-2K`` (open_h3/pipeline.py):
the draft is encoded at its own H3 canvas (768P by default), the latent is
upscaled to the request canvas by the author latent upscaler, and the refine
starts from ``(1 - sigma0) * upscaled + sigma0 * noise``. Audio rows are held
clean (sigma 0) for every step: the draft's own soundtrack by default, or the
encoded silence the reference pass uses.

The stage writes the denoise state and both sigma schedules; the regular H3
latent / timestep preparation stages then leave them untouched, and the H3
Euler loop and decoder run unchanged.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
    MINIMAX_H3_SUPPORTED_FPS,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# The refiner's release schedules (open_h3/sigmas.py): Euler, model time 1 - sigma.
REGENERATE_SHIFT = 6.0
REGENERATE_SCHEDULES = {
    3: [0.909375, 0.8571428571428571, 0.6666666666666666, 0.0],
    2: None,  # shifted_grid(0.7, 2)
}
# the reference encode samples the VAE posterior under seed 42
REGENERATE_VAE_SEED = 42
# trailing copies of the last draft frame (121 + 3 -> the 124-frame H3 grid)
REGENERATE_HELD_FRAMES = 3


def regenerate_shifted_grid(start: float, steps: int) -> list[float]:
    """Euler grid from ``start`` to 0: even spacing in unshifted sigma, then shift 6."""
    shift = REGENERATE_SHIFT
    raw = start / (shift - (shift - 1) * start)
    interior = (raw * (1 - i / steps) for i in range(1, steps))
    return [start] + [shift * x / (1 + (shift - 1) * x) for x in interior] + [0.0]


def regenerate_sigmas(steps: int, start_sigma: float | None) -> list[float]:
    if start_sigma is not None:
        return regenerate_shifted_grid(float(start_sigma), steps)
    if steps == 3:
        return list(REGENERATE_SCHEDULES[3])
    if steps == 2:
        return regenerate_shifted_grid(0.7, 2)
    raise ValueError(
        "Open-H3-Regenerate-2K ships 2- and 3-step schedules; pass start_sigma "
        f"for num_inference_steps={steps}"
    )


def _local_path(uri: str) -> str:
    return uri[len("file://") :] if uri.startswith("file://") else uri


def _has_audio_stream(path: str) -> bool:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return bool(out.strip())


def decode_draft_frames(
    path: str, *, width: int, height: int, frame_count: int
) -> np.ndarray:
    """Draft -> [frame_count, height, width, 3] uint8, as the reference encodes it.

    24 fps, cover-scaled (bilinear) and center-cropped to the canvas, cut or
    padded (last frame held) to ``frame_count - 3`` frames, then the last frame
    repeated three more times.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Open-H3-Regenerate-2K needs ffmpeg to decode the draft")
    real = frame_count - REGENERATE_HELD_FRAMES
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-vf",
            f"fps={MINIMAX_H3_SUPPORTED_FPS},scale={width}:{height}:"
            "force_original_aspect_ratio=increase:flags=bilinear,"
            f"crop={width}:{height},tpad=stop_mode=clone:stop_duration=16",
            "-frames:v",
            str(real),
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    if len(raw) != real * height * width * 3:
        raise ValueError(
            f"could not decode {path} into {real} frames at {width}x{height}"
        )
    frames = np.frombuffer(raw, np.uint8).reshape(real, height, width, 3)
    held = np.repeat(frames[-1:], REGENERATE_HELD_FRAMES, axis=0)
    return np.concatenate([frames, held], axis=0)


class MiniMaxH3RegenerateSourceStage(PipelineStage):
    def __init__(self, *, video_vae, audio_vae, upscaler, pipeline_config) -> None:
        super().__init__()
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.upscaler = upscaler
        self.video_arch = pipeline_config.vae_config.arch_config
        self.audio_arch = pipeline_config.audio_vae_config.arch_config

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(stage_name, "video_vae"),
            ComponentUse(stage_name, "audio_vae"),
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise ValueError("Open-H3-Regenerate-2K requires a resolved t2va plan")
        params = batch.sampling_params
        shape = plan.shape
        frame_count = int(shape["frame_count"])
        latent_t = int(shape["video_latent_t"])
        latent_h, latent_w = int(shape["height"]) // 16, int(shape["width"]) // 16
        audio_t = int(shape["audio_latent_t"])

        source = getattr(params, "source_video", None)
        if source is None and not batch.is_warmup:
            raise ValueError("Open-H3-Regenerate-2K requires source_video")
        if source is None:
            upscaled = torch.zeros(1, 24, latent_t, latent_h, latent_w)
            audio_rows = torch.zeros(2 * audio_t, 32)
        else:
            path = _local_path(str(source))
            upscaled = self._encode_and_upscale(
                path,
                plan,
                source_short_edge=int(params.source_short_edge),
                frame_count=frame_count,
                target=(latent_t, latent_h, latent_w),
            )
            audio_rows = self._encode_audio(
                path if params.keep_source_audio else None,
                frame_count=frame_count,
                audio_t=audio_t,
            )

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
            minimax_h3_patchify_video_latent,
        )

        sigmas = regenerate_sigmas(
            int(batch.num_inference_steps), getattr(params, "start_sigma", None)
        )
        seed = 42 if plan.seed is None else int(plan.seed)
        noise = torch.randn(
            upscaled.shape,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float32,
        )
        start = (1.0 - sigmas[0]) * upscaled + sigmas[0] * noise
        video_rows = minimax_h3_patchify_video_latent(start, patch_size=[1, 2, 2])
        batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY] = {
            "initial_video_rows": video_rows.to(torch.float32),
            "initial_audio_rows": audio_rows,
            "latent_t": latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "audio_t": audio_t,
        }
        batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = {
            "video": sigmas,
            # the soundtrack stays clean (timestep 1) through every step
            "audio": [0.0] * len(sigmas),
        }
        return batch

    def _encode_and_upscale(
        self,
        path: str,
        plan,
        *,
        source_short_edge: int,
        frame_count: int,
        target: tuple[int, int, int],
    ) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
            _cached_latent_mean_std,
            minimax_h3_scoped_encode_rng,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_resolve_spatial_shape,
        )

        canvas = minimax_h3_resolve_spatial_shape(
            width=int(plan.shape["width"]),
            height=int(plan.shape["height"]),
            base_short_edge=source_short_edge,
        )
        frames = decode_draft_frames(
            path,
            width=int(canvas["width"]),
            height=int(canvas["height"]),
            frame_count=frame_count,
        )
        vae = self.video_vae
        parameter = next(vae.parameters())
        prev_dtype = parameter.dtype
        if prev_dtype != torch.float32:
            vae.to(torch.float32)
        try:
            with minimax_h3_scoped_encode_rng(REGENERATE_VAE_SEED, parameter.device):
                z = vae.encode_videos(frames, use_fp16_latent=True)[0]
        finally:
            if prev_dtype != torch.float32:
                vae.to(prev_dtype)
        z = z.float()
        if z.dim() == 4:
            z = z[None]
        mean, std = _cached_latent_mean_std(
            tuple(self.video_arch.latents_mean),
            tuple(self.video_arch.latents_std),
            (1, self.video_arch.latent_channels, 1, 1, 1),
        )
        mean, std = mean.to(z.device), std.to(z.device)
        z = (z - mean) / std
        if int(z.shape[2]) != target[0]:
            raise ValueError(
                f"draft latent has {int(z.shape[2])} frames, expected {target[0]}"
            )
        device = next(self.upscaler.parameters()).device
        # The reference hands the node an already-normalized latent, which the
        # node normalizes once more (D(U(N(z)))); keep that composition.
        upscaled = self.upscaler.upscale(
            z.to(device),
            target_hw=target[1:],
            mean=mean.to(device),
            std=std.to(device),
        )
        return upscaled.cpu()

    def _encode_audio(
        self, path: str | None, *, frame_count: int, audio_t: int
    ) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            MINIMAX_H3_AUDIO_SAMPLE_RATE,
            _audio_resampler,
            _load_waveform,
            minimax_h3_encode_audio_waveform_rows,
        )

        samples = round(
            frame_count / MINIMAX_H3_SUPPORTED_FPS * MINIMAX_H3_AUDIO_SAMPLE_RATE
        )
        waveform = torch.zeros(2, samples)
        if path is not None and _has_audio_stream(path):
            loaded, rate = _load_waveform(
                path,
                material_chain="video_audio.reference_preserve",
                max_duration_seconds=frame_count / MINIMAX_H3_SUPPORTED_FPS,
            )
            if int(rate) != MINIMAX_H3_AUDIO_SAMPLE_RATE:
                loaded = _audio_resampler(int(rate))(loaded)
            loaded = loaded[:, :samples]
            waveform[:, : loaded.shape[-1]] = loaded
        rows, encoded_t = minimax_h3_encode_audio_waveform_rows(
            self.audio_vae, waveform, self.audio_arch
        )
        latent = rows.view(2, encoded_t, -1)
        if encoded_t >= audio_t:
            latent = latent[:, :audio_t]
        else:
            # hold the last latent to fill the plan's audio grid
            latent = torch.cat(
                [latent, latent[:, -1:].expand(-1, audio_t - encoded_t, -1)], dim=1
            )
        return latent.reshape(2 * audio_t, -1).contiguous()


__all__ = [
    "MiniMaxH3RegenerateSourceStage",
    "decode_draft_frames",
    "regenerate_shifted_grid",
    "regenerate_sigmas",
]
