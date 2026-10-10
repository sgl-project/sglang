# SPDX-License-Identifier: Apache-2.0
"""ComfyUI per-step DiT stage for LTX-2.x audio-video (ComfyUI ``ltxav``).

ComfyUI owns the sampler loop, CFG and the VAEs. Each request is either one
``LTXAVModel.forward`` (``[video, audio]`` velocities on ``batch.noise_pred``)
or, with ``extra["ltxav_op"] == "connectors"``, one pass of the text
connectors that ComfyUI runs from ``preprocess_text_embeds``. The DiT runs
with SGLang's native LTX semantics (RoPE coordinates, timestep embeddings).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.ltx_2 import (
    load_comfyui_ltx_connectors,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    bind_comfyui_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs

LTXAV_OP_CONNECTORS = "connectors"
# ComfyUI AudioPatchifier: [B, 8 channels, T, 16 mel bins] <-> [B, T, 128].
AUDIO_CHANNELS = 8
# ComfyUI Embeddings1DConnector pads the prompt with registers to this length.
CONNECTOR_MIN_TOKENS = 1024


def patchify_video(video: torch.Tensor) -> torch.Tensor:
    """[B, C, T, H, W] -> [B, T*H*W, C] (patch size 1)."""
    return video.flatten(2).transpose(1, 2)


def patchify_audio(audio: torch.Tensor) -> torch.Tensor:
    """[B, C, T, F] -> [B, T, C*F]."""
    return audio.transpose(1, 2).flatten(2)


def unpatchify_audio(tokens: torch.Tensor) -> torch.Tensor:
    b, t, _ = tokens.shape
    return tokens.reshape(b, t, AUDIO_CHANNELS, -1).transpose(1, 2)


def per_token_timestep(timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
    """ComfyUI timesteps are [B], a scalar, or per token [B, N, 1]."""
    timestep = timestep.to(torch.float32)
    if timestep.numel() == 1:
        return timestep.reshape(1).expand(batch_size)
    return timestep.reshape(batch_size, -1)


def run_ltxav_connectors(
    video_connector, audio_connector, context: torch.Tensor, video_dim: int
) -> torch.Tensor:
    """ComfyUI ``preprocess_text_embeds(unprocessed=True)`` with SGLang connectors.

    The prompt tokens are left-padded to a register multiple; the connector
    moves them to the front and fills the rest with its learnable registers.
    """
    batch, tokens, _ = context.shape
    registers = video_connector.num_learnable_registers
    total = max(CONNECTOR_MIN_TOKENS, math.ceil(tokens / registers) * registers)
    padded = context.new_zeros(batch, total, context.shape[-1])
    padded[:, total - tokens :] = context
    mask = torch.full((batch, total), -1e6, device=context.device)
    mask[:, total - tokens :] = 0.0
    video, _ = video_connector(padded[..., :video_dim], mask)
    audio, _ = audio_connector(padded[..., video_dim:], mask)
    return torch.cat((video, audio), dim=-1)


def shard_video_frames_for_sp(
    tokens: torch.Tensor, timestep: torch.Tensor, frames: int, rank: int, size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """This SP rank's contiguous latent frames of the [B, T*H*W, C] video tokens.

    The DiT offsets the RoPE time of rank r by r * local frames, so the split
    must be by whole, equal frame blocks; audio stays replicated on every rank.
    """
    if frames % size:
        raise ValueError(
            f"LTX sequence parallelism splits the {frames} latent frames across "
            f"{size} ranks, which needs a frame count divisible by {size}; change "
            "the video length, or run without sp_degree"
        )
    local = frames // size
    per_rank = tokens.shape[1] // size
    rows = slice(rank * per_rank, (rank + 1) * per_rank)
    if timestep.ndim == 2 and timestep.shape[1] == tokens.shape[1]:
        timestep = timestep[:, rows]
    return tokens[:, rows], timestep, local


class LTX2ComfyUIStepStage(PipelineStage):
    """One ComfyUI ``apply_model`` (or connector) call per request."""

    def __init__(self, transformer, model_path: str) -> None:
        super().__init__()
        self.transformer = transformer
        # Loaded with the DiT, while ComfyUI has freed the device for the worker.
        self.connectors = load_comfyui_ltx_connectors(
            model_path, get_local_torch_device(), self._dtype
        )

    def verify_input(self, batch: Req, server_args: ServerArgs):
        bind_comfyui_session(batch)
        return super().verify_input(batch, server_args)

    @property
    def _dtype(self) -> torch.dtype:
        return self.transformer.patchify_proj.bias.dtype

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        bind_comfyui_session(batch)
        extra = batch.extra or {}
        context = batch.prompt_embeds
        if isinstance(context, (list, tuple)):
            context = context[0] if context else None
        if context is None:
            raise ValueError("LTX ComfyUI step requires prompt_embeds")
        context = context.to(self._dtype)
        with (
            torch.no_grad(),
            set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=batch
            ),
        ):
            if extra.get("ltxav_op") == LTXAV_OP_CONNECTORS:
                batch.noise_pred = self._connectors(context)
            else:
                batch.noise_pred = self._step(batch, extra, context)
        return batch

    def _connectors(self, context: torch.Tensor) -> torch.Tensor:
        video_dim = self.transformer.config.cross_attention_dim
        return run_ltxav_connectors(*self.connectors, context, video_dim)

    def _step(
        self, batch: Req, extra: dict[str, Any], context: torch.Tensor
    ) -> list[torch.Tensor]:
        transformer = self.transformer
        video = batch.latents.to(self._dtype)
        audio = batch.audio_latents.to(self._dtype)
        b, _, frames, height, width = video.shape
        v_ts = per_token_timestep(batch.timesteps, b)
        a_ts = per_token_timestep(extra["ltxav_audio_timestep"], b)
        scale = float(transformer.config.timestep_scale_multiplier)
        video_dim = transformer.config.cross_attention_dim
        hidden, sp_size = patchify_video(video), get_sp_world_size()
        if sp_size > 1:
            hidden, v_ts, frames = shard_video_frames_for_sp(
                hidden, v_ts, frames, get_sp_parallel_rank(), sp_size
            )
        video_tokens, audio_tokens = transformer(
            hidden_states=hidden,
            audio_hidden_states=patchify_audio(audio),
            encoder_hidden_states=context[..., :video_dim],
            audio_encoder_hidden_states=context[..., video_dim:],
            timestep=v_ts * scale,
            audio_timestep=a_ts * scale,
            num_frames=frames,
            height=height,
            width=width,
            fps=float(extra.get("ltxav_frame_rate", 25.0)),
            audio_num_frames=int(audio.shape[2]),
            return_latents=False,
            audio_replicated_for_sp=sp_size > 1,
        )
        if sp_size > 1:
            video_tokens = sequence_model_parallel_all_gather(
                video_tokens.contiguous(), dim=1
            )
        video_out = video_tokens.transpose(1, 2).reshape(video.shape)
        return [video_out, unpatchify_audio(audio_tokens)]


__all__ = [
    "LTXAV_OP_CONNECTORS",
    "LTX2ComfyUIStepStage",
    "patchify_audio",
    "patchify_video",
    "per_token_timestep",
    "run_ltxav_connectors",
    "shard_video_frames_for_sp",
    "unpatchify_audio",
]
