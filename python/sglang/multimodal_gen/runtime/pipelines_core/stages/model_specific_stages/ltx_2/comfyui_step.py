# SPDX-License-Identifier: Apache-2.0
"""ComfyUI per-step DiT stage for LTX-2.x audio-video (ComfyUI ``ltxav``).

Each request is one ``LTXAVModel.forward`` (``[video, audio]`` velocities on
``batch.noise_pred``) or, with ``extra["ltxav_connectors"]``, one pass of the
text connectors that ComfyUI runs from ``preprocess_text_embeds``.
"""

from __future__ import annotations

import math

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

# ComfyUI Embeddings1DConnector pads the prompt with registers to this length.
CONNECTOR_MIN_TOKENS = 1024


def per_token_timestep(timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
    """ComfyUI timesteps are [B], a scalar, or per token [B, N, 1]."""
    timestep = timestep.to(torch.float32)
    if timestep.numel() == 1:
        return timestep.reshape(1).expand(batch_size)
    return timestep.reshape(batch_size, -1)


def run_ltxav_connectors(
    video_connector, audio_connector, context: torch.Tensor, video_dim: int
) -> torch.Tensor:
    """ComfyUI ``preprocess_text_embeds(unprocessed=True)``: the prompt is
    left-padded to a register multiple; the connector moves it to the front."""
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
    per_rank = tokens.shape[1] // size
    rows = slice(rank * per_rank, (rank + 1) * per_rank)
    if timestep.ndim == 2 and timestep.shape[1] == tokens.shape[1]:
        timestep = timestep[:, rows]
    return tokens[:, rows], timestep, frames // size


class LTX2ComfyUIStepStage(PipelineStage):
    """One ComfyUI ``apply_model`` (or connector) call per request."""

    def __init__(self, transformer, model_path: str) -> None:
        super().__init__()
        self.transformer = transformer
        self.dtype = transformer.patchify_proj.bias.dtype
        # Loaded with the DiT, while ComfyUI has freed the device for the worker.
        self.connectors = load_comfyui_ltx_connectors(
            model_path, get_local_torch_device(), self.dtype
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        bind_comfyui_session(batch)
        context = batch.prompt_embeds[0].to(self.dtype)
        video_dim = self.transformer.config.cross_attention_dim
        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ):
            if batch.extra.get("ltxav_connectors"):
                batch.noise_pred = run_ltxav_connectors(
                    *self.connectors, context, video_dim
                )
            else:
                batch.noise_pred = self._step(batch, context, video_dim)
        return batch

    def _step(
        self, batch: Req, context: torch.Tensor, video_dim: int
    ) -> list[torch.Tensor]:
        transformer = self.transformer
        video = batch.latents.to(self.dtype)
        audio = batch.audio_latents.to(self.dtype)
        b, _, frames, height, width = video.shape
        v_ts = per_token_timestep(batch.timesteps, b)
        a_ts = per_token_timestep(batch.extra["ltxav_audio_timestep"], b)
        scale = float(transformer.config.timestep_scale_multiplier)
        # Same token layouts as ComfyUI's patchifiers and the native packers:
        # video [B, T*H*W, C], audio [B, T, C*F].
        hidden, sp_size = video.flatten(2).transpose(1, 2), get_sp_world_size()
        if sp_size > 1:
            hidden, v_ts, frames = shard_video_frames_for_sp(
                hidden, v_ts, frames, get_sp_parallel_rank(), sp_size
            )
        video_tokens, audio_tokens = transformer(
            hidden_states=hidden,
            audio_hidden_states=audio.transpose(1, 2).flatten(2),
            encoder_hidden_states=context[..., :video_dim],
            audio_encoder_hidden_states=context[..., video_dim:],
            timestep=v_ts * scale,
            audio_timestep=a_ts * scale,
            num_frames=frames,
            height=height,
            width=width,
            fps=batch.extra["ltxav_frame_rate"],
            audio_num_frames=int(audio.shape[2]),
            return_latents=False,
            audio_replicated_for_sp=sp_size > 1,
        )
        if sp_size > 1:
            video_tokens = sequence_model_parallel_all_gather(
                video_tokens.contiguous(), dim=1
            )
        return [
            video_tokens.transpose(1, 2).reshape(video.shape),
            audio_tokens.unflatten(2, (audio.shape[1], -1)).transpose(1, 2),
        ]
