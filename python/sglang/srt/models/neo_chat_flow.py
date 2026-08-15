# SPDX-License-Identifier: Apache-2.0
"""Native flow-matching modules and tensor transforms for SenseNova U1."""

from __future__ import annotations

import math
from typing import Any

import torch
from sglang.srt.models.neo_chat_vision import NEOVisionModel
from torch import nn


class NEOChatTimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        timesteps: torch.Tensor,
        dimension: int,
        *,
        max_period: float = 10000.0,
    ) -> torch.Tensor:
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                half,
                dtype=torch.float32,
                device=timesteps.device,
            )
            / half
        )
        angles = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
        if dimension % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        frequencies = self.timestep_embedding(
            timesteps,
            self.frequency_embedding_size,
        )
        return self.mlp(frequencies.to(self.mlp[0].weight.dtype))


class NEOChatFlowModules(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if bool(getattr(config, "use_pixel_head", False)):
            raise NotImplementedError("SenseNova U1 use_pixel_head is not supported")
        if int(getattr(config, "fm_head_layers", 2)) > 2:
            raise NotImplementedError("SenseNova U1 deep fm_head is not supported")

        hidden_size = int(config.llm_config.hidden_size)
        patch_size = int(config.patch_size)
        merge_size = int(1 / float(config.downsample_ratio))
        output_size = 3 * (patch_size * merge_size) ** 2

        self.vision_model_mot_gen = NEOVisionModel(config.vision_config)
        self.timestep_embedder = NEOChatTimestepEmbedder(hidden_size)
        self.fm_head = nn.Sequential(
            nn.Linear(hidden_size, 4096, bias=True),
            nn.GELU(),
            nn.Linear(4096, output_size, bias=True),
        )
        self.add_noise_scale_embedding = bool(
            getattr(config, "add_noise_scale_embedding", False)
        )
        if self.add_noise_scale_embedding:
            self.noise_scale_embedder = NEOChatTimestepEmbedder(hidden_size)


def patchify_images(
    images: torch.Tensor,
    patch_size: int,
    *,
    channel_first: bool = False,
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [batch, 3, height, width]")
    if images.shape[2] % patch_size or images.shape[3] % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")

    grid_height = images.shape[2] // patch_size
    grid_width = images.shape[3] // patch_size
    patches = images.reshape(
        images.shape[0],
        3,
        grid_height,
        patch_size,
        grid_width,
        patch_size,
    )
    if channel_first:
        patches = torch.einsum("nchpwq->nhwcpq", patches)
    else:
        patches = torch.einsum("nchpwq->nhwpqc", patches)
    return patches.reshape(
        images.shape[0],
        grid_height * grid_width,
        patch_size**2 * 3,
    )


def unpatchify_images(
    patches: torch.Tensor,
    patch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")

    grid_height = height // patch_size
    grid_width = width // patch_size
    expected_tokens = grid_height * grid_width
    expected_features = patch_size**2 * 3
    if patches.shape[1:] != (expected_tokens, expected_features):
        raise ValueError("patch tensor shape does not match image geometry")

    images = patches.reshape(
        patches.shape[0],
        grid_height,
        grid_width,
        patch_size,
        patch_size,
        3,
    )
    images = torch.einsum("nhwpqc->nchpwq", images)
    return images.reshape(patches.shape[0], 3, height, width)


def apply_u1_time_schedule(
    timesteps: torch.Tensor,
    *,
    image_seq_len: int,
    timestep_shift: float,
    time_schedule: str,
    time_shift_type: str,
    base_shift: float,
    max_shift: float,
    base_image_seq_len: int,
    max_image_seq_len: int,
) -> torch.Tensor:
    schedule = time_schedule
    if timestep_shift != 1:
        schedule = "standard"

    sigma = 1 - timesteps
    if schedule == "standard":
        shift = timestep_shift
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
    elif schedule == "dynamic":
        denominator = max_image_seq_len - base_image_seq_len
        dynamic_shift = (
            base_shift
            if denominator == 0
            else (
                (max_shift - base_shift) / denominator * image_seq_len
                + base_shift
                - (max_shift - base_shift) / denominator * base_image_seq_len
            )
        )
        shift_tensor = timesteps.new_tensor(dynamic_shift)
        if time_shift_type == "exponential":
            shift = torch.exp(shift_tensor)
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        elif time_shift_type == "linear":
            sigma = shift_tensor / (shift_tensor + (1 / sigma - 1))
        else:
            raise ValueError(f"Unsupported time_shift_type: {time_shift_type}")
    else:
        raise ValueError(f"Unsupported time_schedule: {schedule}")
    return 1 - sigma


def compute_u1_noise_scale(
    *,
    grid_height: int,
    grid_width: int,
    merge_size: int,
    noise_scale: float,
    noise_scale_mode: str,
    base_image_seq_len: int,
    max_value: float,
) -> float:
    if noise_scale_mode in {"resolution", "dynamic", "dynamic_sqrt"}:
        scale = math.sqrt(
            (grid_height * grid_width) / (merge_size**2) / float(base_image_seq_len)
        )
        noise_scale *= scale
        if noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
    return min(noise_scale, max_value)


def build_u1_flow_batch_layout(
    positions: torch.Tensor,
    extend_seq_lens: list[int],
    extend_prefix_lens: list[int],
    flow_specs: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(extend_seq_lens) != len(extend_prefix_lens) or len(flow_specs) != len(
        extend_seq_lens
    ):
        raise ValueError("flow layout batch fields must have matching lengths")

    positions = positions.flatten()
    indexes = torch.stack(
        [positions, torch.zeros_like(positions), torch.zeros_like(positions)],
        dim=0,
    )
    image_gen_indicators = torch.zeros_like(positions, dtype=torch.bool)

    token_offset = 0
    for extend_len, prefix_len, spec in zip(
        extend_seq_lens,
        extend_prefix_lens,
        flow_specs,
        strict=True,
    ):
        image_start = int(spec["image_start"])
        image_tokens = int(spec["image_tokens"])
        token_height = int(spec["token_height"])
        token_width = int(spec["token_width"])
        if image_tokens != token_height * token_width:
            raise ValueError("flow image token geometry is inconsistent")
        if prefix_len > image_start:
            raise RuntimeError(
                "flow image tokens were matched by Radix cache; use a unique extra_key"
            )

        absolute_positions = prefix_len + torch.arange(
            extend_len,
            dtype=torch.long,
            device=positions.device,
        )
        request_image_mask = (absolute_positions >= image_start) & (
            absolute_positions < image_start + image_tokens
        )
        visible_image_tokens = int(request_image_mask.sum().item())
        if visible_image_tokens != image_tokens:
            raise ValueError(
                "the complete flow image block must be present in the extend window"
            )

        request_slice = slice(token_offset, token_offset + extend_len)
        request_indexes = indexes[:, request_slice]
        image_offsets = absolute_positions[request_image_mask] - image_start
        request_indexes[0, request_image_mask] = int(spec["image_t_index"])
        request_indexes[1, request_image_mask] = image_offsets // token_width
        request_indexes[2, request_image_mask] = image_offsets % token_width
        image_gen_indicators[request_slice] = request_image_mask
        token_offset += extend_len

    if token_offset != positions.numel():
        raise ValueError("extend lengths do not consume all flow positions")
    return indexes, image_gen_indicators


__all__ = [
    "NEOChatFlowModules",
    "NEOChatTimestepEmbedder",
    "apply_u1_time_schedule",
    "build_u1_flow_batch_layout",
    "compute_u1_noise_scale",
    "patchify_images",
    "unpatchify_images",
]
