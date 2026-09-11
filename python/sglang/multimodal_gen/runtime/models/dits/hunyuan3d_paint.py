# SPDX-License-Identifier: Apache-2.0
"""Native multi-view UNet used by Hunyuan3D Paint."""

from __future__ import annotations

import copy
from typing import Any

import torch
from einops import rearrange
from torch import nn

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.stable_diffusion import (
    BasicTransformerBlock,
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    StableDiffusionAttention,
    StableDiffusionUNet2DConditionModel,
    StableDiffusionUNetConfig,
    StableDiffusionUNetOutput,
    Transformer2DModel,
    UNetMidBlock2DCrossAttn,
)


class Hunyuan3DPaintTransformerBlock(nn.Module):
    def __init__(
        self,
        transformer: BasicTransformerBlock,
        layer_name: str,
        *,
        use_multiview_attention: bool,
        use_reference_attention: bool,
        is_turbo: bool,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.layer_name = layer_name
        self.use_multiview_attention = use_multiview_attention
        self.use_reference_attention = use_reference_attention
        self.is_turbo = is_turbo
        self.attn_multiview = (
            StableDiffusionAttention(
                transformer.dim,
                transformer.num_attention_heads,
                transformer.attention_head_dim,
            )
            if use_multiview_attention
            else None
        )
        self.attn_refview = (
            StableDiffusionAttention(
                transformer.dim,
                transformer.num_attention_heads,
                transformer.attention_head_dim,
            )
            if use_reference_attention
            else None
        )
        if is_turbo:
            self._initialize_added_attention()

    def _initialize_added_attention(self) -> None:
        for attention in (self.attn_multiview, self.attn_refview):
            if attention is None:
                continue
            attention.load_state_dict(self.transformer.attn1.state_dict())
            with torch.no_grad():
                for parameter in attention.to_out[0].parameters():
                    parameter.zero_()

    @staticmethod
    def _broadcast_scale(
        scale: float | torch.Tensor,
        output: torch.Tensor,
        num_views: int,
    ) -> float | torch.Tensor:
        if not isinstance(scale, torch.Tensor):
            return scale
        scale = scale.unsqueeze(1).repeat(1, num_views).reshape(-1)
        for _ in range(output.ndim - 1):
            scale = scale.unsqueeze(-1)
        return scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        options = {} if cross_attention_kwargs is None else cross_attention_kwargs
        num_views = int(options.get("num_in_batch", 1))
        mode = options.get("mode")
        condition_embeddings = options.get("condition_embed_dict")
        if mode is not None and not isinstance(condition_embeddings, dict):
            raise ValueError("Hunyuan3D reference attention requires a shared cache.")

        normalized = self.transformer.norm1(hidden_states)
        hidden_states = hidden_states + self.transformer.attn1(
            normalized, attention_mask=attention_mask
        )

        if mode is not None and "w" in mode:
            condition_embeddings[self.layer_name] = rearrange(
                normalized, "(b n) l c -> b (n l) c", n=num_views
            )

        if mode is not None and "r" in mode and self.use_reference_attention:
            if self.attn_refview is None:
                raise RuntimeError("Reference attention was not initialized.")
            reference = condition_embeddings[self.layer_name]
            reference = reference.unsqueeze(1).repeat(1, num_views, 1, 1)
            reference = rearrange(reference, "b n l c -> (b n) l c")
            reference_output = self.attn_refview(
                normalized, encoder_hidden_states=reference
            )
            reference_scale = self._broadcast_scale(
                1.0 if self.is_turbo else options.get("ref_scale", 1.0),
                reference_output,
                num_views,
            )
            hidden_states = hidden_states + reference_scale * reference_output

        if num_views > 1 and self.use_multiview_attention:
            if self.attn_multiview is None:
                raise RuntimeError("Multiview attention was not initialized.")
            multiview = rearrange(normalized, "(b n) l c -> b (n l) c", n=num_views)
            position_masks = options.get("position_attn_mask")
            position_mask = None
            if isinstance(position_masks, dict):
                position_mask = position_masks.get(multiview.shape[1])
            multiview_output = self.attn_multiview(
                multiview,
                encoder_hidden_states=multiview,
                attention_mask=position_mask,
            )
            multiview_output = rearrange(
                multiview_output, "b (n l) c -> (b n) l c", n=num_views
            )
            multiview_scale = 1.0 if self.is_turbo else options.get("mva_scale", 1.0)
            hidden_states = hidden_states + multiview_scale * multiview_output

        hidden_states = hidden_states + self.transformer.attn2(
            self.transformer.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        return hidden_states + self.transformer.ff(
            self.transformer.norm3(hidden_states)
        )


def _replace_transformer_blocks(
    unet: StableDiffusionUNet2DConditionModel,
    *,
    use_multiview_attention: bool,
    use_reference_attention: bool,
    is_turbo: bool,
) -> None:
    def replace(model: Transformer2DModel, layer_name: str) -> None:
        transformer = model.transformer_blocks[0]
        if not isinstance(transformer, BasicTransformerBlock):
            raise TypeError(
                f"Expected BasicTransformerBlock, got {type(transformer).__name__}."
            )
        model.transformer_blocks[0] = Hunyuan3DPaintTransformerBlock(
            transformer,
            layer_name,
            use_multiview_attention=use_multiview_attention,
            use_reference_attention=use_reference_attention,
            is_turbo=is_turbo,
        )

    for block_index, block in enumerate(unet.down_blocks):
        if not isinstance(block, CrossAttnDownBlock2D):
            continue
        for attention_index, attention in enumerate(block.attentions):
            replace(attention, f"down_{block_index}_{attention_index}_0")

    mid_block = unet.mid_block
    if not isinstance(mid_block, UNetMidBlock2DCrossAttn):
        raise TypeError(f"Unexpected SD2 mid block: {type(mid_block).__name__}.")
    replace(mid_block.attentions[0], "mid_0_0")

    for block_index, block in enumerate(unet.up_blocks):
        if not isinstance(block, CrossAttnUpBlock2D):
            continue
        for attention_index, attention in enumerate(block.attentions):
            replace(attention, f"up_{block_index}_{attention_index}_0")


@torch.no_grad()
def compute_voxel_grid_mask(
    position: torch.Tensor, grid_resolution: int = 8
) -> torch.Tensor:
    position = position.half()
    _, _, _, height, width = position.shape
    if height % grid_resolution != 0 or width % grid_resolution != 0:
        raise ValueError(
            f"Position map {height}x{width} is not divisible by {grid_resolution}."
        )
    valid_mask = (position != 1).all(dim=2, keepdim=True).expand_as(position)
    position = position.masked_fill(~valid_mask, 0)
    position = rearrange(
        position,
        "b n c (nh gh) (nw gw) -> b n nh nw c gh gw",
        nh=grid_resolution,
        nw=grid_resolution,
    )
    valid_mask = rearrange(
        valid_mask,
        "b n c (nh gh) (nw gw) -> b n nh nw c gh gw",
        nh=grid_resolution,
        nw=grid_resolution,
    )
    counts = valid_mask.sum(dim=(-2, -1))
    grid_position = position.sum(dim=(-2, -1)) / counts.clamp(min=1)
    grid_position = grid_position.masked_fill(counts < 5, 0)
    grid_position = rearrange(grid_position, "b n h w c -> b n (h w) c")
    lhs = grid_position.unsqueeze(2).unsqueeze(4)
    rhs = grid_position.unsqueeze(1).unsqueeze(3)
    return torch.linalg.vector_norm(lhs - rhs, dim=-1) < 1.73 / grid_resolution


def compute_multi_resolution_mask(
    position_maps: torch.Tensor,
    grid_resolutions: tuple[int, ...] = (32, 16, 8),
) -> dict[int, torch.Tensor]:
    masks: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for grid_resolution in grid_resolutions:
            mask = compute_voxel_grid_mask(position_maps, grid_resolution)
            mask = rearrange(mask, "b ni nj li lj -> b (ni li) (nj lj)")
            masks[mask.shape[1]] = mask
    return masks


class Hunyuan3DPaintUNet(nn.Module, LayerwiseOffloadableModuleMixin):
    layer_names = [
        "unet.down_blocks",
        "unet.up_blocks",
        "unet_dual.down_blocks",
        "unet_dual.up_blocks",
    ]
    layerwise_offload_dit_group_enabled = True

    def __init__(
        self,
        config: StableDiffusionUNetConfig,
        *,
        is_turbo: bool = False,
    ) -> None:
        super().__init__()
        base_unet = StableDiffusionUNet2DConditionModel(config)
        self.unet = base_unet
        self.unet_dual = copy.deepcopy(base_unet)

        _replace_transformer_blocks(
            self.unet_dual,
            use_multiview_attention=False,
            use_reference_attention=False,
            is_turbo=is_turbo,
        )
        _replace_transformer_blocks(
            self.unet,
            use_multiview_attention=True,
            use_reference_attention=True,
            is_turbo=is_turbo,
        )

        self.unet.conv_in = nn.Conv2d(12, self.unet.conv_in.out_channels, 3, padding=1)
        self.unet.learned_text_clip_gen = nn.Parameter(
            torch.randn(1, 77, config.cross_attention_dim)
        )
        self.unet.learned_text_clip_ref = nn.Parameter(
            torch.randn(1, 77, config.cross_attention_dim)
        )
        self.max_num_ref_images = 5
        self.max_num_generated_images = 44
        time_embedding_dim = config.block_out_channels[0] * 4
        self.unet.class_embedding = nn.Embedding(
            self.max_num_ref_images + self.max_num_generated_images,
            time_embedding_dim,
        )

    @property
    def config(self) -> StableDiffusionUNetConfig:
        return self.unet.config

    @property
    def dtype(self) -> torch.dtype:
        return self.unet.dtype

    @property
    def learned_text_clip_gen(self) -> torch.Tensor:
        return self.unet.learned_text_clip_gen

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        ref_latents: torch.Tensor,
        num_in_batch: int,
        condition_embed_dict: dict[str, torch.Tensor],
        normal_imgs: torch.Tensor | None = None,
        position_imgs: torch.Tensor | None = None,
        camera_info_gen: torch.Tensor,
        camera_info_ref: torch.Tensor,
        ref_scale: float | torch.Tensor = 1.0,
        mva_scale: float | torch.Tensor = 1.0,
        position_attn_mask: dict[int, torch.Tensor] | None = None,
        timestep_cond: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        return_dict: bool = True,
    ) -> StableDiffusionUNetOutput | tuple[torch.Tensor]:
        if timestep_cond is not None or cross_attention_kwargs is not None:
            raise ValueError("Hunyuan3D Paint does not use extra UNet conditioning.")
        if added_cond_kwargs is not None:
            raise ValueError("Hunyuan3D Paint does not use added conditioning.")
        batch_size, num_generated, _, height, width = sample.shape
        if height != width or num_generated != num_in_batch:
            raise ValueError(
                "Hunyuan3D Paint expects square latents and a matching view count."
            )

        camera_gen = rearrange(
            camera_info_gen + self.max_num_ref_images, "b n -> (b n)"
        )
        inputs = [sample]
        if normal_imgs is not None:
            inputs.append(normal_imgs)
        if position_imgs is not None:
            inputs.append(position_imgs)
        sample = rearrange(torch.cat(inputs, dim=2), "b n c h w -> (b n) c h w")
        encoder_gen = encoder_hidden_states.unsqueeze(1).repeat(1, num_generated, 1, 1)
        encoder_gen = rearrange(encoder_gen, "b n l c -> (b n) l c")

        if not condition_embed_dict:
            num_reference = ref_latents.shape[1]
            camera_ref = rearrange(camera_info_ref, "b n -> (b n)")
            reference = rearrange(ref_latents, "b n c h w -> (b n) c h w")
            encoder_ref = self.unet.learned_text_clip_ref.unsqueeze(1).repeat(
                batch_size, num_reference, 1, 1
            )
            encoder_ref = rearrange(encoder_ref, "b n l c -> (b n) l c")
            self.unet_dual(
                reference,
                0,
                encoder_ref,
                class_labels=camera_ref,
                return_dict=False,
                cross_attention_kwargs={
                    "mode": "w",
                    "num_in_batch": num_reference,
                    "condition_embed_dict": condition_embed_dict,
                },
            )

        options: dict[str, Any] = {
            "mode": "r",
            "num_in_batch": num_generated,
            "condition_embed_dict": condition_embed_dict,
            "mva_scale": mva_scale,
            "ref_scale": ref_scale,
        }
        if position_attn_mask is not None:
            options["position_attn_mask"] = position_attn_mask
        return self.unet(
            sample,
            timestep,
            encoder_gen,
            class_labels=camera_gen,
            return_dict=return_dict,
            cross_attention_kwargs=options,
        )


EntryClass = Hunyuan3DPaintUNet
