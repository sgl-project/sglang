# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/radio.py

import logging
import math
from collections.abc import Iterable
from itertools import repeat
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    replace_prefix,
    replace_substrings,
)
from sglang.srt.models.internvl import InternVisionEncoder

logger = logging.getLogger(__name__)

input_dim_t: TypeAlias = int | tuple[int, int]
norm_t: TypeAlias = tuple[float, float, float] | torch.Tensor


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class ClsToken(nn.Module):
    def __init__(
        self,
        ndim: int,
        num_tokens: int = 1,
        enabled: bool = True,
        register_multiple: int | None = None,
        num_registers: int | None = None,
    ):
        super().__init__()

        self.ndim = ndim
        self.enabled = enabled
        self.num_registers = 0
        self.num_tokens = num_tokens
        if enabled:
            if num_registers:
                self.num_registers = num_registers
            elif register_multiple:
                self.num_registers = register_multiple - (
                    num_tokens % register_multiple
                )

            scale = ndim**-0.5
            self.token = nn.Parameter(
                torch.randn(num_tokens + self.num_registers, ndim) * scale
            )

        else:
            self.token = None

        self.num_patches = self.num_tokens + self.num_registers

    def forward(self, x: torch.Tensor):
        if self.token is None:
            return x

        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat(
            [
                token,
                x,
            ],
            dim=1,
        )

        return x


class ViTPatchGenerator(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        input_dims: input_dim_t,
        abs_pos: bool = True,
        normalize_patches: bool = False,
        cls_token: bool = False,
        max_input_dims: input_dim_t | None = None,
        pos_dropout: float = 0.0,
        return_pos_enc: bool = False,
        num_cls_tokens: int = 1,
        register_multiple: int | None = None,
        num_registers: int | None = None,
        patch_bias: bool = False,
        video_temporal_patch_size: int = 1,
        separate_video_embedder: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(input_dims, int):
            input_dims = (input_dims, input_dims)

        if max_input_dims is None:
            max_input_dims = input_dims
        if isinstance(max_input_dims, int):
            max_input_dims = (max_input_dims, max_input_dims)

        max_input_dims = tuple(
            int(math.ceil(d / patch_size) * patch_size) for d in max_input_dims
        )

        self.cpe_mode = max_input_dims != input_dims
        self.pos_dropout = pos_dropout
        self.return_pos_enc = return_pos_enc

        factory = dict(device=device, dtype=dtype)

        self.patch_size = patch_size
        self.abs_pos = abs_pos
        self.embed_dim = embed_dim

        self.num_rows = max_input_dims[0] // patch_size
        self.num_cols = max_input_dims[1] // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols
        self.max_input_dims = max_input_dims

        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(
            patch_size, embed_dim, bias=patch_bias, **factory
        )

        if abs_pos:
            scale = embed_dim**-0.5
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches, embed_dim, **factory) * scale
            )

        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            enabled=cls_token,
            register_multiple=register_multiple,
            num_registers=num_registers,
        )

        self.patch_normalizer = (
            nn.LayerNorm(embed_dim) if normalize_patches else nn.Identity()
        )

        self.video_temporal_patch_size = video_temporal_patch_size
        self.video_embedder = None
        self._video_embedder_loaded = False
        if video_temporal_patch_size > 1 and separate_video_embedder:
            self.video_embedder = nn.Linear(
                3 * video_temporal_patch_size * patch_size * patch_size,
                embed_dim,
                bias=False,
                **factory,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.embed_patches(x)
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        if self.return_pos_enc:
            return patches, pos_enc
        return patches

    def forward_video(self, x: torch.Tensor, temporal_patch_size: int) -> torch.Tensor:
        """Embed video frames with temporal compression via tubelet grouping."""
        assert (
            self.video_embedder is not None
        ), "video_embedder is required for temporal compression"
        T = temporal_patch_size
        num_frames = x.shape[0]

        if num_frames % T != 0:
            pad = T - (num_frames % T)
            x = torch.cat(
                [x, x[-1:].expand(pad, -1, -1, -1)],
                dim=0,
            )

        padded_frames = x.shape[0]
        num_tubelets = padded_frames // T

        patches = self.im_to_patches(x)
        num_spatial = patches.shape[1]
        feat_dim = patches.shape[2]

        patches = patches.reshape(num_tubelets, T, num_spatial, feat_dim)
        patches = patches.permute(0, 2, 1, 3).reshape(
            num_tubelets, num_spatial, T * feat_dim
        )

        patches = self.video_embedder(patches)

        patches, _ = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        return patches

    @property
    def apply_cls_token(self):
        return self.cls_token.enabled

    @property
    def num_cls_tokens(self):
        return self.cls_token.num_tokens

    @property
    def num_cls_patches(self):
        return self.cls_token.num_patches

    @property
    def num_registers(self):
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        return self.num_cls_tokens + self.num_registers

    def _load_embed(self, src_embed: torch.Tensor, targ_embed: nn.Parameter):
        if src_embed.shape != targ_embed.shape:
            src_size = int(math.sqrt(src_embed.shape[1]))

            assert (
                src_size**2 == src_embed.shape[1]
            ), "Unable to interpolate non-square embedding"

            src_embed = rearrange(
                src_embed, "b (h w) c -> b c h w", h=src_size, w=src_size
            )
            src_embed = F.interpolate(
                src_embed,
                size=(self.num_rows, self.num_cols),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            src_embed = rearrange(src_embed, "b c h w -> b (h w) c")
        targ_embed.data.copy_(src_embed)

    def _load_projection(
        self, src_proj_weight: torch.Tensor, targ_proj_weight: torch.Tensor
    ):
        if src_proj_weight.shape != targ_proj_weight.shape:
            src_patch_size = int(math.sqrt(src_proj_weight.shape[1] // 3))

            assert (src_patch_size**2) * 3 == src_proj_weight.shape[
                1
            ], "Unable to interpolate non-square patch size"

            src_proj_weight = rearrange(
                src_proj_weight,
                "b (c h w) -> b c h w",
                c=3,
                h=src_patch_size,
                w=src_patch_size,
            )
            src_proj_weight = F.interpolate(
                src_proj_weight,
                size=(self.patch_size, self.patch_size),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            src_proj_weight = rearrange(src_proj_weight, "b c h w -> b (c h w)")
        targ_proj_weight.data.copy_(src_proj_weight)

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.im_to_patches(x)
        patches = self.embedder(patches)
        return patches

    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: torch.Tensor | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if not self.abs_pos:
            return patches

        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)

        if self.training and self.pos_dropout > 0:
            keeps = (
                torch.rand(
                    patches.shape[0], 1, 1, dtype=pos_enc.dtype, device=pos_enc.device
                )
                > self.pos_dropout
            )
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc

        return patches + pos_enc_drop, pos_enc

    def get_pos_enc(
        self,
        batch_size: int,
        patch_idxs: torch.Tensor | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_size for d in input_size)

        pos_embed = self._get_pos_embeddings(batch_size, input_dims)

        if patch_idxs is None:
            return pos_embed

        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])

        pos_embed = torch.gather(
            pos_embed.expand(patch_idxs.shape[0], -1, -1), dim=1, index=exp_patch_idxs
        )
        return pos_embed

    def _get_pos_embeddings(self, batch_size: int, input_dims: tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pos_embed

        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(
            0, 3, 1, 2
        )

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., : input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, : input_dims[1]]
            return pos_embed

        if self.cpe_mode:
            max_dim = max(input_dims)
            pos_embed = F.interpolate(
                pos_embed.float(),
                size=(max_dim, max_dim),
                align_corners=False,
                mode="bilinear",
            ).to(pos_embed.dtype)

            pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(
                pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear"
            ).to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed


class Im2Patches(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size
        px = x.shape[-1] // self.patch_size
        patches = rearrange(
            x,
            "b c (py yy) (px xx) -> b (py px) (c yy xx)",
            py=py,
            yy=self.patch_size,
            px=px,
            xx=self.patch_size,
        )
        return patches


class ViTPatchLinear(nn.Linear):
    def __init__(self, patch_size: int, embed_dim: int, bias: bool = False, **factory):
        super().__init__(3 * (patch_size**2), embed_dim, bias=bias, **factory)
        self.patch_size = patch_size


class RadioInternVisionModel(nn.Module):
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            to_2tuple(config.patch_size), config.image_size
        )
        max_img_size = int(
            round(config.max_img_size / config.patch_size) * config.patch_size
        )
        video_temporal_patch_size = getattr(config, "video_temporal_patch_size", 1)
        separate_video_embedder = getattr(config, "separate_video_embedder", True)

        self.patch_generator = ViTPatchGenerator(
            config.patch_size,
            config.hidden_size,
            input_dims=self.img_size,
            max_input_dims=max_img_size,
            cls_token=True,
            register_multiple=config.reg_tokens,
            video_temporal_patch_size=video_temporal_patch_size,
            separate_video_embedder=separate_video_embedder,
        )

        self.encoder = InternVisionEncoder(config=config, quant_config=quant_config)

    def _init_img_size(self, patch_size, img_size: int | tuple[int, int]):
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        assert self.patch_generator is not None
        hidden_states = self.patch_generator(x)
        encoder_outputs = self.encoder.forward(inputs_embeds=hidden_states)
        assert isinstance(encoder_outputs, BaseModelOutput)
        return encoder_outputs.last_hidden_state


class RadioModel(nn.Module):
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.model = RadioInternVisionModel(
            config=config,
            quant_config=quant_config,
        )

    def forward(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
        num_frames: int | None = None,
    ) -> torch.FloatTensor:
        if (
            num_frames is not None
            and getattr(self.config, "video_temporal_patch_size", 1) > 1
        ):
            return self._forward_video_temporal(pixel_values, num_frames)
        if isinstance(pixel_values, list):
            return self._forward_dynamic(pixel_values)
        y = self.model(pixel_values)
        return self._extract_final(y)

    def _forward_dynamic(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[int]]:
        """Process variable-size images with ragged packing via cu_seqlens."""
        patch_gen = self.model.patch_generator
        all_patches = []
        seqlens = [0]

        for img in images:
            patches = patch_gen(img)
            seq_len = patches.shape[1]
            all_patches.append(patches.squeeze(0))
            seqlens.append(seqlens[-1] + seq_len)

        hidden = torch.cat(all_patches, dim=0).unsqueeze(0)
        cu_seqlens = torch.tensor(seqlens, dtype=torch.int32, device=hidden.device)

        out = self.model.encoder.forward(inputs_embeds=hidden, cu_seqlens=cu_seqlens)
        features = out.last_hidden_state

        num_skip = patch_gen.num_skip
        per_image_features = []
        num_patches_list = []
        for i in range(len(images)):
            start = seqlens[i] + num_skip
            end = seqlens[i + 1]
            per_image_features.append(features[0, start:end])
            num_patches_list.append(end - start)

        return (
            torch.cat(per_image_features, dim=0).unsqueeze(0),
            num_patches_list,
        )

    def _forward_video_temporal(
        self, pixel_values: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Process video frames with temporal compression (tubelet grouping)."""
        T = self.config.video_temporal_patch_size
        patch_gen = self.model.patch_generator

        patches = patch_gen.forward_video(pixel_values, T)
        num_tubelets = patches.shape[0]
        seq_per_tubelet = patches.shape[1]

        cu_seqlens = torch.arange(
            0,
            (num_tubelets + 1) * seq_per_tubelet,
            seq_per_tubelet,
            dtype=torch.int32,
            device=patches.device,
        )
        packed = patches.reshape(1, -1, patches.shape[-1])

        out = self.model.encoder.forward(inputs_embeds=packed, cu_seqlens=cu_seqlens)
        features = out.last_hidden_state.reshape(num_tubelets, seq_per_tubelet, -1)

        num_skip = patch_gen.num_skip
        return features[:, num_skip:]

    def load_weights(self, weights) -> set[str]:
        remap_substrings = {
            "attn": "attn.attn",
            "qkv": "qkv_proj",
            "blocks": "encoder.layers",
        }
        remap_prefixes = {
            "radio_model.": "",
        }

        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters())

        if isinstance(weights, dict):
            weights_list = list(weights.items())
        else:
            weights_list = list(weights)

        for name, weight in weights_list:
            if not name.startswith("radio_model."):
                # Skip non-radio weights
                continue
            name = replace_substrings(name, remap_substrings)
            name = replace_prefix(name, remap_prefixes)
            if name and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(name)
                if "video_embedder" in name:
                    self.model.patch_generator._video_embedder_loaded = True

        return loaded_params

    def _extract_final(self, y: torch.Tensor):
        # Remove CLS + REGISTERS tokens
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            all_feat = y[:, patch_gen.num_skip :]

        return all_feat
