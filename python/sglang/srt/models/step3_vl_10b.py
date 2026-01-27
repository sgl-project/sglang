# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This is basically a copy from perception_models/core/vision_encoder/pe.py"""

from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from sglang.srt.configs.step3_vl import Step3VLConfig
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.utils import add_prefix

_DEFAULT_NORM_LAYER = partial(nn.LayerNorm, eps=1e-5)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], (
        "feature dimension {} is not of sufficient size to rotate in all the "
        "positions {}".format(t.shape[-1], rot_dim)
    )

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t_left, t, t_right), dim=-1)

    return out.type(dtype)


class PerceptionEncoderRope2D(nn.Module):
    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        theta=10000,
        max_freq=10,
        num_freqs=1,
        theta_rescale_factor=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        cache = self._compute_2d_freqs()
        self.register_buffer("freqs_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: int | float, dim: int) -> torch.Tensor:
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        return freqs

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor):
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def _compute_2d_freqs(self) -> torch.Tensor:
        grid_h_range = torch.arange(self.max_grid_height, dtype=torch.float)
        grid_w_range = torch.arange(self.max_grid_width, dtype=torch.float)
        if self.use_cls_token:
            grid_h_range += 1
            grid_w_range += 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        freqs_h = self._compute_freqs(grid_h_range, inv_freq)[:, None].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs_w = self._compute_freqs(grid_w_range, inv_freq)[None, :].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(
            self.max_grid_height * self.max_grid_width, -1
        )
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)
        freqs = freqs[None, None, ...]
        return freqs

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int], x_shape
    ):
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            if self.use_cls_token:
                positions = torch.cat(
                    [torch.zeros(1, device=q.device), positions + 1], dim=0
                )
                positions = positions.to(torch.long)
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        ori_shape = q.shape
        bs, seq_len, _ = x_shape
        q = q.view(bs, seq_len, -1, self.dim).permute(0, 2, 1, 3)
        k = k.view(bs, seq_len, -1, self.dim).permute(0, 2, 1, 3)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        q = q.permute(0, 2, 1, 3).reshape(ori_shape)
        k = k.permute(0, 2, 1, 3).reshape(ori_shape)
        return q, k


class PerceptionEncoderLayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PerceptionEncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        act_layer: Callable[[], nn.Module],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_dim,
            hidden_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.activation = act_layer
        self.fc2 = RowParallelLinear(
            hidden_dim,
            input_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.activation(x)
        x, _ = self.fc2(x)
        return x


class PerceptionEncoderVisionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_grid_height: int,
        max_grid_width: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        use_cls_token: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.head_dim = d_model // n_head
        self.rope = PerceptionEncoderRope2D(
            dim=self.head_dim,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
        )
        self.attn = VisionAttention(
            embed_dim=d_model,
            num_heads=n_head,
            projection_size=d_model,
            use_qkv_parallel=True,
            proj_bias=True,
            # flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            customized_position_embedding_applier=self.rope,
        )
        self.ls_1 = (
            PerceptionEncoderLayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            PerceptionEncoderLayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = PerceptionEncoderMLP(
            d_model,
            hidden_dim,
            act_layer,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]):
        x = x + self.ls_1(self.attn(self.ln_1(x), position_embeddings=grid_hw))  # hacky
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class PerceptionEncoderVisionTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        max_grid_height: int,
        max_grid_width: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        use_cls_token: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                PerceptionEncoderVisionBlock(
                    d_model=width,
                    n_head=heads,
                    max_grid_height=max_grid_height,
                    max_grid_width=max_grid_width,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    use_cls_token=use_cls_token,
                    quant_config=quant_config,
                    prefix=f"{prefix}.resblocks.{i}",
                )
                for i in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]):
        for block in self.resblocks:
            x = block(x, grid_hw=grid_hw)
        return x


class PerceptionEncoder(nn.Module):
    def __init__(
        self,
        config,
        act_layer: Callable,
        norm_layer: Callable = _DEFAULT_NORM_LAYER,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.patch_size = config.patch_size

        self.output_dim = config.output_dim or config.width
        self.heads = config.heads
        self.width = config.width
        self.layers = config.layers

        self.use_abs_posemb = config.use_abs_posemb
        self.use_cls_token = config.use_cls_token
        self.use_rope2d = config.use_rope2d
        if not self.use_rope2d:
            raise ValueError("use_rope2d must be True")
        self.image_size = config.image_size

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.ln_pre = norm_layer(config.width) if config.use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(self.width) if config.use_ln_post else nn.Identity()

        self.transformer = PerceptionEncoderVisionTransformer(
            config.width,
            config.layers,
            config.heads,
            max_grid_height=self.image_size // self.patch_size,
            max_grid_width=self.image_size // self.patch_size,
            mlp_ratio=config.mlp_ratio,
            ls_init_value=config.ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_cls_token=self.use_cls_token,
            quant_config=quant_config,
            prefix=f"{prefix}.transformer",
        )

        self.vit_downsampler1 = nn.Conv2d(
            config.width, config.width * 2, kernel_size=3, stride=2, padding=1
        )
        self.vit_downsampler2 = nn.Conv2d(
            config.width * 2, config.width * 4, kernel_size=3, stride=2, padding=1
        )

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(
                (self.width**-0.5) * torch.randn(self.width)
            )

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                (self.width**-0.5)
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2,
                    self.width,
                )
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def sample_abs_posemb(self, grid_h: int, grid_w: int):
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width)

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    def forward_features(self, x: torch.Tensor):
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x], dim=1
            )

        if self.use_abs_posemb:
            x = x + self.sample_abs_posemb(grid_h, grid_w)

        x = self.ln_pre(x)
        x = self.transformer(x, grid_hw=(grid_h, grid_w))
        x = self.ln_post(x)

        if self.use_cls_token:
            x = x[:, 1:, :]

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        B, P, C = x.shape
        T = int(P**0.5)
        x = x.transpose(2, 1).contiguous()
        x = x.view(B, C, T, T)

        x = self.vit_downsampler1(x)
        x = self.vit_downsampler2(x)

        B, C, T, T = x.shape
        return x.view(B, -1, T * T).transpose(1, 2)


class StepVLForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Step3VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config
        self.vision_model = PerceptionEncoder(
            config.vision_config,
            ACT2FN[config.vision_config.hidden_act],
            quant_config=quant_config,
            prefix=add_prefix(prefix, "vision_model"),
        )
        self.vit_large_projector = ColumnParallelLinear(
            config.vision_config.width * 4,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix(prefix, "vit_large_projector"),
        )

        self.language_model = Qwen3ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix(prefix, "language_model"),
        )

    def _get_vision_model_output(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.vision_model(input_tensor)

    @property
    def device(self) -> torch.device:
        return self.vit_large_projector.weight.device

    def _flatten_embeddings(self, embeddings) -> torch.Tensor:

        if isinstance(embeddings, torch.Tensor):
            # Flatten all but the last dimension.
            return embeddings.flatten(0, -2)

        return torch.cat(tuple(self._flatten_embeddings(t) for t in embeddings))

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features, _ = self.vit_large_projector(image_features)
        return image_features

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        assert len(items) == 1  # We only have images.

        item = items[0]
        pixel_values = item.feature.type(self.vision_model.dtype).to(self.device)
        num_patches = item.model_specific_data.get("num_patches")
        patch_pixel_values = item.model_specific_data.get("patch_pixel_values", None)
        if patch_pixel_values is not None:
            patch_pixel_values = patch_pixel_values.type(self.vision_model.dtype).to(
                self.device
            )

        image_features = self._get_vision_model_output(pixel_values)
        patch_image_features = (
            self._get_vision_model_output(patch_pixel_values)
            if patch_pixel_values is not None
            else None
        )
        image_features = self._process_image_features(image_features)
        patch_image_features = (
            self._process_image_features(patch_image_features)
            if patch_image_features is not None
            else None
        )
        merged_image_features = []
        cur_patch_idx = 0
        for i, num_patch in enumerate(num_patches):
            cur_feature = []
            if num_patch > 0:
                patch_slice = patch_image_features[
                    cur_patch_idx : cur_patch_idx + num_patch
                ]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))
            cur_feature.append(image_features[i].view(-1, image_features.shape[-1]))
            cur_patch_idx += num_patch
            merged_image_features.append(
                torch.cat(cur_feature) if len(cur_feature) > 1 else cur_feature[0]
            )
        return self._flatten_embeddings(merged_image_features)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model, separating vision and language weights"""
        weights = list(weights)

        # Separate vision tower weights and language model weights
        vision_weights = []
        language_weights = []

        for name, loaded_weight in weights:
            if "vision_model" in name or "vit_large_projector" in name:
                name = name.replace(r".attn.in_proj_weight", r".attn.qkv_proj.weight")
                name = name.replace(r".attn.in_proj_bias", r".attn.qkv_proj.bias")
                name = name.replace(r".attn.out_proj.bias", r".attn.proj.bias")
                name = name.replace(r".attn.out_proj.weight", r".attn.proj.weight")
                name = name.replace(".mlp.c_fc", ".mlp.fc1")
                name = name.replace(".mlp.c_proj", ".mlp.fc2")
                vision_weights.append((name, loaded_weight))
            else:
                # All other weights go to language model
                language_weights.append((name, loaded_weight))

        # Load vision tower weights
        vision_state_dict = dict(vision_weights)
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in vision_state_dict.items():
            if name not in params_dict:
                raise ValueError(f"Weight {name} not found in params_dict")
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # loaded_weight = self._pad_vit_attn_dummy_heads(name, loaded_weight)
            weight_loader(param, loaded_weight)

        # Load language model weights
        if language_weights:
            self.language_model.load_weights(language_weights)


EntryClass = StepVLForConditionalGeneration
