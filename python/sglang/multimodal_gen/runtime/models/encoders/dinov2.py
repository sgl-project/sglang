# SPDX-License-Identifier: Apache-2.0
# Adapted from Hugging Face Transformers DINOv2.

from __future__ import annotations

import collections.abc
import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class Dinov2PatchEmbeddings(nn.Module):
    def __init__(self, config: Dinov2Config):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        self.image_size = (
            tuple(image_size)
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        self.patch_size = (
            tuple(patch_size)
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        self.num_channels = config.num_channels
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[1] != self.num_channels:
            raise ValueError(
                f"DINOv2 expects {self.num_channels} image channels, "
                f"got {pixel_values.shape[1]}"
            )
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class Dinov2Embeddings(nn.Module):
    def __init__(self, config: Dinov2Config):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.use_mask_token = config.use_mask_token
        if self.use_mask_token:
            self.mask_token = nn.Parameter(torch.empty(1, config.hidden_size))
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(
            torch.empty(
                1,
                self.patch_embeddings.num_patches + 1,
                config.hidden_size,
            )
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size

    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        source_size = math.isqrt(num_positions)
        if source_size * source_size != num_positions:
            raise ValueError(
                f"DINOv2 position embedding has {num_positions} non-square patches"
            )

        if isinstance(self.patch_size, collections.abc.Iterable):
            patch_height, patch_width = self.patch_size
        else:
            patch_height = patch_width = self.patch_size
        target_height = height // patch_height
        target_width = width // patch_width
        dim = embeddings.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(
            1, source_size, source_size, dim
        ).permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = F.interpolate(
            patch_pos_embed.float(),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        ).to(target_dtype)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        if bool_masked_pos is not None and self.use_mask_token:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1),
                self.mask_token.to(embeddings.dtype).unsqueeze(0),
                embeddings,
            )

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(
            embeddings, height, width
        )
        return self.dropout(embeddings)


def _linear_output(linear: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    output = linear(hidden_states)
    return output[0] if isinstance(output, tuple) else output


class Dinov2SelfAttention(nn.Module):
    def __init__(self, config: Dinov2Config, *, use_tensor_parallel: bool):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"DINOv2 hidden size {config.hidden_size} is not divisible by "
                f"{config.num_attention_heads} attention heads"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        self.scaling = self.attention_head_size**-0.5
        tp_size = get_tp_world_size() if use_tensor_parallel else 1
        if self.num_attention_heads % tp_size != 0:
            raise ValueError(
                f"DINOv2 heads {self.num_attention_heads} are not divisible by "
                f"TP size {tp_size}"
            )
        self.use_tensor_parallel = use_tensor_parallel
        self.num_heads = self.num_attention_heads // tp_size
        self.hidden_size = self.num_heads * self.attention_head_size
        if use_tensor_parallel:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.attention_head_size,
                total_num_heads=self.num_attention_heads,
                total_num_kv_heads=self.num_attention_heads,
                bias=config.qkv_bias,
            )
        else:
            self.query = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.key = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
            self.value = nn.Linear(
                config.hidden_size, self.all_head_size, bias=config.qkv_bias
            )
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.attention_head_size,
            num_kv_heads=self.num_heads,
            softmax_scale=self.scaling,
            causal=False,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.FA2,
                AttentionBackendEnum.TORCH_SDPA,
            },
            allow_cudnn_sdp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        if self.use_tensor_parallel:
            qkv = _linear_output(self.qkv_proj, hidden_states)
            query, key, value = qkv.split([self.hidden_size] * 3, dim=-1)
        else:
            query = self.query(hidden_states)
            key = self.key(hidden_states)
            value = self.value(hidden_states)
        shape = (
            batch_size,
            seq_len,
            self.num_heads,
            self.attention_head_size,
        )
        query = query.view(shape)
        key = key.view(shape)
        value = value.view(shape)
        context = self.attn(query, key, value)
        return context.reshape(batch_size, seq_len, self.hidden_size)


class Dinov2SelfOutput(nn.Module):
    def __init__(self, config: Dinov2Config, *, use_tensor_parallel: bool):
        super().__init__()
        self.dense = (
            RowParallelLinear(config.hidden_size, config.hidden_size)
            if use_tensor_parallel
            else nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(_linear_output(self.dense, hidden_states))


class Dinov2Attention(nn.Module):
    def __init__(self, config: Dinov2Config, *, use_tensor_parallel: bool):
        super().__init__()
        self.attention = Dinov2SelfAttention(
            config, use_tensor_parallel=use_tensor_parallel
        )
        self.output = Dinov2SelfOutput(config, use_tensor_parallel=use_tensor_parallel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(hidden_states))


class Dinov2LayerScale(nn.Module):
    def __init__(self, config: Dinov2Config):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.lambda1


class Dinov2MLP(nn.Module):
    def __init__(self, config: Dinov2Config, *, use_tensor_parallel: bool):
        super().__init__()
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = (
            ColumnParallelLinear(config.hidden_size, hidden_features)
            if use_tensor_parallel
            else nn.Linear(config.hidden_size, hidden_features)
        )
        self.activation = get_act_fn(config.hidden_act)
        self.fc2 = (
            RowParallelLinear(hidden_features, config.hidden_size)
            if use_tensor_parallel
            else nn.Linear(hidden_features, config.hidden_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = _linear_output(self.fc1, hidden_states)
        hidden_states = self.activation(hidden_states)
        return _linear_output(self.fc2, hidden_states)


class Dinov2SwiGLUFFN(nn.Module):
    def __init__(self, config: Dinov2Config, *, use_tensor_parallel: bool):
        super().__init__()
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.weights_in = (
            MergedColumnParallelLinear(
                config.hidden_size,
                [hidden_features, hidden_features],
            )
            if use_tensor_parallel
            else nn.Linear(config.hidden_size, 2 * hidden_features)
        )
        self.weights_out = (
            RowParallelLinear(hidden_features, config.hidden_size)
            if use_tensor_parallel
            else nn.Linear(hidden_features, config.hidden_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = _linear_output(self.weights_in, hidden_states)
        gate, value = hidden_states.chunk(2, dim=-1)
        return _linear_output(self.weights_out, F.silu(gate) * value)


class Dinov2DropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        keep_prob = 1.0 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        random_tensor = torch.rand(
            shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        random_tensor = torch.floor(random_tensor + keep_prob)
        return hidden_states.div(keep_prob) * random_tensor


class Dinov2Layer(nn.Module):
    def __init__(self, config: Dinov2Config, *, use_tensor_parallel: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2Attention(
            config, use_tensor_parallel=use_tensor_parallel
        )
        self.layer_scale1 = Dinov2LayerScale(config)
        self.drop_path = (
            Dinov2DropPath(config.drop_path_rate)
            if config.drop_path_rate > 0.0
            else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = (
            Dinov2SwiGLUFFN(config, use_tensor_parallel=use_tensor_parallel)
            if config.use_swiglu_ffn
            else Dinov2MLP(config, use_tensor_parallel=use_tensor_parallel)
        )
        self.layer_scale2 = Dinov2LayerScale(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(self.norm1(hidden_states))
        hidden_states = (
            self.drop_path(self.layer_scale1(attention_output)) + hidden_states
        )
        mlp_output = self.mlp(self.norm2(hidden_states))
        return self.drop_path(self.layer_scale2(mlp_output)) + hidden_states


class Dinov2Encoder(nn.Module):
    def __init__(self, config: Dinov2Config, *, use_tensor_parallel: bool):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                Dinov2Layer(config, use_tensor_parallel=use_tensor_parallel)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        all_hidden_states = [] if output_hidden_states else None
        for layer in self.layer:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        return hidden_states, (
            tuple(all_hidden_states) if all_hidden_states is not None else None
        )


class Dinov2Model(nn.Module, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    layer_names = ["encoder.layer"]

    def __init__(
        self,
        config: Dinov2Config | dict[str, Any],
        *,
        use_tensor_parallel: bool | None = None,
    ):
        super().__init__()
        if isinstance(config, dict):
            config = Dinov2Config.from_dict(config)
        if use_tensor_parallel is None:
            use_tensor_parallel = model_parallel_is_initialized()
        self.config = config
        self.use_tensor_parallel = use_tensor_parallel
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config, use_tensor_parallel=use_tensor_parallel)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embeddings.projection.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embeddings.projection.weight.dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseEncoderOutput:
        if pixel_values is None:
            raise ValueError("DINOv2 requires pixel_values")
        output_hidden_states = bool(
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        hidden_states = self.embeddings(pixel_values, bool_masked_pos)
        hidden_states, all_hidden_states = self.encoder(
            hidden_states,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = self.layernorm(hidden_states)
        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states[:, 0],
            hidden_states=all_hidden_states,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        stacked_params_mapping = (
            (
                ".attention.attention.qkv_proj.",
                ".attention.attention.query.",
                "q",
            ),
            (
                ".attention.attention.qkv_proj.",
                ".attention.attention.key.",
                "k",
            ),
            (
                ".attention.attention.qkv_proj.",
                ".attention.attention.value.",
                "v",
            ),
        )
        params = dict(self.named_parameters())
        parallel_weight_loaders = {
            f"{module_name}.{param_name}": module.weight_loader
            for module_name, module in self.named_modules()
            if isinstance(module, (ColumnParallelLinear, RowParallelLinear))
            for param_name, _ in module.named_parameters(recurse=False)
        }
        loaded = set()
        for name, tensor in weights:
            for target_name, source_name, shard_id in (
                stacked_params_mapping if self.use_tensor_parallel else ()
            ):
                if source_name not in name:
                    continue
                name = name.replace(source_name, target_name)
                param = params[name]
                parallel_weight_loaders[name](param, tensor, shard_id)
                break
            else:
                try:
                    param = params[name]
                except KeyError as exc:
                    raise ValueError(f"Unexpected DINOv2 weight: {name}") from exc
                weight_loader = parallel_weight_loaders.get(name, default_weight_loader)
                weight_loader(param, tensor)
            loaded.add(name)

        missing = set(params) - loaded
        if missing:
            examples = sorted(missing)[:8]
            raise RuntimeError(
                f"DINOv2 checkpoint is missing {len(missing)} parameters: {examples}"
            )
        return loaded


EntryClass = Dinov2Model
