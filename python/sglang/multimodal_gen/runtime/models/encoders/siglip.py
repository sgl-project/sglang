# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM and Hugging Face Transformers SigLIP implementations.

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import QuickGELU
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.common import add_prefix

ActivationFactory = Callable[[], nn.Module]


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=self.patch_embedding.weight.dtype)
        )
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        return embeddings + self.position_embedding(self.position_ids)


class SiglipMLP(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        activation_factory: ActivationFactory = QuickGELU,
        tensor_parallel: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tensor_parallel = tensor_parallel
        if tensor_parallel:
            self.fc1 = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix("fc1", prefix),
            )
            self.fc2 = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("fc2", prefix),
            )
        else:
            if quant_config is not None:
                raise ValueError("SigLIP quantization requires tensor parallel layers")
            self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
            self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = activation_factory()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        if self.tensor_parallel:
            hidden_states = hidden_states[0]
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        if self.tensor_parallel:
            hidden_states = hidden_states[0]
        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        tensor_parallel: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"SigLIP hidden size {hidden_size} is not divisible by {num_heads} heads"
            )

        self.tensor_parallel = tensor_parallel
        self.hidden_size = hidden_size
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        if tensor_parallel:
            tp_size = get_tp_world_size()
            if num_heads % tp_size != 0:
                raise ValueError(
                    f"SigLIP heads {num_heads} are not divisible by TP size {tp_size}"
                )
            self.num_heads_per_partition = num_heads // tp_size
            self.embed_dim_per_partition = self.num_heads_per_partition * self.head_dim
            self.qkv_proj = QKVParallelLinear(
                hidden_size=hidden_size,
                head_size=self.head_dim,
                total_num_heads=num_heads,
                total_num_kv_heads=num_heads,
                bias=True,
                quant_config=quant_config,
                prefix=add_prefix("qkv_proj", prefix),
            )
            self.out_proj = RowParallelLinear(
                input_size=hidden_size,
                output_size=hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=add_prefix("out_proj", prefix),
            )
        else:
            if quant_config is not None:
                raise ValueError("SigLIP quantization requires tensor parallel layers")
            self.num_heads_per_partition = num_heads
            self.embed_dim_per_partition = hidden_size
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, hidden_size)

        attention_kwargs = {}
        if not tensor_parallel:
            attention_kwargs = {
                "supported_attention_backends": {
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.FA2,
                    AttentionBackendEnum.TORCH_SDPA,
                },
                "compute_dtype": self.projection_dtype,
                "allow_cudnn_sdp": True,
            }
        self.attn = LocalAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads_per_partition,
            softmax_scale=self.scaling,
            causal=False,
            **attention_kwargs,
        )

    @property
    def projection_dtype(self) -> torch.dtype:
        if self.tensor_parallel:
            return self.qkv_proj.weight.dtype
        return self.q_proj.weight.dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel:
            qkv, _ = self.qkv_proj(hidden_states)
            query, key, value = qkv.split([self.embed_dim_per_partition] * 3, dim=-1)
        else:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

        batch_size, seq_len = hidden_states.shape[:2]
        shape = (batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        query = query.view(shape)
        key = key.view(shape)
        value = value.view(shape)
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.embed_dim_per_partition
        )
        hidden_states = self.out_proj(hidden_states)
        if self.tensor_parallel:
            hidden_states = hidden_states[0]
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        activation_factory: ActivationFactory = QuickGELU,
        tensor_parallel: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        norm_factory = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        self.layer_norm1 = norm_factory(config.hidden_size)
        self.layer_norm2 = norm_factory(config.hidden_size)
        self.self_attn = SiglipAttention(
            config.hidden_size,
            config.num_attention_heads,
            tensor_parallel=tensor_parallel,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = SiglipMLP(
            config,
            activation_factory=activation_factory,
            tensor_parallel=tensor_parallel,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states))
        return residual + hidden_states


class SiglipEncoder(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        activation_factory: ActivationFactory = QuickGELU,
        tensor_parallel: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                SiglipEncoderLayer(
                    config,
                    activation_factory=activation_factory,
                    tensor_parallel=tensor_parallel,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        activation_factory: ActivationFactory = QuickGELU,
        tensor_parallel: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(
            config,
            activation_factory=activation_factory,
            tensor_parallel=tensor_parallel,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values.to(self.device))
        if self.encoder.layers:
            encoder_dtype = self.encoder.layers[0].self_attn.projection_dtype
            if hidden_states.dtype != encoder_dtype:
                hidden_states = hidden_states.to(encoder_dtype)
        hidden_states = self.encoder(hidden_states)
        return self.post_layernorm(hidden_states)


class SiglipVisionModel(nn.Module, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    layer_names = ["vision_model.encoder.layers"]

    def __init__(
        self,
        config: Any,
        *,
        activation_factory: ActivationFactory = QuickGELU,
        tensor_parallel: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(
            config,
            activation_factory=activation_factory,
            tensor_parallel=tensor_parallel,
            quant_config=quant_config,
            prefix=add_prefix("vision_model", prefix),
        )

    @property
    def device(self) -> torch.device:
        return self.vision_model.device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)
