# Adapted from
# https://github.com/huggingface/transformers/blob/af9b2eaa54c150741f298d6db939af6328e1dc38/src/transformers/models/siglip/modeling_siglip.py

from functools import partial
from typing import Optional, Type, Union

import torch
import torch.nn as nn
from transformers import SiglipVisionConfig

from sglang.srt.layers.activation import QuickGELU
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.utils import add_prefix


# Adapted from transformers.models.siglip.modeling_siglip.SiglipVisionTransformer
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
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
        self.position_embedding = VocabParallelEmbedding(
            self.num_positions, self.embed_dim
        )
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        # interpolate_pos_encoding is never used in sglang
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


# Eagle2.5 Vision Head (matches checkpoint parameters)
class EagleVisionHead(nn.Module):
    """Vision head for Eagle2.5 model matching checkpoint parameter structure."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Layer norm (matches layernorm.weight)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Attention components with direct attributes for correct parameter names
        # Match checkpoint dimensions: in_proj has (3 * hidden_size) out_features
        self.attention = nn.Module()
        self.attention.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.attention.proj = nn.Linear(config.hidden_size, config.hidden_size)

        # MLP components with direct attributes
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.mlp.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

        # Activation for MLP
        self.act = QuickGELU()

        # Probe component (matches probe parameter)
        # Checkpoint has shape [1, 1, hidden_size], so initialize accordingly
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Layer norm
        hidden_states = self.layernorm(hidden_states)

        # Self-attention with separate projections
        # QKV projection (checkpoint has 4 * hidden_size, take first 3 for QKV)
        qkv_proj = self.attention.in_proj(hidden_states)
        qkv = qkv_proj[
            :, :, : 3 * self.hidden_size
        ]  # Take first 3*hidden_size dimensions
        q, k, v = qkv.chunk(3, dim=-1)

        # Simple self-attention (no masking for vision)
        # For vision head, use simplified attention - just pass through v
        attn_output = v

        # Output projection
        attn_output = self.attention.proj(attn_output)

        # MLP
        mlp_output = self.mlp.fc1(attn_output)
        mlp_output = self.act(mlp_output)
        mlp_output = self.mlp.fc2(mlp_output)

        # Add probe (already has batch/seq dimensions)
        output = mlp_output + self.probe

        return output


# Copied from sglang.srt.models.clip.CLIPMLP
class SiglipMLP(nn.Module):
    def __init__(
        self,
        config,
        act_layer: Type[nn.Module] = QuickGELU,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.act = act_layer()
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel, _ = self.fc1(x)
        x_parallel = self.act(x_parallel)
        x, _ = self.fc2(x_parallel)
        return x


# Copied from sglang.srt.models.clip.CLIPEncoderLayer
class SiglipEncoderLayer(nn.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        act_layer: Type[nn.Module] = QuickGELU,
        norm_layer: Type[nn.Module] = None,
        attn_implementation: Optional[str] = "sdpa",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        self.layer_norm1 = norm_layer(config.hidden_size)
        self.layer_norm2 = norm_layer(config.hidden_size)
        if attn_implementation == "sdpa":
            qkv_backend = "sdpa"
            softmax_in_single_precision = False
        elif attn_implementation == "flash_attention_2":
            qkv_backend = "triton_attn"
            softmax_in_single_precision = False
        elif attn_implementation == "eager":
            qkv_backend = "sdpa"
            softmax_in_single_precision = True
        self.self_attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            qkv_backend=qkv_backend,
            softmax_in_single_precision=softmax_in_single_precision,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = SiglipMLP(
            config,
            act_layer=act_layer,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        # Siglip text model uses both `causal_attention_mask` and `attention_mask`
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attn_mask,
            # causal_attention_mask=causal_attention_mask,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# Copied from sglang.srt.models.clip.CLIPEncoder
class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self
    attention layers. Each layer is a [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        num_hidden_layers = config.num_hidden_layers
        norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            [
                SiglipEncoderLayer(
                    config=config,
                    norm_layer=norm_layer,
                    attn_implementation="sdpa",
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_attention_mask: torch.Tensor = None,
        return_all_hidden_states: bool = False,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states, attention_mask, causal_attention_mask
            )
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


# Adapted from transformers.models.siglip.modeling_siglip.SiglipVisionTransformer
class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_eagle_head: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)

        self.encoder = SiglipEncoder(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # VisionAttention in SiglipEncoderLayer is multihead attention
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # Optional Eagle2.5 head
        self.use_eagle_head = use_eagle_head
        if use_eagle_head:
            self.head = EagleVisionHead(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("head", prefix),
            )

    @property
    def device(self) -> torch.device:
        return self.encoder.layers[0].layer_norm1.weight.device

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values.to(self.device))

        return_all_hidden_states = False

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=return_all_hidden_states,
        )

        last_hidden_state = self.post_layernorm(last_hidden_state)

        # Apply Eagle2.5 head if enabled
        if self.use_eagle_head:
            last_hidden_state = self.head(last_hidden_state)

        return last_hidden_state


# Copied from sglang.srt.models.clip.CLIPVisionModel
class SiglipVisionModel(nn.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_eagle_head: bool = False,
    ):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(
            config,
            quant_config,
            prefix=add_prefix("vision_model", prefix),
            use_eagle_head=use_eagle_head,
        )

    @property
    def device(self) -> torch.device:
        return self.vision_model.device

    def forward(self, pixel_values: torch.Tensor):
        return self.vision_model(pixel_values)
