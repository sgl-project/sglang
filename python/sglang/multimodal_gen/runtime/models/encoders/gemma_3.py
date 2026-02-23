# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from sglang: python/sglang/srt/models/gemma3_causal.py

import logging
from functools import partial
from typing import Any, Iterable, Optional, Set, Tuple

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.gemma_3 import Gemma3Config
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import GeluAndMul
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.rotary_embedding import get_rope
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.utils.common import add_prefix

logger = logging.getLogger(__name__)


def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Gemma3Attention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Gemma3Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(
            config.text_config, "head_dim", self.hidden_size // self.total_num_heads
        )

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.text_config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.text_config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=config.text_config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.is_sliding = (
            config.text_config.layer_types[layer_id] == "sliding_attention"
        )

        # Initialize the rotary embedding.
        if self.is_sliding:
            # Local attention.
            self.rope_theta = config.text_config.rope_local_base_freq
            rope_scaling = None  # Default
            # sliding window
            self.sliding_window = get_attention_sliding_window_size(config.text_config)
            # (left, right) = (window, 0) effectively for causal
            self.window_size = (self.sliding_window, 0)
        else:
            # Global attention.
            self.rope_theta = config.text_config.rope_theta
            rope_scaling = config.text_config.rope_scaling
            self.sliding_window = None
            self.window_size = (-1, -1)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.text_config.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        # Local Attention not support attention mask, we use global attention instead.
        # self.attn = LocalAttention(
        #     self.num_heads,
        #     self.head_dim,
        #     self.num_kv_heads,
        #     softmax_scale=self.scaling,
        #     causal=True,
        #     supported_attention_backends=config._supported_attention_backends,
        #     window_size=self.window_size,
        # )

        # Gemma3 adds normalization for q and k
        self.q_norm = Gemma3RMSNorm(
            dim=self.head_dim, eps=config.text_config.rms_norm_eps
        )
        self.k_norm = Gemma3RMSNorm(
            dim=self.head_dim, eps=config.text_config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q, k = self.rotary_emb(positions, q, k)

        # TODO(FlamingoPg): Support LocalAttention
        query = q.transpose(1, 2)
        key = k.transpose(1, 2)
        value = v.transpose(1, 2)

        attn_mask = torch.zeros(
            (seq_len, seq_len),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        causal = torch.triu(
            torch.ones(
                (seq_len, seq_len), device=hidden_states.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        attn_mask = attn_mask.masked_fill(causal, float("-inf"))
        if self.is_sliding and self.sliding_window is not None:
            idx = torch.arange(seq_len, device=hidden_states.device)
            dist = idx[None, :] - idx[:, None]
            too_far = dist > self.sliding_window
            attn_mask = attn_mask.masked_fill(too_far, float("-inf"))

        key_pad = ~attention_mask.to(torch.bool)
        attn_mask = attn_mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
        attn_mask = attn_mask.masked_fill(
            key_pad[:, None, None, :].expand(batch_size, 1, seq_len, seq_len),
            float("-inf"),
        )

        attn_kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": 0.0,
            "is_causal": False,
            "scale": self.scaling,
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, **attn_kwargs
        )
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        output, _ = self.o_proj(attn_output)
        return output


class Gemma3DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Gemma3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.self_attn = Gemma3Attention(
            layer_id=layer_id,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.text_config.num_attention_heads,
            num_kv_heads=getattr(
                config.text_config,
                "num_key_value_heads",
                config.text_config.num_attention_heads,
            ),
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.text_config.intermediate_size,
            hidden_act=config.text_config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = Gemma3RMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # Gemma3 uses "sandwich norm":
        # x = x + norm(attn(norm(x)))
        # So we treat input hidden_states as the residual base.

        if residual is not None:
            hidden_states = hidden_states + residual
            residual = None

        residual_input = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual_input + hidden_states

        # MLP
        residual_mlp = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual_mlp + hidden_states

        return hidden_states, None


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: Optional[float] = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


# --- Siglip Vision Model Implementation ---


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
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
        # Use simple Embedding for position embeddings (usually small enough)
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
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
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SiglipMLP(nn.Module):
    def __init__(
        self,
        config,
        act_layer: type[nn.Module] = QuickGELU,
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


class SiglipAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        tp_size = get_tp_world_size()
        self.head_dim = hidden_size // num_heads
        self.num_heads_per_partition = num_heads // tp_size
        self.scaling = self.head_dim**-0.5

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

        self.attn = LocalAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads_per_partition,
            softmax_scale=self.scaling,
            causal=False,  # Bidirectional for Vision
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.hidden_size // get_tp_world_size()] * 3, dim=-1)

        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)

        attn_output = self.attn(q, k, v)

        attn_output = attn_output.reshape(
            batch_size, seq_len, self.hidden_size // get_tp_world_size()
        )

        output, _ = self.out_proj(attn_output)
        return output


class SiglipEncoderLayer(nn.Module):
    def __init__(
        self,
        config,
        act_layer: type[nn.Module] = QuickGELU,
        norm_layer: type[nn.Module] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        self.layer_norm1 = norm_layer(config.hidden_size)
        self.layer_norm2 = norm_layer(config.hidden_size)
        self.self_attn = SiglipAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
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
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(
        self,
        config,
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
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
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
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @property
    def device(self) -> torch.device:
        return self.encoder.layers[0].layer_norm1.weight.device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values.to(self.device))
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(
            config, quant_config, prefix=add_prefix("vision_model", prefix)
        )

    @property
    def device(self) -> torch.device:
        return self.vision_model.device

    def forward(self, pixel_values: torch.Tensor):
        return self.vision_model(pixel_values)


class Gemma3MultiModalProjector(nn.Module):
    """Projector for Gemma3 multimodal."""

    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(
                config.vision_config.hidden_size, config.text_config.hidden_size
            )
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_size = vision_outputs.shape

        # Reshape for pooling
        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, hidden_size, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        # Apply pooling
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        # Apply normalization
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        # Project to text embedding space
        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.mm_input_projection_weight
        )

        return projected_vision_outputs.type_as(vision_outputs)


class Gemma3TextModel(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        # TODO(yinfan.1024) support text encoding model quant later
        self.quant_config = None

        # Use VocabParallelEmbedding
        from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        self.vocab_size = config.text_config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=config.text_config.vocab_size,
            quant_config=self.quant_config,
        )
        self.embed_scale = config.text_config.hidden_size**0.5

        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(
                    layer_id=i,
                    config=config,
                    quant_config=self.quant_config,
                    prefix=f"{config.text_config.prefix}.layers.{i}",
                )
                for i in range(config.text_config.num_hidden_layers)
            ]
        )

        self.norm = Gemma3RMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.embed_scale

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        residual = None

        if position_ids is None:
            position_ids = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
        position_ids = position_ids + 1

        all_hidden_states: tuple[Any, ...] | None = () if output_hidden_states else None

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            hidden_states, residual = layer(
                position_ids,
                hidden_states,
                residual,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        output = BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

        return output

    def load_weights(self, weights: Any) -> set[str]:
        # Copied from LlamaModel.load_weights but adapted
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def _load_with_shard_id(
            weight_loader, param, loaded_weight: torch.Tensor, shard_id
        ) -> None:
            """Call param.weight_loader with best-effort shard_id normalization.

            Different fused-QKV implementations expect different shard_id types:
            - Some expect strings: "q"/"k"/"v"
            - Some expect integer indices: 0/1/2
            We try the provided shard_id first, then fall back between str/int forms.
            """
            try:
                weight_loader(param, loaded_weight, shard_id)
                return
            except (AssertionError, TypeError):
                pass

            # Fall back between common representations.
            if isinstance(shard_id, str):
                mapping = {"q": 0, "k": 1, "v": 2}
                if shard_id in mapping:
                    weight_loader(param, loaded_weight, mapping[shard_id])
                    return
                if shard_id.isdigit():
                    weight_loader(param, loaded_weight, int(shard_id))
                    return
            elif isinstance(shard_id, int):
                mapping = {0: "q", 1: "k", 2: "v"}
                if shard_id in mapping:
                    weight_loader(param, loaded_weight, mapping[shard_id])
                    return

            # Re-raise with a clearer message.
            raise TypeError(
                f"Unsupported shard_id={shard_id!r} for weight_loader={weight_loader} "
                f"(param={getattr(param, 'name', '<param>')})."
            )

        stacked_params_mapping = getattr(
            getattr(self.config, "arch_config", object()),
            "stacked_params_mapping",
            None,
        )
        if stacked_params_mapping is None:
            stacked_params_mapping = [
                # Fused QKV shards; downstream loaders may want "q/k/v" or 0/1/2.
                (".qkv_proj", ".q_proj", "q"),
                (".qkv_proj", ".k_proj", "k"),
                (".qkv_proj", ".v_proj", "v"),
                (".gate_up_proj", ".gate_proj", 0),
                (".gate_up_proj", ".up_proj", 1),
            ]

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # The config has stacked_params_mapping
            for (
                param_name,
                weight_name,
                shard_id,
            ) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                _load_with_shard_id(weight_loader, param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


class Gemma3ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Gemma3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.text_config = config.text_config

        # Vision Tower
        self.vision_tower = SiglipVisionModel(
            config=config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        # Projector
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

        # Text Model
        self.language_model = Gemma3TextModel(config)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.Tensor:
        image_token_index = int(getattr(self.config, "image_token_index", -1))
        if image_token_index < 0:
            image_token_index = int(getattr(self.text_config, "image_token_index", -1))
        special_image_mask = input_ids == image_token_index
        n_image_tokens = int(special_image_mask.sum().item())
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        n_image_features = int(image_features.shape[0] * image_features.shape[1])
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor | None = None,
        **kwargs,
    ):
        vocab_size = int(self.language_model.vocab_size)
        image_token_index = int(getattr(self.config, "image_token_index", -1))
        if image_token_index < 0:
            image_token_index = int(getattr(self.text_config, "image_token_index", -1))

        if input_ids is not None and image_token_index >= vocab_size:
            special_image_mask = input_ids == image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        inputs_embeds = self.language_model.get_input_embeddings(llm_input_ids)

        if pixel_values is not None:
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.reshape(
                    -1,
                    pixel_values.shape[2],
                    pixel_values.shape[3],
                    pixel_values.shape[4],
                )
            elif pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            elif pixel_values.dim() != 4:
                raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

            vision_outputs = self.vision_tower(pixel_values)
            image_features = self.multi_modal_projector(vision_outputs)
            image_features = image_features.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        return self.language_model.forward(
            llm_input_ids, inputs_embeds=inputs_embeds, **kwargs
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters())

        def _load_with_shard_id(
            weight_loader, param, loaded_weight: torch.Tensor, shard_id
        ) -> None:
            """Call param.weight_loader with best-effort shard_id normalization.

            Different fused-QKV implementations expect different shard_id types:
            - Some expect strings: "q"/"k"/"v"
            - Some expect integer indices: 0/1/2
            We try the provided shard_id first, then fall back between str/int forms.
            """
            try:
                weight_loader(param, loaded_weight, shard_id)
                return
            except (AssertionError, TypeError):
                pass

            # Fall back between common representations.
            if isinstance(shard_id, str):
                mapping = {"q": 0, "k": 1, "v": 2}
                if shard_id in mapping:
                    weight_loader(param, loaded_weight, mapping[shard_id])
                    return
                if shard_id.isdigit():
                    weight_loader(param, loaded_weight, int(shard_id))
                    return
            elif isinstance(shard_id, int):
                mapping = {0: "q", 1: "k", 2: "v"}
                if shard_id in mapping:
                    weight_loader(param, loaded_weight, mapping[shard_id])
                    return

            raise TypeError(
                f"Unsupported shard_id={shard_id!r} for weight_loader={weight_loader} "
                f"(param={getattr(param, 'name', '<param>')})."
            )

        # Separate weights
        language_model_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []

        for name, loaded_weight in weights:
            # Handle prefix mapping if needed
            # HF weights might be "model.vision_tower...", "model.language_model..."

            if "vision_tower" in name or "vision_model" in name:
                # Load vision tower weights
                # Map name to local name
                local_name = name
                if "model.vision_tower" in name:
                    local_name = name.replace("model.vision_tower", "vision_tower")
                elif "vision_tower" in name:
                    pass  # already correct prefix if matching self.vision_tower
                elif local_name.startswith("vision_model."):
                    local_name = (
                        "vision_tower.vision_model."
                        + local_name[len("vision_model.") :]
                    )

                # We need to map HF Siglip names to our Siglip implementation
                # Our Siglip: vision_tower.vision_model.encoder.layers...
                # HF Siglip: vision_model.encoder.layers...

                # If loading from Gemma3 checkpoint, it usually has "model.vision_tower.vision_model..."

                if local_name in params_dict:
                    param = params_dict[local_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(local_name)
                else:
                    qkv_shard_id = None
                    fused_name = None
                    if ".self_attn.q_proj." in local_name:
                        fused_name = local_name.replace(
                            ".self_attn.q_proj.", ".self_attn.qkv_proj."
                        )
                        qkv_shard_id = "q"
                    elif ".self_attn.k_proj." in local_name:
                        fused_name = local_name.replace(
                            ".self_attn.k_proj.", ".self_attn.qkv_proj."
                        )
                        qkv_shard_id = "k"
                    elif ".self_attn.v_proj." in local_name:
                        fused_name = local_name.replace(
                            ".self_attn.v_proj.", ".self_attn.qkv_proj."
                        )
                        qkv_shard_id = "v"

                    if fused_name is not None and fused_name in params_dict:
                        param = params_dict[fused_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        _load_with_shard_id(
                            weight_loader, param, loaded_weight, qkv_shard_id
                        )
                        loaded_params.add(fused_name)
                        continue

                    if ".self_attn.proj." in local_name:
                        candidate = local_name.replace(
                            ".self_attn.proj.", ".self_attn.out_proj."
                        )
                        if candidate in params_dict:
                            param = params_dict[candidate]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                            loaded_params.add(candidate)
                            continue
                    if ".self_attn.out_proj." in local_name:
                        candidate = local_name.replace(
                            ".self_attn.out_proj.", ".self_attn.proj."
                        )
                        if candidate in params_dict:
                            param = params_dict[candidate]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                            loaded_params.add(candidate)
                            continue

                    # Try to find match
                    suffix = local_name.split("vision_tower.")[-1]
                    # Try adding vision_model
                    candidate = f"vision_tower.vision_model.{suffix}"
                    if candidate in params_dict:
                        param = params_dict[candidate]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(candidate)

            elif "multi_modal_projector" in name:
                local_name = name
                if "model.multi_modal_projector" in name:
                    local_name = name.replace(
                        "model.multi_modal_projector", "multi_modal_projector"
                    )

                if local_name in params_dict:
                    param = params_dict[local_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(local_name)

            elif "language_model" in name or "model.language_model" in name:
                # Strip prefix for language model
                # If name is "model.language_model.model.layers.0...", we want "model.layers.0..." for Gemma3ForCausalLM
                # Gemma3ForCausalLM has .model (Gemma3TextModel) and .lm_head

                # HF: model.language_model.model.layers...
                # Ours: language_model.model.layers...

                # We pass (name, weight) to language_model.load_weights
                # We should strip "model.language_model." or "language_model."

                suffix = name
                if "model.language_model." in name:
                    suffix = name.replace("model.language_model.", "")
                elif "language_model." in name:
                    suffix = name.replace("language_model.", "")
                if suffix.startswith("model."):
                    suffix = suffix[len("model.") :]

                language_model_weights.append((suffix, loaded_weight))

            else:
                # Fallback for other weights (maybe direct lm_head if not nested?)
                other_weights.append((name, loaded_weight))

        if language_model_weights:
            lm_loaded = self.language_model.load_weights(language_model_weights)
            loaded_params.update({f"language_model.{n}" for n in lm_loaded})

        return loaded_params

    def get_attention_sliding_window_size(self):
        if self.text_config is not None and hasattr(
            self.text_config, "get_attention_sliding_window_size"
        ):
            return self.text_config.get_attention_sliding_window_size()
        sliding_window = getattr(self.text_config, "sliding_window", None)
        if sliding_window is None:
            sliding_window = getattr(self.config, "sliding_window", None)
        if sliding_window is None:
            return None
        return int(sliding_window) - 1


EntryClass = Gemma3ForConditionalGeneration
