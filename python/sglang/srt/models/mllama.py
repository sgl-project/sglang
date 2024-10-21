# Adapted from:
# https://github.com/vllm-project/vllm/blob/7193774b1ff8603ad5bf4598e5efba0d9a39b436/vllm/model_executor/models/mllama.py
"""PyTorch Mllama model."""
import math
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers.models.mllama.configuration_mllama as config_mllama
import vllm.distributed.parallel_state as ps
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from transformers.models.mllama.modeling_mllama import (
    _prepare_aspect_ratio_attention_mask,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaMLP


class ColumnParallelConv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        bias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self._linear = ColumnParallelLinear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        x, _ = self._linear(x)
        return x


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):

    def __init__(self, config: config_mllama.MllamaVisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size
        )
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * self.gate.tanh()

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: config_mllama.MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1,
            self.max_num_tiles * self.num_patches * self.hidden_size,
        )

    def forward(
        self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor
    ) -> torch.Tensor:
        # position embeddings
        gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(
            1, 1, self.num_patches, self.hidden_size
        )

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


class MllamaVisionSdpaAttention(nn.Module):
    def __init__(self, config: config_mllama.MllamaVisionConfig):
        super().__init__()

        model_parallel_size = get_tensor_model_parallel_world_size()
        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        self.num_local_heads = self.num_heads // model_parallel_size
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=False,
            input_is_parallel=True,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_state)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(
            q.shape[0], q.shape[1], self.num_local_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            k.shape[0], k.shape[1], self.num_local_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            v.shape[0], v.shape[1], self.num_local_heads, self.head_dim
        ).transpose(1, 2)

        # TODO: remove padding in image encoder
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            attn_output.shape[0], attn_output.shape[1], -1
        )
        output, _ = self.o_proj(attn_output)
        return output


class MllamaVisionMLP(nn.Module):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class MllamaVisionEncoderLayer(nn.Module):
    def __init__(
        self, config: config_mllama.MllamaVisionConfig, is_gated: bool = False
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size

        self.self_attn = MllamaVisionSdpaAttention(config)
        self.mlp = MllamaVisionMLP(config)

        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=config.norm_eps
        )

        # there used to be an if else here, no code path
        if is_gated:
            self.gate_attn = nn.Parameter(torch.ones(1) * math.pi / 4)
            self.gate_ffn = nn.Parameter(torch.ones(1) * math.pi / 4)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask=attention_mask)
        gate_attn = 1 if not self.is_gated else self.gate_attn.tanh()
        hidden_state = residual + gate_attn * hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        gate_ffn = 1 if not self.is_gated else self.gate_ffn.tanh()
        hidden_state = residual + gate_ffn * hidden_state

        return hidden_state


class MllamaVisionEncoder(nn.Module):
    def __init__(
        self,
        config: config_mllama.MllamaVisionConfig,
        num_layers=32,
        is_gated=False,
        output_hidden_states=None,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [MllamaVisionEncoderLayer(config, is_gated) for _ in range(num_layers)]
        )
        self.output_hidden_states = output_hidden_states or []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        encoder_states = ()

        for i, encoder_layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
            )

        if len(self.layers) - 1 in self.output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return hidden_states, encoder_states


class MllamaVisionModel(nn.Module):
    def __init__(self, config: config_mllama.MllamaVisionConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.in_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = ColumnParallelConv2dPatch(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            config, is_gated=True
        )
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            config, is_gated=True
        )

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)

        # encoders
        self.transformer = MllamaVisionEncoder(
            config,
            config.num_hidden_layers,
            is_gated=False,
            output_hidden_states=config.intermediate_layers_indices,
        )
        self.global_transformer = MllamaVisionEncoder(
            config, config.num_global_layers, is_gated=True
        )

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = (
            pixel_values.shape
        )

        pixel_values = pixel_values.reshape(
            batch_size * num_concurrent_media * num_tiles, num_channels, height, width
        )
        aspect_ratio_ids = aspect_ratio_ids.reshape(
            batch_size * num_concurrent_media, -1
        )

        # patch embedding
        patch_embeds = self.patch_embedding(
            pixel_values.to(self.layernorm_pre.weight.dtype)
        )
        hidden_state = patch_embeds
        hidden_state = ps.get_tp_group().all_gather(hidden_state)

        # tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, -1, dim
        )
        hidden_state = self.pre_tile_positional_embedding(
            hidden_state, aspect_ratio_ids
        )

        # apply cls token
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media * num_tiles, num_patches, dim
        )
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # apply position embeddings
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches, dim
        )
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        # apply encoder
        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (
            0,
            0,
            0,
            num_padding_patches,
        )  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        attention_mask = aspect_ratio_mask.reshape(
            batch_size * num_concurrent_media, -1
        )
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.layernorm_pre.weight.dtype,
        )

        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state, intermediate_hidden_states = output[0], output[1]
        intermediate_hidden_states = torch.stack(intermediate_hidden_states, dim=-1)

        # apply global encoder
        hidden_state = self.layernorm_post(hidden_state)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        )
        hidden_state = self.post_tile_positional_embedding(
            hidden_state, aspect_ratio_ids
        )
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles * (num_patches + num_padding_patches),
            dim,
        )
        hidden_state = self.global_transformer(
            hidden_state, attention_mask=attention_mask
        )[0]
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        )
        hidden_state = hidden_state[:, :, :slice_index]

        # adding intermediate layer outputs
        hidden_state = hidden_state.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, dim
        )
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            -1,
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)
        return hidden_state


class MllamaTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MllamaTextCrossAttention(nn.Module):
    def __init__(
        self,
        config: Optional[config_mllama.MllamaTextConfig] = None,
        layer_id: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.config.num_attention_heads
        self.num_local_heads = self.num_heads // self.model_parallel_size
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_local_key_value_heads = (
            self.num_key_value_heads // self.model_parallel_size
        )
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_id = layer_id
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_local_size = self.num_local_heads * self.head_dim
        self.kv_local_size = self.num_local_key_value_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
        )
        # vllm.model_executor.layers.layernorm.RMSNorm has precision issue,
        # use huggingface's instead
        self.q_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scaling = self.head_dim**-0.5

        self.attn = RadixAttention(
            self.num_local_heads,
            self.head_dim,
            self.scaling,
            self.num_local_key_value_heads,
            layer_id=layer_id,
            is_cross_attention=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cross_attention_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv_dec, _ = self.qkv_proj(hidden_states)
        q, _, _ = qkv_dec.split(
            [self.q_local_size, self.kv_local_size, self.kv_local_size], dim=-1
        )
        if cross_attention_states is None:
            k = None
            v = None
        else:
            qkv_enc, _ = self.qkv_proj(cross_attention_states)
            _, k, v = qkv_enc.split(
                [self.q_local_size, self.kv_local_size, self.kv_local_size], dim=-1
            )
            k = k.view(-1, self.num_local_key_value_heads, self.head_dim)
            v = v.view(-1, self.num_local_key_value_heads, self.head_dim)
            k = self.k_norm(k)
        q = q.view(-1, self.num_local_heads, self.head_dim)
        q = self.q_norm(q)

        output = self.attn(q, k, v, forward_batch)
        out, _ = self.o_proj(output)
        return out


class MllamaCrossAttentionDecoderLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention
    and feedforward."""

    def __init__(
        self,
        config: config_mllama.MllamaTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.cross_attn = MllamaTextCrossAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))

        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            forward_batch=forward_batch,
        )
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states
        return hidden_states


class MllamaTextModel(nn.Module):
    config_class = config_mllama.MllamaTextConfig
    base_model_prefix = "model"

    def __init__(
        self,
        config: config_mllama.MllamaTextConfig,
        quant_config: Optional[QuantizationConfig],
        cache_config=None,
    ):
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size + 8, config.hidden_size
        )
        self.cross_attention_layers = config.cross_attention_layers

        layers = []
        for layer_id in range(config.num_hidden_layers):
            if layer_id in self.cross_attention_layers:
                layers.append(
                    MllamaCrossAttentionDecoderLayer(
                        config, layer_id, quant_config=quant_config
                    )
                )
            else:
                # TODO: force LlamaDecoderLayer to config.attention_bias=False
                layers.append(
                    LlamaDecoderLayer(
                        config, quant_config=quant_config, layer_id=layer_id
                    )
                )

        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]],
        forward_batch: ForwardBatch,
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for _, decoder_layer in enumerate(self.layers):
            if isinstance(decoder_layer, MllamaCrossAttentionDecoderLayer):
                if not skip_cross_attention:
                    hidden_states = decoder_layer(
                        hidden_states=hidden_states,
                        cross_attention_states=cross_attention_states,
                        cross_attention_mask=cross_attention_mask,
                        full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                        forward_batch=forward_batch,
                    )
            elif isinstance(decoder_layer, LlamaDecoderLayer):
                hidden_states, residual = decoder_layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=None,
                )
                hidden_states = hidden_states + residual
            else:
                raise ValueError(f"Unknown decoder layer type {type(decoder_layer)}")
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MllamaForCausalLM(nn.Module):
    config_class = config_mllama.MllamaTextConfig
    base_model_prefix = "language_model"
    _no_split_modules = [
        "MllamaCrossAttentionDecoderLayer",
        "MllamaSelfAttentionDecoderLayer",
    ]

    def __init__(
        self,
        config: config_mllama.MllamaTextConfig,
        quant_config: Optional[QuantizationConfig],
        cache_config=None,
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.model = MllamaTextModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]],
        forward_batch: ForwardBatch,
        skip_cross_attention: bool,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            forward_batch=forward_batch,
            skip_cross_attention=skip_cross_attention,
        )
        return hidden_states


class MllamaForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: config_mllama.MllamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config=None,
    ):
        super().__init__()
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )
        self.image_size = config.vision_config.image_size

        self.vision_model = MllamaVisionModel(config.vision_config)
        self.language_model = MllamaForCausalLM(
            config.text_config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )
        self.logits_processor = LogitsProcessor(config.text_config)
        self.capture_mode = False

    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):
        pixel_values = image_inputs.pixel_values
        pad_values = image_inputs.pad_values

        num_concurrent_media, num_tiles = pixel_values.shape[1:3]
        num_patches = self.vision_model.num_patches
        image_len = num_concurrent_media * num_tiles * num_patches
        image_inputs.num_image_tokens = image_len

        pad_ids = pad_values * ((image_len + len(pad_values)) // len(pad_values))

        return pad_ids[:image_len] + input_ids

    def _batch_image_inputs(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode() or all(forward_batch.encoder_cached):
            return None, None, None, None

        # pixel_values: shape (bs, num_image, num_tiles, 3, image_res, image_res)
        max_num_images = max_num_tiles = bs = 0
        for i, im in enumerate(forward_batch.image_inputs):
            if not forward_batch.encoder_cached[i] and im is not None:
                max_num_images = max(max_num_images, im.pixel_values.shape[1])
                max_num_tiles = max(max_num_tiles, im.pixel_values.shape[2])
                bs += 1

        if max_num_images * max_num_tiles * bs == 0:
            return None, None, None, None

        with forward_batch.out_cache_loc.device:
            batched_images = torch.zeros(
                bs,
                max_num_images,
                max_num_tiles,
                3,
                self.image_size,
                self.image_size,
                dtype=torch.float32,
            )
            batched_ar_ids = torch.ones(
                bs, max_num_images, dtype=torch.int64, device="cuda"
            )
            batched_ar_mask = torch.zeros(
                bs, max_num_images, max_num_tiles, dtype=torch.int64
            )
            i = 0
            encoder_lens_need = []
            for k, im in enumerate(forward_batch.image_inputs):
                if forward_batch.encoder_cached[k] or im is None:
                    continue

                encoder_lens_need.append(forward_batch.encoder_lens[k])
                for j in range(im.pixel_values.shape[1]):
                    img = im.pixel_values[0, j]
                    num_tiles = img.shape[0]
                    batched_images[i, j, :num_tiles] = img
                    batched_ar_ids[i, j] = im.aspect_ratio_ids[0, j]
                    batched_ar_mask[i, j, :num_tiles] = im.aspect_ratio_mask[0, j]
                i += 1

        return batched_images, batched_ar_ids, batched_ar_mask, encoder_lens_need

    def flat_encoder_result(
        self, cross_attention_states: torch.Tensor, encoder_lens_need: List[int]
    ):
        # NOTE: not all encoders need computation, some are cached
        head_dim = cross_attention_states.shape[-1]
        total_encoder_len = sum(encoder_lens_need)
        cross_attention_states_flat = torch.zeros(
            total_encoder_len,
            head_dim,
            device=cross_attention_states.device,
            dtype=cross_attention_states.dtype,
        )

        i = start_pos = 0
        for encoder_len in encoder_lens_need:
            if encoder_len == 0:
                continue
            end_pos = start_pos + encoder_len
            cross_attention_states_flat[start_pos:end_pos] = cross_attention_states[i][
                :encoder_len
            ]
            i += 1
            start_pos += encoder_len

        return cross_attention_states_flat

    def get_full_text_row_masked_out_mask(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode():
            full_text_row_masked_out_mask = forward_batch.encoder_lens != 0
        else:
            full_text_row_masked_out_mask = torch.ones(
                forward_batch.extend_seq_lens.sum(), dtype=torch.bool
            )
            start_pos = 0

            for seq_len, encoder_len in zip(
                forward_batch.seq_lens.tolist(), forward_batch.encoder_lens_cpu
            ):
                if encoder_len == 0:
                    full_text_row_masked_out_mask[start_pos : start_pos + seq_len] = (
                        False
                    )
                start_pos += encoder_len

            full_text_row_masked_out_mask = full_text_row_masked_out_mask.to(
                forward_batch.seq_lens.device
            )

        return full_text_row_masked_out_mask.reshape(-1, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        batched_images, batched_ar_ids, batched_ar_mask, encoder_lens_need = (
            self._batch_image_inputs(forward_batch)
        )

        # TODO: support multi-image by this mask
        cross_attention_mask = None
        cross_attention_states = None

        if self.capture_mode:
            # NOTE: when doing cuda graph capture, we do not want to skip cross attention
            # Make is a constant value to avoid cuda graph capture issue
            skip_cross_attention = False
        else:
            # NOTE: we do not need image_inputs when prefill
            assert len(forward_batch.encoder_lens) == len(forward_batch.seq_lens)
            assert len(forward_batch.encoder_lens_cpu) == len(forward_batch.seq_lens)
            skip_cross_attention = forward_batch.encoder_lens.max() == 0

        if not skip_cross_attention:
            full_text_row_masked_out_mask = self.get_full_text_row_masked_out_mask(
                forward_batch
            )
        else:
            full_text_row_masked_out_mask = None

        if batched_images is not None:
            # NOTE: llama's reference implementation runs vision model on CPU
            cross_attention_states = self.vision_model(
                batched_images, batched_ar_ids, batched_ar_mask
            )
            cross_attention_states = self.multi_modal_projector(cross_attention_states)

            bs, _, _, _, image_token_dim = cross_attention_states.shape
            cross_attention_states = cross_attention_states.view(
                bs, -1, image_token_dim
            )

            cross_attention_states = self.flat_encoder_result(
                cross_attention_states, encoder_lens_need
            )

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            forward_batch=forward_batch,
            skip_cross_attention=skip_cross_attention,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.language_model.lm_head.weight, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        updated_params = set()
        for name, loaded_weight in weights:
            if "patch_embedding.weight" in name:
                name = name.replace(
                    "patch_embedding.weight", "patch_embedding._linear.weight"
                )
                loaded_weight = loaded_weight.view(loaded_weight.shape[0], -1)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                updated_params.add(name)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict.pop(name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = MllamaForConditionalGeneration
