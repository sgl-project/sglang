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
"""SGLang-native TP-sharded audio encoder for Gemma 4.

Architecture: Conformer-based USM (Universal Speech Model) with SSCP convolution
projection. Adapted from gemma3n_audio.py with Gemma 4 specific changes:
  - Activation clamping (clippable linears) on all conformer linears
  - per_dim_key_scale in attention
  - LayerNorm (not CumulativeGroupNorm) in SSCP convolution blocks
  - Semicausal SSCP padding
  - Mask propagation through SSCP
  - Output projection (hidden_size -> output_proj_dims)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Gemma4AudioConfig

from sglang.srt.layers.clippable_linear import (
    ClippableColumnParallelLinear,
    ClippableGLUParallelLinear,
    ClippableQKVParallelLinear,
    ClippableRowParallelLinear,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs

# SSCP convolution constants (no longer in config.json, never varied across models)
_SSCP_INPUT_FEAT_SIZE = 128
_SSCP_CONV_KERNEL_SIZES = ((3, 3), (3, 3))
_SSCP_CONV_STRIDE_SIZES = ((2, 2), (2, 2))

# ---------------------------------------------------------------------------
# Relative Position Embedding
# ---------------------------------------------------------------------------


class Gemma4AudioRelativePositionEmbedding(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        tp_size = get_attention_tp_size()
        total_num_heads = config.num_attention_heads
        self.channels = config.hidden_size
        self.head_dim = self.channels // total_num_heads
        self.num_heads = total_num_heads // tp_size
        self.max_backward = max(0, config.attention_context_left - 1)
        self.max_forward = config.attention_context_right

        self.pos_proj = ColumnParallelLinear(
            self.channels,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("pos_proj", prefix),
        )

        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.channels // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales) * -log_timescale_increment
        )
        self.register_buffer(
            "inv_timescales",
            inv_timescales.float().unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _get_timing_signal_1d_pos(
        self, position: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        assert position.ndim == 2
        position = position.float().unsqueeze(-1)
        scaled_time = position * self.inv_timescales.to(
            device=position.device, dtype=torch.float32
        )
        timing_signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1
        )
        return timing_signal.type(dtype)

    def _relative_shift(
        self,
        term_bd_before_shift: torch.Tensor,
        batch_size: int,
        num_heads: int,
        num_query_blocks: int,
        query_block_size: int,
        key_context_size: int,
        max_span_plus_1: int,
    ) -> torch.Tensor:
        pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1
        padding_tuple = (0, pad_amount_last_dim)

        term_bd_padded = F.pad(term_bd_before_shift, padding_tuple)
        term_bd_reshaped = term_bd_padded.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size * (key_context_size + 1),
            )
        )
        term_bd_sliced = term_bd_reshaped[
            :, :, :, : query_block_size * key_context_size
        ]
        term_bd_shifted = term_bd_sliced.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size,
                key_context_size,
            )
        )
        return term_bd_shifted

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        batch_size, num_query_blocks, query_block_size, num_heads, head_dim = (
            queries.shape
        )
        _, _, key_context_size, _, _ = keys.shape

        pos_indices = torch.arange(
            self.max_backward, -self.max_forward - 1, -1, device=queries.device
        ).unsqueeze(0)
        max_span_plus_1 = pos_indices.shape[1]

        sin_emb_timing_signal = self._get_timing_signal_1d_pos(
            pos_indices, dtype=queries.dtype
        )
        # pos_proj is a ColumnParallelLinear (no implicit dtype promotion);
        # project in weight dtype, then cast back to queries' dtype for the matmuls.
        projected_sin_emb, _ = self.pos_proj(
            sin_emb_timing_signal.to(self.pos_proj.weight.dtype)
        )
        projected_sin_emb = projected_sin_emb.to(queries.dtype)
        sin_emb = projected_sin_emb.reshape(
            1, max_span_plus_1, self.num_heads, self.head_dim
        ).squeeze(0)

        queries_p = queries.permute(0, 3, 1, 2, 4)
        keys_p_t = keys.permute(0, 3, 1, 4, 2)
        term_ac = torch.matmul(queries_p, keys_p_t)

        q_permuted = queries.permute(0, 3, 1, 2, 4)
        s_permuted = sin_emb.permute(1, 2, 0)
        q_reshaped = q_permuted.reshape(
            batch_size, num_heads, num_query_blocks * query_block_size, head_dim
        )
        term_bd_unshifed_matmul = torch.matmul(q_reshaped, s_permuted)
        term_bd_unshifed = term_bd_unshifed_matmul.reshape(
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        )

        term_bd_shifted = self._relative_shift(
            term_bd_unshifed,
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
            max_span_plus_1,
        )

        return term_ac + term_bd_shifted


# ---------------------------------------------------------------------------
# Local Dot-Product Attention (with per_dim_key_scale)
# ---------------------------------------------------------------------------


class Gemma4AudioAttention(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        tp_size = get_attention_tp_size()
        total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // total_num_heads
        self.num_heads = total_num_heads // tp_size

        self.chunk_size = config.attention_chunk_size
        self.max_future_horizon = config.attention_context_right
        self.max_past_horizon = max(0, config.attention_context_left - 1)
        self.attention_logits_soft_cap = config.attention_logit_cap
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )

        self.relative_position_embedding = Gemma4AudioRelativePositionEmbedding(
            config,
            quant_config,
            prefix=add_prefix("relative_position_embedding", prefix),
        )
        self.per_dim_scale = nn.Parameter(torch.zeros((self.head_dim,)))

        self.qkv = ClippableQKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)

        self.register_buffer(
            "softcap",
            torch.tensor(self.attention_logits_soft_cap).float(),
            persistent=False,
        )

    # ------ block / context helpers (identical to Gemma3n) ------------------

    def _pad_dim1(
        self, x: torch.Tensor, dim10_val: int, dim11_val: int
    ) -> torch.Tensor:
        padding_tuple = [0] * x.ndim * 2
        dim_idx_from_end = x.ndim - 2
        start_idx_for_dim = 2 * dim_idx_from_end
        padding_tuple[start_idx_for_dim] = dim10_val
        padding_tuple[start_idx_for_dim + 1] = dim11_val
        return F.pad(x, tuple(padding_tuple))

    def _convert_to_block(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        b, t = shape[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size
        if (padding_len := num_blocks * self.chunk_size - t) > 0:
            x = self._pad_dim1(x, 0, padding_len)
        permute_dims = (b, num_blocks, self.chunk_size) + shape[2:]
        return x.reshape(permute_dims).contiguous()

    def _extract_block_context(self, x: torch.Tensor) -> torch.Tensor:
        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        x = self._pad_dim1(x, pad_left, pad_right)
        frame_len = self.context_size
        frame_step = self.chunk_size
        x_unfolded = x.unfold(dimension=1, size=frame_len, step=frame_step)
        if x.ndim > 2 and x_unfolded.ndim > 3:
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)
        return x_unfolded.contiguous()

    # ------ forward ---------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.BoolTensor,
        causal_valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        q, k, v = self.qkv(x)
        qkv_shape = (*x.shape[:-1], self.num_heads, self.head_dim)
        query_states = q.float().reshape(qkv_shape).contiguous()
        key_states = k.float().reshape(qkv_shape).contiguous()
        value_states = v.float().reshape(qkv_shape).contiguous()

        per_dim_scale_sp = F.softplus(self.per_dim_scale)
        broadcast_shape = (1, 1, 1, self.head_dim)
        query_states = (
            query_states * self.q_scale * per_dim_scale_sp.view(broadcast_shape)
        )

        key_states = key_states * self.k_scale

        batch_size, q_time = query_states.shape[:2]

        query_blocks = self._convert_to_block(query_states)
        key_blocks = self._extract_block_context(key_states)
        value_blocks = self._extract_block_context(value_states)
        num_query_blocks = query_blocks.shape[1]

        original_valid_mask = ~mask
        extracted_valid_mask_blocks = self._extract_block_context(original_valid_mask)

        if (
            extracted_valid_mask_blocks.ndim == 4
            and extracted_valid_mask_blocks.shape[0] == batch_size
            and extracted_valid_mask_blocks.shape[1] == num_query_blocks
            and extracted_valid_mask_blocks.shape[2]
            * extracted_valid_mask_blocks.shape[3]
            == self.context_size
        ):
            extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
                batch_size, num_query_blocks, self.context_size
            )

        condition_from_input_validity = extracted_valid_mask_blocks.unsqueeze(
            1
        ).unsqueeze(-2)
        condition_from_causality = (
            causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        final_condition_for_where = torch.logical_and(
            condition_from_input_validity,
            condition_from_causality.to(condition_from_input_validity.device),
        )

        logits = self.relative_position_embedding(query_blocks, key_blocks)

        softcap_val = self.softcap.to(logits.device)
        logits = logits / softcap_val
        logits = torch.tanh(logits)
        logits = logits * softcap_val

        logits = torch.where(
            final_condition_for_where,
            logits,
            self.config.attention_invalid_logits_value,
        )

        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(
            dtype=value_blocks.dtype
        )

        b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
        h_dim = value_blocks.shape[-1]
        prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
        v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
        result_bmm = torch.bmm(prob_bun, v_bun)
        context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(
            0, 1, 3, 2, 4
        )
        context_vectors = context_vectors.reshape(
            batch_size,
            num_query_blocks * self.chunk_size,
            self.num_heads,
            self.head_dim,
        )
        context_vectors = context_vectors[:, :q_time]
        return context_vectors


# ---------------------------------------------------------------------------
# SSCP (Sub-Sample Convolution Projection)
# ---------------------------------------------------------------------------


class Gemma4AudioSSCPConvBlock(nn.Module):
    """Single 2D conv block with LayerNorm and semicausal padding."""

    def __init__(
        self,
        config: Gemma4AudioConfig,
        idx: int,
        input_freq_dim: int,
    ):
        super().__init__()
        self.config = config

        conv_channels = config.subsampling_conv_channels
        in_channels = 1 if idx == 0 else conv_channels[idx - 1]
        out_channels = conv_channels[idx]
        kernel_t, kernel_f = _SSCP_CONV_KERNEL_SIZES[idx]
        stride_t, stride_f = _SSCP_CONV_STRIDE_SIZES[idx]
        self.time_stride = stride_t

        # Semicausal padding (hardcoded — streaming is not supported)
        pad_t_top = kernel_t // 2
        pad_t_bottom = kernel_t // 2

        pad_f_left = 1
        pad_f_right = 1

        self.manual_padding = (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_t, kernel_f),
            stride=(stride_t, stride_f),
            padding=(0, 0),
            bias=False,
        )

        f_in_padded = input_freq_dim + pad_f_left + pad_f_right
        self.f_out_conv = (f_in_padded - kernel_f) // stride_f + 1

        self.norm = nn.LayerNorm(
            [out_channels],
            eps=config.rms_norm_eps,
            elementwise_affine=True,
            bias=False,
        )
        self.activation = nn.ReLU()

    def forward(
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_for_fill = audio_mel_mask.unsqueeze(1).unsqueeze(-1)
        audio_encodings = audio_encodings.masked_fill(mask_for_fill, 0.0)

        audio_encodings_padded = F.pad(
            audio_encodings, self.manual_padding, mode="constant", value=0.0
        ).to(self.conv.weight.dtype)
        audio_encodings_conv = self.conv(audio_encodings_padded)

        output_mask = audio_mel_mask[:, :: self.time_stride][
            :, : audio_encodings_conv.shape[2]
        ]

        x = audio_encodings_conv.permute(0, 2, 3, 1)
        x_normed = self.norm(x)
        audio_encodings_normed = x_normed.permute(0, 3, 1, 2).contiguous()
        return self.activation(audio_encodings_normed), output_mask


class Gemma4AudioSubSampleConvProjection(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        conv_channels = config.subsampling_conv_channels

        current_f = _SSCP_INPUT_FEAT_SIZE
        calculated_f_out_dims = []

        for i in range(2):
            kernel_h, kernel_w = _SSCP_CONV_KERNEL_SIZES[i]
            stride_h, stride_w = _SSCP_CONV_STRIDE_SIZES[i]

            pad_f_left = 1
            pad_f_right = 1
            f_in_padded = current_f + pad_f_left + pad_f_right
            f_out = (f_in_padded - kernel_w) // stride_w + 1
            calculated_f_out_dims.append(f_out)
            current_f = f_out

        self.conv_0 = Gemma4AudioSSCPConvBlock(
            idx=0,
            input_freq_dim=_SSCP_INPUT_FEAT_SIZE,
            config=config,
        )
        self.conv_1 = Gemma4AudioSSCPConvBlock(
            idx=1,
            input_freq_dim=calculated_f_out_dims[0],
            config=config,
        )

        final_c_out = conv_channels[-1]
        final_f_out = calculated_f_out_dims[-1]
        self.input_proj_in_features = final_c_out * final_f_out

        self.input_proj_linear = RowParallelLinear(
            self.input_proj_in_features,
            config.hidden_size,
            bias=False,
            input_is_parallel=False,
            quant_config=quant_config,
            prefix=add_prefix("input_proj_linear", prefix),
        )

    def forward(
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_encodings_reshaped = audio_encodings.unsqueeze(1)
        x, mask = self.conv_0(audio_encodings_reshaped, audio_mel_mask)
        x, mask = self.conv_1(x, mask)
        b, c_out, t_out, f_out = x.shape
        x_permuted = x.permute(0, 2, 3, 1).contiguous()
        output_flattened = x_permuted.reshape(b, t_out, f_out * c_out)
        output, _ = self.input_proj_linear(output_flattened)
        return output, mask


# ---------------------------------------------------------------------------
# Conformer Blocks
# ---------------------------------------------------------------------------


class Gemma4AudioConformerAttention(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.post_in_features = config.hidden_size

        self.register_buffer(
            "gradient_clipping",
            torch.tensor(config.gradient_clipping),
            persistent=False,
        )

        self.pre_attn_norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)
        self.attn = Gemma4AudioAttention(
            config, quant_config, prefix=add_prefix("attn", prefix)
        )
        self.post = ClippableRowParallelLinear(
            self.post_in_features,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("post", prefix),
        )
        self.post_norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)

    def forward(
        self,
        audio_encodings: torch.Tensor,
        audio_mel_mask: torch.BoolTensor,
        causal_valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        audio_encodings_input_to_attn = audio_encodings
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings_norm = self.pre_attn_norm(audio_encodings)
        audio_encodings_attn_out = self.attn(
            audio_encodings_norm, audio_mel_mask, causal_valid_mask
        )

        b, t, num_heads, head_dim = audio_encodings_attn_out.shape
        audio_encodings_reshaped = audio_encodings_attn_out.reshape(
            b, t, num_heads * head_dim
        ).to(dtype=audio_encodings_input_to_attn.dtype)

        audio_encodings = self.post(audio_encodings_reshaped)
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        return audio_encodings_input_to_attn + self.post_norm(audio_encodings)


class Gemma4AudioConformerFeedForward(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.register_buffer(
            "gradient_clipping",
            torch.tensor(config.gradient_clipping),
            persistent=False,
        )

        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)
        self.ffw_layer_1 = ClippableColumnParallelLinear(
            config.hidden_size,
            config.hidden_size * 4,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("ffw_layer_1", prefix),
        )
        self.ffw_layer_2 = ClippableRowParallelLinear(
            config.hidden_size * 4,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("ffw_layer_2", prefix),
        )
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)
        self.post_layer_scale = config.residual_weight

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        residual = audio_encodings
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings = self.ffw_layer_1(audio_encodings)
        audio_encodings = F.silu(audio_encodings)
        audio_encodings = self.ffw_layer_2(audio_encodings)
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.post_layer_norm(audio_encodings)
        return residual + (audio_encodings * self.post_layer_scale)


class Gemma4AudioConformerLightConv1d(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.causal_padding = config.conv_kernel_size - 1
        tp_size = get_attention_tp_size()
        hidden_per_tp = config.hidden_size // tp_size

        self.register_buffer(
            "gradient_clipping",
            torch.tensor(config.gradient_clipping),
            persistent=False,
        )

        self.pre_layer_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, scale_shift=0.0
        )
        self.linear_start = ClippableGLUParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_start", prefix),
        )
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=hidden_per_tp,
            out_channels=hidden_per_tp,
            kernel_size=config.conv_kernel_size,
            stride=1,
            padding=0,
            groups=hidden_per_tp,
            bias=False,
        )
        self.conv_norm = Gemma4RMSNorm(
            hidden_per_tp, eps=config.rms_norm_eps, scale_shift=0.0
        )

        tp_rank = get_attention_tp_rank()

        def _shard_dim0(param, loaded_weight, _rank=tp_rank, _tp=tp_size):
            shard = param.shape[0]
            loaded_weight = loaded_weight.narrow(0, _rank * shard, shard)
            param.data.copy_(loaded_weight)

        set_weight_attrs(self.depthwise_conv1d.weight, {"weight_loader": _shard_dim0})
        set_weight_attrs(self.conv_norm.weight, {"weight_loader": _shard_dim0})

        self.linear_end = ClippableRowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_end", prefix),
        )

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        audio_encodings_residual = audio_encodings

        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings = self.linear_start(audio_encodings)

        audio_encodings_permuted = audio_encodings.permute(0, 2, 1)
        audio_encodings_permuted_padded = F.pad(
            audio_encodings_permuted, (self.causal_padding, 0)
        )
        audio_encodings = self.depthwise_conv1d(audio_encodings_permuted_padded)
        audio_encodings = audio_encodings.permute(0, 2, 1)
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.conv_norm(audio_encodings)
        audio_encodings = F.silu(audio_encodings)
        audio_encodings = self.linear_end(audio_encodings)
        return audio_encodings + audio_encodings_residual


class Gemma4AudioConformerBlock(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.ffw_layer_start = Gemma4AudioConformerFeedForward(
            config, quant_config, prefix=add_prefix("ffw_layer_start", prefix)
        )
        self.attention = Gemma4AudioConformerAttention(
            config, quant_config, prefix=add_prefix("attention", prefix)
        )
        self.lconv1d = Gemma4AudioConformerLightConv1d(
            config, quant_config, prefix=add_prefix("lconv1d", prefix)
        )
        self.ffw_layer_end = Gemma4AudioConformerFeedForward(
            config, quant_config, prefix=add_prefix("ffw_layer_end", prefix)
        )
        self.register_buffer(
            "gradient_clipping",
            torch.tensor(config.gradient_clipping),
            persistent=False,
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, scale_shift=0.0)

    def forward(
        self,
        audio_encodings: torch.Tensor,
        audio_mel_mask: torch.BoolTensor,
        causal_valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        audio_encodings = self.ffw_layer_start(audio_encodings)
        audio_encodings = self.attention(
            audio_encodings, audio_mel_mask, causal_valid_mask
        )
        validity_mask_for_lconv = ~audio_mel_mask
        audio_encodings_for_lconv_input = (
            audio_encodings
            * validity_mask_for_lconv.unsqueeze(-1).to(audio_encodings.dtype)
        )
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)

        audio_encodings = self.ffw_layer_end(audio_encodings)
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        return self.norm(audio_encodings)


# ---------------------------------------------------------------------------
# Top-level Encoder
# ---------------------------------------------------------------------------


class Gemma4AudioEncoder(nn.Module):
    """SGLang-native TP-sharded Gemma 4 audio encoder (USM Conformer + SSCP)."""

    def __init__(
        self,
        config: Gemma4AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(
            config, quant_config, prefix=add_prefix("subsample_conv_projection", prefix)
        )
        self.conformer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma4AudioConformerBlock(
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("conformer", prefix),
        )

        if config.output_proj_dims is not None:
            self.output_proj = RowParallelLinear(
                config.hidden_size,
                config.output_proj_dims,
                bias=True,
                input_is_parallel=False,
                quant_config=quant_config,
                prefix=add_prefix("output_proj", prefix),
            )
        else:
            self.output_proj = None

        # Precompute causal_valid_mask — depends only on static config values.
        chunk_size = config.attention_chunk_size
        max_future_horizon = config.attention_context_right
        max_past_horizon = max(0, config.attention_context_left - 1)
        upper_diagonal = max_past_horizon + max_future_horizon
        context_size = chunk_size + max_past_horizon + max_future_horizon

        lower_causal_mask = torch.tril(
            torch.ones((context_size, chunk_size), dtype=torch.bool),
            diagonal=0,
        ).T
        upper_causal_mask = torch.tril(
            torch.ones((chunk_size, context_size), dtype=torch.bool),
            diagonal=upper_diagonal,
        )
        local_causal_valid_mask = torch.ones(
            (chunk_size, context_size), dtype=torch.bool
        )
        self.register_buffer(
            "causal_valid_mask",
            local_causal_valid_mask * lower_causal_mask * upper_causal_mask,
            persistent=False,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Encode a batch of mel spectrograms.

        Args:
            audio_mel: [batch, num_frames, mel_bins]
            audio_mel_mask: [batch, num_frames], True = padding

        Returns:
            audio_encodings: [batch, reduced_frames, hidden_size/output_proj_dims]
            audio_mel_mask: [batch, reduced_frames], True = padding
        """
        audio_encodings, current_mask = self.subsample_conv_projection(
            audio_mel, audio_mel_mask
        )

        for block in self.conformer:
            audio_encodings = block(
                audio_encodings, current_mask, self.causal_valid_mask
            )

        if self.output_proj is not None:
            audio_encodings, _ = self.output_proj(audio_encodings)

        if current_mask.shape[1] != audio_encodings.shape[1]:
            target_len = audio_encodings.shape[1]
            if target_len > current_mask.shape[1]:
                current_mask = F.pad(
                    current_mask, (0, target_len - current_mask.shape[1]), value=True
                )
            else:
                current_mask = current_mask[:, :target_len]

        audio_encodings = audio_encodings.masked_fill(current_mask.unsqueeze(-1), 0.0)
        return audio_encodings, current_mask
