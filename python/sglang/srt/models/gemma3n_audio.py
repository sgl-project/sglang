import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Gemma3nAudioConfig, PreTrainedModel

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.gemma3n_causal import Gemma3nRMSNorm
from sglang.srt.utils import add_prefix, make_layers


class Gemma3nCumulativeGroupNorm(nn.Module):
    """Applies Group Normalization cumulatively over the time dimension.

    This layer normalizes the input by calculating the mean and variance
    cumulatively over the time dimension (dim 1). The statistics are computed
    over all feature dimensions (specified by `feature_dims` and `num_channels`)
    for elements marked as valid by the optional `mask`.

    If a `mask` is provided (True for valid, False for invalid/padded),
    invalid time steps do not contribute to the statistics calculation, and
    their corresponding output values are zeroed out.

    Scale and bias, if enabled, are applied per-channel (last dimension).
    This behavior is similar to JAX's `GroupNormalization` with `num_groups=1`
    and `cumulative=True`.
    """

    def __init__(
        self,
        num_channels: int,  # Number of channels (size of the last dimension)
        feature_dims: Sequence[
            int
        ],  # Sizes of non-channel feature dimensions, e.g., (H, W) for input [B,T,H,W,C]
        eps: float = 1e-3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps

        # Scale parameter depends only on the channel dimension
        self.weight = nn.Parameter(torch.ones(num_channels))

        # Axes for normalization: all dimensions except Batch (0) and Time (1).
        # For input [B, T, *feature_dims, C], these are dims from 2 onwards.
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies cumulative group norm, optionally using a mask.

        Args:
          x: Input tensor, shape [B, T, *feature_dims, C].
          mask: Optional boolean mask, shape [B, T]. True indicates a valid
            (non-padded) time step. If None, all time steps are considered valid.

        Returns:
          Normalized tensor with the same shape as x.
        """
        expected_input_suffix = self.feature_dims + (self.num_channels,)
        if x.shape[2:] != expected_input_suffix:
            raise ValueError(
                f"Input tensor shape suffix {x.shape[2:]} does not match expected"
                f" suffix (feature_dims + num_channels) {expected_input_suffix}"
            )

        input_dtype = x.dtype
        # Calculations are performed in float32 for numerical stability.
        calc_dtype = torch.float32
        x_calc = x.to(calc_dtype)

        # Prepare a broadcastable mask (`mask_calc`).
        # If no mask is provided, treat all elements as valid
        # (mask_calc is all ones).
        # Otherwise, expand the [B, T] mask to [B, T, 1, ..., 1] for broadcasting.
        mask_calc = torch.ones_like(x_calc, dtype=calc_dtype)

        # Cumulative Statistics Calculation
        # 1. Sum of values over reduction axes at each time step.
        sum_values_at_t = torch.sum(x_calc, dim=self.reduction_axes, keepdim=True)
        # 2. Cumulative sum of values over time.
        cum_sum_values = torch.cumsum(sum_values_at_t, dim=1)

        # 3. Count of valid elements in the normalization group at each time step.
        #    (A "group" here consists of all features at a given Batch, Time).
        elements_in_group_at_t = torch.sum(
            mask_calc, dim=self.reduction_axes, keepdim=True
        )
        # 4. Cumulative count of valid elements over time.
        cum_count_elements = torch.cumsum(elements_in_group_at_t, dim=1)
        # Avoid division by zero if all preceding elements were masked.
        safe_cum_count_elements = torch.clamp(cum_count_elements, min=1.0)

        # 5. Cumulative mean.
        cum_mean = cum_sum_values / safe_cum_count_elements

        # 6. Sum of squared differences from the cumulative mean.
        #    Only sum for valid elements: (x_calc - cum_mean)^2 * mask_calc.
        #    Using x_calc here for the difference, as cum_mean already accounts for masking.
        squared_diff_from_mean = (x_calc - cum_mean).pow(2)
        sum_sq_diff_at_t = torch.sum(
            squared_diff_from_mean, dim=self.reduction_axes, keepdim=True
        )

        # 7. Cumulative sum of squared differences over time.
        cum_sum_sq_diff = torch.cumsum(sum_sq_diff_at_t, dim=1)

        # 8. Cumulative variance.
        cum_variance = cum_sum_sq_diff / safe_cum_count_elements

        # Normalize the input using the calculated cumulative statistics:
        # (x - E[x]) / sqrt(Var[x] + eps)
        normalized_x = (x_calc - cum_mean) * torch.rsqrt(cum_variance + self.eps)

        # Apply affine transformation (scale and bias) if enabled.
        # Scale and bias are applied per-channel (last dimension).
        scale = self.weight.to(calc_dtype)
        # Reshape for broadcasting: [C] -> [1, ..., 1, C]
        scale_view_shape = [1] * (x.dim() - 1) + [self.num_channels]
        normalized_x = normalized_x * scale.view(scale_view_shape)

        # Zero out outputs for time steps that were originally masked (where mask_calc is 0).
        # This ensures padded/invalid positions in the input result in zero output.
        final_output = normalized_x * mask_calc

        return final_output.to(input_dtype)


class Gemma3nAudioRelativePositionEmbedding(nn.Module):
    def __init__(
        self,
        config: Gemma3nAudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.channels = self.config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = max(0, self.config.conf_attention_context_left - 1)
        self.max_forward = self.config.conf_attention_context_right

        self.pos_proj = ColumnParallelLinear(
            self.channels,
            self.num_heads * self.head_dim,
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
        """Performs the relative shift."""
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
        projected_sin_emb, _ = self.pos_proj(sin_emb_timing_signal)
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


class Gemma3nAudioAttention(nn.Module):
    """Local dot product self-attention for audio."""

    def __init__(
        self,
        config: Gemma3nAudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.chunk_size = self.config.conf_attention_chunk_size
        self.max_future_horizon = self.config.conf_attention_context_right
        self.max_past_horizon = max(0, self.config.conf_attention_context_left - 1)
        self.attention_logits_soft_cap = self.config.conf_attention_logit_cap
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )

        self.relative_position_embedding = Gemma3nAudioRelativePositionEmbedding(
            config,
            quant_config,
            prefix=add_prefix("relative_position_embedding", prefix),
        )
        self.per_dim_scale = nn.Parameter(torch.zeros((self.head_dim,)))

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        q_scale = self.head_dim**-0.5
        r_softplus_0 = 1.0 / F.softplus(torch.tensor(0.0))
        self.register_buffer(
            "q_scale", (q_scale * r_softplus_0).clone().detach(), persistent=False
        )

        # Create local causal mask
        lower_causal_mask = torch.tril(
            torch.ones((self.context_size, self.chunk_size), dtype=torch.bool),
            diagonal=0,
        ).T
        upper_causal_mask = torch.tril(
            torch.ones((self.chunk_size, self.context_size), dtype=torch.bool),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = torch.ones(
            (self.chunk_size, self.context_size), dtype=torch.bool
        )
        local_causal_valid_mask = (
            local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        )
        self.register_buffer(
            "local_causal_valid_mask", local_causal_valid_mask, persistent=False
        )

        self.register_buffer(
            "softcap",
            torch.tensor(self.attention_logits_soft_cap).float(),
            persistent=False,
        )

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
        """Turns a sequence to non overlapping blocks."""
        shape = x.shape
        b, t = shape[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size

        if (padding_len := num_blocks * self.chunk_size - t) > 0:
            x = self._pad_dim1(x, 0, padding_len)

        permute_dims = (b, num_blocks, self.chunk_size) + shape[2:]
        x = x.reshape(permute_dims).contiguous()
        return x

    def _extract_block_context(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts temporal context for every block."""
        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        x = self._pad_dim1(x, pad_left, pad_right)

        frame_len = self.context_size
        frame_step = self.chunk_size

        x_unfolded = x.unfold(dimension=1, size=frame_len, step=frame_step)

        if x.ndim > 2 and x_unfolded.ndim > 3:
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)

        return x_unfolded.contiguous()

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # Project to Q, K, V
        qkv, _ = self.qkv_proj(x)
        query_states, key_states, value_states = qkv.chunk(chunks=3, dim=-1)

        # Reshape
        query_states = query_states.reshape(
            *x.shape[:-1], self.num_heads, self.head_dim
        ).contiguous()
        key_states = key_states.reshape(
            *x.shape[:-1], self.num_heads, self.head_dim
        ).contiguous()
        value_states = value_states.reshape(
            *x.shape[:-1], self.num_heads, self.head_dim
        ).contiguous()

        # Apply per-dim scale
        per_dim_scale_sp = F.softplus(self.per_dim_scale)
        broadcast_shape = (1, 1, 1, self.head_dim)
        per_dim_scale_sp_broadcast = per_dim_scale_sp.view(broadcast_shape)
        query_states = query_states * self.q_scale * per_dim_scale_sp_broadcast

        batch_size, q_time = query_states.shape[:2]

        # Convert to blocks
        query_blocks = self._convert_to_block(query_states)
        key_blocks = self._extract_block_context(key_states)
        value_blocks = self._extract_block_context(value_states)
        num_query_blocks = query_blocks.shape[1]

        # Create mask for valid positions
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
            self.local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        final_condition_for_where = torch.logical_and(
            condition_from_input_validity,
            condition_from_causality.to(condition_from_input_validity.device),
        )

        # Compute attention scores
        logits = self.relative_position_embedding(query_blocks, key_blocks)

        # Apply attention logit softcap
        softcap_val = self.softcap.to(logits.device)
        logits = logits / softcap_val
        logits = torch.tanh(logits)
        logits = logits * softcap_val

        # Apply the combined mask.
        # final_condition_for_where will broadcast with logits [B,N,U,W,C]
        logits = torch.where(
            final_condition_for_where, logits, torch.finfo(logits.dtype).min
        )

        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(
            dtype=value_blocks.dtype
        )

        # context_vectors is adapted from jax.numpy.einsum("BNuwc,BucNH->BuwNH", ...)
        b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
        h_dim = value_blocks.shape[-1]
        prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
        v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
        result_bmm = torch.bmm(prob_bun, v_bun)
        context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(
            0, 1, 3, 2, 4
        )
        context_vectors = context_vectors.reshape(
            (
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            )
        )
        context_vectors = context_vectors[:, :q_time]

        return context_vectors


class Gemma3nAudioSSCPConvBlock(nn.Module):
    """A single convolution block for the SubSampleConvProjection."""

    def __init__(
        self,
        config: Gemma3nAudioConfig,
        idx: int,
        input_freq_dim: int,
        manual_padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.manual_padding = manual_padding

        in_channels = 1 if idx == 0 else self.config.sscp_conv_channel_size[idx - 1]
        out_channels = self.config.sscp_conv_channel_size[idx]
        kernel_h, kernel_w = self.config.sscp_conv_kernel_size[idx]
        stride_h, stride_w = self.config.sscp_conv_stride_size[idx]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=(0, 0),  # Manual padding is used
            bias=False,
        )

        f_in_padded = input_freq_dim + self.manual_padding[0] + self.manual_padding[1]
        f_out_conv = (f_in_padded - kernel_w) // stride_w + 1

        self.norm = Gemma3nCumulativeGroupNorm(
            num_channels=out_channels,
            feature_dims=(f_out_conv,),
            eps=self.config.sscp_conv_group_norm_eps,
        )

        self.activation = nn.ReLU()

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        audio_encodings_padded = F.pad(
            audio_encodings, self.manual_padding, mode="constant", value=0.0
        )
        audio_encodings_conv = self.conv(audio_encodings_padded)
        x_for_norm = audio_encodings_conv.permute(0, 2, 3, 1).contiguous()
        x_normed = self.norm(x_for_norm)
        audio_encodings_normed = x_normed.permute(0, 3, 1, 2).contiguous()
        return self.activation(audio_encodings_normed)


class Gemma3nAudioSubSampleConvProjection(nn.Module):
    def __init__(
        self,
        config: Gemma3nAudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        current_f_for_block_input = config.input_feat_size
        calculated_block_padding = []
        calculated_f_out_dims = []

        for i in range(2):  # Assuming 2 conv layers
            kernel_h, kernel_w = config.sscp_conv_kernel_size[i]
            stride_h, stride_w = config.sscp_conv_stride_size[i]

            # Padding for Time (Height for Conv2d) - REVERSE_CAUSAL like
            pad_t_top = 0
            pad_t_bottom = kernel_h - 1

            # Frequency Padding (Width for Conv2d)
            pad_f_left = 1
            pad_f_right = 1

            manual_padding_tuple = (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom)
            calculated_block_padding.append(manual_padding_tuple)

            f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right
            f_out_after_conv = (f_in_padded - kernel_w) // stride_w + 1
            calculated_f_out_dims.append(f_out_after_conv)
            current_f_for_block_input = f_out_after_conv

        self.conv_0 = Gemma3nAudioSSCPConvBlock(
            idx=0,
            input_freq_dim=config.input_feat_size,
            config=config,
            manual_padding=calculated_block_padding[0],
            quant_config=quant_config,
            prefix=add_prefix("conv_0", prefix),
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(
            idx=1,
            input_freq_dim=calculated_f_out_dims[0],
            config=config,
            manual_padding=calculated_block_padding[1],
            quant_config=quant_config,
            prefix=add_prefix("conv_1", prefix),
        )

        final_c_out = config.sscp_conv_channel_size[-1]
        final_f_out = calculated_f_out_dims[-1]
        self.input_proj_in_features = final_c_out * final_f_out

        self.input_proj_linear = RowParallelLinear(
            self.input_proj_in_features,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("input_proj_linear", prefix),
        )

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        audio_encodings_reshaped = audio_encodings.unsqueeze(1)
        x = self.conv_0(audio_encodings_reshaped)
        x = self.conv_1(x)
        b, c_out, t_out, f_out = x.shape
        x_permuted = x.permute(0, 2, 3, 1).contiguous()
        output_flattened = x_permuted.view(b, t_out, f_out * c_out)
        output, _ = self.input_proj_linear(output_flattened)
        return output


class Gemma3nAudioConformerAttention(nn.Module):
    def __init__(
        self,
        config: Gemma3nAudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        head_dim = self.config.hidden_size // self.config.conf_num_attention_heads
        self.post_in_shape = (self.config.conf_num_attention_heads, head_dim)
        self.post_in_features = self.config.hidden_size

        self.register_buffer(
            "gradient_clipping",
            torch.tensor(self.config.gradient_clipping),
            persistent=False,
        )

        self.pre_attn_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.attn = Gemma3nAudioAttention(
            config, quant_config, prefix=add_prefix("attn", prefix)
        )
        self.post = RowParallelLinear(
            self.post_in_features,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("post", prefix),
        )
        self.post_norm = Gemma3nRMSNorm(self.config.hidden_size)

    def forward(
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> torch.Tensor:
        audio_encodings_input_to_attn = audio_encodings
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings_norm = self.pre_attn_norm(audio_encodings)
        audio_encodings_attn_out = self.attn(audio_encodings_norm, audio_mel_mask)

        b, t, num_heads, head_dim = audio_encodings_attn_out.shape
        audio_encodings_reshaped = audio_encodings_attn_out.reshape(
            b, t, num_heads * head_dim
        )

        audio_encodings, _ = self.post(audio_encodings_reshaped)
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        return audio_encodings_input_to_attn + self.post_norm(audio_encodings)


class Gemma3nAudioConformerFeedForward(nn.Module):
    def __init__(
        self,
        config: Gemma3nAudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.register_buffer(
            "gradient_clipping",
            torch.tensor(self.config.gradient_clipping),
            persistent=False,
        )

        self.pre_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.ffw_layer_1 = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size * 4,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("ffw_layer_1", prefix),
        )
        self.ffw_layer_2 = RowParallelLinear(
            self.config.hidden_size * 4,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("ffw_layer_2", prefix),
        )
        self.post_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.post_layer_scale = torch.tensor(self.config.conf_residual_weight)

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        residual = audio_encodings
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings, _ = self.ffw_layer_1(audio_encodings)
        audio_encodings = F.silu(audio_encodings)
        audio_encodings, _ = self.ffw_layer_2(audio_encodings)
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.post_layer_norm(audio_encodings)
        return residual + (audio_encodings * self.post_layer_scale)


class Gemma3nAudioConformerLightConv1d(nn.Module):
    def __init__(
        self,
        config: Gemma3nAudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.pre_layer_norm = Gemma3nRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.linear_start = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_start", prefix),
        )

        self.depthwise_conv1d = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.conf_conv_kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            groups=self.config.hidden_size,  # Depthwise
            bias=False,
        )
        self.register_buffer(
            "gradient_clipping",
            torch.tensor(self.config.gradient_clipping),
            persistent=False,
        )
        self.conv_norm = Gemma3nRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.linear_end = RowParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("linear_end", prefix),
        )

        self.causal_padding = self.config.conf_conv_kernel_size - 1

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        audio_encodings_residual = audio_encodings  # Save for residual connection

        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings, _ = self.linear_start(audio_encodings)
        audio_encodings = F.glu(audio_encodings, dim=-1)

        # Permute for Conv1d: [B, T, D] -> [B, D, T]
        audio_encodings_permuted = audio_encodings.permute(0, 2, 1)
        # Apply manual causal padding
        audio_encodings_permuted_padded = F.pad(
            audio_encodings_permuted, (self.causal_padding, 0)
        )
        audio_encodings = self.depthwise_conv1d(audio_encodings_permuted_padded)
        # Permute back: [B, D, T_out] -> [B, T_out, D]
        audio_encodings = audio_encodings.permute(0, 2, 1)
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        audio_encodings = self.conv_norm(audio_encodings)
        audio_encodings = F.silu(audio_encodings)
        audio_encodings, _ = self.linear_end(audio_encodings)
        output = audio_encodings + audio_encodings_residual
        return output


class Gemma3nAudioConformerBlock(nn.Module):
    def __init__(
        self,
        config: Gemma3nAudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(
            config, quant_config, prefix=add_prefix("ffw_layer_start", prefix)
        )
        self.attention = Gemma3nAudioConformerAttention(
            config, quant_config, prefix=add_prefix("attention", prefix)
        )
        self.lconv1d = Gemma3nAudioConformerLightConv1d(
            config, quant_config, prefix=add_prefix("lconv1d", prefix)
        )
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(
            config, quant_config, prefix=add_prefix("ffw_layer_end", prefix)
        )
        self.register_buffer(
            "gradient_clipping",
            torch.tensor(self.config.gradient_clipping),
            persistent=False,
        )
        self.norm = Gemma3nRMSNorm(self.config.hidden_size)

    def forward(
        self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> torch.Tensor:
        audio_encodings = self.ffw_layer_start(audio_encodings)
        audio_encodings = self.attention(audio_encodings, audio_mel_mask)
        validity_mask_for_lconv = ~audio_mel_mask  # True for valid
        audio_encodings_for_lconv_input = (
            audio_encodings
            * validity_mask_for_lconv.unsqueeze(-1).to(audio_encodings.dtype)
        )
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)

        audio_encodings = self.ffw_layer_end(audio_encodings)
        audio_encodings = torch.clamp(
            audio_encodings, -self.gradient_clipping, self.gradient_clipping
        )
        output = self.norm(audio_encodings)
        return output


class Gemma3nAudioEncoder(PreTrainedModel):
    """A Universal Speech Encoder -- https://arxiv.org/abs/2303.01037"""

    config_class = Gemma3nAudioConfig

    def __init__(
        self,
        config: Gemma3nAudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.config = config

        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(
            config, quant_config, prefix=add_prefix("subsample_conv_projection", prefix)
        )
        self.conformer = make_layers(
            config.conf_num_hidden_layers,
            lambda idx, prefix: Gemma3nAudioConformerBlock(
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("conformer", prefix),
        )

    def forward(
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Encodes a batch of MELs.

        Args:
            audio_mel: a torch.Tensor of shape [batch, num_frames, mel_bins].
            audio_mel_mask: a torch.BoolTensor of shape [batch, num_frames].

        Returns:
            audio_encodings: a torch.Tensor of shape
                `[batch_size, reduced_time_frames, hidden_size]`
            audio_mel_mask: a torch.BoolTensor of shape [batch, reduced_time_frames].
        """
        audio_encodings = self.subsample_conv_projection(
            audio_mel
        )  # audio_encodings: [B, T_sub, D]

        # Subsample the input audio_mel_mask to match the time dimension of audio_encodings (T_sub)
        t_sub = audio_encodings.shape[1]

        time_stride_product = 1
        for stride_pair_idx in range(len(self.config.sscp_conv_stride_size)):
            time_stride_product *= self.config.sscp_conv_stride_size[stride_pair_idx][0]

        # Create indices for gathering from the original mask.
        # These indices map to original time steps corresponding to the start of each
        # receptive field in the subsampled output.
        indices = (
            torch.arange(t_sub, device=audio_mel_mask.device) * time_stride_product
        )
        indices = torch.clamp(indices, max=audio_mel_mask.shape[1] - 1)

        # Expand indices for batch compatibility if B > 1 and indices is 1D.
        if audio_mel_mask.ndim > 1 and indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(
                audio_mel_mask.shape[0], -1
            )  # [B, T_sub]
        elif (
            audio_mel_mask.ndim == indices.ndim
            and audio_mel_mask.shape[0] == 1
            and indices.shape[0] != 1
            and t_sub == indices.shape[0]
        ):
            # Handle case where B=1 but indices became [T_sub] instead of [1, T_sub]
            indices = indices.unsqueeze(0)

        current_mask = torch.gather(audio_mel_mask, 1, indices)  # [B, T_sub]

        # Fallback: Ensure mask length matches feature length after gather.
        if current_mask.shape[1] != t_sub:
            if current_mask.shape[1] > t_sub:
                current_mask = current_mask[:, :t_sub]
            else:  # current_mask.shape[1] < t_sub
                padding_needed = t_sub - current_mask.shape[1]
                current_mask = F.pad(
                    current_mask, (0, padding_needed), value=True
                )  # Pad with True (masked)

        for i, block in enumerate(self.conformer):
            audio_encodings = block(
                audio_encodings, current_mask
            )  # Pass the processed mask

        if self.config.conf_reduction_factor > 1:
            audio_encodings = audio_encodings[:, :: self.config.conf_reduction_factor]
            # Reduce the mask as well
            current_mask = current_mask[:, :: self.config.conf_reduction_factor]

        # Final masking of audio_encodings based on the final current_mask
        # Ensure current_mask length matches the finally reduced audio_encodings length
        if current_mask.shape[1] != audio_encodings.shape[1]:
            target_len = audio_encodings.shape[1]
            mask_current_len = current_mask.shape[1]
            if target_len > mask_current_len:
                padding_needed = target_len - mask_current_len
                current_mask = F.pad(current_mask, (0, padding_needed), value=True)
            elif mask_current_len > target_len:  # mask is longer
                current_mask = current_mask[:, :target_len]

        audio_encodings = audio_encodings.masked_fill(current_mask.unsqueeze(-1), 0.0)
        return audio_encodings, current_mask
