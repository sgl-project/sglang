# Copyright 2025 Xiaomi Corporation.
import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

from .configuration_audio_tokenizer import MiMoAudioTokenizerConfig
from .modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    apply_rotary_pos_emb,
    dynamic_rope_update,
)
from .quantization import ResidualVectorQuantizer


def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(
        bsz, tgt_len, 1
    )
    unpacking_index = torch.cumsum(sequence_mask.to(torch.int64).view(-1), dim=0) - 1
    return sequence_mask, unpacking_index


def unpack_hidden_states(
    hidden_states, lengths, sequence_mask=None, unpacking_index=None
):
    bsz = lengths.shape[0]
    if sequence_mask is None or unpacking_index is None:
        sequence_mask, unpacking_index = get_sequence_mask(hidden_states, lengths)
    hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
        bsz, torch.max(lengths), hidden_states.shape[-1]
    )
    hidden_states = torch.where(sequence_mask, hidden_states, 0)
    return hidden_states


def get_position_ids(lengths):
    total_len = lengths.sum()
    offset = torch.cat([torch.zeros(1).to(lengths), lengths[:-1].cumsum(dim=0)])
    offset = torch.repeat_interleave(offset, lengths)
    position_ids = torch.arange(0, total_len).to(offset) - offset
    return position_ids


@dataclass
class StreamingConfig:
    seg_point: int = field(default=60 * 25)
    process_seg_point: bool = field(default=True)
    left_overlap: int = field(default=10 * 25)
    right_overlap: int = field(default=40)
    seg_point_left_overlap: int = field(default=0)


@dataclass
class StreamingCache:
    hidden_states: List[torch.Tensor] = field(default=None)
    processed_lengths: List[int] = field(default=None)


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        original_dtype = x.dtype
        S = mag.float() * (x.float() + 1j * y.float())
        audio = self.istft(S)
        audio = audio.to(original_dtype)
        return audio


class RotaryEmbedding(nn.Module):
    def __init__(self, base, dim, max_seq_len, rope_type="default", device=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            device=device, base=base, dim=dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[:, None].float().expand(-1, 1).to(x.device)
        position_ids_expanded = position_ids[None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


LAYER_NORM = {"LayerNorm": nn.LayerNorm, "RMSNorm": RMSNorm}


# 替换代码部分
def sdpa_varlen_replacement(
    query_states,
    key_states,
    value_states,
    seq_len,
    bsz,
    num_heads,
    head_dim,
    causal=True,
):
    """
    使用 PyTorch 原生 SDPA 替换 flash_attn_varlen_func
    """
    # 1. 将压平的 (total_q, n_heads, d) 还原为 (bsz, n_heads, max_seqlen, d)
    # 注意：SDPA 要求输入形状为 (batch, heads, seq_len, head_dim)
    max_seqlen = max(seq_len)

    # 初始化输出张量
    # 这里我们假设输入的 query_states 等已经是 (total_q, n_heads, d) 格式
    # 我们需要根据 cu_len 或 seq_len 将其拆回 batch 维度

    # 简易做法：创建一个 padding 后的 tensor
    def unflatten_and_pad(states, seq_lens, bsz, n_heads, d):
        output = torch.zeros(
            bsz, n_heads, max_seqlen, d, device=states.device, dtype=states.dtype
        )
        start_idx = 0
        for i, length in enumerate(seq_lens):
            # 提取当前 sequence 并放入 output
            # states 形状假设为 [total_len, n_heads, d]
            output[i, :, :length, :] = states[start_idx : start_idx + length].transpose(
                0, 1
            )
            start_idx += length
        return output

    q = unflatten_and_pad(query_states, seq_len, bsz, num_heads, head_dim)
    k = unflatten_and_pad(key_states, seq_len, bsz, num_heads, head_dim)
    v = unflatten_and_pad(value_states, seq_len, bsz, num_heads, head_dim)

    # 2. 准备 Mask (处理变长 padding)
    # 如果 causal=True，SDPA 内部支持 is_causal 参数
    # 但由于我们有 padding，最好结合 key_padding_mask
    mask = None
    if not causal:
        # 创建 padding mask: (bsz, 1, 1, max_seqlen)
        mask = (
            torch.arange(max_seqlen, device=q.device)[None, :]
            < torch.tensor(seq_len, device=q.device)[:, None]
        )
        mask = mask.view(bsz, 1, 1, max_seqlen)  # 广播到 heads 和 query 维度

    # 3. 调用原生 SDPA
    # 如果是 causal 且没有额外的 padding mask，可以直接设 is_causal=True
    # 但因为我们是手动 padding 的，为了安全，建议显式构造 mask 或确保 padding 部分不参与计算
    attn_output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=causal
    )

    # 4. 将结果还原回原本的序列格式 (去掉 padding)
    # 原代码最后接的是 .reshape(bsz, self.embed_dim)，说明它期望输出是 (bsz, seq_len, hidden)
    # 或者是 total_len 展平后的格式。

    # 按照你原图中 flash_attn_varlen_func 的返回习惯，通常是 [total_len, heads, d]
    # 如果后续代码直接用 reshape(bsz, -1)，这里我们需要处理一下：

    # 将 (bsz, heads, max_len, d) -> (bsz, max_len, heads, d) -> (bsz, max_len, hidden)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(-1, -1), causal=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: torch.Tensor,
        rope_position_embeddings=None,
    ):
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(
            bsz, self.num_heads, self.head_dim
        )

        if rope_position_embeddings is not None:
            cos, sin = rope_position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(
            torch.int32
        )
        max_seqlen = torch.max(seq_len).to(torch.int32).detach()
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_len,
            cu_len,
            max_seqlen,
            max_seqlen,
            causal=self.causal,
            window_size=self.window_size,
        )
        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class TransformerLayer(nn.Module):
    def __init__(
        self,
        act,
        d_model,
        encoder_attention_heads,
        encoder_ffn_dim,
        causal,
        ln_type="LayerNorm",
        attn_window_size=(-1, -1),
    ):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = Attention(
            self.embed_dim, encoder_attention_heads, attn_window_size, causal
        )

        self.self_attn_layer_norm = LAYER_NORM[ln_type](self.embed_dim)

        self.activation_fn = act
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = LAYER_NORM[ln_type](self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: torch.Tensor,
        rope_position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, seq_len, rope_position_embeddings=rope_position_embeddings
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if (
            hidden_states.dtype == torch.float16
            or hidden_states.dtype == torch.bfloat16
        ) and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states


class TransformerVocos(nn.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.max_source_positions = (
            self.config.max_audio_seconds
            * self.config.sampling_rate
            // self.config.hop_length
        )
        self.embeddings = nn.Linear(config.n_mels, config.vocoder_dim, bias=False)

        self.poisition_embedding = RotaryEmbedding(
            config.rope_theta,
            config.vocoder_dim // config.vocoder_attention_heads,
            self.max_source_positions,
            self.config.rope_type,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    ACT2FN[self.config.activation_function],
                    self.config.vocoder_dim,
                    self.config.vocoder_attention_heads,
                    self.config.vocoder_intermediate_dim,
                    causal=False,
                    ln_type=self.config.ln_type,
                    attn_window_size=self.config.vocoder_attn_window_size,
                )
                for _ in range(self.config.vocoder_num_layers)
            ]
        )

        self.layer_norm = LAYER_NORM[self.config.ln_type](self.config.vocoder_dim)
        self.hop_size = self.config.hop_length
        self.head = ISTFTHead(
            self.config.vocoder_dim,
            self.config.nfft,
            self.config.hop_length,
            self.config.vocoder_padding,
        )

    def forward(self, x: torch.Tensor, input_length):
        x = x.transpose(1, 2)
        attention_mask, unpacking_index = get_sequence_mask(x, input_length)
        x = torch.masked_select(x, attention_mask).view(
            torch.sum(input_length), self.config.n_mels
        )
        x = self.embeddings(x)
        position_ids = torch.arange(0, x.size(0), device=x.device, dtype=torch.long)
        rope_position_embeddings = self.poisition_embedding(x, position_ids)
        for idx, layer in enumerate(self.layers):
            x = layer(
                x, input_length, rope_position_embeddings=rope_position_embeddings
            )

        x = self.layer_norm(x)
        x = unpack_hidden_states(x, input_length, attention_mask, unpacking_index)
        x = self.head(x)
        output_length = input_length * self.hop_size
        return x[:, None, :], output_length


class AudioEncoder(nn.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        config._attn_implementation = "flash_attention_2"
        self.config = config
        self.max_source_positions = (
            config.max_audio_seconds * config.sampling_rate // config.hop_length
        ) // config.stride_size
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.skip_layer_idx = config.encoder_skip_layer_id
        self.conv1 = nn.Conv1d(
            config.n_mels, config.d_model, kernel_size=config.kernel_size, padding=1
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.kernel_size,
            stride=config.stride_size,
            padding=1,
        )

        self.position_embedding = RotaryEmbedding(
            config.rope_theta,
            config.d_model // config.encoder_attention_heads,
            self.max_source_positions,
            config.rope_type,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    ACT2FN[config.activation_function],
                    config.d_model,
                    config.encoder_attention_heads,
                    config.encoder_ffn_dim,
                    causal=self.config.encoder_causal,
                    ln_type=self.config.ln_type,
                    attn_window_size=self.config.encoder_attn_window_size,
                )
                for _ in range(config.encoder_layers)
            ]
        )

        self.layer_norm = LAYER_NORM[config.ln_type](config.d_model)

        if self.config.avg_pooler != 1:
            self.down_sample_layer = nn.Sequential(
                nn.Conv1d(
                    config.d_model,
                    config.d_model,
                    config.avg_pooler,
                    config.avg_pooler,
                    bias=False,
                ),
                nn.GELU(),
            )
            self.down_sample_norm = LAYER_NORM[config.ln_type](config.d_model)
        else:
            self.down_sample_layer = None

        if self.config.num_quantizers != 0:
            self.quantizer = ResidualVectorQuantizer(
                dimension=self.config.d_model,
                n_q=self.config.num_quantizers,
                bins=self.config.codebook_size,
                threshold_ema_dead_code=self.config.threshold_ema_dead_code,
            )
        else:
            self.quantizer = None

    def get_features(self, input_features, output_length):
        input_features = input_features.to(self.conv1.weight)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        bsz, tgt_len, _ = inputs_embeds.size()

        hidden_states = inputs_embeds

        position_ids = get_position_ids(output_length).long().to(input_features.device)
        rope_position_embeddings = self.position_embedding(input_features, position_ids)

        attention_mask, unpacking_index = get_sequence_mask(
            hidden_states, output_length
        )

        hidden_states = torch.masked_select(hidden_states, attention_mask).view(
            torch.sum(output_length), self.config.d_model
        )

        skip_connect_hidden_states = 0.0
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                output_length,
                rope_position_embeddings=rope_position_embeddings,
            )
            if (self.skip_layer_idx is not None) and idx == self.skip_layer_idx - 1:
                skip_connect_hidden_states = hidden_states.clone()

        hidden_states += skip_connect_hidden_states
        hidden_states = self.layer_norm(hidden_states)

        if self.down_sample_layer is not None:
            hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
                bsz, tgt_len, self.config.d_model
            )
            if hidden_states.size(1) % self.config.avg_pooler:
                pad_len = (
                    self.config.avg_pooler
                    - hidden_states.size(1) % self.config.avg_pooler
                )
                hidden_states = torch.nn.functional.pad(
                    hidden_states, (0, 0, 0, pad_len), mode="constant", value=0.0
                )
                tgt_len += pad_len
            tgt_len = tgt_len // self.config.avg_pooler
            hidden_states = self.down_sample_layer(hidden_states.transpose(1, 2))
            output_length = (
                output_length // self.config.avg_pooler
                + (output_length % self.config.avg_pooler != 0).int()
            )
            hidden_states = hidden_states.transpose(1, 2)
            attention_mask, unpacking_index = get_sequence_mask(
                hidden_states, output_length
            )
            hidden_states = torch.masked_select(hidden_states, attention_mask).view(
                torch.sum(output_length), self.config.d_model
            )
            hidden_states = self.down_sample_norm(hidden_states)

        return (
            hidden_states,
            output_length,
            attention_mask,
            unpacking_index,
            tgt_len,
            bsz,
        )

    def get_output_length(self, mel_len):
        tgt_len = mel_len + 3 - self.config.kernel_size
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1

    @torch.no_grad()
    def encode(
        self,
        input_features,
        input_lens=None,
        output_length=None,
        return_codes_only=False,
        n_q=None,
        use_quantizer=True,
    ):
        if output_length is None:
            output_length = self.get_output_length(input_lens)
        input_features = unpack_hidden_states(input_features, input_lens)
        hidden_states, output_length, attention_mask, unpacking_index, tgt_len, bsz = (
            self.get_features(
                input_features=input_features.transpose(1, 2),
                output_length=output_length,
            )
        )

        dtype = hidden_states.dtype

        if use_quantizer and self.quantizer is not None:
            self.quantizer.float()

            codes = self.quantizer.encode(hidden_states.float(), n_q=n_q)
            if return_codes_only:
                return codes, output_length
            hidden_states = self.quantizer.decode(codes)
            hidden_states = hidden_states.to(dtype)
        else:
            codes = None

        hidden_states_packed = hidden_states.clone()

        # unpacking
        hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
            bsz, tgt_len, self.config.d_model
        )
        hidden_states = torch.where(attention_mask, hidden_states, 0)
        return hidden_states, hidden_states_packed, output_length, codes

    @torch.no_grad()
    def decode_vq(self, codes):
        self.quantizer.float()
        hidden_states = self.quantizer.decode(codes)

        return hidden_states


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, hidden_states, input_length, output_dim=None):
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        bsz = input_length.shape[0]

        if output_dim is None:
            output_dim = hidden_states.dim()
        if hidden_states.dim() <= 2:  # unpack sequence to 3d
            sequence_mask, unpacking_index = get_sequence_mask(
                hidden_states, input_length
            )
            hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
                bsz, torch.max(input_length), self.in_channels
            )
            hidden_states = torch.where(sequence_mask, hidden_states, 0)

        hidden_states = hidden_states.transpose(2, 1)  # (N, L, C) -> (N, C, L)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(2, 1)  # (N, C, L) -> (N, L, C)

        casual_padding_right = max(0, kernel_size - stride)
        hidden_states = hidden_states[
            :, : hidden_states.shape[1] - casual_padding_right, :
        ]
        output_length = (input_length - 1) * stride + kernel_size - casual_padding_right
        sequence_mask, _ = get_sequence_mask(hidden_states, output_length)
        if output_dim <= 2:
            hidden_states = torch.masked_select(hidden_states, sequence_mask).view(
                -1, self.out_channels
            )
        else:
            hidden_states = torch.where(sequence_mask, hidden_states, 0)
            hidden_states = hidden_states[:, : torch.max(output_length), :]
        return hidden_states, output_length


class AudioDecoder(nn.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.max_source_positions = (
            self.config.max_audio_seconds
            * self.config.sampling_rate
            // self.config.hop_length
        )

        if self.config.avg_pooler != 1:
            self.dconv1 = CausalConvTranspose1d(
                self.config.d_model,
                self.config.d_model,
                self.config.avg_pooler,
                self.config.avg_pooler,
            )
        else:
            self.dconv1 = None

        self.position_embedding = RotaryEmbedding(
            config.rope_theta,
            config.d_model // config.decoder_attention_heads,
            self.max_source_positions,
            config.rope_type,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    ACT2FN[self.config.activation_function],
                    self.config.d_model,
                    self.config.decoder_attention_heads,
                    self.config.decoder_ffn_dim,
                    causal=self.config.decoder_causal,
                    ln_type=self.config.ln_type,
                    attn_window_size=self.config.decoder_attn_window_size,
                )
                for _ in range(self.config.decoder_layers)
            ]
        )
        self.layer_norm = LAYER_NORM[config.ln_type](self.config.d_model)
        self.dconv2 = CausalConvTranspose1d(
            self.config.d_model,
            self.config.n_mels,
            self.config.decoder_kernel_size,
            self.config.decoder_stride_size,
        )
        self.vocoder = TransformerVocos(config)

    def forward(
        self,
        audio_embed,
        input_length,
    ):
        assert audio_embed.shape[-1] == self.config.d_model
        audio_embed = audio_embed.to(self.layer_norm.weight)

        if self.dconv1 is not None:
            audio_embed, output_length = self.dconv1(
                audio_embed, input_length, output_dim=3
            )
            _, tgt_len, _ = audio_embed.size()
        else:
            output_length = input_length
            tgt_len = audio_embed.size(0)

        hidden_states = audio_embed

        position_ids = get_position_ids(output_length).long().to(hidden_states.device)
        rope_position_embeddings = self.position_embedding(hidden_states, position_ids)

        # packing hidden states
        attention_mask, _ = get_sequence_mask(hidden_states, output_length)
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(
            torch.sum(output_length), self.config.d_model
        )

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                output_length,
                rope_position_embeddings=rope_position_embeddings,
            )

        hidden_states = self.layer_norm(hidden_states)

        coarse_mel, output_length = self.dconv2(
            hidden_states, output_length, output_dim=3
        )

        recon_wav, wav_length = self.vocoder(
            x=coarse_mel.transpose(1, 2),
            input_length=output_length,
        )

        return recon_wav


class MiMoAudioTokenizer(PreTrainedModel):
    config_class = MiMoAudioTokenizerConfig

    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__(config)
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.encoder = AudioEncoder(config=config)
        self.decoder = AudioDecoder(config=config)
        self.downsample_rate = int(self.config.hop_length * 2 * self.config.avg_pooler)

    def get_output_length(self, mel_len):
        tgt_len = mel_len + 3 - self.config.kernel_size
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1

    @torch.no_grad()
    def encode(self, mels, input_lens, use_quantizer=True):
        input_features = mels
        encoder_output_length = self.get_output_length(input_lens)
        hidden_states, hidden_states_packed, encoder_output_length, codes = (
            self.encoder.encode(
                input_features, input_lens=input_lens, use_quantizer=use_quantizer
            )
        )
        return hidden_states, hidden_states_packed, encoder_output_length, codes

    @torch.no_grad()
    def decode(self, codes):
        hidden_states = self.encoder.decode_vq(codes)
        output = self.decoder(
            hidden_states,
            torch.tensor([hidden_states.size(0)], device=hidden_states.device),
        )
        return output

    @torch.no_grad()
    def streaming_decode(
        self,
        codes_chunks,
        chunk_input_lengths,
        history_cache=StreamingCache(),
        streaming_config=StreamingConfig(),
        last_chunk=False,
    ):
        hidden_states = self.encoder.decode_vq(codes_chunks)
        input_lengths = []
        input_hidden_states = []
        start_idx = 0
        cache_hidden_states = []
        for i, input_length in enumerate(chunk_input_lengths):
            sample_hidden_states = hidden_states[start_idx : start_idx + input_length]
            start_idx += input_length
            if history_cache.hidden_states is not None:
                sample_hidden_states = torch.cat(
                    [history_cache.hidden_states[i], sample_hidden_states], dim=0
                )
                input_length += history_cache.hidden_states[i].size(0)
            input_hidden_states.append(sample_hidden_states)
            cache_hidden_states.append(sample_hidden_states.clone())
            input_lengths.append(input_length)
        input_hidden_states = torch.cat(input_hidden_states, dim=0)
        input_lengths = torch.tensor(input_lengths, device=hidden_states.device)
        output = self.decoder(input_hidden_states, input_lengths)
        return_wavs = []
        frames_per_token = (
            self.config.avg_pooler * self.config.stride_size * self.config.hop_length
        )
        processed_lengths = []
        for i, wav in enumerate(output):
            wav = wav.float().detach().cpu()
            start_idx = (
                history_cache.processed_lengths[i]
                if history_cache.processed_lengths is not None
                else 0
            )
            if last_chunk:
                return_wavs.append(wav[:, start_idx * frames_per_token :])
                new_processed_length = input_lengths[i].item()
            elif input_lengths[i].item() <= streaming_config.right_overlap:
                return_wavs.append(None)
                new_processed_length = 0
            else:
                end_idx = input_lengths[i].item() - streaming_config.right_overlap
                wav = wav[:, start_idx * frames_per_token : end_idx * frames_per_token]
                return_wavs.append(wav)
                new_processed_length = end_idx
                if input_lengths[i].item() > streaming_config.left_overlap:
                    cache_hidden_states[i] = cache_hidden_states[i][
                        -streaming_config.left_overlap :
                    ]
                    new_processed_length -= (
                        input_lengths[i].item() - streaming_config.left_overlap
                    )
            processed_lengths.append(new_processed_length)
        history_cache.hidden_states = cache_hidden_states
        history_cache.processed_lengths = processed_lengths

        return return_wavs, history_cache
