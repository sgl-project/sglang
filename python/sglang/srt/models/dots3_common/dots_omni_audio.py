"""Dots-path speech encoder for inference only (single GPU).

Ported from cybertron_alm ``dots_audio_encoder/modeling_whisper.py``.
Upstream ``WhisperEncoder`` is exposed as :class:`DotsSpeechEncoder`.
"""

import math
from functools import lru_cache
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.audio_utils import mel_filter_bank
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.utils import logging

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.srt.layers.rotary_embedding.utils import rotate_neox

logger = logging.get_logger(__name__)

__all__ = [
    "DotsSpeechEncoder",
    "DotsWhisperConfig",
    "OmniAudioConfig",
    "OmniAudioModel",
    "compute_audio_token_length",
]


class DotsWhisperConfig(WhisperConfig):
    """Whisper configuration with the fields required by the Dots encoder."""

    def __init__(
        self,
        *,
        use_causal: bool = False,
        use_rms_norm: bool = False,
        use_latent_input: bool = False,
        use_conv2d_stem: bool = False,
        latent_dim: int | None = None,
        downsample_hidden_size: int = 480,
        use_rope: bool = False,
        rope_parameters: dict | None = None,
        conv_chunksize: int = 500,
        conv_stem_gradient_checkpointing: bool = False,
        conv_bucket_step: int | None = None,
        conv_bucket_max_elements: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_causal = use_causal
        self.use_rms_norm = use_rms_norm
        self.use_latent_input = use_latent_input
        self.use_conv2d_stem = use_conv2d_stem
        self.latent_dim = latent_dim
        self.downsample_hidden_size = downsample_hidden_size
        self.use_rope = use_rope
        self.rope_parameters = rope_parameters or {}
        self.conv_chunksize = conv_chunksize
        self.conv_stem_gradient_checkpointing = conv_stem_gradient_checkpointing
        self.conv_bucket_step = conv_bucket_step
        self.conv_bucket_max_elements = conv_bucket_max_elements


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def swiglu(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(x1) * x2


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rope_parameters: dict,
        base_seq_len: int = 0,
    ):
        super().__init__()
        self.partial_rotary_factor = float(
            rope_parameters.get("partial_rotary_factor", 1.0)
        )
        self.rope_theta = float(rope_parameters.get("rope_theta", 10000.0))
        self.rope_type = rope_parameters.get("rope_type", "default")
        rotary_dim = int(head_dim * self.partial_rotary_factor)
        # keep even rotary dimension to align cos/sin pairs
        self.rotary_dim = (rotary_dim // 2) * 2
        self.attention_scaling = 1.0
        if self.rope_type != "default":
            logger.warning(
                f"RoPE type {self.rope_type} not implemented in WhisperLv3, fallback to default."
            )
        if self.rotary_dim > 0:
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                    / max(self.rotary_dim, 1)
                )
            )
        else:
            inv_freq = torch.tensor([])
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.base_seq_len = max(0, int(base_seq_len))
        self._cache: (
            tuple[int, torch.dtype, torch.device, torch.Tensor, torch.Tensor] | None
        ) = None

    @torch.no_grad()
    def get_cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.rotary_dim == 0:
            return None, None
        seq_len = position_ids.shape[-1]
        if position_ids.shape[0] == 1 and self._cache is not None:
            cached_seq_len, cached_dtype, cached_device, cached_cos, cached_sin = (
                self._cache
            )
            if (
                cached_seq_len >= seq_len
                and cached_dtype == dtype
                and cached_device == device
            ):
                return cached_cos[:, :seq_len, :], cached_sin[:, :seq_len, :]

        if position_ids.shape[0] == 1:
            cache_seq_len = max(seq_len, self.base_seq_len)
            position_ids = torch.arange(cache_seq_len, device=device)[None, :]
        # [1, D/2, 1]
        if self.inv_freq.dtype != torch.float32:
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(
                        0, self.rotary_dim, 2, dtype=torch.float, device=device
                    )
                    / max(self.rotary_dim, 1)
                )
            )
        else:
            inv_freq = self.inv_freq.to(device=device)
        inv_freq_expanded = inv_freq[None, :, None]
        # [B, 1, T]
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = device.type if isinstance(device, torch.device) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        cos = cos.to(dtype)
        sin = sin.to(dtype)
        if position_ids.shape[0] == 1:
            self._cache = (position_ids.shape[-1], dtype, device, cos, sin)
            return cos[:, :seq_len, :], sin[:, :seq_len, :]
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if q.dim() not in (3, 4):
        return q, k

    rotary_dim = cos.shape[-1]
    unsqueeze_dim = 2 if q.dim() == 4 else 1
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_neox(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_neox(k_rot) * sin)
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward_flash_attn(
        self,
        hidden_states,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        output_attentions=False,
        rotary_cos=None,
        rotary_sin=None,
    ):
        """Dense eager attention with SGLang FA3 for packed variable-length input."""
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if cu_seqlens_q is None:
            bsz, tgt_len, _ = hidden_states.size()
            query_states = query_states.view(
                bsz, tgt_len, self.num_heads, self.head_dim
            )
            key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
            value_states = value_states.view(
                bsz, tgt_len, self.num_heads, self.head_dim
            )
            if rotary_cos is not None and rotary_sin is not None:
                cos = rotary_cos[:, :tgt_len, :]
                sin = rotary_sin[:, :tgt_len, :]
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
            attn_output, attn_probs = self._eager_attention(
                query_states, key_states, value_states, output_attentions
            )
            attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        else:
            query_states = query_states.view(-1, self.num_heads, self.head_dim)
            key_states = key_states.view(-1, self.num_heads, self.head_dim)
            value_states = value_states.view(-1, self.num_heads, self.head_dim)
            if rotary_cos is not None and rotary_sin is not None:
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, rotary_cos, rotary_sin
                )
            result = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                softmax_scale=self.scaling,
                causal=self.is_decoder,
                return_softmax_lse=output_attentions,
            )
            if isinstance(result, tuple):
                attn_output = result[0]
                attn_probs = result[1] if output_attentions else None
            else:
                attn_output = result
                attn_probs = None
            attn_output = attn_output.view(-1, self.embed_dim)

        return self.out_proj(attn_output), attn_probs

    def _eager_attention(self, query, key, value, output_attentions):
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        if self.is_decoder:
            seq_len = scores.shape[-1]
            causal_mask = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=scores.device
            ).triu(1)
            scores.masked_fill_(causal_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(probabilities, value).transpose(1, 2)
        return output, probabilities if output_attentions else None


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: DotsWhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        use_causal = config.use_causal
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=use_causal,  # enable causal attention when use_causal=True
        )
        self.use_causal = use_causal
        norm_cls = RMSNorm if config.use_rms_norm else nn.LayerNorm
        self.self_attn_layer_norm = norm_cls(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = (
            ACT2FN[config.activation_function]
            if config.activation_function != "swiglu"
            else swiglu
        )
        self.use_swiglu = config.activation_function == "swiglu"
        self.activation_dropout = config.activation_dropout
        ffn_dim = config.encoder_ffn_dim
        fc1_out = ffn_dim * 2 if self.use_swiglu else ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, fc1_out)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.final_layer_norm = norm_cls(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_kv: torch.Tensor = None,
        max_seqlen_q: int | None = None,
        max_seqlen_kv: int | None = None,
        output_attentions: bool = False,
        rotary_cos: torch.Tensor | None = None,
        rotary_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights = self.self_attn.forward_flash_attn(
            hidden_states=hidden_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            output_attentions=output_attentions,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.use_swiglu:
            hidden_states = swiglu(self.fc1(hidden_states))
        else:
            hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DotsSpeechPreTrainedModel(PreTrainedModel):
    config_class = DotsWhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = False
    _no_split_modules: ClassVar[list[str]] = ["WhisperEncoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class DotsSpeechEncoder(DotsSpeechPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: DotsWhisperConfig):
        if not isinstance(config, DotsWhisperConfig):
            raise TypeError(
                "DotsSpeechEncoder requires DotsWhisperConfig, not a plain "
                "transformers.WhisperConfig."
            )
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.use_latent_input = config.use_latent_input
        self.use_causal = config.use_causal
        self.use_conv2d_stem = config.use_conv2d_stem
        latent_dim = config.latent_dim
        if self.use_latent_input and latent_dim is None:
            raise ValueError(
                "DotsSpeechEncoder: use_latent_input=True requires config.latent_dim"
            )
        if self.use_conv2d_stem and self.use_latent_input:
            raise ValueError(
                "DotsSpeechEncoder: use_conv2d_stem and use_latent_input are mutually exclusive"
            )
        self.num_mel_bins = (
            latent_dim if self.use_latent_input and latent_dim is not None else 128
        )
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if self.use_conv2d_stem:
            # Conv2D stem: 3 layers of stride-2 for 8x downsampling
            dhs = config.downsample_hidden_size
            # Causal: keep freq padding, remove time padding (handled by pad(14,0) in forward)
            conv_padding = (1, 0) if self.use_causal else 1
            self.conv2d1 = nn.Conv2d(
                1, dhs, kernel_size=3, stride=2, padding=conv_padding
            )
            self.conv2d2 = nn.Conv2d(
                dhs, dhs, kernel_size=3, stride=2, padding=conv_padding
            )
            self.conv2d3 = nn.Conv2d(
                dhs, dhs, kernel_size=3, stride=2, padding=conv_padding
            )
            # After 3x stride-2 on freq=128: 128→64→32→16; linear projects dhs*16 → embed_dim
            freq_after = self.num_mel_bins
            for _ in range(3):
                freq_after = (freq_after + 1) // 2
            self.conv_out = nn.Linear(dhs * freq_after, embed_dim, bias=False)
            self.conv1 = None
            self.conv2 = None
        elif self.use_latent_input:
            # Latent path keeps length (stride=1) and applies GLU after conv1
            self.conv1 = nn.Conv1d(
                self.num_mel_bins, embed_dim * 2, kernel_size=3, stride=1, padding=1
            )
            self.conv2 = nn.Conv1d(
                embed_dim, embed_dim, kernel_size=3, stride=1, padding=1
            )
        else:
            self.conv1 = nn.Conv1d(
                self.num_mel_bins, embed_dim, kernel_size=3, padding=1
            )
            self.conv2 = nn.Conv1d(
                embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
            )

        self.use_rope = config.use_rope
        rope_parameters = config.rope_parameters
        self.rope_parameters = rope_parameters
        if self.use_rope:
            head_dim = embed_dim // config.encoder_attention_heads
            self.rotary_embedding = RotaryEmbedding(
                head_dim,
                rope_parameters,
                base_seq_len=self.max_source_positions,
            )
            self.embed_positions = None
        else:
            self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        norm_cls = RMSNorm if config.use_rms_norm else nn.LayerNorm
        self.layer_norm = norm_cls(config.d_model)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        if self.use_conv2d_stem:
            return self.conv2d1
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    @staticmethod
    def _temporal_mask(feat, valid_lens):
        """Zero out temporal positions >= valid_lens. feat: [B,C,F,T], valid_lens: [B] tensor."""
        T = feat.shape[-1]
        mask = torch.arange(T, device=feat.device)[None, :] < valid_lens[:, None]
        return mask[:, None, None, :]

    def _conv2d_stem_one_chunk(self, chunk, chunk_valid_mel_lens=None):
        """Run 3x Conv2d(stride=2) + GELU with per-layer masking."""
        # Causal: left-pad time by 14 = 2 + 4 + 8 (receptive field of the 3-layer
        # stride-2 stack mapped back to input time). Combined with conv padding=(1,0),
        # this makes the whole conv2d stem strictly causal.
        if self.use_causal:
            chunk = nn.functional.pad(chunk, (14, 0))
            if chunk_valid_mel_lens is not None:
                chunk_valid_mel_lens = chunk_valid_mel_lens + 14
        # Step 0: mask mel — silence_mel(-1.5) → 0
        if chunk_valid_mel_lens is not None:
            chunk = chunk * self._temporal_mask(chunk, chunk_valid_mel_lens)
        chunk = nn.functional.gelu(self.conv2d1(chunk))
        # Step 1: mask conv1 output — gelu(conv(0)+bias) → 0
        if chunk_valid_mel_lens is not None:
            chunk_valid_mel_lens = (chunk_valid_mel_lens + 1) // 2
            chunk = chunk * self._temporal_mask(chunk, chunk_valid_mel_lens)
        chunk = nn.functional.gelu(self.conv2d2(chunk))
        # Step 2: mask conv2 output — gelu(bias) → 0
        if chunk_valid_mel_lens is not None:
            chunk_valid_mel_lens = (chunk_valid_mel_lens + 1) // 2
            chunk = chunk * self._temporal_mask(chunk, chunk_valid_mel_lens)
        chunk = nn.functional.gelu(self.conv2d3(chunk))
        # Step 3: mask conv3 output — gelu(bias) → 0
        if chunk_valid_mel_lens is not None:
            chunk_valid_mel_lens = (chunk_valid_mel_lens + 1) // 2
            chunk = chunk * self._temporal_mask(chunk, chunk_valid_mel_lens)
        return chunk

    def _forward_conv2d_stem(
        self, input_features, input_seq_lens=None, audio_sample_lens=None
    ):
        """Conv2D stem: 3x Conv2d(stride=2) → 8x downsample. Returns [B, T/8, embed_dim]."""
        x = input_features.unsqueeze(1)  # [B, 1, 128, T]

        if audio_sample_lens is not None:
            hop_length = 160
            if not isinstance(audio_sample_lens, torch.Tensor):
                audio_sample_lens = torch.tensor(audio_sample_lens, device=x.device)
            valid_mel_lens = audio_sample_lens.to(x.device) // hop_length
        else:
            valid_mel_lens = None
        x = self._conv2d_stem_one_chunk(x, valid_mel_lens)
        # [B, dhs, F_out, T/8] → [B, T/8, dhs*freq_after] → [B, T/8, embed_dim]
        batch, channels, frequency, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * frequency)
        inputs_embeds = self.conv_out(x)

        return inputs_embeds

    def _forward_conv1d_stem(self, input_features):
        """Standard Conv1d stem: conv1 + conv2(stride=2), 2x downsample. Returns [B, T/2, embed_dim]."""
        if self.use_causal:
            x = nn.functional.pad(input_features, (2, 0))
            x = nn.functional.gelu(self.conv1(x))
            x = nn.functional.pad(x, (2, 0))
            x = nn.functional.gelu(self.conv2(x))
        else:
            x = nn.functional.gelu(self.conv1(input_features))
            x = nn.functional.gelu(self.conv2(x))
        return x.permute(0, 2, 1)

    def _forward_latent_stem(self, input_features):
        """Latent input stem: conv1+GLU + conv2, stride=1, no downsample. Returns [B, T, embed_dim]."""
        if self.use_causal:
            x = nn.functional.pad(input_features, (2, 0))
            x = nn.functional.glu(self.conv1(x), dim=1)
            x = nn.functional.pad(x, (2, 0))
            x = nn.functional.gelu(self.conv2(x))
        else:
            x = nn.functional.glu(self.conv1(input_features), dim=1)
            x = nn.functional.gelu(self.conv2(x))
        return x.permute(0, 2, 1)

    def forward(
        self,
        input_features,
        input_seq_lens=None,
        audio_sample_lens=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Mel `[B, n_mels, T]` → encoder hidden states. Inference-only."""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if self.use_conv2d_stem:
            inputs_embeds = self._forward_conv2d_stem(
                input_features, input_seq_lens, audio_sample_lens
            )
        elif self.use_latent_input:
            inputs_embeds = self._forward_latent_stem(input_features)
        else:
            inputs_embeds = self._forward_conv1d_stem(input_features)
        rotary_cos = None
        rotary_sin = None
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )[None, :]
        if self.use_rope:
            rotary_cos, rotary_sin = self.rotary_embedding.get_cos_sin(
                position_ids, inputs_embeds.dtype, inputs_embeds.device
            )
            hidden_states = inputs_embeds
        else:
            embed_pos = self.embed_positions.weight[: inputs_embeds.shape[1]]
            hidden_states = inputs_embeds + embed_pos

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if input_seq_lens is not None:
            # Build varlen metadata and pack valid tokens without per-sample cat loops.
            # Read max on whatever device the caller provided: when it is a CPU
            # tensor this avoids a device->host sync (one per forward).
            max_seqlen_q = int(input_seq_lens.max().item())
            input_seq_lens = input_seq_lens.to(
                device=hidden_states.device, dtype=torch.long
            )
            B, S, D = hidden_states.shape
            max_seqlen_kv = max_seqlen_q
            cu_seqlens_q = torch.nn.functional.pad(
                input_seq_lens.cumsum(0, dtype=torch.int32), (1, 0)
            )
            cu_seqlens_kv = cu_seqlens_q

            token_positions = torch.arange(S, device=hidden_states.device)[None, :]
            valid_token_mask = token_positions < input_seq_lens[:, None]
            hidden_states = hidden_states[valid_token_mask]
            if rotary_cos is not None and rotary_sin is not None:
                packed_positions = token_positions.expand(B, S)[valid_token_mask]
                rotary_cos = rotary_cos.squeeze(0).index_select(0, packed_positions)
                rotary_sin = rotary_sin.squeeze(0).index_select(0, packed_positions)
        else:
            cu_seqlens_q = None
            cu_seqlens_kv = None
            max_seqlen_q = None
            max_seqlen_kv = None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                output_attentions=output_attentions,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if input_seq_lens is not None:
            # Recover packed varlen output directly on device.
            recover_positions = torch.arange(max_seqlen_q, device=hidden_states.device)[
                None, :
            ]
            recover_mask = recover_positions < input_seq_lens[:, None]
            recovered = hidden_states.new_zeros(B, max_seqlen_q, D)
            recovered[recover_mask] = hidden_states
            hidden_states = recovered

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
DEFAULT_CHUNK_LENGTH_S = 60
DEFAULT_CONV_TEMPORAL_STRIDE = 8
DEFAULT_MERGE_FACTOR = 1
N_SAMPLES = DEFAULT_CHUNK_LENGTH_S * SAMPLE_RATE


class OmniAudioConfig:
    def __init__(self, **kwargs):
        self.encoder_type = kwargs.get("encoder_type", "dots")
        self.whisper_config = kwargs.get("whisper_config", {})
        self.whisper_adapter_in_dim = kwargs.get(
            "whisper_adapter_in_dim", kwargs.get("adapter_in_dim", 1280)
        )
        self.whisper_adapter_out_dim = kwargs.get(
            "whisper_adapter_out_dim", kwargs.get("adapter_out_dim", 2048)
        )
        self.sampling_rate = kwargs.get("sampling_rate", SAMPLE_RATE)
        self.audio_comp_start = kwargs.get("audio_comp_start", "<|audio_comp_start|>")
        self.audio_comp_span = kwargs.get("audio_comp_span", "<|audio_comp_pad|>")
        self.audio_comp_end = kwargs.get("audio_comp_end", "<|audio_comp_end|>")
        self.merge_factor = kwargs.get("merge_factor", DEFAULT_MERGE_FACTOR)
        self.chunk_seconds = kwargs.get("chunk_seconds", DEFAULT_CHUNK_LENGTH_S)
        self.use_conv2d_stem = kwargs.get("use_conv2d_stem", True)
        self.use_latent_input = kwargs.get("use_latent_input", False)
        self.latent_dim = kwargs.get("latent_dim")
        self.use_rope = kwargs.get("use_rope", True)
        self.use_rms_norm = kwargs.get("use_rms_norm", True)
        self.use_causal = kwargs.get("use_causal", False)
        self.downsample_hidden_size = kwargs.get("downsample_hidden_size", 480)
        self.conv_chunksize = kwargs.get("conv_chunksize", 500)
        self.conv_stem_gradient_checkpointing = kwargs.get(
            "conv_stem_gradient_checkpointing", False
        )
        self.conv_bucket_step = kwargs.get("conv_bucket_step", None)
        self.conv_bucket_max_elements = kwargs.get("conv_bucket_max_elements", None)
        self.rope_parameters = kwargs.get(
            "rope_parameters",
            {
                "partial_rotary_factor": 0.5,
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        )

    @property
    def conv_temporal_stride(self) -> int:
        return 8 if self.use_conv2d_stem else 2

    @property
    def token_stride(self) -> int:
        return HOP_LENGTH * self.conv_temporal_stride * self.merge_factor

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_seconds * SAMPLE_RATE)

    @property
    def chunk_mel_frames(self) -> int:
        return int(self.chunk_seconds * 100)


def pad_or_trim(array, length=N_SAMPLES, axis=-1):
    if array.shape[axis] > length:
        array = array.index_select(
            dim=axis, index=torch.arange(length, device=array.device)
        )
    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    return array


@lru_cache(maxsize=4)
def _mel_filters(device, n_mels=128):
    filters = mel_filter_bank(
        num_frequency_bins=1 + N_FFT // 2,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=float(SAMPLE_RATE) / 2.0,
        sampling_rate=SAMPLE_RATE,
        norm="slaney",
        mel_scale="slaney",
    )
    return torch.from_numpy(filters).T.contiguous().float().to(device)


@lru_cache(maxsize=4)
def _hann_window(device):
    # Generate on CPU (default) then move, to preserve the exact reference
    # window values. Direct on-device generation changes numerics slightly.
    return torch.hann_window(N_FFT).to(device)


def log_mel_spectrogram(audio, n_mels=128):
    window = _hann_window(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def compute_audio_token_length(
    num_samples,
    *,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: int = DEFAULT_CHUNK_LENGTH_S,
    hop_length: int = HOP_LENGTH,
    conv_temporal_stride: int = DEFAULT_CONV_TEMPORAL_STRIDE,
    merge_factor: int = DEFAULT_MERGE_FACTOR,
):
    stride = hop_length * conv_temporal_stride * merge_factor
    total = 0
    time_step = 0
    while time_step * sample_rate < num_samples:
        chunk_len = min(
            num_samples - time_step * sample_rate, chunk_seconds * sample_rate
        )
        total += math.ceil(chunk_len / stride)
        time_step += chunk_seconds
    return total


class DotsEncoderWithMask(nn.Module):
    def __init__(self, config: OmniAudioConfig):
        super().__init__()
        whisper_config_kwargs = dict(config.whisper_config)
        whisper_config_kwargs.update(
            use_rope=config.use_rope,
            rope_parameters=config.rope_parameters,
            use_rms_norm=config.use_rms_norm,
            use_causal=config.use_causal,
            use_conv2d_stem=config.use_conv2d_stem,
            use_latent_input=config.use_latent_input,
            latent_dim=config.latent_dim,
            downsample_hidden_size=config.downsample_hidden_size,
            conv_chunksize=config.conv_chunksize,
            conv_stem_gradient_checkpointing=(config.conv_stem_gradient_checkpointing),
            conv_bucket_step=config.conv_bucket_step,
            conv_bucket_max_elements=config.conv_bucket_max_elements,
        )
        whisper_config = DotsWhisperConfig(**whisper_config_kwargs)

        self.speech_encoder = DotsSpeechEncoder(whisper_config)
        self.merge_factor = config.merge_factor
        self.chunk_seconds = config.chunk_seconds
        self.chunk_samples = config.chunk_samples
        self.chunk_mel_frames = config.chunk_mel_frames
        self.conv_temporal_stride = config.conv_temporal_stride

    @property
    def device(self):
        return next(self.speech_encoder.parameters()).device

    def _forward_speech_encoder(
        self,
        mel_features: torch.Tensor,
        input_seq_lens: torch.Tensor,
        audio_sample_lens: list[int],
    ) -> torch.Tensor:
        """Run the eager speech encoder without server-side slicing/batching."""
        mel_features = mel_features.to(dtype=torch.bfloat16, device=self.device)
        return self.speech_encoder(
            mel_features,
            return_dict=True,
            input_seq_lens=input_seq_lens,
            audio_sample_lens=audio_sample_lens,
        ).last_hidden_state

    def encode_waveform(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        segments = []
        time_step = 0
        while time_step * SAMPLE_RATE < audio_waveform.shape[0]:
            segments.append(
                audio_waveform[
                    time_step * SAMPLE_RATE : (time_step + self.chunk_seconds)
                    * SAMPLE_RATE
                ]
            )
            time_step += self.chunk_seconds

        mel_features = []
        token_lens = []
        audio_sample_lens = []
        for audio_segment in segments:
            segment_length = audio_segment.shape[0]
            token_len = (segment_length - 1) // (
                HOP_LENGTH * self.conv_temporal_stride * self.merge_factor
            ) + 1
            pad_audio = pad_or_trim(audio_segment.flatten(), length=self.chunk_samples)
            mel = log_mel_spectrogram(pad_audio)
            assert mel.shape[1] == self.chunk_mel_frames
            mel_features.append(mel)
            token_lens.append(token_len)
            audio_sample_lens.append(segment_length)

        mel_features = torch.stack(mel_features, dim=0)
        # Keep input_seq_lens on CPU: the conv2d bucket path reads it via
        # ``.item()`` and CPU scalars avoid device->host syncs. The encoder's
        # varlen path moves it to the device itself.
        input_seq_lens = torch.tensor(token_lens, dtype=torch.long) * self.merge_factor
        audio_embedding = self._forward_speech_encoder(
            mel_features, input_seq_lens, audio_sample_lens
        )

        chunk_embeddings = []
        for idx, token_len in enumerate(token_lens):
            chunk_embeddings.append(
                audio_embedding[idx, : token_len * self.merge_factor, :]
            )
        return torch.cat(chunk_embeddings, dim=0).unsqueeze(0)


class AudioAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class OmniAudioModel(nn.Module):
    def __init__(self, config: OmniAudioConfig):
        super().__init__()
        if config.encoder_type != "dots":
            raise ValueError("Dots omni only supports encoder_type='dots'")
        self.merge_factor = config.merge_factor
        self.audio_adapter = AudioAdapter(
            config.whisper_adapter_in_dim,
            config.whisper_adapter_out_dim,
        )

        self.dots_encoder = DotsEncoderWithMask(config)

    @property
    def device(self):
        return self.dots_encoder.device

    def _merge_embeddings(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.merge_factor <= 1:
            return embedding
        return embedding.reshape(
            embedding.shape[0],
            embedding.shape[1] // self.merge_factor,
            embedding.shape[2] * self.merge_factor,
        )

    def _encode_single_audio(self, audio_waveform):
        embedding = self.dots_encoder.encode_waveform(audio_waveform)
        embedding = self._merge_embeddings(embedding)
        embedding = self.audio_adapter(embedding)
        return embedding.squeeze(0)

    def _split_waveforms(self, audio_inputs, lengths):
        # Convert lengths to CPU ints once so the per-audio slicing below stays
        # on the host and does not trigger a device->host sync per audio.
        if isinstance(audio_inputs, list):
            return audio_inputs
        lengths_list = lengths.tolist()
        waveforms = []
        audio_start = 0
        for length in lengths_list:
            waveforms.append(audio_inputs[audio_start : audio_start + length])
            audio_start += length
        return waveforms

    def forward(self, audio_inputs, lengths):
        waveforms = self._split_waveforms(audio_inputs, lengths)
        all_embeddings = []
        token_lengths = []
        for waveform in waveforms:
            embedding = self._encode_single_audio(waveform)
            all_embeddings.append(embedding)
            token_lengths.append(embedding.shape[0])
        return torch.cat(all_embeddings, dim=0), token_lengths
