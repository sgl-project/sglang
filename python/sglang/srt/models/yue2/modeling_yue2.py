"""YuE2 AR–NAR Mixture-of-Transformers, with checkpoint-compatible names.

This module is self contained for Transformers ``trust_remote_code`` loading.
It imports no CUDA extension and implements the released model architecture.
``generate`` returns token IDs; the package pipeline supplies song generation.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


def sdpa(query, key, value, *, attn_mask=None, is_causal=False):
    """Use native grouped-query attention, including a portable MPS fallback."""
    grouped = query.shape[1] != key.shape[1]
    if grouped and query.device.type == "mps":
        # PyTorch's MPS attention does not implement enable_gqa on every release.
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        grouped = False
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, is_causal=is_causal,
        enable_gqa=grouped,
    )


def _causal_mask(attention_mask, cache_position, key_length, batch_size):
    """Physical cache slots are causal; RoPE positions may exclude padding."""
    device = cache_position.device
    visible = torch.arange(key_length, device=device)[None, :] <= cache_position[:, None]
    visible = visible[None, None].expand(batch_size, 1, -1, -1)
    if attention_mask is None:
        return visible
    mask = attention_mask.to(device=device)
    if mask.ndim == 2:
        if mask.shape[0] != batch_size or mask.shape[1] > key_length:
            raise ValueError("attention_mask must cover the batch and used cache slots")
        # Static cache has unused capacity after the supplied 2D padding mask.
        if mask.shape[1] < key_length:
            mask = F.pad(mask, (0, key_length - mask.shape[1]), value=0)
        return visible & mask[:, None, None, :].bool()
    if mask.ndim != 4 or mask.shape[-2:] != visible.shape[-2:]:
        raise ValueError("Expected a 2D padding mask or a matching 4D attention mask")
    if mask.dtype == torch.bool:
        return visible & mask
    return mask.masked_fill(~visible, float("-inf"))

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════


class YuE2Config(PretrainedConfig):
    model_type = "yue2"

    _hf_fields = frozenset({
        "model_type", "architectures", "auto_map", "transformers_version",
        "dtype", "torch_dtype", "return_dict", "output_hidden_states",
        "output_attentions", "use_cache", "tie_word_embeddings", "torchscript",
        "is_decoder", "is_encoder_decoder", "add_cross_attention",
        "bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id",
        "attn_implementation",
    })

    def to_dict(self):
        return {key: value for key, value in super().to_dict().items()
                if key in self._hf_fields or key in self._inference_fields}

    _inference_fields = frozenset(['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'intermediate_size', 'vocab_size', 'rms_norm_eps', 'rope_theta', 'max_position_embeddings', 'tie_word_embeddings', 'latent_type', 'latent_dim', 'max_latent_frames', 'timestep_shift'])

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 6144,
        vocab_size: int = 184704,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 24576,
        tie_word_embeddings: bool = False,
        # Acoustic inference architecture
        latent_type: str = "vae",
        latent_dim: int = 64,
        max_latent_frames: int = 24576,
        timestep_shift: float = 1.0,
        **kwargs,
    ):
        if latent_type != "vae":
            raise ValueError("YuE2 inference supports only latent_type='vae'")
        # Serialize only the documented model and Transformers configuration.
        kwargs = {key: value for key, value in kwargs.items() if key in self._hf_fields}
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.latent_type = latent_type
        self.latent_dim = latent_dim
        self.max_latent_frames = max_latent_frames
        self.timestep_shift = timestep_shift


# ══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ══════════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._inv_freq: Optional[torch.Tensor] = None

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._inv_freq is None or self._inv_freq.device != position_ids.device:
            self._inv_freq = 1.0 / (self.base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=position_ids.device) / self.head_dim
            ))
        pos = position_ids.float().unsqueeze(-1)
        angles = pos * self._inv_freq
        return angles.cos(), angles.sin()


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, config: YuE2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def project_qkv(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, normalize, and apply RoPE. No SDPA, no KV cache, no O proj.

        Returns Q [B,T,num_heads,hd], K [B,T,num_kv_heads,hd], V [B,T,num_kv_heads,hd].
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        q = _apply_rotary(q, rc, rs)
        k = _apply_rotary(k, rc, rs)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[DynamicCache] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.project_qkv(x, cos, sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, layer_idx, {"cache_position": cache_position})

        if attention_mask is not None:
            out = sdpa(q, k, v, attn_mask=attention_mask[..., :k.shape[2]])
        else:
            out = sdpa(q, k, v, is_causal=(T > 1 and k.shape[2] == T))
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config: YuE2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Transformer layer with full MoT: dual attention projections + dual MLP."""

    def __init__(self, config: YuE2Config):
        super().__init__()
        # AR attention path
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        # NAR attention path (separate Q/K/V/O + layernorms)
        self.nar_input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_self_attn = Attention(config)
        # AR MLP path
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)
        # NAR MLP path
        self.nar_pre_mlp_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[DynamicCache] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        ar_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ar_mask is not None:
            mask_3d = ar_mask.unsqueeze(-1)  # [B, S, 1]
            mask_4d = ar_mask.unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]

            # Per-type input layernorm
            ln_ar = self.input_layernorm(x)
            ln_nar = self.nar_input_layernorm(x)

            # Per-type QKV projection (both process all tokens)
            q_ar, k_ar, v_ar = self.self_attn.project_qkv(ln_ar, cos, sin)
            q_nar, k_nar, v_nar = self.nar_self_attn.project_qkv(ln_nar, cos, sin)

            # Merge Q/K/V per-position: AR positions use AR projections, NAR use NAR
            query = torch.where(mask_4d, q_ar, q_nar)    # [B, S, num_heads, hd]
            # K/V have num_kv_heads (fewer), same mask broadcast works
            key = torch.where(mask_4d, k_ar, k_nar)      # [B, S, num_kv_heads, hd]
            value = torch.where(mask_4d, v_ar, v_nar)

            # Transpose to [B, H, S, D] for SDPA
            B, S = x.shape[:2]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # Shared attention with hybrid mask
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.to(query.dtype)
            core_out = sdpa(query, key, value, attn_mask=attention_mask)
            core_out = core_out.transpose(1, 2).reshape(B, S, -1)

            # Per-type O projection, merge by mask
            o_ar = self.self_attn.o_proj(core_out)
            o_nar = self.nar_self_attn.o_proj(core_out)
            h = torch.where(mask_3d, o_ar, o_nar)
            x = x + h

            # Per-type MLP
            ar_out = self.mlp(self.post_attention_layernorm(x))
            nar_out = self.nar_mlp(self.nar_pre_mlp_layernorm(x))
            mlp_out = torch.where(mask_3d, ar_out, nar_out)
        else:
            # AR-only mode (generation): use AR path only
            h = self.self_attn(self.input_layernorm(x), cos, sin, past_key_value, layer_idx,
                               attention_mask, cache_position)
            x = x + h
            mlp_out = self.mlp(self.post_attention_layernorm(x))

        x = x + mlp_out
        return x


# ══════════════════════════════════════════════════════════════════════════════
# NAR auxiliary modules
# ══════════════════════════════════════════════════════════════════════════════


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → MLP → hidden_size (same as modules.py)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(next(self.parameters()).dtype))


class AudioPositionEmbedding(nn.Module):
    """Non-learnable 1D sinusoidal PE for audio latent frames."""

    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(max_frames, hidden_size)
        position = torch.arange(0, max_frames, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, position_ids):
        return self.pe[position_ids]


# ══════════════════════════════════════════════════════════════════════════════
# Static KV Cache
# ══════════════════════════════════════════════════════════════════════════════


class StaticKVCache:
    """Bounded, append-only cache for the explicit single-request AR loop.

    Returns views of the used prefix and never reallocates/copies its history.
    Standard HF ``generate`` also supports Transformers' own StaticCache.
    """

    def __init__(
        self, num_layers: int, batch_size: int, num_kv_heads: int,
        max_seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = [
            torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.value_cache: List[torch.Tensor] = [
            torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    def get_seq_length(self, layer_idx=0) -> int:
        return self._seen_tokens

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        T = key_states.shape[2]
        pos = self._seen_tokens
        end = pos + T
        if end > self.max_seq_len:
            raise ValueError(f"KV cache capacity {self.max_seq_len} exceeded by {end}; generation was not shortened")
        self.key_cache[layer_idx][:, :, pos:end] = key_states
        self.value_cache[layer_idx][:, :, pos:end] = value_states
        if layer_idx == self.num_layers - 1:
            self._seen_tokens = end
        return self.key_cache[layer_idx][:, :, :end], self.value_cache[layer_idx][:, :, :end]

    def reset(self):
        self._seen_tokens = 0

    def reorder_cache(self, beam_idx):
        self.key_cache = [v.index_select(0, beam_idx.to(v.device)) for v in self.key_cache]
        self.value_cache = [v.index_select(0, beam_idx.to(v.device)) for v in self.value_cache]


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════


class Backbone(nn.Module):
    """Transformer backbone with MoT dual MLP."""

    def __init__(self, config: YuE2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        ar_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, past_key_values if use_cache else None,
                      layer_idx=i, attention_mask=attention_mask, ar_mask=ar_mask,
                      cache_position=cache_position)

        return self.norm(x), past_key_values


class YuE2PreTrainedModel(PreTrainedModel):
    config_class = YuE2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.01)


class YuE2ForCausalLM(YuE2PreTrainedModel, GenerationMixin):
    """YuE2 model: AR causal LM (generate) + NAR flow matching (ODE)."""

    def __init__(self, config: YuE2Config):
        super().__init__(config)
        self.model = Backbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # NAR auxiliary
        self.llm2vae = nn.Linear(config.hidden_size, config.latent_dim)
        self.vae2llm = nn.Linear(config.latent_dim, config.hidden_size)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.latent_pos_embed = AudioPositionEmbedding(config.max_latent_frames, config.hidden_size)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # ── AR forward (standard causal LM, KV cached) ───────────────────

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids or inputs_embeds")
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        tensor = input_ids if input_ids is not None else inputs_embeds
        batch_size, seq_len = tensor.shape[:2]
        if not seq_len:
            raise ValueError("Input must contain at least one token")
        device = tensor.device
        past_len = past_key_values.get_seq_length() if past_key_values is not None and use_cache else 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + seq_len, device=device)
        else:
            cache_position = cache_position.to(device=device, dtype=torch.long)
        if cache_position.ndim != 1 or cache_position.numel() != seq_len:
            raise ValueError("cache_position must identify each current token's physical cache slot")
        if position_ids is None:
            if attention_mask is not None and attention_mask.ndim == 2:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                position_ids = position_ids[:, -seq_len:].to(device)
            else:
                position_ids = cache_position[None]
        else:
            position_ids = position_ids.to(device=device, dtype=torch.long)
        if position_ids.shape[-1] != seq_len:
            raise ValueError("position_ids must cover the current input tokens")

        key_length = past_len + seq_len
        if use_cache and past_key_values is not None and hasattr(past_key_values, "get_max_cache_shape"):
            capacity = past_key_values.get_max_cache_shape()
            if capacity is not None and capacity > 0:
                key_length = capacity
        # No explicit mask is needed for unpadded prefill or single-token dynamic
        # decode. Chunked prefill needs bottom-right causal alignment; a full
        # static cache additionally needs to hide all unfilled slots.
        needs_mask = attention_mask is not None or key_length != past_len + seq_len or (past_len > 0 and seq_len > 1)
        causal_mask = _causal_mask(attention_mask, cache_position, key_length, batch_size) if needs_mask else None
        hidden_states, past_key_values = self.model(
            input_ids=input_ids, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            attention_mask=causal_mask, inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError("logits_to_keep must be nonnegative")
            selected = hidden_states[:, -logits_to_keep:, :] if logits_to_keep else hidden_states
        else:
            selected = hidden_states[:, logits_to_keep.to(device), :]
        if labels is not None and selected.shape[1] != hidden_states.shape[1]:
            raise ValueError("Loss computation requires logits_to_keep=0")
        logits = self.lm_head(selected)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits, past_key_values) if use_cache else (logits,)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values if use_cache else None)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, cache_position=None, position_ids=None, **kwargs,
    ):
        """Keep physical cache slots separate from padding-aware RoPE positions."""
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            total = inputs_embeds.shape[1] if inputs_embeds is not None and past_len == 0 else input_ids.shape[1]
            count = max(total - past_len, 1) if past_len else total
            cache_position = torch.arange(past_len, past_len + count, device=input_ids.device)
        count = cache_position.numel()
        use_embeds = inputs_embeds is not None and past_len == 0
        if use_embeds:
            current_ids, current_embeds = None, inputs_embeds[:, -count:]
        else:
            current_ids, current_embeds = input_ids[:, -count:].contiguous(), None
        if position_ids is None and attention_mask is not None and attention_mask.ndim == 2:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        if position_ids is not None:
            position_ids = position_ids[:, -count:].contiguous()
        return {
            "input_ids": current_ids, "inputs_embeds": current_embeds,
            "past_key_values": past_key_values, "attention_mask": attention_mask,
            "position_ids": position_ids, "cache_position": cache_position,
            "use_cache": kwargs.get("use_cache", True),
            "logits_to_keep": kwargs.get("logits_to_keep", 1),
        }

    # ── NAR velocity (flow matching, no KV cache) ────────────────────

    def _shift_t_value(self, t_value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_sig = torch.sigmoid(torch.tensor(t_value, dtype=dtype, device=device))
        shift = self.config.timestep_shift
        return shift * t_sig / (1 + (shift - 1) * t_sig)

    @torch.no_grad()
    def nar_velocity(
        self,
        tokens: torch.LongTensor,
        ar_mask: torch.BoolTensor,
        nar_mask: torch.BoolTensor,
        nar_content_mask: torch.BoolTensor,
        x_t: torch.Tensor,
        t_value: float,
        nar_cond_end: int = 0,
    ) -> torch.Tensor:
        """Compute v_theta(x_t, t) — flow-matching velocity field.

        Args:
            tokens: [1, S] full sequence (AR + NAR tokens)
            ar_mask: [1, S] True for AR positions
            nar_mask: [1, S] True for NAR positions
            nar_content_mask: [1, S] True for actual latent positions (not LATENT_START/END)
            x_t: [T_lat, D] current ODE state
            t_value: raw timestep (will be sigmoid-shifted)
            nar_cond_end: if > 0, NAR only sees positions < nar_cond_end (text-only mode)
        Returns:
            v_pred: [T_lat, D] predicted velocity
        """
        device = tokens.device
        dtype = next(self.parameters()).dtype
        B, S = tokens.shape

        # 1. Token embeddings
        token_emb = self.model.embed_tokens(tokens)  # [B, S, H]

        # 2. Build latent hidden for ALL NAR positions (START + content + END)
        # Training injects vae2llm(x_t) + time_emb + pos_emb at ALL NAR positions,
        # including LATENT_START (clean=0) and LATENT_END (clean=0).
        # NAR position IDs via cumsum: START=0, content=[1..T_lat], END=T_lat+1.
        t_shifted = self._shift_t_value(t_value, device, dtype)
        T_lat = x_t.shape[0]

        nar_indices = nar_mask[0].nonzero(as_tuple=True)[0]  # all NAR positions
        content_indices = nar_content_mask[0].nonzero(as_tuple=True)[0]
        N_nar = nar_indices.shape[0]  # START + T_lat + END

        # Build x_t for all NAR positions: zeros for START/END, actual x_t for content
        x_nar = torch.zeros(N_nar, x_t.shape[1], device=device, dtype=dtype)
        x_nar[1:1 + T_lat] = x_t.to(dtype)  # content frames at positions [1, T_lat]

        latent_hidden_nar = self.vae2llm(x_nar.unsqueeze(0))  # [1, N_nar, H]

        # Timestep embedding (same t for all NAR positions)
        time_emb = self.time_embedder(t_shifted.expand(N_nar)).unsqueeze(0)
        latent_hidden_nar = latent_hidden_nar + time_emb

        # Position embedding: cumsum-style [0, 1, 2, ..., N_nar-1]
        pos_ids = torch.arange(N_nar, device=device).clamp(max=self.config.max_latent_frames - 1)
        pos_emb = self.latent_pos_embed(pos_ids).unsqueeze(0)
        latent_hidden_nar = latent_hidden_nar + pos_emb

        # Inject at ALL NAR positions (matching training's torch.where)
        token_emb[0, nar_indices] = latent_hidden_nar[0]

        # 3. Build hybrid attention mask [B, 1, S, S]
        # AR→AR: causal, NAR→AR: full, NAR→NAR: bidirectional, AR→NAR: blocked
        ar_q = ar_mask.unsqueeze(2).float()   # [B, S, 1]
        ar_k = ar_mask.unsqueeze(1).float()   # [B, 1, S]
        nar_q = nar_mask.unsqueeze(2).float()
        nar_k = nar_mask.unsqueeze(1).float()
        causal = torch.tril(torch.ones(S, S, device=device))

        if nar_cond_end > 0:
            # Codec dropout: NAR only sees positions < nar_cond_end (text) + NAR
            text_k = torch.zeros(1, 1, S, device=device)
            text_k[0, 0, :nar_cond_end] = 1.0
            mask = (ar_q * ar_k * causal) + (nar_q * text_k) + (nar_q * nar_k)
        else:
            mask = (ar_q * ar_k * causal) + (nar_q * ar_k) + (nar_q * nar_k)
        # Convert to additive: 0 → attend, -inf → block
        attn_mask = mask.unsqueeze(1)  # [B, 1, S, S]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf")).masked_fill(attn_mask > 0, 0.0)

        # 4. Position IDs + RoPE
        position_ids = torch.arange(S, device=device).unsqueeze(0)

        # 5. Forward through decoder (with MoT routing)
        ar_mask_bt = ar_mask  # [B, S] bool for MoT routing
        hidden_states, _ = self.model(
            inputs_embeds=token_emb, position_ids=position_ids,
            use_cache=False, attention_mask=attn_mask, ar_mask=ar_mask_bt,
        )

        # 6. NAR head at content positions
        nar_pred = self.llm2vae(hidden_states)  # [B, S, D]
        v_pred = nar_pred[0, content_indices]  # [T_lat, D]
        return v_pred

# Keep custom code + auto_map when a local user calls save_pretrained as well
# as when the release builder creates a Hub repository.
YuE2Config.register_for_auto_class()
YuE2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
