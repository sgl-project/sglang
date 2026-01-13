# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""
Minimal Qwen3 DFlash draft model.

Key pattern: Q from noise, K/V from [context + noise], non-causal attention.
"""

from typing import Iterable, Tuple

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import RMSNorm3D, build_target_layer_ids


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class DFlashAttention(nn.Module):
    """DFlash attention: Q from noise, K/V from [ctx + noise], non-causal."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        noise: torch.Tensor,
        ctx: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        """
        Args:
            noise: [bsz, q_len, hidden]
            ctx: [bsz, ctx_len, hidden] - already normalized
            cos, sin: [bsz, ctx_len + q_len, head_dim]
        """
        bsz, q_len, _ = noise.shape
        ctx_len = ctx.shape[1]

        # Q from noise
        q = self.q_proj(noise).view(bsz, q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        q = q.transpose(1, 2)

        # K/V from [ctx, noise]
        kv_input = torch.cat([ctx, noise], dim=1)
        k = self.k_proj(kv_input).view(
            bsz, ctx_len + q_len, self.num_kv_heads, self.head_dim
        )
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(
            bsz, ctx_len + q_len, self.num_kv_heads, self.head_dim
        )
        k = k.transpose(1, 2)
        v = (
            self.v_proj(kv_input)
            .view(bsz, ctx_len + q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Rotary: Q uses last q_len positions, K uses all
        cos_q, sin_q = cos[:, -q_len:].unsqueeze(1), sin[:, -q_len:].unsqueeze(1)
        cos_k, sin_k = cos.unsqueeze(1), sin.unsqueeze(1)
        q = (q * cos_q) + (rotate_half(q) * sin_q)
        k = (k * cos_k) + (rotate_half(k) * sin_k)

        # GQA expansion
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Non-causal attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out)


def norm_3d(layer, x):
    """Apply RMSNorm to 3D tensor by flattening/unflattening."""
    shape = x.shape
    return layer(x.view(-1, shape[-1])).view(shape)


class DFlashLayer(nn.Module):
    """DFlash decoder layer."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.attn = DFlashAttention(config, layer_idx)
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.act = nn.SiLU()

    def forward(
        self,
        noise: torch.Tensor,
        ctx: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        # Attention with residual
        h = norm_3d(self.input_norm, noise)
        noise = noise + self.attn(h, ctx, cos, sin)
        # MLP with residual
        h = norm_3d(self.post_norm, noise)
        noise = noise + self.down_proj(self.act(self.gate_proj(h)) * self.up_proj(h))
        return noise


class Qwen3ForCausalLMDFlash(nn.Module):
    """Minimal DFlash draft model."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # Target layer IDs for hidden state extraction
        self.target_layer_ids = build_target_layer_ids(
            getattr(config, "num_target_layers", 36),
            config.num_hidden_layers,
        )

        # Feature compression
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm3D(config.hidden_size, eps=config.rms_norm_eps)

        # Layers
        self.layers = nn.ModuleList(
            [DFlashLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embedding
        self.rope_theta = getattr(config, "rope_theta", 1000000)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Embed/head from target
        self._embed = None
        self._head = None
        self.block_size = getattr(config, "block_size", 16)

    def get_rotary(self, positions: torch.Tensor, device, dtype):
        freqs = torch.einsum("bi,j->bij", positions.float(), self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor):
        self._embed = embed
        self._head = head

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(ids, self._embed)

    def get_input_embeddings(self):
        return self._embed

    def forward(
        self,
        noise_emb: torch.Tensor,  # [bsz, block_size, hidden]
        target_hidden: torch.Tensor,  # [bsz, ctx_len, num_layers * hidden]
        position_ids: torch.Tensor,  # [bsz, ctx_len + block_size]
    ) -> torch.Tensor:
        # Project and normalize target hidden
        ctx = self.fc(target_hidden)
        ctx = self.hidden_norm(ctx)

        # Rotary embeddings
        cos, sin = self.get_rotary(position_ids, noise_emb.device, noise_emb.dtype)

        # Forward through layers
        h = noise_emb
        for layer in self.layers:
            h = layer(h, ctx, cos, sin)

        # Output logits
        h = norm_3d(self.norm, h)
        return torch.matmul(h, self._head.t())

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights with name mapping from checkpoint to our model."""
        params = dict(self.named_parameters())
        loaded = set()

        for name, w in weights:
            # Map checkpoint names to our model names
            mapped = name
            mapped = mapped.replace("self_attn.", "attn.")
            mapped = mapped.replace("mlp.gate_proj", "gate_proj")
            mapped = mapped.replace("mlp.up_proj", "up_proj")
            mapped = mapped.replace("mlp.down_proj", "down_proj")
            mapped = mapped.replace("input_layernorm", "input_norm")
            mapped = mapped.replace("post_attention_layernorm", "post_norm")

            if mapped in params:
                default_weight_loader(params[mapped], w)
                loaded.add(mapped)

        # Report missing
        missing = set(params.keys()) - loaded
        if missing:
            print(f"DFlash: {len(missing)} params not loaded: {list(missing)[:5]}")


# HuggingFace compatibility - must match architectures in config.json
class DFlashDraftModel(Qwen3ForCausalLMDFlash):
    """DFlashDraftModel - matches HF config architectures field."""

    pass


EntryClass = DFlashDraftModel
