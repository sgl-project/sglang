# SPDX-License-Identifier: Apache-2.0
"""BRIA Fibo 2.0 diffusion transformer.

A single-stream DiT in the Z-Image family, extended in three ways:

* a Perceiver resampler turns the first text bundle into ``num_gist_tokens``
  gist tokens, a count derived per prompt from the text length,
* the blocks listed in ``injection_layer_ids`` also cross-attend to their own
  text bundle,
* adaLN modulation is per token: tokens marked noisy by ``noise_mask`` take the
  modulation of the real timestep, the others (gist and context tokens, which
  are clean) take the modulation of t = 0. That is how editing conditions a
  generation on the clean latents of the images to edit.

The whole image plus every context image share one sequence and one patch
embedding. Their positions come from a three-axis RoPE, so the token order only
affects float rounding, not the result.

Parameter names are the checkpoint's diffusers names, so weights load without a
remap. q/k/v and w1/w3 are deliberately kept as separate projections: a
cross-attention reads a ``cap_feat_dim`` wide text bundle with a ``dim`` wide
query, so those cannot share a stacked projection.

This file targets the single-GPU path. Tensor/sequence parallelism, an
attention-backend switch (the reference uses ``USPAttention`` elsewhere in the
tree) and the batched variable-gist path are follow-ups; the batched case is
rejected explicitly rather than half-supported.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.bria_fibo2 import BriaFibo2DitConfig
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.zimage import RopeEmbedder
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

ADALN_EMBED_DIM = 256


def compute_gist_count(
    text_lengths: Sequence[int],
    min_gist: int,
    max_gist: int,
    gist_step: int,
    min_text_len: int,
    max_text_len: int,
) -> list[int]:
    """Number of gist tokens per prompt, from its text length in tokens.

    Lengths between ``min_text_len`` and ``max_text_len`` map linearly onto
    ``[min_gist, max_gist]``, rounded up to a multiple of ``gist_step``.
    """
    span = max_text_len - min_text_len
    counts = []
    for n in text_lengths:
        ratio = min(1.0, max(0.0, (float(n) - min_text_len) / span))
        raw = min_gist + ratio * (max_gist - min_gist)
        quantized = int(math.ceil(raw / gist_step)) * gist_step
        counts.append(max(min_gist, min(max_gist, quantized)))
    return counts


class BriaFibo2RMSNorm(nn.Module):
    """RMSNorm matching the reference implementation.

    The variance is taken in float32. ``zimage.ZImageRMSNorm`` deliberately keeps
    Z-Image's native bf16 behaviour instead, so the two are not interchangeable.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


def _mask_to_bool(mask: torch.Tensor) -> torch.Tensor:
    """SDPA wants a bool (or float) mask; the stages carry it as long/bool."""
    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point():
        return mask
    return mask != 0


def normalize_encoder_attention_mask(
    mask: Any, device: torch.device
) -> Optional[torch.Tensor]:
    """Normalize the text mask to one ``(batch, text_len)`` tensor.

    The diffusion stages thread prompt masks **per sample** (`Req.prompt_attention_mask`
    is a list), which is convenient for the encoders but not for the transformer;
    accept either form so the caller does not have to convert.
    """
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = mask if mask.device == device else mask.to(device)
        return _mask_to_bool(mask)
    if isinstance(mask, (list, tuple)):
        items = [m for m in mask if m is not None]
        if not items or len(items) != len(mask):
            return None
        tensors = [
            _mask_to_bool(
                (m if isinstance(m, torch.Tensor) else torch.as_tensor(m)).to(device)
            )
            for m in items
        ]
        if len(tensors) == 1:
            single = tensors[0]
            return single if single.dim() == 2 else single.reshape(1, -1)
        if all(t.dim() == 2 and t.shape[0] == 1 for t in tensors):
            return torch.cat(tensors, dim=0)
        return torch.stack(tensors, dim=0)
    return None


def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Broadcast the noisy or the clean value onto each token of the sequence."""
    noise_mask_expanded = noise_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate adjacent channel pairs, the complex multiply of the reference.

    ``x`` is ``[batch, seq_len, heads, head_dim]``; ``cos`` and ``sin`` are
    ``[1, seq_len, head_dim // 2]``. The rotation is computed in float32 and cast
    back, as the reference does.
    """
    dtype = x.dtype
    x_float = x.float()
    x_even = x_float[..., 0::2]
    x_odd = x_float[..., 1::2]
    cos = cos.unsqueeze(2).float()
    sin = sin.unsqueeze(2).float()
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(dtype)


class BriaFibo2TimestepEmbedder(nn.Module):
    def __init__(
        self,
        out_size: int,
        mid_size: Optional[int] = None,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class BriaFibo2FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BriaFibo2Attention(nn.Module):
    """Self-attention (with RoPE) or cross-attention to ``encoder_hidden_states``."""

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        kv_dim: Optional[int] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        kv_dim = kv_dim or query_dim
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(kv_dim, heads * dim_head, bias=False)
        # The reference keeps the output projection in a ModuleList, which the
        # checkpoint reflects as `to_out.0`
        self.to_out = nn.ModuleList(
            [nn.Linear(heads * dim_head, query_dim, bias=False)]
        )
        self.norm_q = BriaFibo2RMSNorm(dim_head, eps=eps)
        self.norm_k = BriaFibo2RMSNorm(dim_head, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        query = self.to_q(hidden_states)
        kv_source = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        key = self.to_k(kv_source)
        value = self.to_v(kv_source)

        query = query.view(batch_size, seq_len, self.heads, self.head_dim)
        key = key.view(batch_size, -1, self.heads, self.head_dim)
        value = value.view(batch_size, -1, self.heads, self.head_dim)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            query = apply_rotary_emb(query, cos, sin)
            key = apply_rotary_emb(key, cos, sin)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # A 2D mask marks the real keys, so it broadcasts over query and heads
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out[0](out)


class BriaFibo2TransformerBlock(nn.Module):
    """A Z-Image block, plus optional text cross-attention and per-token modulation."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        norm_eps: float,
        cross_attention_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.attention = BriaFibo2Attention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.feed_forward = BriaFibo2FeedForward(dim, (dim * 8 // 3) // 64 * 64)

        self.attention_norm1 = BriaFibo2RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = BriaFibo2RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = BriaFibo2RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = BriaFibo2RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
        )

        if cross_attention_dim is not None:
            self.cross_attn_norm1 = BriaFibo2RMSNorm(dim, eps=norm_eps)
            self.cross_attention = BriaFibo2Attention(
                dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps
            )
            self.cross_attn_norm2 = BriaFibo2RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        else:
            self.cross_attention = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        adaln_input: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]

        if noise_mask is not None:
            # Per-token modulation: noisy and clean tokens get different values
            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)
            (
                scale_msa_noisy,
                gate_msa_noisy,
                scale_mlp_noisy,
                gate_mlp_noisy,
            ) = mod_noisy.chunk(4, dim=1)
            (
                scale_msa_clean,
                gate_msa_clean,
                scale_mlp_clean,
                gate_mlp_clean,
            ) = mod_clean.chunk(4, dim=1)

            gate_msa_noisy, gate_mlp_noisy = (
                gate_msa_noisy.tanh(),
                gate_mlp_noisy.tanh(),
            )
            gate_msa_clean, gate_mlp_clean = (
                gate_msa_clean.tanh(),
                gate_mlp_clean.tanh(),
            )
            scale_msa_noisy, scale_mlp_noisy = (
                1.0 + scale_msa_noisy,
                1.0 + scale_mlp_noisy,
            )
            scale_msa_clean, scale_mlp_clean = (
                1.0 + scale_msa_clean,
                1.0 + scale_mlp_clean,
            )

            scale_msa = select_per_token(
                scale_msa_noisy, scale_msa_clean, noise_mask, seq_len
            )
            scale_mlp = select_per_token(
                scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len
            )
            gate_msa = select_per_token(
                gate_msa_noisy, gate_msa_clean, noise_mask, seq_len
            )
            gate_mlp = select_per_token(
                gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len
            )
        else:
            # Global modulation: one value broadcast over the whole sequence
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            attention_mask=attn_mask,
            freqs_cis=freqs_cis,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)

        if self.cross_attention is not None:
            cross_out = self.cross_attention(
                self.cross_attn_norm1(x),
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(cross_out)

        x = x + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        )
        return x


class BriaFibo2PerceiverLayer(nn.Module):
    """Gist tokens read the text bundle, then each other, then an FFN."""

    def __init__(
        self, dim: int, n_heads: int, cross_attention_dim: int, norm_eps: float
    ) -> None:
        super().__init__()
        self.cross_attn_norm = BriaFibo2RMSNorm(dim, eps=norm_eps)
        self.cross_attention = BriaFibo2Attention(
            dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps
        )
        self.self_attn_norm = BriaFibo2RMSNorm(dim, eps=norm_eps)
        self.self_attention = BriaFibo2Attention(
            dim, n_heads, dim // n_heads, eps=norm_eps
        )
        self.ffn_norm = BriaFibo2RMSNorm(dim, eps=norm_eps)
        self.ffn = BriaFibo2FeedForward(dim, (dim * 8 // 3) // 64 * 64)

    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latents = latents + self.cross_attention(
            self.cross_attn_norm(latents),
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        latents = latents + self.self_attention(
            self.self_attn_norm(latents), attention_mask=attention_mask
        )
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class BriaFibo2PerceiverResampler(nn.Module):
    """Squeezes the text bundle into ``num_gist_tokens`` gist tokens."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        cross_attention_dim: int,
        num_queries: int,
        num_layers: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                BriaFibo2PerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = BriaFibo2RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        num_gist_tokens: int,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = encoder_hidden_states.shape[0]
        latents = (
            self.latent_queries[:num_gist_tokens]
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        for layer in self.layers:
            latents = layer(
                latents, encoder_hidden_states, encoder_attention_mask, attention_mask
            )
        return self.norm_out(latents)


class BriaFibo2FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaLN_modulation(c)
        return self.linear(self.norm_final(x) * scale.unsqueeze(1))


class BriaFibo2Transformer2DModel(CachableDiT):
    param_names_mapping = BriaFibo2DitConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = (
        BriaFibo2DitConfig().arch_config.reverse_param_names_mapping
    )
    # Single-GPU implementation: nothing is sharded, so FSDP has no layer groups
    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    packed_modules_mapping: dict = {}

    def __init__(
        self,
        config: BriaFibo2DitConfig,
        hf_config: dict[str, Any],
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        arch_config = self.config
        self.dim = arch_config.dim
        self.in_channels = arch_config.in_channels
        self.out_channels = arch_config.out_channels
        self.patch_size = arch_config.patch_size
        self.n_heads = arch_config.num_attention_heads
        self.hidden_size = arch_config.dim
        self.num_attention_heads = arch_config.num_attention_heads
        self.num_channels_latents = arch_config.in_channels
        self.gradient_checkpointing = False

        self.x_embedder = nn.Linear(
            self.patch_size * self.patch_size * self.in_channels, self.dim, bias=True
        )
        self.t_embedder = BriaFibo2TimestepEmbedder(
            min(self.dim, ADALN_EMBED_DIM), mid_size=1024
        )
        self.rope_embedder = RopeEmbedder(
            theta=arch_config.rope_theta,
            axes_dims=list(arch_config.axes_dims),
            axes_lens=list(arch_config.axes_lens),
        )

        self.noise_refiner = nn.ModuleList(
            [
                BriaFibo2TransformerBlock(self.dim, self.n_heads, arch_config.norm_eps)
                for _ in range(arch_config.n_refiner_layers)
            ]
        )
        self.perceiver = BriaFibo2PerceiverResampler(
            self.dim,
            self.n_heads,
            arch_config.cap_feat_dim,
            arch_config.max_num_gist_tokens,
            arch_config.perceiver_num_layers,
            arch_config.norm_eps,
        )
        injection_layer_ids = set(arch_config.injection_layer_ids)
        self.layers = nn.ModuleList(
            [
                BriaFibo2TransformerBlock(
                    self.dim,
                    self.n_heads,
                    arch_config.norm_eps,
                    cross_attention_dim=(
                        arch_config.cap_feat_dim
                        if layer_id in injection_layer_ids
                        else None
                    ),
                )
                for layer_id in range(arch_config.num_layers)
            ]
        )
        self.final_layer = BriaFibo2FinalLayer(
            self.dim, self.patch_size * self.patch_size * self.out_channels
        )

    @staticmethod
    def create_coordinate_grid(
        size: Sequence[int],
        start: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if start is None:
            start = (0 for _ in size)
        axes = [
            torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
            for x0, span in zip(start, size)
        ]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """2x2 latent patches become tokens, features ordered (channel, dy, dx)."""
        batch_size, channels, height, width = latents.shape
        patch_size = self.patch_size
        latents = latents.view(
            batch_size,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        return latents.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size, (height // patch_size) * (width // patch_size), -1
        )

    def _unpatchify(
        self, tokens: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        patch_size = self.patch_size
        height_tokens, width_tokens = height // patch_size, width // patch_size
        tokens = tokens.view(
            batch_size,
            height_tokens,
            width_tokens,
            self.out_channels,
            patch_size,
            patch_size,
        )
        return tokens.permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size, self.out_channels, height, width
        )

    @staticmethod
    def _unsqueeze_freqs(
        freqs: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = freqs
        return cos.unsqueeze(0), sin.unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        guidance: Any = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        source_latents: Optional[Sequence[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: clean-image latents being denoised, ``(B, in_channels, H, W)``.
            encoder_hidden_states: ``(B, 1 + len(injection_layer_ids), text_len, cap_feat_dim)``.
                Bundle 0 feeds the Perceiver, bundle k + 1 feeds injection block k.
            timestep: noise level in ``[0, 1]``, 1 is pure noise.
            encoder_attention_mask: bool mask ``(B, text_len)`` of the real text tokens.
            source_latents: normalized latents of the images to edit, one
                ``(B, in_channels, h, w)`` tensor per image, in order.
        """
        arch_config = self.config
        encoder_attention_mask = normalize_encoder_attention_mask(
            encoder_attention_mask, hidden_states.device
        )
        batch_size, _, height, width = hidden_states.shape
        patch_size = self.patch_size
        height_tokens, width_tokens = height // patch_size, width // patch_size
        num_image_tokens = height_tokens * width_tokens
        device = hidden_states.device

        # 1. Timestep embeddings. Image tokens take the real timestep, the clean
        # tokens (gist and context) take t = 0. `timestep` arrives in the
        # scheduler's units, so rescale it into the [0, t_scale] domain the
        # checkpoint's embedder was trained on.
        timestep = timestep.to(hidden_states.dtype) * (
            arch_config.t_scale / arch_config.scheduler_timesteps
        )
        adaln_noisy = self.t_embedder(timestep)
        adaln_clean = self.t_embedder(torch.zeros_like(timestep))

        # 2. Patchify the image and the context images, which share the patch
        # embedding. The image sits on RoPE plane `max_num_gist_tokens + 1`
        # whatever the gist count is; context image k sits on the k-th plane
        # after it with its own rows and columns.
        image_plane = arch_config.max_num_gist_tokens + 1
        tokens = [self._patchify(hidden_states)]
        ids = [
            self.create_coordinate_grid(
                (1, height_tokens, width_tokens), (image_plane, 0, 0), device
            )
        ]
        for k, latents in enumerate(source_latents or []):
            tokens.append(self._patchify(latents))
            context_size = (
                1,
                latents.shape[2] // patch_size,
                latents.shape[3] // patch_size,
            )
            ids.append(
                self.create_coordinate_grid(
                    context_size, (image_plane + k + 1, 0, 0), device
                )
            )
        hidden_states = self.x_embedder(torch.cat(tokens, dim=1))
        image_freqs = self._unsqueeze_freqs(
            self.rope_embedder(torch.cat([grid.flatten(0, 2) for grid in ids]))
        )

        # 3. Noise refiner. With context images, only the image tokens are noisy.
        refiner_noise_mask = None
        if source_latents:
            refiner_noise_mask = torch.zeros(
                batch_size, hidden_states.shape[1], dtype=torch.long, device=device
            )
            refiner_noise_mask[:, :num_image_tokens] = 1
        for block in self.noise_refiner:
            hidden_states = block(
                hidden_states,
                None,
                image_freqs,
                adaln_input=adaln_noisy,
                noise_mask=refiner_noise_mask,
                adaln_noisy=adaln_noisy,
                adaln_clean=adaln_clean,
            )

        # 4. Gist count per prompt from its text length, then the Perceiver turns
        # the first bundle into gist tokens.
        if encoder_attention_mask is not None:
            text_lengths = encoder_attention_mask.sum(dim=1).tolist()
        else:
            text_lengths = [encoder_hidden_states.shape[2]] * batch_size
        gist_counts = compute_gist_count(
            text_lengths,
            min_gist=arch_config.min_num_gist_tokens,
            max_gist=arch_config.max_num_gist_tokens,
            gist_step=arch_config.gist_step,
            min_text_len=arch_config.gist_min_text_len,
            max_text_len=arch_config.gist_max_text_len,
        )
        if min(gist_counts) != max(gist_counts):
            # The per-prompt gist padding path reorders the sequence and needs a
            # varlen-aware attention backend. Reject it instead of silently
            # producing wrong tokens.
            raise ValueError(
                "BriaFibo2 does not support a batch whose prompts need different "
                f"numbers of gist tokens yet (got {sorted(set(gist_counts))}). "
                "Send these prompts as separate requests."
            )
        num_gist_tokens = gist_counts[0]
        gist = self.perceiver(
            encoder_hidden_states[:, 0], num_gist_tokens, encoder_attention_mask
        )
        gist_ids = self.create_coordinate_grid(
            (num_gist_tokens, 1, 1), (1, 0, 0), device
        ).flatten(0, 2)

        # 5. One stream: gist tokens first, then the image, then the context,
        # the order the model was trained with. Positions come from RoPE, so the
        # order only affects float rounding.
        gist_freqs = self._unsqueeze_freqs(self.rope_embedder(gist_ids))
        image_start = num_gist_tokens
        hidden_states = torch.cat([gist, hidden_states], dim=1)
        freqs_cis = (
            torch.cat([gist_freqs[0], image_freqs[0]], dim=1),
            torch.cat([gist_freqs[1], image_freqs[1]], dim=1),
        )
        noise_mask = torch.zeros(
            batch_size, hidden_states.shape[1], dtype=torch.long, device=device
        )
        noise_mask[:, image_start : image_start + num_image_tokens] = 1

        # 6. The main blocks. Injection blocks read their own bundle.
        bundle_index = {
            layer_id: index + 1
            for index, layer_id in enumerate(arch_config.injection_layer_ids)
        }
        for layer_id, block in enumerate(self.layers):
            text = (
                encoder_hidden_states[:, bundle_index[layer_id]]
                if layer_id in bundle_index
                else None
            )
            hidden_states = block(
                hidden_states,
                None,
                freqs_cis,
                noise_mask=noise_mask,
                adaln_noisy=adaln_noisy,
                adaln_clean=adaln_clean,
                encoder_hidden_states=text,
                encoder_attention_mask=encoder_attention_mask,
            )

        # 7. Final layer on the image tokens only, then unpatchify.
        hidden_states = self.final_layer(
            hidden_states[:, image_start : image_start + num_image_tokens], adaln_noisy
        )
        return self._unpatchify(hidden_states, height, width)


EntryClass = BriaFibo2Transformer2DModel

__all__ = [
    "BriaFibo2Attention",
    "BriaFibo2PerceiverResampler",
    "BriaFibo2Transformer2DModel",
    "BriaFibo2TransformerBlock",
    "compute_gist_count",
    "select_per_token",
]
