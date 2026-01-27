# Copied and adapted from: https://github.com/Tencent-Hunyuan/Hunyuan3D-2


# SPDX-License-Identifier: Apache-2.0
# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the respective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
Hunyuan3D DiT (Diffusion Transformer) model for 3D shape generation.

This module implements the core DiT architecture used in Hunyuan3D for
generating 3D latent representations from image conditions.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


class Timesteps(nn.Module):
    """Generate sinusoidal timestep embeddings."""

    def __init__(
        self,
        num_channels: int,
        downscale_freq_shift: float = 0.0,
        scale: int = 1,
        max_period: int = 10000,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        cond_proj_dim=None,
        out_size=None,
    ):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(
                cond_proj_dim, frequency_embedding_size, bias=False
            )

        self.time_embed = Timesteps(hidden_size)

    def forward(self, t, condition):

        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)

        # t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)

        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

        self.with_dca = with_decoupled_ca
        if self.with_dca:
            self.kv_proj_dca = nn.Linear(kdim, 2 * qdim, bias=qkv_bias)
            self.k_norm_dca = (
                norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape  # [b, s1, D]

        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim :, :]
            kv_dca = self.kv_proj_dca(context_dca).view(
                b, self.dca_dim, 2, self.num_heads, self.head_dim
            )
            k_dca, v_dca = kv_dca.unbind(dim=2)  # [b, s, h, d]
            k_dca = self.k_norm_dca(k_dca)
            y = y[:, : (token_len - self.dca_dim), :]

        _, s2, c = y.shape  # [b, s2, 1024]
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)  # [b, s1, h, d]
        k = k.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]
        v = v.view(b, s2, self.num_heads, self.head_dim)  # [b, s2, h, d]

        q = self.q_norm(q)
        k = self.k_norm(k)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
            q, k, v = map(
                lambda t: rearrange(t, "b n h d -> b h n d", h=self.num_heads),
                (q, k, v),
            )
            context = (
                F.scaled_dot_product_attention(q, k, v)
                .transpose(1, 2)
                .reshape(b, s1, -1)
            )

        if self.with_dca:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                k_dca, v_dca = map(
                    lambda t: rearrange(t, "b n h d -> b h n d", h=self.num_heads),
                    (k_dca, v_dca),
                )
                context_dca = (
                    F.scaled_dot_product_attention(q, k_dca, v_dca)
                    .transpose(1, 2)
                    .reshape(b, s1, -1)
                )

            context = context + self.dca_weight * context_dca

        out = self.out_proj(context)  # context.reshape - B, L1, -1

        return out


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        # This assertion is aligned with flash attention
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // self.num_heads // 3
        qkv = qkv.view(1, -1, self.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [b, h, s, d]
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [b, h, s, d]
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)  # [b, h, s, d]
        k = self.k_norm(k)  # [b, h, s, d]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
            x = F.scaled_dot_product_attention(q, k, v)
            x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.out_proj(x)
        return x


class GEGLU(nn.Module):
    """GELU activation wrapper for MoE experts."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Project to out_features, which will be split in half for GEGLU
        self.proj = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.gelu(self.proj(x))


class MoEExpert(nn.Module):
    """Single expert in MoE block."""

    def __init__(self, hidden_size, ff_inner_dim, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            GEGLU(hidden_size, ff_inner_dim, bias=bias),
            nn.Identity(),  # Placeholder for index alignment (net.1)
            nn.Linear(ff_inner_dim, hidden_size, bias=bias),
        )

    def forward(self, x):
        return self.net[2](self.net[0](x))


class MoEBlock(nn.Module):
    """Mixture of Experts block with shared experts."""

    def __init__(
        self,
        hidden_size,
        num_experts=8,
        moe_top_k=2,
        dropout=0.0,
        activation_fn="gelu",
        final_dropout=False,
        ff_inner_dim=None,
        ff_bias=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k

        if ff_inner_dim is None:
            ff_inner_dim = int(hidden_size * 4.0)

        # Expert modules
        self.experts = nn.ModuleList(
            [
                MoEExpert(hidden_size, ff_inner_dim, bias=ff_bias)
                for _ in range(num_experts)
            ]
        )

        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Shared experts
        self.shared_experts = MoEExpert(hidden_size, ff_inner_dim, bias=ff_bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D]
        """
        B, L, D = x.shape

        # Compute gating scores
        gate_logits = self.gate(x)  # [B, L, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.moe_top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Normalize

        # Compute expert outputs
        x_flat = x.view(-1, D)  # [B*L, D]
        output = torch.zeros_like(x_flat)

        for k in range(self.moe_top_k):
            expert_indices = top_k_indices[:, :, k].view(-1)  # [B*L]
            expert_weights = top_k_probs[:, :, k].view(-1, 1)  # [B*L, 1]

            for e in range(self.num_experts):
                mask = expert_indices == e
                if mask.sum() == 0:
                    continue
                expert_input = x_flat[mask]
                expert_output = self.experts[e](expert_input)
                output[mask] += expert_weights[mask] * expert_output

        # Add shared expert output
        shared_output = self.shared_experts(x_flat)
        output = output + shared_output

        return output.view(B, L, D)


class HunYuanDiTBlock(nn.Module):
    """Transformer block for HunYuan DiT."""

    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        text_states_dim=1024,
        use_flash_attn=False,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        init_scale=1.0,
        qkv_bias=True,
        skip_connection=True,
        timested_modulate=False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        mlp_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        use_ele_affine = True

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(
            hidden_size, elementwise_affine=use_ele_affine, eps=1e-6
        )
        self.attn1 = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=qk_norm_layer,
        )

        # ========================= FFN =========================
        self.norm2 = norm_layer(
            hidden_size, elementwise_affine=use_ele_affine, eps=1e-6
        )

        # ========================= Add =========================
        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(c_emb_size, hidden_size, bias=True)
            )

        # ========================= Cross-Attention =========================
        self.attn2 = CrossAttention(
            hidden_size,
            text_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=qk_norm_layer,
            with_decoupled_ca=with_decoupled_ca,
            decoupled_ca_dim=decoupled_ca_dim,
            decoupled_ca_weight=decoupled_ca_weight,
            init_scale=init_scale,
        )
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.use_moe = use_moe
        if self.use_moe:
            self.moe = MoEBlock(
                hidden_size,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                dropout=0.0,
                activation_fn="gelu",
                final_dropout=False,
                ff_inner_dim=int(hidden_size * mlp_ratio),
                ff_bias=True,
            )
        else:
            self.mlp = MLP(width=hidden_size)

    def forward(self, x, c=None, text_states=None, skip_value=None):
        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # Self-Attention
        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa

        attn_out = self.attn1(self.norm1(x))
        x = x + attn_out

        # Cross-Attention
        x = x + self.attn2(self.norm2(x), text_states)

        # FFN Layer
        mlp_inputs = self.norm3(x)

        if self.use_moe:
            x = x + self.moe(mlp_inputs)
        else:
            x = x + self.mlp(mlp_inputs)

        return x


class AttentionPool(nn.Module):
    """Attention pooling for conditioning."""

    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)
        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class FinalLayer(nn.Module):
    """The final layer of HunYuanDiT."""

    def __init__(self, final_hidden_size, out_channels):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.norm_final = nn.LayerNorm(
            final_hidden_size, elementwise_affine=True, eps=1e-6
        )
        self.linear = nn.Linear(final_hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x


class HunYuanDiTPlain(nn.Module):
    """HunYuan DiT Plain model with optional MoE layers.

    This model uses a standard DiT architecture with self-attention,
    cross-attention, and optional MoE MLP layers.

    Args:
        input_size: Number of input latent tokens.
        in_channels: Number of input channels per token.
        hidden_size: Hidden dimension of transformer.
        context_dim: Dimension of conditioning context.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Whether to use bias in QKV projections.
        qk_norm: Whether to use QK normalization.
        qk_norm_type: Type of QK normalization ('rms' or 'layer').
        text_len: Length of text conditioning.
        num_moe_layers: Number of MoE layers (from the end).
        num_experts: Number of experts in MoE layers.
        moe_top_k: Top-k routing for MoE.
    """

    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        context_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_type="layer",
        qk_norm_type="rms",
        qk_norm=False,
        text_len=257,
        with_decoupled_ca=False,
        additional_cond_hidden_state=768,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        use_pos_emb=False,
        use_attention_pooling=True,
        guidance_cond_proj_dim=None,
        qkv_bias=True,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == "layer" else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == "rms" else nn.LayerNorm
        self.context_dim = context_dim

        self.with_decoupled_ca = with_decoupled_ca
        self.decoupled_ca_dim = decoupled_ca_dim
        self.decoupled_ca_weight = decoupled_ca_weight
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling
        self.guidance_cond_proj_dim = guidance_cond_proj_dim

        self.text_len = text_len

        # Input embedder
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)

        # Time embedder
        self.t_embedder = TimestepEmbedder(
            hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim
        )

        # Position embedding (optional)
        if self.use_pos_emb:
            self.register_buffer("pos_embed", torch.zeros(1, input_size, hidden_size))
            pos = np.arange(self.input_size, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Attention pooling (optional)
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.pooler = AttentionPool(
                self.text_len, context_dim, num_heads=8, output_dim=1024
            )
            self.extra_embedder = nn.Sequential(
                nn.Linear(1024, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size, bias=True),
            )

        # Decoupled cross-attention projection (optional)
        if with_decoupled_ca:
            self.additional_cond_hidden_state = additional_cond_hidden_state
            self.additional_cond_proj = nn.Sequential(
                nn.Linear(additional_cond_hidden_state, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, 1024, bias=True),
            )

        # HunYuanDiT Blocks
        self.blocks = nn.ModuleList(
            [
                HunYuanDiTBlock(
                    hidden_size=hidden_size,
                    c_emb_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    text_states_dim=context_dim,
                    qk_norm=qk_norm,
                    norm_layer=self.norm,
                    qk_norm_layer=self.qk_norm,
                    skip_connection=layer > depth // 2,
                    with_decoupled_ca=with_decoupled_ca,
                    decoupled_ca_dim=decoupled_ca_dim,
                    decoupled_ca_weight=decoupled_ca_weight,
                    qkv_bias=qkv_bias,
                    use_moe=True if depth - layer <= num_moe_layers else False,
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                )
                for layer in range(depth)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x, t, contexts, **kwargs):
        """Forward pass for denoising.

        Args:
            x: Noisy latent tokens [B, N, C].
            t: Timesteps [B].
            contexts: Dictionary with 'main' key containing conditioning.
            **kwargs: Additional arguments including 'guidance_cond'.

        Returns:
            Predicted noise or velocity [B, N, C].
        """
        cond = contexts["main"]

        # Time embedding
        t = self.t_embedder(t, condition=kwargs.get("guidance_cond"))

        # Input embedding
        x = self.x_embedder(x)

        # Position embedding
        if self.use_pos_emb:
            pos_embed = self.pos_embed.to(x.dtype)
            x = x + pos_embed

        # Attention pooling for conditioning
        if self.use_attention_pooling:
            extra_vec = self.pooler(cond, None)
            c = t + self.extra_embedder(extra_vec)  # [B, D]
        else:
            c = t

        # Decoupled cross-attention
        if self.with_decoupled_ca:
            additional_cond = self.additional_cond_proj(contexts["additional"])
            cond = torch.cat([cond, additional_cond], dim=1)

        # Concatenate time embedding with input
        x = torch.cat([c, x], dim=1)

        # Transformer blocks with skip connections
        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        # Final layer
        x = self.final_layer(x)
        return x

    def load_weights(self, weights: dict[str, torch.Tensor]) -> set[str]:
        """Load weights from a state dict."""
        loaded = set()
        missing, unexpected = self.load_state_dict(weights, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading HunYuanDiTPlain: {missing}")
        if unexpected:
            logger.warning(
                f"Unexpected keys when loading HunYuanDiTPlain: {unexpected}"
            )
        loaded.update(k for k in weights.keys() if k not in unexpected)
        return loaded


# Entry class for model registry
EntryClass = HunYuanDiTPlain
