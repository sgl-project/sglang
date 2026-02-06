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

from sglang.multimodal_gen.configs.models.dits.hunyuan3d import (
    Hunyuan3DDiTArchConfig,
    Hunyuan3DDiTConfig,
    Hunyuan3DPlainDiTArchConfig,
    Hunyuan3DPlainDiTConfig,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.encoders.hunyuan3d import (
    get_1d_sincos_pos_embed_from_grid,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
    """
    Cross-attention module with SGLang attention backend support.
    """

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
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
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

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

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

        if supported_attention_backends is None:
            supported_attention_backends = {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }

        self.local_attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

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
            self.local_attn_dca = LocalAttention(
                num_heads=num_heads,
                head_size=self.head_dim,
                causal=False,
                supported_attention_backends=supported_attention_backends,
            )

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        """
        b, s1, c = x.shape

        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim :, :]
            kv_dca = self.kv_proj_dca(context_dca).view(
                b, self.dca_dim, 2, self.num_heads, self.head_dim
            )
            k_dca, v_dca = kv_dca.unbind(dim=2)
            k_dca = self.k_norm_dca(k_dca)
            y = y[:, : (token_len - self.dca_dim), :]

        _, s2, _ = y.shape
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        q = q.view(b, s1, self.num_heads, self.head_dim)
        k = k.view(b, s2, self.num_heads, self.head_dim)
        v = v.view(b, s2, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        context = self.local_attn(q, k, v)
        context = context.flatten(2)

        if self.with_dca:
            context_dca = self.local_attn_dca(q, k_dca, v_dca)
            context_dca = context_dca.flatten(2)
            context = context + self.dca_weight * context_dca

        out = self.out_proj(context)

        return out


class Attention(nn.Module):
    """
    Self-attention module with SGLang attention backend support.
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

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

        if supported_attention_backends is None:
            supported_attention_backends = {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        self.local_attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(self, x):
        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = self.local_attn(q, k, v)
        x = x.flatten(2)

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
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
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
            supported_attention_backends=supported_attention_backends,
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
            supported_attention_backends=supported_attention_backends,
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


class HunYuanDiTPlain(CachableDiT, OffloadableDiTMixin):
    """HunYuan DiT Plain model with optional MoE layers.

    This model uses a standard DiT architecture with self-attention,
    cross-attention, and optional MoE MLP layers.
    """

    # External module path aliases for compatibility with Hunyuan3D configs
    _aliases = [
        "hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain",
        "hy3dshape.models.denoisers.hunyuan3ddit.HunYuanDiTPlain",
    ]

    param_names_mapping = {}

    @classmethod
    def build_config_from_params(cls, params: dict) -> Hunyuan3DPlainDiTConfig:
        """Build a DiTConfig from YAML-style parameter dict."""
        field_mapping = {"num_heads": "num_attention_heads", "depth": "num_layers"}
        arch_kwargs = {}
        for k, v in params.items():
            if k in ("supported_attention_backends",):
                continue
            mapped = field_mapping.get(k, k)
            arch_kwargs[mapped] = v
        return Hunyuan3DPlainDiTConfig(
            arch_config=Hunyuan3DPlainDiTArchConfig(**arch_kwargs)
        )

    def __init__(
        self,
        config: Hunyuan3DPlainDiTConfig,
        hf_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(config=config, hf_config=hf_config or {}, **kwargs)
        arch = config.arch_config

        input_size = arch.input_size
        in_channels = arch.in_channels
        hidden_size = arch.hidden_size
        depth = arch.num_layers
        num_heads = arch.num_attention_heads
        context_dim = arch.context_dim
        mlp_ratio = arch.mlp_ratio
        norm_type = arch.norm_type
        qk_norm_type = arch.qk_norm_type
        qk_norm = arch.qk_norm
        text_len = arch.text_len
        with_decoupled_ca = arch.with_decoupled_ca
        additional_cond_hidden_state = arch.additional_cond_hidden_state
        decoupled_ca_dim = arch.decoupled_ca_dim
        decoupled_ca_weight = arch.decoupled_ca_weight
        use_pos_emb = arch.use_pos_emb
        use_attention_pooling = arch.use_attention_pooling
        guidance_cond_proj_dim = arch.guidance_cond_proj_dim
        qkv_bias = arch.qkv_bias
        num_moe_layers = arch.num_moe_layers
        num_experts = arch.num_experts
        moe_top_k = arch.moe_top_k
        supported_attention_backends = arch._supported_attention_backends

        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_channels_latents = in_channels
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
                    supported_attention_backends=supported_attention_backends,
                )
                for layer in range(depth)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        # OffloadableDiTMixin
        self.layer_names = ["blocks"]

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


# =============================================================================
# Hunyuan3D2DiT (Flux-style architecture for Hunyuan3D-2.0)
# Copied from: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
# =============================================================================

from dataclasses import dataclass
from typing import List, Optional, Tuple


def _flux_timestep_embedding(
    t: torch.Tensor, dim, max_period=10000, time_factor: float = 1000.0
):
    """
    Create sinusoidal timestep embeddings for Flux-style model.
    :param t: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class _FluxGELU(nn.Module):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class _FluxMLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class _FluxRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class _FluxQKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = _FluxRMSNorm(dim)
        self.key_norm = _FluxRMSNorm(dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class _FluxSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = _FluxQKNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

        if supported_attention_backends is None:
            supported_attention_backends = {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        self.local_attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v_for_norm = v.transpose(1, 2)
        q, k = self.norm(q, k, v_for_norm)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        x = self.local_attn(q, k, v)
        x = x.flatten(2)
        x = self.proj(x)
        return x


@dataclass
class _FluxModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class _FluxModulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(
        self, vec: torch.Tensor
    ) -> Tuple[_FluxModulationOut, Optional[_FluxModulationOut]]:
        out = self.lin(F.silu(vec))[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)

        return (
            _FluxModulationOut(*out[:3]),
            _FluxModulationOut(*out[3:]) if self.is_double else None,
        )


class _FluxDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.img_mod = _FluxModulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = _FluxSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            supported_attention_backends=supported_attention_backends,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            _FluxGELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = _FluxModulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = _FluxSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            supported_attention_backends=supported_attention_backends,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            _FluxGELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        if supported_attention_backends is None:
            supported_attention_backends = {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        self.local_attn_joint = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift

        B, img_L, _ = img_modulated.shape
        img_qkv = self.img_attn.qkv(img_modulated)
        img_qkv = img_qkv.view(B, img_L, 3, self.num_heads, self.head_dim)
        img_q, img_k, img_v = img_qkv[:, :, 0], img_qkv[:, :, 1], img_qkv[:, :, 2]
        img_q_t = img_q.transpose(1, 2)
        img_k_t = img_k.transpose(1, 2)
        img_v_t = img_v.transpose(1, 2)
        img_q_t, img_k_t = self.img_attn.norm(img_q_t, img_k_t, img_v_t)
        img_q = img_q_t.transpose(1, 2)
        img_k = img_k_t.transpose(1, 2)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_L = txt_modulated.shape[1]
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_qkv = txt_qkv.view(B, txt_L, 3, self.num_heads, self.head_dim)
        txt_q, txt_k, txt_v = txt_qkv[:, :, 0], txt_qkv[:, :, 1], txt_qkv[:, :, 2]
        txt_q_t = txt_q.transpose(1, 2)
        txt_k_t = txt_k.transpose(1, 2)
        txt_v_t = txt_v.transpose(1, 2)
        txt_q_t, txt_k_t = self.txt_attn.norm(txt_q_t, txt_k_t, txt_v_t)
        txt_q = txt_q_t.transpose(1, 2)
        txt_k = txt_k_t.transpose(1, 2)

        q = torch.cat((txt_q, img_q), dim=1)
        k = torch.cat((txt_k, img_k), dim=1)
        v = torch.cat((txt_v, img_v), dim=1)

        attn = self.local_attn_joint(q, k, v)
        attn = attn.flatten(2)

        txt_attn, img_attn = attn[:, :txt_L], attn[:, txt_L:]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class _FluxSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = _FluxQKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = _FluxGELU(approximate="tanh")
        self.modulation = _FluxModulation(hidden_size, double=False)

        if supported_attention_backends is None:
            supported_attention_backends = {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        self.local_attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor
    ) -> torch.Tensor:
        mod, _ = self.modulation(vec)

        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        q_t, k_t = self.norm(q_t, k_t, v_t)
        q = q_t.transpose(1, 2)
        k = k_t.transpose(1, 2)

        attn = self.local_attn(q, k, v)
        attn = attn.flatten(2)

        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class _FluxLastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class Hunyuan3D2DiT(CachableDiT, OffloadableDiTMixin):
    """Hunyuan3D DiT model (Flux-style architecture for Hunyuan3D-2.0)."""

    _aliases = ["hy3dgen.shapegen.models.Hunyuan3DDiT"]

    param_names_mapping = {}

    @classmethod
    def build_config_from_params(cls, params: dict) -> Hunyuan3DDiTConfig:
        """Build a DiTConfig from YAML-style parameter dict."""
        field_mapping = {
            "num_heads": "num_attention_heads",
            "depth": "num_layers",
            "depth_single_blocks": "num_single_layers",
        }
        arch_kwargs = {}
        for k, v in params.items():
            if k in ("ckpt_path", "supported_attention_backends"):
                continue
            mapped = field_mapping.get(k, k)
            if k == "axes_dim" and isinstance(v, list):
                v = tuple(v)
            arch_kwargs[mapped] = v
        return Hunyuan3DDiTConfig(arch_config=Hunyuan3DDiTArchConfig(**arch_kwargs))

    def __init__(
        self,
        config: Hunyuan3DDiTConfig,
        hf_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(config=config, hf_config=hf_config or {}, **kwargs)
        arch = config.arch_config

        in_channels = arch.in_channels
        context_in_dim = arch.context_in_dim
        hidden_size = arch.hidden_size
        mlp_ratio = arch.mlp_ratio
        num_heads = arch.num_attention_heads
        depth = arch.num_layers
        depth_single_blocks = arch.num_single_layers
        axes_dim = list(arch.axes_dim)
        theta = arch.theta
        qkv_bias = arch.qkv_bias
        time_factor = arch.time_factor
        guidance_embed = arch.guidance_embed
        supported_attention_backends = arch._supported_attention_backends

        self.in_channels = in_channels
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.axes_dim = axes_dim
        self.theta = theta
        self.qkv_bias = qkv_bias
        self.time_factor = time_factor
        self.out_channels = self.in_channels
        self.num_channels_latents = self.in_channels
        self.guidance_embed = guidance_embed

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.latent_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = _FluxMLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.cond_in = nn.Linear(context_in_dim, self.hidden_size)
        self.guidance_in = (
            _FluxMLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if guidance_embed
            else nn.Identity()
        )

        self.double_blocks = nn.ModuleList(
            [
                _FluxDoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    supported_attention_backends=supported_attention_backends,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                _FluxSingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    supported_attention_backends=supported_attention_backends,
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = _FluxLastLayer(self.hidden_size, 1, self.out_channels)

        # OffloadableDiTMixin
        self.layer_names = ["double_blocks", "single_blocks"]

    def forward(
        self,
        x,
        t,
        contexts,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for denoising."""

        cond = contexts["main"]

        latent = self.latent_in(x)

        t_emb = _flux_timestep_embedding(t, 256, self.time_factor).to(
            dtype=latent.dtype
        )

        vec = self.time_in(t_emb)

        if self.guidance_embed:
            guidance = kwargs.get("guidance", None)
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(
                _flux_timestep_embedding(guidance, 256, self.time_factor)
            )

        cond = self.cond_in(cond)

        pe = None

        # Double blocks
        for i, block in enumerate(self.double_blocks):
            latent, cond = block(img=latent, txt=cond, vec=vec, pe=pe)
        latent = torch.cat((cond, latent), 1)

        # Single blocks
        for i, block in enumerate(self.single_blocks):
            latent = block(latent, vec=vec, pe=pe)

        latent = latent[:, cond.shape[1] :, ...]
        latent = self.final_layer(latent, vec)
        return latent


# =============================================================================
# Hunyuan3D Paint UNet model for texture generation
# Copied from: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
# =============================================================================

import copy
import json
import os as _os

from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import Attention as DiffusersAttention
from diffusers.models.transformers.transformer_2d import BasicTransformerBlock


def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
):
    """Feed forward with chunking to save memory.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/unet/modules.py
    """
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]}"
            f"has to be divisible by chunk size: {chunk_size}."
            f" Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


class SGLangAttentionWrapper(torch.nn.Module):
    """Drop-in replacement for DiffusersAttention that uses sglang's attention backend."""

    _SUPPORTED_BACKENDS = {AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA}

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: int | None = None,
        out_bias: bool = True,
    ) -> None:
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.query_dim = query_dim
        cross_attention_dim = cross_attention_dim or query_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, query_dim, bias=out_bias), nn.Dropout(dropout)]
        )

        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            get_attn_backend,
        )

        attn_backend = get_attn_backend(
            dim_head, torch.float16, self._SUPPORTED_BACKENDS
        )
        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=dim_head**-0.5,
            num_kv_heads=heads,
            causal=False,
        )
        self._attn_backend_name = attn_backend.get_enum().name

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        B, N_q, _ = hidden_states.shape
        _, N_kv, _ = encoder_hidden_states.shape

        q = self.to_q(hidden_states).view(B, N_q, self.heads, self.dim_head)
        k = self.to_k(encoder_hidden_states).view(B, N_kv, self.heads, self.dim_head)
        v = self.to_v(encoder_hidden_states).view(B, N_kv, self.heads, self.dim_head)

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        ctx = get_forward_context()
        out = self.attn_impl.forward(q, k, v, attn_metadata=ctx.attn_metadata)
        out = out.reshape(B, N_q, self.inner_dim)

        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class Basic2p5DTransformerBlock(torch.nn.Module):
    """2.5D Transformer block with Multiview Attention (MVA) and Reference View Attention (RVA)."""

    def __init__(
        self,
        transformer: BasicTransformerBlock,
        layer_name: str,
        use_ma: bool = True,
        use_ra: bool = True,
        is_turbo: bool = False,
        use_sglang_attn: bool = True,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.layer_name = layer_name
        self.use_ma = use_ma
        self.use_ra = use_ra
        self.is_turbo = is_turbo
        self.use_sglang_attn = use_sglang_attn and not is_turbo

        attn_cls = (
            SGLangAttentionWrapper if self.use_sglang_attn else DiffusersAttention
        )
        attn_kwargs = dict(
            query_dim=self.dim,
            heads=self.num_attention_heads,
            dim_head=self.attention_head_dim,
            dropout=self.dropout,
            bias=self.attention_bias,
            cross_attention_dim=None,
            upcast_attention=self.attn1.upcast_attention,
            out_bias=True,
        )
        if self.use_sglang_attn:
            attn_kwargs.pop("upcast_attention")

        if self.use_ma:
            self.attn_multiview = attn_cls(**attn_kwargs)

        if self.use_ra:
            self.attn_refview = attn_cls(**attn_kwargs)

        if self.is_turbo:
            self._initialize_attn_weights()

    def _initialize_attn_weights(self):
        """Initialize attention weights for turbo mode."""
        if self.use_ma:
            self.attn_multiview.load_state_dict(self.attn1.state_dict())
            with torch.no_grad():
                for layer in self.attn_multiview.to_out:
                    for param in layer.parameters():
                        param.zero_()
        if self.use_ra:
            self.attn_refview.load_state_dict(self.attn1.state_dict())
            with torch.no_grad():
                for layer in self.attn_refview.to_out:
                    for param in layer.parameters():
                        param.zero_()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: dict = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward pass with MVA and RVA support."""
        batch_size = hidden_states.shape[0]

        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        num_in_batch = cross_attention_kwargs.pop("num_in_batch", 1)
        mode = cross_attention_kwargs.pop("mode", None)

        if not self.is_turbo:
            mva_scale = cross_attention_kwargs.pop("mva_scale", 1.0)
            ref_scale = cross_attention_kwargs.pop("ref_scale", 1.0)
        else:
            position_attn_mask = cross_attention_kwargs.pop("position_attn_mask", None)
            position_voxel_indices = cross_attention_kwargs.pop(
                "position_voxel_indices", None
            )
            mva_scale = 1.0
            ref_scale = 1.0

        condition_embed_dict = cross_attention_kwargs.pop("condition_embed_dict", None)

        # Normalization
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # Self-attention
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Reference Attention - Write mode
        if mode is not None and "w" in mode:
            condition_embed_dict[self.layer_name] = rearrange(
                norm_hidden_states, "(b n) l c -> b (n l) c", n=num_in_batch
            )

        # Reference Attention - Read mode
        if mode is not None and "r" in mode and self.use_ra:
            condition_embed = (
                condition_embed_dict[self.layer_name]
                .unsqueeze(1)
                .repeat(1, num_in_batch, 1, 1)
            )
            condition_embed = rearrange(condition_embed, "b n l c -> (b n) l c")

            attn_output = self.attn_refview(
                norm_hidden_states,
                encoder_hidden_states=condition_embed,
                attention_mask=None,
                **cross_attention_kwargs,
            )

            if not self.is_turbo:
                ref_scale_timing = ref_scale
                if isinstance(ref_scale, torch.Tensor):
                    ref_scale_timing = (
                        ref_scale.unsqueeze(1).repeat(1, num_in_batch).view(-1)
                    )
                    for _ in range(attn_output.ndim - 1):
                        ref_scale_timing = ref_scale_timing.unsqueeze(-1)

            hidden_states = ref_scale_timing * attn_output + hidden_states

            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        # Multiview Attention
        if num_in_batch > 1 and self.use_ma:
            multivew_hidden_states = rearrange(
                norm_hidden_states, "(b n) l c -> b (n l) c", n=num_in_batch
            )

            if self.is_turbo:
                position_mask = None
                if position_attn_mask is not None:
                    if multivew_hidden_states.shape[1] in position_attn_mask:
                        position_mask = position_attn_mask[
                            multivew_hidden_states.shape[1]
                        ]
                position_indices = None
                if position_voxel_indices is not None:
                    if multivew_hidden_states.shape[1] in position_voxel_indices:
                        position_indices = position_voxel_indices[
                            multivew_hidden_states.shape[1]
                        ]
                attn_output = self.attn_multiview(
                    multivew_hidden_states,
                    encoder_hidden_states=multivew_hidden_states,
                    attention_mask=position_mask,
                    position_indices=position_indices,
                    **cross_attention_kwargs,
                )
            else:
                attn_output = self.attn_multiview(
                    multivew_hidden_states,
                    encoder_hidden_states=multivew_hidden_states,
                    **cross_attention_kwargs,
                )

            attn_output = rearrange(
                attn_output, "b (n l) c -> (b n) l c", n=num_in_batch
            )

            hidden_states = mva_scale * attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        # GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )

            hidden_states = attn_output + hidden_states

        # Feed-forward
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


@torch.no_grad()
def compute_voxel_grid_mask(position: torch.Tensor, grid_resolution: int = 8):
    """Compute voxel grid mask for position-aware attention.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/unet/modules.py
    """
    position = position.half()
    B, N, _, H, W = position.shape
    assert H % grid_resolution == 0 and W % grid_resolution == 0

    valid_mask = (position != 1).all(dim=2, keepdim=True)
    valid_mask = valid_mask.expand_as(position)
    position[valid_mask == False] = 0

    position = rearrange(
        position,
        "b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w",
        num_h=grid_resolution,
        num_w=grid_resolution,
    )
    valid_mask = rearrange(
        valid_mask,
        "b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w",
        num_h=grid_resolution,
        num_w=grid_resolution,
    )

    grid_position = position.sum(dim=(-2, -1))
    count_masked = valid_mask.sum(dim=(-2, -1))

    grid_position = grid_position / count_masked.clamp(min=1)
    grid_position[count_masked < 5] = 0

    grid_position = grid_position.permute(0, 1, 4, 2, 3)
    grid_position = rearrange(grid_position, "b n c h w -> b n (h w) c")

    grid_position_expanded_1 = grid_position.unsqueeze(2).unsqueeze(4)
    grid_position_expanded_2 = grid_position.unsqueeze(1).unsqueeze(3)

    distances = torch.norm(grid_position_expanded_1 - grid_position_expanded_2, dim=-1)

    weights = distances
    grid_distance = 1.73 / grid_resolution

    weights = weights < grid_distance

    return weights


def compute_multi_resolution_mask(
    position_maps: torch.Tensor, grid_resolutions: List[int] = [32, 16, 8]
) -> dict:
    """Compute multi-resolution position attention masks.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/unet/modules.py
    """
    position_attn_mask = {}
    with torch.no_grad():
        for grid_resolution in grid_resolutions:
            position_mask = compute_voxel_grid_mask(position_maps, grid_resolution)
            position_mask = rearrange(
                position_mask, "b ni nj li lj -> b (ni li) (nj lj)"
            )
            position_attn_mask[position_mask.shape[1]] = position_mask
    return position_attn_mask


@torch.no_grad()
def compute_discrete_voxel_indice(
    position: torch.Tensor, grid_resolution: int = 8, voxel_resolution: int = 128
):
    """Compute discrete voxel indices for position encoding.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/unet/modules.py
    """
    position = position.half()
    B, N, _, H, W = position.shape
    assert H % grid_resolution == 0 and W % grid_resolution == 0

    valid_mask = (position != 1).all(dim=2, keepdim=True)
    valid_mask = valid_mask.expand_as(position)
    position[valid_mask == False] = 0

    position = rearrange(
        position,
        "b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w",
        num_h=grid_resolution,
        num_w=grid_resolution,
    )
    valid_mask = rearrange(
        valid_mask,
        "b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w",
        num_h=grid_resolution,
        num_w=grid_resolution,
    )

    grid_position = position.sum(dim=(-2, -1))
    count_masked = valid_mask.sum(dim=(-2, -1))

    grid_position = grid_position / count_masked.clamp(min=1)
    grid_position[count_masked < 5] = 0

    grid_position = grid_position.permute(0, 1, 4, 2, 3).clamp(0, 1)
    voxel_indices = grid_position * (voxel_resolution - 1)
    voxel_indices = torch.round(voxel_indices).long()
    return voxel_indices


def compute_multi_resolution_discrete_voxel_indice(
    position_maps: torch.Tensor,
    grid_resolutions: List[int] = [64, 32, 16, 8],
    voxel_resolutions: List[int] = [512, 256, 128, 64],
) -> dict:
    """Compute multi-resolution discrete voxel indices.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/unet/modules.py
    """
    voxel_indices = {}
    with torch.no_grad():
        for grid_resolution, voxel_resolution in zip(
            grid_resolutions, voxel_resolutions
        ):
            voxel_indice = compute_discrete_voxel_indice(
                position_maps, grid_resolution, voxel_resolution
            )
            voxel_indice = rearrange(voxel_indice, "b n c h w -> b (n h w) c")
            voxel_indices[voxel_indice.shape[1]] = {
                "voxel_indices": voxel_indice,
                "voxel_resolution": voxel_resolution,
            }
    return voxel_indices


class UNet2p5DConditionModel(torch.nn.Module):
    """2.5D UNet for multi-view texture generation.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/unet/modules.py

    This model wraps a standard UNet2DConditionModel and adds:
    - Multiview Attention (MVA) for cross-view consistency
    - Reference View Attention (RVA) for conditioning
    - Camera embeddings for view-aware generation
    - Dual-stream architecture for reference encoding
    """

    def __init__(self, unet: UNet2DConditionModel) -> None:
        super().__init__()
        self.unet = unet

        self.use_ma = True
        self.use_ra = True
        self.use_camera_embedding = True
        self.use_dual_stream = True
        self.is_turbo = False

        if self.use_dual_stream:
            self.unet_dual = copy.deepcopy(unet)
            self.init_attention(self.unet_dual)
        self.init_attention(
            self.unet, use_ma=self.use_ma, use_ra=self.use_ra, is_turbo=self.is_turbo
        )
        self.init_condition()
        self.init_camera_embedding()

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, **kwargs):
        """Load a pretrained UNet2p5DConditionModel.

        Args:
            pretrained_model_name_or_path: Path to the pretrained model directory.
            **kwargs: Additional arguments (e.g., dtype).

        Returns:
            Loaded UNet2p5DConditionModel instance.
        """
        torch_dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", torch.float32))
        config_path = _os.path.join(pretrained_model_name_or_path, "config.json")
        unet_ckpt_path = _os.path.join(
            pretrained_model_name_or_path, "diffusion_pytorch_model.bin"
        )

        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        unet = UNet2DConditionModel(**config)
        unet = UNet2p5DConditionModel(unet)
        unet_ckpt = torch.load(unet_ckpt_path, map_location="cpu", weights_only=True)
        unet.load_state_dict(unet_ckpt, strict=True)
        unet = unet.to(torch_dtype)
        return unet

    def init_condition(self):
        """Initialize condition-related modules."""
        self.unet.conv_in = torch.nn.Conv2d(
            12,  # 4 (latent) + 4 (normal) + 4 (position)
            self.unet.conv_in.out_channels,
            kernel_size=self.unet.conv_in.kernel_size,
            stride=self.unet.conv_in.stride,
            padding=self.unet.conv_in.padding,
            dilation=self.unet.conv_in.dilation,
            groups=self.unet.conv_in.groups,
            bias=self.unet.conv_in.bias is not None,
        )

        self.unet.learned_text_clip_gen = nn.Parameter(torch.randn(1, 77, 1024))
        self.unet.learned_text_clip_ref = nn.Parameter(torch.randn(1, 77, 1024))

    def init_camera_embedding(self):
        """Initialize camera embedding module."""
        if self.use_camera_embedding:
            time_embed_dim = 1280
            self.max_num_ref_image = 5
            self.max_num_gen_image = 12 * 3 + 4 * 2
            self.unet.class_embedding = nn.Embedding(
                self.max_num_ref_image + self.max_num_gen_image, time_embed_dim
            )

    def init_attention(
        self,
        unet: UNet2DConditionModel,
        use_ma: bool = False,
        use_ra: bool = False,
        is_turbo: bool = False,
        use_sglang_attn: bool = True,
    ):
        """Initialize attention blocks with MVA and RVA support."""
        block_kwargs = dict(
            use_ma=use_ma,
            use_ra=use_ra,
            is_turbo=is_turbo,
            use_sglang_attn=use_sglang_attn,
        )

        # Down blocks
        for down_block_i, down_block in enumerate(unet.down_blocks):
            if (
                hasattr(down_block, "has_cross_attention")
                and down_block.has_cross_attention
            ):
                for attn_i, attn in enumerate(down_block.attentions):
                    for transformer_i, transformer in enumerate(
                        attn.transformer_blocks
                    ):
                        if isinstance(transformer, BasicTransformerBlock):
                            attn.transformer_blocks[transformer_i] = (
                                Basic2p5DTransformerBlock(
                                    transformer,
                                    f"down_{down_block_i}_{attn_i}_{transformer_i}",
                                    **block_kwargs,
                                )
                            )

        # Mid block
        if (
            hasattr(unet.mid_block, "has_cross_attention")
            and unet.mid_block.has_cross_attention
        ):
            for attn_i, attn in enumerate(unet.mid_block.attentions):
                for transformer_i, transformer in enumerate(attn.transformer_blocks):
                    if isinstance(transformer, BasicTransformerBlock):
                        attn.transformer_blocks[transformer_i] = (
                            Basic2p5DTransformerBlock(
                                transformer,
                                f"mid_{attn_i}_{transformer_i}",
                                **block_kwargs,
                            )
                        )

        # Up blocks
        for up_block_i, up_block in enumerate(unet.up_blocks):
            if (
                hasattr(up_block, "has_cross_attention")
                and up_block.has_cross_attention
            ):
                for attn_i, attn in enumerate(up_block.attentions):
                    for transformer_i, transformer in enumerate(
                        attn.transformer_blocks
                    ):
                        if isinstance(transformer, BasicTransformerBlock):
                            attn.transformer_blocks[transformer_i] = (
                                Basic2p5DTransformerBlock(
                                    transformer,
                                    f"up_{up_block_i}_{attn_i}_{transformer_i}",
                                    **block_kwargs,
                                )
                            )

        if use_sglang_attn and (use_ma or use_ra):
            backend = "unknown"
            for block in self._iter_2p5d_blocks(unet):
                for attr in ("attn_multiview", "attn_refview"):
                    wrapper = getattr(block, attr, None)
                    if isinstance(wrapper, SGLangAttentionWrapper):
                        backend = wrapper._attn_backend_name
                        break
                if backend != "unknown":
                    break
            count = sum(1 for _ in self._iter_2p5d_blocks(unet))
            logger.info(
                "Initialized %d Basic2p5DTransformerBlocks with sglang %s attention",
                count,
                backend,
            )

    @staticmethod
    def _iter_2p5d_blocks(unet):
        """Yield all Basic2p5DTransformerBlock instances in a UNet."""
        for block_group in (unet.down_blocks, [unet.mid_block], unet.up_blocks):
            for block in block_group:
                if not hasattr(block, "attentions"):
                    continue
                for attn in block.attentions:
                    for tb in attn.transformer_blocks:
                        if isinstance(tb, Basic2p5DTransformerBlock):
                            yield tb

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        down_intrablock_additional_residuals=None,
        down_block_res_samples=None,
        mid_block_res_sample=None,
        **cached_condition,
    ):
        """Forward pass for multi-view texture generation.

        Args:
            sample: Input latents [B, N_gen, C, H, W]
            timestep: Diffusion timestep
            encoder_hidden_states: Text encoder hidden states
            cached_condition: Cached conditions including:
                - normal_imgs: Normal map latents [B, N_gen, C, H, W]
                - position_imgs: Position map latents [B, N_gen, C, H, W]
                - ref_latents: Reference image latents [B, N_ref, C, H, W]
                - camera_info_gen: Camera indices for generated views
                - camera_info_ref: Camera indices for reference views
        """
        B, N_gen, _, H, W = sample.shape
        assert H == W

        if self.use_camera_embedding:
            camera_info_gen = (
                cached_condition["camera_info_gen"] + self.max_num_ref_image
            )
            camera_info_gen = rearrange(camera_info_gen, "b n -> (b n)")
        else:
            camera_info_gen = None

        # Concatenate latents with normal and position maps
        sample = [sample]
        if "normal_imgs" in cached_condition:
            sample.append(cached_condition["normal_imgs"])
        if "position_imgs" in cached_condition:
            sample.append(cached_condition["position_imgs"])
        sample = torch.cat(sample, dim=2)

        sample = rearrange(sample, "b n c h w -> (b n) c h w")

        encoder_hidden_states_gen = encoder_hidden_states.unsqueeze(1).repeat(
            1, N_gen, 1, 1
        )
        encoder_hidden_states_gen = rearrange(
            encoder_hidden_states_gen, "b n l c -> (b n) l c"
        )

        # Process reference images for RVA
        if self.use_ra:
            if "condition_embed_dict" in cached_condition:
                condition_embed_dict = cached_condition["condition_embed_dict"]
            else:
                condition_embed_dict = {}
                ref_latents = cached_condition["ref_latents"]
                N_ref = ref_latents.shape[1]

                if self.use_camera_embedding:
                    camera_info_ref = cached_condition["camera_info_ref"]
                    camera_info_ref = rearrange(camera_info_ref, "b n -> (b n)")
                else:
                    camera_info_ref = None

                ref_latents = rearrange(ref_latents, "b n c h w -> (b n) c h w")

                encoder_hidden_states_ref = self.unet.learned_text_clip_ref.unsqueeze(
                    1
                ).repeat(B, N_ref, 1, 1)
                encoder_hidden_states_ref = rearrange(
                    encoder_hidden_states_ref, "b n l c -> (b n) l c"
                )

                noisy_ref_latents = ref_latents
                timestep_ref = 0

                if self.use_dual_stream:
                    unet_ref = self.unet_dual
                else:
                    unet_ref = self.unet

                unet_ref(
                    noisy_ref_latents,
                    timestep_ref,
                    encoder_hidden_states=encoder_hidden_states_ref,
                    class_labels=camera_info_ref,
                    return_dict=False,
                    cross_attention_kwargs={
                        "mode": "w",
                        "num_in_batch": N_ref,
                        "condition_embed_dict": condition_embed_dict,
                    },
                )
                cached_condition["condition_embed_dict"] = condition_embed_dict
        else:
            condition_embed_dict = None

        mva_scale = cached_condition.get("mva_scale", 1.0)
        ref_scale = cached_condition.get("ref_scale", 1.0)

        if self.is_turbo:
            position_attn_mask = cached_condition.get("position_attn_mask", None)
            position_voxel_indices = cached_condition.get(
                "position_voxel_indices", None
            )
            cross_attention_kwargs_ = {
                "mode": "r",
                "num_in_batch": N_gen,
                "condition_embed_dict": condition_embed_dict,
                "position_attn_mask": position_attn_mask,
                "position_voxel_indices": position_voxel_indices,
                "mva_scale": mva_scale,
                "ref_scale": ref_scale,
            }
        else:
            cross_attention_kwargs_ = {
                "mode": "r",
                "num_in_batch": N_gen,
                "condition_embed_dict": condition_embed_dict,
                "mva_scale": mva_scale,
                "ref_scale": ref_scale,
            }

        return self.unet(
            sample,
            timestep,
            encoder_hidden_states_gen,
            *args,
            class_labels=camera_info_gen,
            down_intrablock_additional_residuals=(
                [
                    s.to(dtype=self.unet.dtype)
                    for s in down_intrablock_additional_residuals
                ]
                if down_intrablock_additional_residuals is not None
                else None
            ),
            down_block_additional_residuals=(
                [s.to(dtype=self.unet.dtype) for s in down_block_res_samples]
                if down_block_res_samples is not None
                else None
            ),
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=self.unet.dtype)
                if mid_block_res_sample is not None
                else None
            ),
            return_dict=False,
            cross_attention_kwargs=cross_attention_kwargs_,
        )


# Entry class for model registry
EntryClass = [HunYuanDiTPlain, Hunyuan3D2DiT, UNet2p5DConditionModel]
