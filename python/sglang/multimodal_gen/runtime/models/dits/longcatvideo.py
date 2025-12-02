# SPDX-License-Identifier: Apache-2.0
"""
Native LongCat Video DiT implementation using FastVideo conventions.

This is a Phase 2 reimplementation that replaces the third_party wrapper
with native FastVideo layers for better performance and integration.
"""

from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.configs.models.dits.longcatvideo import LongCatVideoConfig
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, FP32LayerNorm
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.rotary_embedding_3d import RotaryPositionalEmbedding3D
from sglang.multimodal_gen.runtime.layers.attention.layer import UlyssesAttention, LocalAttention
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.third_party.longcatvideo.block_sparse_attention.bsa_interface import flash_attn_bsa_3d

# ============================================================================
# Embeddings
# ============================================================================

class PatchEmbed3D(nn.Module):
    """
    3D patch embedding using Conv3d.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        embed_dim: int = 4096,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            [B, N, C] where N = (T/pt) * (H/ph) * (W/pw)
        """
        # Padding if needed
        _, _, T, H, W = x.shape
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if T % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        x = self.proj(x)  # [B, C, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


class TimestepEmbedder(nn.Module):
    """
    Sinusoidal timestep embedding + MLP projection.
    """

    def __init__(
        self,
        frequency_embedding_size: int = 256,
        adaln_tembed_dim: int = 512,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size

        # Use FastVideo's ReplicatedLinear
        self.linear_1 = ReplicatedLinear(
            frequency_embedding_size,
            adaln_tembed_dim,
            bias=True,
            params_dtype=dtype,
        )
        self.act = nn.SiLU()
        self.linear_2 = ReplicatedLinear(
            adaln_tembed_dim,
            adaln_tembed_dim,
            bias=True,
            params_dtype=dtype,
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor, latent_shape: tuple | None = None) -> torch.Tensor:
        """
        Args:
            t: [B] or [B, T] timesteps
            latent_shape: (T, H, W) for temporal expansion
        Returns:
            [B, T, C]
        """
        # Sinusoidal embedding in FP32
        t_freq = self.timestep_embedding(t.flatten(), self.frequency_embedding_size)

        # Cast to model dtype before MLP
        # Handle LoRA wrapper if present
        linear_layer = self.linear_1.base_layer if hasattr(self.linear_1, 'base_layer') else self.linear_1
        target_dtype = linear_layer.weight.dtype
        if t_freq.dtype != target_dtype:
            t_freq = t_freq.to(target_dtype)

        # MLP projection
        t_emb, _ = self.linear_1(t_freq)
        t_emb = self.act(t_emb)
        t_emb, _ = self.linear_2(t_emb)

        # Reshape if needed
        if latent_shape is not None and len(t.shape) > 1:
            B = t.shape[0]
            T = latent_shape[0]
            t_emb = t_emb.reshape(B, T, -1)

        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Caption embedding with MLP projection and optional text compaction.
    """

    def __init__(
        self,
        caption_channels: int = 4096,
        hidden_size: int = 4096,
        text_tokens_zero_pad: bool = True,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.text_tokens_zero_pad = text_tokens_zero_pad

        # Two-layer MLP using ReplicatedLinear
        self.linear_1 = ReplicatedLinear(
            caption_channels,
            hidden_size,
            bias=True,
            params_dtype=dtype,
        )
        self.act = nn.SiLU()
        self.linear_2 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=True,
            params_dtype=dtype,
        )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: [B, N_text, C_text] or [B, 1, N_text, C_text]
            encoder_attention_mask: [B, N_text] or [B, 1, 1, N_text]
        Returns:
            y: [B, N_text, C] - standard padded representation (like other models)
        """
        # Handle extra dimension from wrapper
        if len(encoder_hidden_states.shape) == 4:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)

        # Project
        y, _ = self.linear_1(encoder_hidden_states)
        y = self.act(y)
        y, _ = self.linear_2(y)  # [B, N_text, C]

        # Handle attention masking - just zero out padded tokens if requested
        if encoder_attention_mask is not None:
            # Remove extra dimensions
            if len(encoder_attention_mask.shape) == 4:
                encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            elif len(encoder_attention_mask.shape) == 3:
                encoder_attention_mask = encoder_attention_mask.squeeze(1)

            # Zero out padded tokens if requested
            if self.text_tokens_zero_pad:
                y = y * encoder_attention_mask.unsqueeze(-1)

        # Return standard format [B, N_text, C] - no compaction!
        return y


# ============================================================================
# Attention Modules (Placeholders for now)
# ============================================================================

class LongCatSelfAttention(nn.Module):
    """
    Self-attention with 3D RoPE support and optional BSA.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        config: LongCatVideoConfig,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Separate Q/K/V projections (not fused like original)
        self.to_q = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)
        self.to_k = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)
        self.to_v = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)

        # Per-head RMS normalization
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype or torch.float32)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype or torch.float32)

        # Output projection
        self.to_out = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)

        # 3D RoPE
        self.rope_3d = RotaryPositionalEmbedding3D(head_dim=self.head_dim)

        # BSA configuration
        self.enable_bsa = getattr(config, 'enable_bsa', False)
        self.bsa_params = getattr(config, 'bsa_params', None)

        # FastVideo attention backend (used when BSA is disabled)
        self.attn = UlyssesAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            supported_attention_backends=config._supported_attention_backends,
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, N, C]
        latent_shape: tuple,  # (T, H, W)
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with 3D RoPE and optional BSA.
        """
        B, N, C = x.shape
        T, H, W = latent_shape

        # Project to Q/K/V
        q, _ = self.to_q(x)
        k, _ = self.to_k(x)
        v, _ = self.to_v(x)

        # Reshape to heads: [B, N, num_heads, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        # Per-head RMS normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # For RoPE: need [B, num_heads, N, head_dim]
        q_rope = q.transpose(1, 2)
        k_rope = k.transpose(1, 2)

        # Apply 3D RoPE
        q_rope, k_rope = self.rope_3d(q_rope, k_rope, grid_size=latent_shape)

        # Transpose back: [B, N, num_heads, head_dim] or [B, H, N, D] for BSA
        q = q_rope.transpose(1, 2)
        k = k_rope.transpose(1, 2)

        # === Attention: BSA or standard ===
        if self.enable_bsa and T > 1:  # Only use BSA for multi-frame videos
            # BSA expects [B, H, S, D] format
            q_bsa = q.transpose(1, 2).contiguous()  # [B, num_heads, N, head_dim]
            k_bsa = k.transpose(1, 2).contiguous()
            v_bsa = v.transpose(1, 2).contiguous()

            # Handle SP split: BSA operates on per-rank spatial dimensions
            # Replicate LongCat's cp_split_hw logic exactly
            from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size
            sp_size = get_sp_world_size()
            if sp_size > 1:
                # Calculate optimal 2D split (same as LongCat's get_optimal_split)
                factors = []
                for i in range(1, int(sp_size ** 0.5) + 1):
                    if sp_size % i == 0:
                        factors.append([i, sp_size // i])
                cp_split_hw = min(factors, key=lambda x: abs(x[0] - x[1]))

                # Split H and W dimensions by their respective factors
                T_bsa, H_bsa, W_bsa = latent_shape
                assert H_bsa % cp_split_hw[0] == 0 and W_bsa % cp_split_hw[1] == 0, \
                    f"H {H_bsa} must be divisible by {cp_split_hw[0]}, W {W_bsa} must be divisible by {cp_split_hw[1]}"
                H_bsa = H_bsa // cp_split_hw[0]
                W_bsa = W_bsa // cp_split_hw[1]
                latent_shape_bsa = (T_bsa, H_bsa, W_bsa)
            else:
                latent_shape_bsa = latent_shape

            # Call BSA with per-rank latent shape
            out = flash_attn_bsa_3d(
                q_bsa, k_bsa, v_bsa,
                latent_shape_q=latent_shape_bsa,
                latent_shape_k=latent_shape_bsa,
                **self.bsa_params
            )  # [B, num_heads, N, head_dim]

            # Transpose back: [B, N, num_heads, head_dim]
            out = out.transpose(1, 2)
        else:
            # Standard attention: [B, N, num_heads, head_dim]
            out, _ = self.attn(q, k, v)

        # Reshape and project out
        out = out.reshape(B, N, C)
        out, _ = self.to_out(out)

        return out


class LongCatCrossAttention(nn.Module):
    """
    Cross-attention for text conditioning (standard implementation like other models).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        config: LongCatVideoConfig,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Separate Q/K/V projections
        self.to_q = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)
        self.to_k = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)
        self.to_v = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)

        # Per-head RMS normalization
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype or torch.float32)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype or torch.float32)

        # Output projection
        self.to_out = ReplicatedLinear(dim, dim, bias=True, params_dtype=dtype)

        # Cross-attention uses LocalAttention (FastVideo standard)
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=config.arch_config._supported_attention_backends,
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, N_img, C]
        context: torch.Tensor,  # [B, N_text, C]
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention (standard implementation).

        Args:
            x: Image tokens [B, N_img, C]
            context: Text tokens [B, N_text, C] (standard padded format)
        """
        B, N_img, C = x.shape

        # Project Q, K, V (standard cross-attention like WanVideo/StepVideo/Cosmos)
        q, _ = self.to_q(x)
        k, _ = self.to_k(context)
        v, _ = self.to_v(context)

        N_text = context.shape[1]

        # Reshape to heads
        q = q.view(B, N_img, self.num_heads, self.head_dim)
        k = k.view(B, N_text, self.num_heads, self.head_dim)
        v = v.view(B, N_text, self.num_heads, self.head_dim)

        # Per-head RMS normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Run cross-attention using FastVideo's LocalAttention
        # LocalAttention handles different q and k/v sequence lengths automatically
        out = self.attn(q, k, v)  # [B, N_img, num_heads, head_dim]

        # Reshape and project out
        out = out.reshape(B, N_img, C)
        out, _ = self.to_out(out)

        return out


# ============================================================================
# Feed-Forward Network
# ============================================================================

class LongCatSwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network using FastVideo's ReplicatedLinear.

    FFN(x) = down(gate(x) * SiLU(up(x)))
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Three projections for SwiGLU (no bias as per original)
        self.w1 = ReplicatedLinear(dim, hidden_dim, bias=False, params_dtype=dtype)  # gate
        self.w3 = ReplicatedLinear(dim, hidden_dim, bias=False, params_dtype=dtype)  # up
        self.w2 = ReplicatedLinear(hidden_dim, dim, bias=False, params_dtype=dtype)  # down

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SiLU(w1(x)) * w3(x) -> w2 (matching original LongCat)
        """
        w1_out, _ = self.w1(x)
        w3_out, _ = self.w3(x)
        combined = self.act(w1_out) * w3_out
        out, _ = self.w2(combined)
        return out


# ============================================================================
# Modulation Utilities
# ============================================================================

def modulate_fp32(norm: nn.Module, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply modulation in FP32 for numerical stability (matching original LongCat).

    shift and scale should already be FP32 from torch.amp.autocast context.
    """
    # Ensure modulation params are FP32 (should be from autocast)
    assert shift.dtype == torch.float32 and scale.dtype == torch.float32, \
        f"shift and scale must be FP32, got {shift.dtype} and {scale.dtype}"

    orig_dtype = x.dtype

    # Normalize and modulate in FP32
    x_norm = norm(x.to(torch.float32))
    x_mod = x_norm * (scale + 1) + shift

    return x_mod.to(orig_dtype)


# ============================================================================
# Transformer Block
# ============================================================================

class LongCatTransformerBlock(nn.Module):
    """
    Single-stream transformer block with:
    - AdaLN modulation (FP32)
    - Self-attention
    - Cross-attention
    - SwiGLU FFN
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        config: LongCatVideoConfig,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # AdaLN modulation (6 parameters: scale/shift for attn & ffn, gate for residual)
        self.adaln_linear_1 = ReplicatedLinear(
            adaln_tembed_dim,
            6 * hidden_size,
            bias=True,
            params_dtype=dtype,
        )
        self.adaln_act = nn.SiLU()

        # Normalization layers (CRITICAL: Use LayerNorm not RMSNorm like original!)
        # Original LongCat uses LayerNorm_FP32 with elementwise_affine=False
        self.norm_attn = FP32LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.norm_ffn = FP32LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        # Cross-attention norm has elementwise_affine=True (has weight and bias)
        self.norm_cross = FP32LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)

        # Self-attention
        self.self_attn = LongCatSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            config=config,
            dtype=dtype,
        )

        # Cross-attention
        self.cross_attn = LongCatCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            config=config,
            dtype=dtype,
        )

        # SwiGLU FFN
        ffn_hidden_dim = int(hidden_size * mlp_ratio * 2 / 3)
        # Round up to nearest multiple of 256
        ffn_hidden_dim = 256 * ((ffn_hidden_dim + 255) // 256)

        self.ffn = LongCatSwiGLUFFN(
            dim=hidden_size,
            hidden_dim=ffn_hidden_dim,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, N, C]
        context: torch.Tensor,  # [B, N_text, C]
        t: torch.Tensor,  # [B, T, C_t]
        latent_shape: tuple,  # (T, H, W)
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with AdaLN modulation.
        """
        B, N, C = x.shape
        T, H, W = latent_shape
        x_orig_dtype = x.dtype  # Save for later casting

        # === AdaLN Modulation (CRITICAL: FP32 for stability like original) ===
        # Use autocast to compute modulation params in FP32
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            t_mod = self.adaln_act(t)
            mod_params, _ = self.adaln_linear_1(t_mod)
            # Ensure FP32 output (needed when LoRA is applied)
            if mod_params.dtype != torch.float32:
                mod_params = mod_params.float()
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                mod_params.unsqueeze(2).chunk(6, dim=-1)  # [B, T, 1, C]

        # === Self-Attention ===
        x_norm = modulate_fp32(self.norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa)
        x_norm = x_norm.view(B, N, C)

        attn_out = self.self_attn(x_norm, latent_shape=latent_shape)

        # Residual with gating (CRITICAL: FP32 like original, then cast back)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_msa * attn_out.view(B, T, -1, C)).view(B, N, C)
        x = x.to(x_orig_dtype)

        # === Cross-Attention ===
        x_norm_cross = self.norm_cross(x)
        cross_out = self.cross_attn(x_norm_cross, context)
        x = x + cross_out

        # === FFN ===
        x_norm_ffn = modulate_fp32(self.norm_ffn, x.view(B, T, -1, C), shift_mlp, scale_mlp)
        x_norm_ffn = x_norm_ffn.view(B, N, C)

        ffn_out = self.ffn(x_norm_ffn)

        # Residual with gating (CRITICAL: FP32 like original, then cast back)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + (gate_mlp * ffn_out.view(B, T, -1, C)).view(B, N, C)
        x = x.to(x_orig_dtype)

        return x


# ============================================================================
# Final Layer
# ============================================================================

class FinalLayer(nn.Module):
    """
    Final output projection with AdaLN modulation.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        adaln_tembed_dim: int,
        patch_size: tuple[int, int, int],
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # AdaLN for final layer (2 parameters: scale and shift)
        self.adaln_linear = ReplicatedLinear(
            adaln_tembed_dim,
            2 * hidden_size,
            bias=True,
            params_dtype=dtype,
        )
        self.adaln_act = nn.SiLU()

        # CRITICAL: Use LayerNorm not RMSNorm! (matches original)
        self.norm = FP32LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        # Output projection
        num_patch = patch_size[0] * patch_size[1] * patch_size[2]
        self.proj = ReplicatedLinear(
            hidden_size,
            num_patch * out_channels,
            bias=True,
            params_dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, N, C]
        t: torch.Tensor,  # [B, T, C_t]
        latent_shape: tuple,
    ) -> torch.Tensor:
        """
        Returns: [B, N, out_channels * patch_size^3]
        """
        B, N, C = x.shape
        T, _, _ = latent_shape

        # AdaLN modulation (FP32 for stability like original)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            t_mod = self.adaln_act(t)
            mod_params, _ = self.adaln_linear(t_mod)
            # Ensure FP32 output (needed when LoRA is applied)
            if mod_params.dtype != torch.float32:
                mod_params = mod_params.float()
            shift, scale = mod_params.unsqueeze(2).chunk(2, dim=-1)

        # Modulate
        x = modulate_fp32(self.norm, x.view(B, T, -1, C), shift, scale)
        x = x.reshape(B, N, C)

        # Project
        x, _ = self.proj(x)

        return x


# ============================================================================
# Main Model
# ============================================================================

class LongCatTransformer3DModel(CachableDiT):
    """
    Native LongCat Video Transformer using FastVideo layers.

    This is a Phase 2 implementation that replaces third_party dependencies.
    """

    # FSDP sharding: shard at each transformer block
    _fsdp_shard_conditions = [
        lambda n, m: "blocks" in n and n.split(".")[-1].isdigit(),
    ]
    # torch.compile optimization: compile each transformer block for speedup
    _compile_conditions = [
        lambda n, m: "blocks" in n and n.split(".")[-1].isdigit(),
    ]

    # Parameter name mapping (for weight conversion)
    param_names_mapping = {}  # Will be defined in config
    reverse_param_names_mapping = {}
    lora_param_names_mapping = {}

    # Supported attention backends
    _supported_attention_backends = (
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    )

    def __init__(self, config: LongCatVideoConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)

        # Extract architecture parameters
        self.hidden_size = config.hidden_size  # 4096
        self.num_attention_heads = config.num_attention_heads  # 32
        self.depth = config.depth  # 48
        self.mlp_ratio = config.mlp_ratio  # 4
        self.in_channels = config.in_channels  # 16
        self.out_channels = config.out_channels  # 16
        self.num_channels_latents = config.in_channels
        self.patch_size = config.patch_size  # [1, 2, 2]

        # Embeddings
        self.patch_embed = PatchEmbed3D(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
        )

        self.time_embedder = TimestepEmbedder(
            frequency_embedding_size=config.frequency_embedding_size,
            adaln_tembed_dim=config.adaln_tembed_dim,
        )

        self.caption_embedder = CaptionEmbedder(
            caption_channels=config.caption_channels,
            hidden_size=self.hidden_size,
            text_tokens_zero_pad=getattr(config, 'text_tokens_zero_pad', True),
        )

        # Transformer blocks (48 blocks)
        self.blocks = nn.ModuleList([
            LongCatTransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                mlp_ratio=self.mlp_ratio,
                adaln_tembed_dim=config.adaln_tembed_dim,
                config=config,
            )
            for _ in range(self.depth)
        ])

        # Output projection
        self.final_layer = FinalLayer(
            hidden_size=self.hidden_size,
            out_channels=self.out_channels,
            adaln_tembed_dim=config.adaln_tembed_dim,
            patch_size=self.patch_size,
        )

    def enable_bsa(self):
        """Enable BSA for all self-attention layers."""
        for block in self.blocks:
            block.self_attn.enable_bsa = True

    def disable_bsa(self):
        """Disable BSA for all self-attention layers."""
        for block in self.blocks:
            block.self_attn.enable_bsa = False

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, C, T, H, W]
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],  # [B, N_text, C_text]
        timestep: torch.LongTensor,  # [B] or [B, T]
        encoder_attention_mask: torch.Tensor | None = None,  # [B, N_text]
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: float | None = None,  # Unused, for API compatibility
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with FastVideo parameter ordering.

        NOTE: This follows FastVideo convention:
              (hidden_states, encoder_hidden_states, timestep)
        """
        B, _, T, H, W = hidden_states.shape

        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        # Handle list of encoder outputs (take first one)
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        # 1. Patch embedding
        x = self.patch_embed(hidden_states)  # [B, N, C]

         # 2. Timestep embedding
        # Expand timestep from [B] to [B, T] if needed
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)  # [B, T]

        t = self.time_embedder(timestep.flatten(), latent_shape=(N_t, N_h, N_w))
        if t.ndim == 2:
            t = t.reshape(B, N_t, -1)  # [B, T, C_t]

        # 3. Caption embedding (standard format, no compaction)
        context = self.caption_embedder(
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )  # [B, N_text, C]

        # 4. Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(
                x, context, t,
                latent_shape=(N_t, N_h, N_w)
            )

        # 5. Output projection
        output = self.final_layer(x, t, latent_shape=(N_t, N_h, N_w))

        # Reshape to [B, C_out, T, H, W]
        output = self.unpatchify(output, N_t, N_h, N_w)

        # Cast to float32 for better accuracy (as per original)
        output = output.to(torch.float32)

        return output

    def unpatchify(self, x: torch.Tensor, N_t: int, N_h: int, N_w: int) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] where C = T_p * H_p * W_p * C_out
        Returns:
            [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

EntryClass = LongCatTransformer3DModel
