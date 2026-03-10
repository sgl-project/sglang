# SPDX-License-Identifier: Apache-2.0
# Adapted from Helios diffusers transformer:
# https://github.com/BestWishYsh/Helios
"""
Helios Transformer 3D model for video generation.

Implements the HeliosTransformer3DModel with multi-term memory patches,
3D rotary position embeddings, and per-block scale-shift modulation.
"""

import math
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.helios import HeliosConfig
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    RMSNorm,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    ModulateProjection,
    PatchEmbed,
    TimestepEmbedder,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def pad_for_3d_conv(x, kernel_size):
    """Pad input to make it divisible by kernel_size using replicate mode."""
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x, kernel_size):
    """Average pooling for 3D downsampling."""
    return F.avg_pool3d(x, kernel_size, stride=kernel_size)


def apply_rotary_emb_transposed(hidden_states, freqs_cis):
    """Apply rotary positional embeddings with transposed cos/sin format."""
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
    out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
    return out.type_as(hidden_states)


# ---------------------------------------------------------------------------
# Output norm
# ---------------------------------------------------------------------------


class HeliosOutputNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, hidden_states, temb, original_context_length):
        temb = temb[:, -original_context_length:, :]
        shift, scale = (
            self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
        ).chunk(2, dim=2)
        shift = shift.squeeze(2).to(hidden_states.device)
        scale = scale.squeeze(2).to(hidden_states.device)
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = (
            self.norm(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (3D)
# ---------------------------------------------------------------------------


class HeliosRotaryPosEmbed(nn.Module):
    """3D rotary position embeddings for (time, height, width)."""

    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta
        # Store as plain attributes (not buffers) to avoid meta-device issues
        # during FSDP loading. They'll be re-created on the correct device in forward.
        self._freqs_base_t = None
        self._freqs_base_y = None
        self._freqs_base_x = None

    def _get_freqs_base(self, dim):
        return 1.0 / (
            self.theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim)
        )

    def _ensure_freqs_base(self, device):
        """Lazily create frequency bases on the correct device."""
        if self._freqs_base_t is None or self._freqs_base_t.device != device:
            self._freqs_base_t = self._get_freqs_base(self.DT).to(device)
            self._freqs_base_y = self._get_freqs_base(self.DY).to(device)
            self._freqs_base_x = self._get_freqs_base(self.DX).to(device)

    @torch.no_grad()
    def get_frequency_batched(self, freqs_base, pos):
        freqs = torch.einsum("d,bthw->dbthw", freqs_base, pos)
        freqs = freqs.repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    @lru_cache(maxsize=32)
    def _get_spatial_meshgrid(self, height, width, device_str):
        device = torch.device(device_str)
        grid_y_coords = torch.arange(height, device=device, dtype=torch.float32)
        grid_x_coords = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y_coords, grid_x_coords, indexing="ij")
        return grid_y, grid_x

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device):
        self._ensure_freqs_base(device)
        batch_size = frame_indices.shape[0]
        num_frames = frame_indices.shape[1]

        frame_indices = frame_indices.to(device=device, dtype=torch.float32)
        grid_y, grid_x = self._get_spatial_meshgrid(height, width, str(device))

        grid_t = frame_indices[:, :, None, None].expand(
            batch_size, num_frames, height, width
        )
        grid_y_batch = grid_y[None, None, :, :].expand(batch_size, num_frames, -1, -1)
        grid_x_batch = grid_x[None, None, :, :].expand(batch_size, num_frames, -1, -1)

        freqs_cos_t, freqs_sin_t = self.get_frequency_batched(
            self._freqs_base_t, grid_t
        )
        freqs_cos_y, freqs_sin_y = self.get_frequency_batched(
            self._freqs_base_y, grid_y_batch
        )
        freqs_cos_x, freqs_sin_x = self.get_frequency_batched(
            self._freqs_base_x, grid_x_batch
        )

        result = torch.cat(
            [
                freqs_cos_t,
                freqs_cos_y,
                freqs_cos_x,
                freqs_sin_t,
                freqs_sin_y,
                freqs_sin_x,
            ],
            dim=0,
        )
        return result.permute(1, 0, 2, 3, 4)


# ---------------------------------------------------------------------------
# Condition Embedder
# ---------------------------------------------------------------------------


class HeliosTimeTextEmbedding(nn.Module):
    """Condition embedder combining timestep and text embeddings."""

    def __init__(self, dim, time_freq_dim, time_proj_dim, text_embed_dim):
        super().__init__()
        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu"
        )
        self.time_modulation = ModulateProjection(dim, factor=6, act_layer="silu")
        self.text_embedder = MLP(
            text_embed_dim, dim, dim, bias=True, act_type="gelu_pytorch_tanh"
        )

    def forward(
        self, timestep, encoder_hidden_states, is_return_encoder_hidden_states=True
    ):
        temb = self.time_embedder(timestep)
        timestep_proj = self.time_modulation(temb)

        if encoder_hidden_states is not None and is_return_encoder_hidden_states:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


# ---------------------------------------------------------------------------
# Self-Attention for Helios
# ---------------------------------------------------------------------------


class HeliosSelfAttention(nn.Module):
    """Self-attention with RMSNorm Q/K, optional history key amplification."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_heads, tp_size)

        self.to_q = ColumnParallelLinear(
            dim, dim, bias=True, gather_output=False, quant_config=quant_config
        )
        self.to_k = ColumnParallelLinear(
            dim, dim, bias=True, gather_output=False, quant_config=quant_config
        )
        self.to_v = ColumnParallelLinear(
            dim, dim, bias=True, gather_output=False, quant_config=quant_config
        )
        self.to_out = RowParallelLinear(
            dim, dim, bias=True, reduce_results=True, quant_config=quant_config
        )
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.tp_rmsnorm = tp_size > 1

        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            is_cross_attention=False,
        )

        self.is_amplify_history = is_amplify_history
        if is_amplify_history:
            if history_scale_mode == "scalar":
                self.history_key_scale = nn.Parameter(torch.ones(1))
            elif history_scale_mode == "per_head":
                self.history_key_scale = nn.Parameter(torch.ones(num_heads))
            else:
                raise ValueError(f"Unknown history_scale_mode: {history_scale_mode}")
            self.history_scale_mode = history_scale_mode
            self.max_scale = 10.0

    def forward(self, hidden_states, rotary_emb=None, original_context_length=None):
        q, _ = self.to_q(hidden_states)
        k, _ = self.to_k(hidden_states)
        v, _ = self.to_v(hidden_states)

        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            q = self.norm_q(q)
            k = self.norm_k(k)

        q = q.unflatten(2, (self.local_num_heads, self.head_dim))
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        if rotary_emb is not None:
            q = apply_rotary_emb_transposed(q, rotary_emb)
            k = apply_rotary_emb_transposed(k, rotary_emb)

        if self.is_amplify_history and original_context_length is not None:
            history_seq_len = hidden_states.shape[1] - original_context_length
            if history_seq_len > 0:
                scale_key = 1.0 + torch.sigmoid(self.history_key_scale) * (
                    self.max_scale - 1.0
                )
                if self.history_scale_mode == "per_head":
                    scale_key = scale_key.view(1, 1, -1, 1)
                k = torch.cat(
                    [k[:, :history_seq_len] * scale_key, k[:, history_seq_len:]],
                    dim=1,
                )

        x = self.attn(q, k, v)
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x


# ---------------------------------------------------------------------------
# Cross-Attention for Helios
# ---------------------------------------------------------------------------


class HeliosCrossAttention(nn.Module):
    """Cross-attention with RMSNorm Q/K normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_heads, tp_size)

        self.to_q = ColumnParallelLinear(
            dim, dim, bias=True, gather_output=False, quant_config=quant_config
        )
        self.to_k = ColumnParallelLinear(
            dim, dim, bias=True, gather_output=False, quant_config=quant_config
        )
        self.to_v = ColumnParallelLinear(
            dim, dim, bias=True, gather_output=False, quant_config=quant_config
        )
        self.to_out = RowParallelLinear(
            dim, dim, bias=True, reduce_results=True, quant_config=quant_config
        )
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.tp_rmsnorm = tp_size > 1

        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            is_cross_attention=True,
        )

    def forward(self, hidden_states, encoder_hidden_states):
        q, _ = self.to_q(hidden_states)
        k, _ = self.to_k(encoder_hidden_states)
        v, _ = self.to_v(encoder_hidden_states)

        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            q = self.norm_q(q)
            k = self.norm_k(k)

        q = q.unflatten(2, (self.local_num_heads, self.head_dim))
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        x = self.attn(q, k, v)
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class HeliosTransformerBlock(nn.Module):
    """
    Single transformer block with self-attention, cross-attention, FFN,
    and scale-shift modulation from timestep embeddings.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        guidance_cross_attn: bool = True,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = HeliosSelfAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            is_amplify_history=is_amplify_history,
            history_scale_mode=history_scale_mode,
            quant_config=quant_config,
        )

        # 2. Cross-attention
        self.attn2 = HeliosCrossAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            quant_config=quant_config,
        )
        self.self_attn_residual_norm = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = MLP(
            dim, ffn_dim, act_type="gelu_pytorch_tanh", quant_config=quant_config
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # 4. Guidance cross-attention flag
        self.guidance_cross_attn = guidance_cross_attn

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        original_context_length=None,
    ):
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states, rotary_emb, original_context_length
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        if self.guidance_cross_attn:
            history_seq_len = hidden_states.shape[1] - original_context_length
            history_hidden_states, current_hidden_states = torch.split(
                hidden_states, [history_seq_len, original_context_length], dim=1
            )
            norm_hidden_states = self.self_attn_residual_norm(
                current_hidden_states.float()
            ).type_as(current_hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
            current_hidden_states = current_hidden_states + attn_output
            hidden_states = torch.cat(
                [history_hidden_states, current_hidden_states], dim=1
            )
        else:
            norm_hidden_states = self.self_attn_residual_norm(
                hidden_states.float()
            ).type_as(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class HeliosTransformer3DModel(CachableDiT, OffloadableDiTMixin):
    """
    Helios Transformer 3D model for video generation.

    Implements multi-scale history patches, 3D RoPE, and chunked denoising
    with zero_history_timestep and guidance_cross_attn.
    """

    _fsdp_shard_conditions = HeliosConfig()._fsdp_shard_conditions
    _compile_conditions = HeliosConfig()._compile_conditions
    _supported_attention_backends = HeliosConfig()._supported_attention_backends
    param_names_mapping = HeliosConfig().param_names_mapping
    reverse_param_names_mapping = HeliosConfig().reverse_param_names_mapping
    lora_param_names_mapping = HeliosConfig().lora_param_names_mapping

    def __init__(
        self,
        config: HeliosConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.inner_dim = inner_dim

        # Helios-specific config
        self.zero_history_timestep = config.zero_history_timestep
        self.has_multi_term_memory_patch = config.has_multi_term_memory_patch
        self.guidance_cross_attn = config.guidance_cross_attn

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Rotary position embeddings
        self.rope = HeliosRotaryPosEmbed(
            rope_dim=config.rope_dim, theta=config.rope_theta
        )

        # 3. Multi-term memory patches
        if self.has_multi_term_memory_patch:
            self.patch_short = nn.Conv3d(
                config.in_channels,
                inner_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            )
            self.patch_mid = nn.Conv3d(
                config.in_channels,
                inner_dim,
                kernel_size=tuple(2 * p for p in config.patch_size),
                stride=tuple(2 * p for p in config.patch_size),
            )
            self.patch_long = nn.Conv3d(
                config.in_channels,
                inner_dim,
                kernel_size=tuple(4 * p for p in config.patch_size),
                stride=tuple(4 * p for p in config.patch_size),
            )

        # 4. Condition embeddings
        self.condition_embedder = HeliosTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=config.text_dim,
        )

        # 5. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                HeliosTransformerBlock(
                    dim=inner_dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_attention_heads,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                    guidance_cross_attn=config.guidance_cross_attn,
                    is_amplify_history=config.is_amplify_history,
                    history_scale_mode=config.history_scale_mode,
                    quant_config=quant_config,
                )
                for _ in range(config.num_layers)
            ]
        )

        # 6. Output norm & projection
        self.norm_out = HeliosOutputNorm(inner_dim, config.eps)
        self.proj_out = ColumnParallelLinear(
            inner_dim,
            config.out_channels * math.prod(config.patch_size),
            bias=True,
            gather_output=True,
            quant_config=quant_config,
        )

        self.cnt = 0
        self.__post_init__()
        self.layer_names = ["blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        # Stage 1 history inputs
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        **kwargs,
    ) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.patch_size

        # 1. Patch embed the noisy latents
        hidden_states = self.patch_embedding(hidden_states)
        _, _, post_patch_num_frames, post_patch_height, post_patch_width = (
            hidden_states.shape
        )

        if indices_hidden_states is None:
            indices_hidden_states = (
                torch.arange(0, post_patch_num_frames)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 2. Compute rotary embeddings
        rotary_emb = self.rope(
            frame_indices=indices_hidden_states,
            height=post_patch_height,
            width=post_patch_width,
            device=hidden_states.device,
        )
        rotary_emb = rotary_emb.flatten(2).transpose(1, 2)
        original_context_length = hidden_states.shape[1]

        # 3. Process short history
        if (
            latents_history_short is not None
            and indices_latents_history_short is not None
        ):
            latents_history_short = latents_history_short.to(hidden_states)
            latents_history_short = self.patch_short(latents_history_short)
            _, _, _, H1, W1 = latents_history_short.shape
            latents_history_short = latents_history_short.flatten(2).transpose(1, 2)

            rotary_emb_history_short = self.rope(
                frame_indices=indices_latents_history_short,
                height=H1,
                width=W1,
                device=latents_history_short.device,
            )
            rotary_emb_history_short = rotary_emb_history_short.flatten(2).transpose(
                1, 2
            )
            hidden_states = torch.cat([latents_history_short, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_short, rotary_emb], dim=1)

        # 4. Process mid history
        if latents_history_mid is not None and indices_latents_history_mid is not None:
            latents_history_mid = latents_history_mid.to(hidden_states)
            latents_history_mid = pad_for_3d_conv(latents_history_mid, (2, 4, 4))
            latents_history_mid = self.patch_mid(latents_history_mid)
            latents_history_mid = latents_history_mid.flatten(2).transpose(1, 2)

            rotary_emb_history_mid = self.rope(
                frame_indices=indices_latents_history_mid,
                height=H1,
                width=W1,
                device=latents_history_mid.device,
            )
            rotary_emb_history_mid = pad_for_3d_conv(rotary_emb_history_mid, (2, 2, 2))
            rotary_emb_history_mid = center_down_sample_3d(
                rotary_emb_history_mid, (2, 2, 2)
            )
            rotary_emb_history_mid = rotary_emb_history_mid.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_mid, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_mid, rotary_emb], dim=1)

        # 5. Process long history
        if (
            latents_history_long is not None
            and indices_latents_history_long is not None
        ):
            latents_history_long = latents_history_long.to(hidden_states)
            latents_history_long = pad_for_3d_conv(latents_history_long, (4, 8, 8))
            latents_history_long = self.patch_long(latents_history_long)
            latents_history_long = latents_history_long.flatten(2).transpose(1, 2)

            rotary_emb_history_long = self.rope(
                frame_indices=indices_latents_history_long,
                height=H1,
                width=W1,
                device=latents_history_long.device,
            )
            rotary_emb_history_long = pad_for_3d_conv(
                rotary_emb_history_long, (4, 4, 4)
            )
            rotary_emb_history_long = center_down_sample_3d(
                rotary_emb_history_long, (4, 4, 4)
            )
            rotary_emb_history_long = rotary_emb_history_long.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_long, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_long, rotary_emb], dim=1)

        history_context_length = hidden_states.shape[1] - original_context_length

        # 6. Compute condition embeddings
        if indices_hidden_states is not None and self.zero_history_timestep:
            timestep_t0 = torch.zeros(
                (1,), dtype=timestep.dtype, device=timestep.device
            )
            temb_t0, timestep_proj_t0, _ = self.condition_embedder(
                timestep_t0,
                encoder_hidden_states,
                is_return_encoder_hidden_states=False,
            )
            temb_t0 = temb_t0.unsqueeze(1).expand(
                batch_size, history_context_length, -1
            )
            timestep_proj_t0 = (
                timestep_proj_t0.unflatten(-1, (6, -1))
                .view(1, 6, 1, -1)
                .expand(batch_size, -1, history_context_length, -1)
            )

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states
        )
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))

        if indices_hidden_states is not None and not self.zero_history_timestep:
            main_repeat_size = hidden_states.shape[1]
        else:
            main_repeat_size = original_context_length
        temb = temb.view(batch_size, 1, -1).expand(batch_size, main_repeat_size, -1)
        timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(
            batch_size, 6, main_repeat_size, -1
        )

        if indices_hidden_states is not None and self.zero_history_timestep:
            temb = torch.cat([temb_t0, temb], dim=1)
            timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)

        if timestep_proj.ndim == 4:
            timestep_proj = timestep_proj.permute(0, 2, 1, 3)

        # 7. Transformer blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                original_context_length,
            )

        self.cnt += 1

        # 8. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb, original_context_length)
        hidden_states, _ = self.proj_out(hidden_states)

        # 9. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output


EntryClass = HeliosTransformer3DModel
