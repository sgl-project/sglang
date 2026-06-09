# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Omni transformer.

Dual-pathway DiT: an Understanding (UND) pathway runs causal self-attention
over the text tokens once and caches its K/V; a Generation (GEN) pathway
cross-attends from noisy visual tokens to that cache at every denoising step.
"""

import math
from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.cosmos3video import Cosmos3VideoConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_world_size,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm,
    apply_qk_norm_rope,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    Qwen3VLTextRotaryEmbedding,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import timestep_embedding
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils import add_prefix

logger = init_logger(__name__)


# -----------------------------------------------------------------------------
# mRoPE position ID computation (Qwen3VL-style)
# -----------------------------------------------------------------------------


def compute_mrope_position_ids_text(
    num_tokens: int,
    temporal_offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Generate 3D mRoPE position IDs for text tokens.

    Text tokens: all three axes (T, H, W) share the same monotonically
    increasing position IDs: (0,0,0), (1,1,1), (2,2,2), ...

    Returns:
        (position_ids [3, num_tokens], next_temporal_offset)
    """
    ids = torch.arange(num_tokens, dtype=torch.long, device=device) + temporal_offset
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def compute_mrope_position_ids_vision(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    device: torch.device,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for vision tokens.

    Creates a (T, H, W) position grid. Spatial indices reset to 0
    per vision segment (Qwen3VL-style).
    Flattened in T-major order.

    When the token rate (``temporal_compression_factor``) differs from the
    base-fps rate (``base_temporal_compression_factor``), the two no longer
    cancel: action tokens run at frame rate (factor 1) while their temporal
    positions are scaled by the video factor. ``base_temporal_compression_factor``
    defaults to ``temporal_compression_factor`` so vision/sound are unchanged.

    Returns:
        (position_ids [3, grid_t * grid_h * grid_w], next_temporal_offset)
    """
    fps_modulation = fps is not None and grid_t > 1

    if fps_modulation:
        tps = fps / temporal_compression_factor
        effective_base_tcf = (
            base_temporal_compression_factor
            if base_temporal_compression_factor is not None
            else temporal_compression_factor
        )
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32, device=device)
        t_index = (
            ((frame_indices + start_frame_offset) / tps * base_tps + temporal_offset)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
        )
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long, device=device)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = (
        torch.arange(grid_h, dtype=torch.long, device=device)
        .view(1, -1, 1)
        .expand(grid_t, -1, grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(grid_w, dtype=torch.long, device=device)
        .view(1, 1, -1)
        .expand(grid_t, grid_h, -1)
        .flatten()
    )

    if fps_modulation:
        mrope_ids = torch.stack(
            [t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0
        )
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_offset = math.ceil(mrope_ids.max().item()) + 1
    return mrope_ids, next_offset


def compute_mrope_position_ids_sound(
    grid_t: int,
    temporal_offset: int | float,
    sound_latent_fps: float,
    device: torch.device,
    base_fps: float = 24.0,
    temporal_compression_factor_sound: int = 1,
) -> tuple[torch.Tensor, int | float]:
    """mRoPE position IDs for sound tokens: a (T, 1, 1) grid."""
    return compute_mrope_position_ids_vision(
        grid_t=grid_t,
        grid_h=1,
        grid_w=1,
        temporal_offset=temporal_offset,
        device=device,
        fps=sound_latent_fps,
        base_fps=base_fps,
        temporal_compression_factor=temporal_compression_factor_sound,
    )


def compute_mrope_position_ids_action(
    grid_t: int,
    temporal_offset: int | float,
    action_fps: float | None,
    device: torch.device,
    base_fps: float = 24.0,
    base_temporal_compression_factor: int = 4,
    start_frame_offset: int = 1,
) -> tuple[torch.Tensor, int | float]:
    """mRoPE position IDs for action tokens: a (T, 1, 1) grid.

    Action tokens run at frame rate (``temporal_compression_factor=1``) but
    their positions are scaled by the video's ``base_temporal_compression_factor``
    so they share the video's temporal coordinate frame. ``start_frame_offset=1``
    (default) shifts them one frame ahead so they align with the video frame
    they condition on.
    """
    return compute_mrope_position_ids_vision(
        grid_t=grid_t,
        grid_h=1,
        grid_w=1,
        temporal_offset=temporal_offset,
        device=device,
        fps=action_fps,
        base_fps=base_fps,
        temporal_compression_factor=1,
        base_temporal_compression_factor=base_temporal_compression_factor,
        start_frame_offset=start_frame_offset,
    )


# -----------------------------------------------------------------------------
# Qwen3-style RoPE functions
# -----------------------------------------------------------------------------


def _apply_qwen3_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    cos_sin_cache: torch.Tensor,
    rope_cache_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_qk_norm_rope(
        q=q.contiguous(),
        k=k.contiguous(),
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
        positions=rope_cache_positions,
    )


def _apply_qwen3_rope_from_cache(
    q: torch.Tensor, k: torch.Tensor, cos_sin_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = q.shape[:2]
    half = q.shape[-1] // 2
    cos = cos_sin_cache[:, :half].view(batch_size, seq_len, 1, half).to(q.dtype)
    sin = cos_sin_cache[:, half:].view(batch_size, seq_len, 1, half).to(q.dtype)

    q1 = q[..., :half]
    q2 = q[..., half:]
    q_out = torch.empty_like(q)
    q_out[..., :half] = q1 * cos - q2 * sin
    q_out[..., half:] = q2 * cos + q1 * sin

    k1 = k[..., :half]
    k2 = k[..., half:]
    k_out = torch.empty_like(k)
    k_out[..., :half] = k1 * cos - k2 * sin
    k_out[..., half:] = k2 * cos + k1 * sin
    return q_out, k_out


def _apply_qwen3_qk_norm_rope_split(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k = apply_qk_norm(q.contiguous(), k.contiguous(), q_norm, k_norm, head_dim)
    return _apply_qwen3_rope_from_cache(q, k, cos_sin_cache)


# -----------------------------------------------------------------------------
# Action domain-aware projection
# -----------------------------------------------------------------------------


class DomainAwareLinear(nn.Module):
    """Per-domain linear projection for action conditioning.

    Maintains one weight matrix and bias per embodiment domain via embedding
    tables, enabling multi-domain robot action generation from a shared
    backbone.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(
            self.fc.weight.view(num_domains, output_size, input_size)
        )
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if domain_id.ndim == 0:
            domain_id = domain_id.unsqueeze(0)
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        weight = self.fc(domain_id).view(
            domain_id.shape[0], self.input_size, self.output_size
        )
        bias = self.bias(domain_id).view(domain_id.shape[0], self.output_size)
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        return torch.bmm(x, weight) + bias.unsqueeze(1)


# -----------------------------------------------------------------------------
# Cosmos3 Timestep Embedder
# -----------------------------------------------------------------------------


class Cosmos3TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Uses ReplicatedLinear for consistency with other SGLang models and
    to support quantization (though timestep embedders are typically excluded).
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        timestep_scale: float = 0.001,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.timestep_scale = timestep_scale
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        self.max_period = max_period

        # Use ReplicatedLinear for consistency (typically excluded from quantization)
        self.linear_1 = ReplicatedLinear(
            frequency_embedding_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_1", prefix),
        )
        self.act = nn.SiLU()
        self.linear_2 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_2", prefix),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: [B] timestep values

        Returns:
            [B, hidden_size] timestep embeddings
        """
        # Scale timestep
        t_scaled = t * self.timestep_scale

        # Compute sinusoidal embeddings in fp32
        t_freq = timestep_embedding(
            t_scaled,
            self.frequency_embedding_size,
            self.max_period,
            dtype=torch.float32,
        )

        # Project through MLP
        # When fp8-quantized, weight.dtype is float8_e4m3fn — keep input in
        # float32 (the quant kernel handles input quantization internally).
        w_dtype = self.linear_1.weight.dtype
        if w_dtype.is_floating_point and w_dtype.itemsize >= 2:
            x = t_freq.to(w_dtype)
        else:
            x = t_freq  # already float32 from timestep_embedding
        x, _ = self.linear_1(x)
        x = self.act(x)
        x, _ = self.linear_2(x)
        return x


# -----------------------------------------------------------------------------
# Cosmos3 Gated MLP
# -----------------------------------------------------------------------------


class Cosmos3GatedMLP(nn.Module):
    """Gated MLP (SwiGLU-style) for Cosmos3."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate_up))
        return out


# -----------------------------------------------------------------------------
# Cosmos3 UND Causal Attention
# -----------------------------------------------------------------------------


class Cosmos3CausalAttention(nn.Module):
    """Understanding pathway: causal self-attention on text tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.tp_size = get_tp_world_size()
        if num_attention_heads % self.tp_size != 0:
            raise ValueError(
                "Cosmos3CausalAttention requires num_attention_heads divisible "
                f"by tp_size, got {num_attention_heads=} {self.tp_size=}."
            )
        if num_key_value_heads % self.tp_size != 0:
            raise ValueError(
                "Cosmos3CausalAttention requires num_key_value_heads divisible "
                f"by tp_size, got {num_key_value_heads=} {self.tp_size=}."
            )
        self.local_num_attention_heads = num_attention_heads // self.tp_size
        self.local_num_key_value_heads = num_key_value_heads // self.tp_size

        self.q_size = num_attention_heads * head_dim
        self.kv_size = num_key_value_heads * head_dim
        self.to_qkv = MergedColumnParallelLinear(
            hidden_size,
            [self.q_size, self.kv_size, self.kv_size],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_qkv", prefix),
        )
        self.to_out = RowParallelLinear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("to_out", prefix),
        )

        # Per-head QK norm.
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rope_cache_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with KV cache return.

        Returns:
            (output, K, V) where K/V are TP-local, post-norm, post-RoPE
        """
        batch_size, seq_len = hidden_states.shape[:2]

        qkv, _ = self.to_qkv(hidden_states)
        qkv = qkv.view(
            batch_size,
            seq_len,
            self.local_num_attention_heads + 2 * self.local_num_key_value_heads,
            self.head_dim,
        )
        q = qkv[:, :, : self.local_num_attention_heads, :]
        k = qkv[
            :,
            :,
            self.local_num_attention_heads : self.local_num_attention_heads
            + self.local_num_key_value_heads,
            :,
        ]
        v = qkv[
            :,
            :,
            self.local_num_attention_heads + self.local_num_key_value_heads :,
            :,
        ]

        q = F.rms_norm(
            q, (self.head_dim,), self.norm_q.weight, self.norm_q.variance_epsilon
        )
        k = F.rms_norm(
            k, (self.head_dim,), self.norm_k.weight, self.norm_k.variance_epsilon
        )
        q, k = _apply_qwen3_rope_from_cache(q, k, cos_sin_cache)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        out, _ = self.to_out(out)
        return out, k, v


# -----------------------------------------------------------------------------
# Cosmos3 GEN Cross Attention
# -----------------------------------------------------------------------------


class Cosmos3CrossAttention(nn.Module):
    """Generation pathway: cross-attention where visual Q attends to all K/V."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
        supported_attention_backends: set | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.tp_size = get_tp_world_size()
        if num_attention_heads % self.tp_size != 0:
            raise ValueError(
                "Cosmos3CrossAttention requires num_attention_heads divisible "
                f"by tp_size, got {num_attention_heads=} {self.tp_size=}."
            )
        if num_key_value_heads % self.tp_size != 0:
            raise ValueError(
                "Cosmos3CrossAttention requires num_key_value_heads divisible "
                f"by tp_size, got {num_key_value_heads=} {self.tp_size=}."
            )
        self.local_num_attention_heads = num_attention_heads // self.tp_size
        self.local_num_key_value_heads = num_key_value_heads // self.tp_size

        self.q_size = num_attention_heads * head_dim
        self.kv_size = num_key_value_heads * head_dim
        self.to_qkv = MergedColumnParallelLinear(
            hidden_size,
            [self.q_size, self.kv_size, self.kv_size],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_qkv", prefix),
        )
        self.to_out = RowParallelLinear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("to_out", prefix),
        )

        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        self.attn = USPAttention(
            num_heads=self.local_num_attention_heads,
            head_size=head_dim,
            num_kv_heads=self.local_num_key_value_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rope_cache_positions: torch.Tensor,
        use_fused_qk_norm_rope: bool,
    ) -> torch.Tensor:
        """Cross-attention from GEN to cached UND K/V.

        Args:
            hidden_states: [B, S_gen_local, hidden_size] visual tokens (may be sharded)
            k_und: [B, S_und, H_kv_local, D] UND keys (replicated over SP)
            v_und: [B, S_und, H_kv_local, D] UND values (replicated over SP)
            cos_sin_cache: [B*S_gen_local, D] local rows of [cos, sin]
            rope_cache_positions: identity row positions into cos_sin_cache
        """
        batch_size, seq_len_gen = hidden_states.shape[:2]

        qkv, _ = self.to_qkv(hidden_states)
        qkv = qkv.view(
            batch_size,
            seq_len_gen,
            self.local_num_attention_heads + 2 * self.local_num_key_value_heads,
            self.head_dim,
        )
        q = qkv[:, :, : self.local_num_attention_heads, :]
        k = qkv[
            :,
            :,
            self.local_num_attention_heads : self.local_num_attention_heads
            + self.local_num_key_value_heads,
            :,
        ]
        v = qkv[
            :,
            :,
            self.local_num_attention_heads + self.local_num_key_value_heads :,
            :,
        ]

        if use_fused_qk_norm_rope:
            q, k = _apply_qwen3_qk_norm_rope(
                q,
                k,
                self.norm_q,
                self.norm_k,
                self.head_dim,
                cos_sin_cache,
                rope_cache_positions,
            )
        else:
            q, k = _apply_qwen3_qk_norm_rope_split(
                q, k, self.norm_q, self.norm_k, self.head_dim, cos_sin_cache
            )

        # K/V = [text (replicated on every SP rank) | image (sharded same as Q)].
        # USPAttention routes through the registered attention backend (FA, sage,
        # …) and handles the Ulysses all-to-all when SP > 1.
        out = self.attn.forward_with_replicated_kv_prefix(q, k_und, v_und, k, v)
        out = out.reshape(batch_size, seq_len_gen, -1)
        out, _ = self.to_out(out)
        return out


# -----------------------------------------------------------------------------
# Cosmos3 UND Decoder Layer
# -----------------------------------------------------------------------------


class Cosmos3UndDecoderLayer(nn.Module):
    """Understanding pathway decoder layer: causal self-attention + MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        layer_idx: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.self_attn = Cosmos3CausalAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            prefix=add_prefix("self_attn", prefix),
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Cosmos3GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=add_prefix("mlp", prefix),
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rope_cache_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (hidden_states, K, V) where K/V are for GEN cross-attention
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, k, v = self.self_attn(
            hidden_states, cos_sin_cache, rope_cache_positions
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, k, v


# -----------------------------------------------------------------------------
# Cosmos3 GEN Decoder Layer
# -----------------------------------------------------------------------------


class Cosmos3GenDecoderLayer(nn.Module):
    """Generation pathway decoder layer: cross-attention + MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        layer_idx: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
        supported_attention_backends: set | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.cross_attention = Cosmos3CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            prefix=add_prefix("cross_attention", prefix),
            quant_config=quant_config,
            supported_attention_backends=supported_attention_backends,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Cosmos3GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=add_prefix("mlp", prefix),
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rope_cache_positions: torch.Tensor,
        use_fused_qk_norm_rope: bool,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fused add+rmsnorm: each `(hidden_states, residual) = norm(...)`
        # collapses the residual add and RMSNorm into one kernel. The
        # caller threads `residual` across layers and resolves it before
        # the post-loop all-gather + `norm_moe_gen`.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.cross_attention(
            hidden_states,
            k_und,
            v_und,
            cos_sin_cache,
            rope_cache_positions,
            use_fused_qk_norm_rope,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# -----------------------------------------------------------------------------
# Cosmos3 Language Model (UND pathway)
# -----------------------------------------------------------------------------


class Cosmos3LanguageModel(nn.Module):
    """Understanding pathway: processes text tokens and caches K/V."""

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        vocab_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        mrope_section: tuple[int, int, int],
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            params_dtype=torch.bfloat16,
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        self.layers = nn.ModuleList(
            [
                Cosmos3UndDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    layer_idx=i,
                    prefix=f"layers.{i}",
                    quant_config=quant_config,
                )
                for i in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Process text tokens and return per-layer K/V cache.

        Args:
            text_ids: [B, S] token IDs
            text_mask: [B, S] float mask (1=real, 0=pad)
            position_ids: [3, B, S] mRoPE position IDs

        Returns:
            List of (K, V) per layer for GEN cross-attention
        """
        hidden = self.embed_tokens(text_ids)
        mask_3d = text_mask.unsqueeze(-1)
        cos_sin_cache, rope_cache_positions = self.rotary_emb.build_rope_cache_inputs(
            position_ids, cache_dtype=hidden.dtype
        )

        cached_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden = hidden * mask_3d
            hidden, k, v = layer(hidden, cos_sin_cache, rope_cache_positions)
            cached_kv.append((k, v))

        return cached_kv


# -----------------------------------------------------------------------------
# Cosmos3 Omni Transformer
# -----------------------------------------------------------------------------


class Cosmos3OmniTransformer(CachableDiT, LayerwiseOffloadableModuleMixin):
    """Cosmos3 Omni transformer.

    Dual-pathway architecture:
    - Understanding (UND): causal LM processing text
    - Generation (GEN): cross-attention from visual to UND K/V
    """

    _fsdp_shard_conditions = Cosmos3VideoConfig()._fsdp_shard_conditions
    _compile_conditions = Cosmos3VideoConfig()._compile_conditions
    _supported_attention_backends = Cosmos3VideoConfig()._supported_attention_backends
    param_names_mapping = Cosmos3VideoConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = (
        Cosmos3VideoConfig().arch_config.reverse_param_names_mapping
    )
    lora_param_names_mapping = Cosmos3VideoConfig().arch_config.lora_param_names_mapping

    def __init__(
        self,
        config: Cosmos3VideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.num_hidden_layers = arch.num_hidden_layers
        self.num_attention_heads = arch.num_attention_heads
        self.num_key_value_heads = arch.num_key_value_heads
        self.head_dim = arch.head_dim
        self.intermediate_size = arch.intermediate_size
        self.latent_patch_size = arch.latent_patch_size
        self.latent_channel = arch.latent_channel
        self.num_channels_latents = arch.out_channels
        self.patch_latent_dim = (self.latent_patch_size**2) * self.latent_channel
        self.timestep_scale = arch.timestep_scale
        self.base_fps = arch.base_fps
        self.temporal_compression_factor = arch.temporal_compression_factor
        self.temporal_margin = arch.unified_3d_mrope_temporal_modality_margin
        self.sound_latent_fps = arch.sound_latent_fps
        self.temporal_compression_factor_sound = arch.temporal_compression_factor_sound
        self.rms_norm_eps = arch.rms_norm_eps

        # Ulysses sequence parallelism. When CFG-parallel is also enabled
        # the SP group only spans ranks that share a CFG context (cond or
        # uncond), so ``sp_size`` here is the per-context shard count.
        self.sp_size = get_sp_world_size()
        self.sp_group = get_sp_group() if self.sp_size > 1 else None
        self.sp_rank = self.sp_group.rank_in_group if self.sp_group else 0
        if self.sp_size > 1:
            logger.info(
                f"Cosmos3 SP enabled: sp_size={self.sp_size}, sp_rank={self.sp_rank}"
            )

        # Language model (UND pathway)
        self.language_model = Cosmos3LanguageModel(
            hidden_size=arch.hidden_size,
            num_hidden_layers=arch.num_hidden_layers,
            num_attention_heads=arch.num_attention_heads,
            num_key_value_heads=arch.num_key_value_heads,
            head_dim=arch.head_dim,
            intermediate_size=arch.intermediate_size,
            vocab_size=arch.vocab_size,
            rms_norm_eps=arch.rms_norm_eps,
            rope_theta=arch.rope_theta,
            mrope_section=arch.mrope_section,
            quant_config=quant_config,
        )

        # Latent projection layers - ReplicatedLinear for quantization support
        self.proj_in = ReplicatedLinear(
            self.patch_latent_dim,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix="proj_in",
        )
        self.proj_out = ReplicatedLinear(
            self.hidden_size,
            self.patch_latent_dim,
            bias=True,
            quant_config=quant_config,
            prefix="proj_out",
        )

        self.sound_dim = arch.sound_dim
        self.audio_proj_in = ReplicatedLinear(
            self.sound_dim,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix="audio_proj_in",
        )
        self.audio_proj_out = ReplicatedLinear(
            self.hidden_size,
            self.sound_dim,
            bias=True,
            quant_config=quant_config,
            prefix="audio_proj_out",
        )
        self.audio_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        if arch.action_gen:
            self.action_dim = arch.action_dim
            self.num_embodiment_domains = arch.num_embodiment_domains
            self.action_proj_in = DomainAwareLinear(
                self.action_dim,
                self.hidden_size,
                self.num_embodiment_domains,
            )
            self.action_proj_out = DomainAwareLinear(
                self.hidden_size,
                self.action_dim,
                self.num_embodiment_domains,
            )
            self.action_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        # Timestep embedder
        self.time_embedder = Cosmos3TimestepEmbedder(
            hidden_size=self.hidden_size,
            frequency_embedding_size=arch.frequency_embedding_size,
            timestep_scale=arch.timestep_scale,
            prefix="time_embedder",
            quant_config=quant_config,
        )

        # Generation layers (GEN pathway)
        self.gen_layers = nn.ModuleList(
            [
                Cosmos3GenDecoderLayer(
                    hidden_size=arch.hidden_size,
                    num_attention_heads=arch.num_attention_heads,
                    num_key_value_heads=arch.num_key_value_heads,
                    head_dim=arch.head_dim,
                    intermediate_size=arch.intermediate_size,
                    rms_norm_eps=arch.rms_norm_eps,
                    layer_idx=i,
                    prefix=f"gen_layers.{i}",
                    quant_config=quant_config,
                    supported_attention_backends=arch._supported_attention_backends,
                )
                for i in range(arch.num_hidden_layers)
            ]
        )

        # Output norm
        self.norm_moe_gen = RMSNorm(self.hidden_size, eps=arch.rms_norm_eps)

        # Cached K/V from UND pathway - dict keyed by cache_key for CFG support
        # This allows maintaining separate caches for conditional and unconditional
        # prompts, avoiding recomputation on every denoising step
        self.cached_kv: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.cached_gen_rope_inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        self.__post_init__()

        self.layer_names = ["gen_layers", "language_model.layers"]

    def _pad_to_patch_size(self, H: int, W: int) -> tuple[int, int, int, int]:
        """Compute padded spatial dims aligned to patch_size."""
        p = self.latent_patch_size
        H_padded = ((H + p - 1) // p) * p
        W_padded = ((W + p - 1) // p) * p
        return H_padded // p, W_padded // p, H_padded, W_padded

    def patchify(self, latents: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """Convert latents to patches: [B, C, T, H, W] -> [B, T*Hp*Wp, p*p*C]."""
        B = latents.shape[0]
        p = self.latent_patch_size
        C = self.latent_channel
        Hp, Wp, H_padded, W_padded = self._pad_to_patch_size(H, W)

        if H_padded != H or W_padded != W:
            latents = F.pad(latents, (0, W_padded - W, 0, H_padded - H))

        x = latents.reshape(B, C, T, Hp, p, Wp, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)  # [B, T, Hp, Wp, p, p, C]
        return x.reshape(B, T * Hp * Wp, p * p * C)

    def unpatchify(self, tokens: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """Convert patches back to latents: [B, T*Hp*Wp, p*p*C] -> [B, C, T, H, W]."""
        B = tokens.shape[0]
        p = self.latent_patch_size
        C = self.latent_channel
        Hp, Wp, H_padded, W_padded = self._pad_to_patch_size(H, W)

        x = tokens.reshape(B, T, Hp, Wp, p, p, C)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)  # [B, C, T, Hp, p, Wp, p]
        x = x.reshape(B, C, T, H_padded, W_padded)

        if H_padded != H or W_padded != W:
            x = x[:, :, :, :H, :W]
        return x

    def _compute_rope_position_ids(
        self,
        text_mask: torch.Tensor,
        T: int,
        Hp: int,
        Wp: int,
        fps: float | None,
        device: torch.device,
        sound_frames: int = 0,
        action_frames: int = 0,
        action_fps: float | None = None,
        action_start_frame_offset: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mRoPE position IDs for UND text and GEN visual + action + sound tokens."""
        B = text_mask.shape[0]
        S_text = text_mask.shape[1]
        text_lengths = text_mask.sum(dim=1).long()
        effective_fps = fps if fps is not None and T > 1 else None

        text_pos_list = []
        vis_pos_list = []
        for b in range(B):
            real_len = int(text_lengths[b].item())
            t_pos, t_offset = compute_mrope_position_ids_text(
                real_len, temporal_offset=0, device=device
            )
            media_offset = t_offset + self.temporal_margin
            v_pos, _ = compute_mrope_position_ids_vision(
                T,
                Hp,
                Wp,
                temporal_offset=media_offset,
                device=device,
                fps=effective_fps,
                base_fps=self.base_fps,
                temporal_compression_factor=self.temporal_compression_factor,
            )
            if action_frames > 0:
                a_pos, _ = compute_mrope_position_ids_action(
                    action_frames,
                    temporal_offset=media_offset,
                    action_fps=action_fps,
                    device=device,
                    base_fps=self.base_fps,
                    base_temporal_compression_factor=self.temporal_compression_factor,
                    start_frame_offset=action_start_frame_offset,
                )
                pos_dtype = torch.promote_types(v_pos.dtype, a_pos.dtype)
                v_pos = torch.cat([v_pos.to(pos_dtype), a_pos.to(pos_dtype)], dim=1)
            if sound_frames > 0:
                s_pos, _ = compute_mrope_position_ids_sound(
                    sound_frames,
                    temporal_offset=media_offset,
                    sound_latent_fps=self.sound_latent_fps,
                    device=device,
                    base_fps=self.base_fps,
                    temporal_compression_factor_sound=self.temporal_compression_factor_sound,
                )
                pos_dtype = torch.promote_types(v_pos.dtype, s_pos.dtype)
                v_pos = torch.cat([v_pos.to(pos_dtype), s_pos.to(pos_dtype)], dim=1)
            if real_len < S_text:
                t_pos = torch.cat(
                    [
                        t_pos,
                        torch.zeros(
                            3, S_text - real_len, dtype=t_pos.dtype, device=device
                        ),
                    ],
                    dim=1,
                )
            text_pos_list.append(t_pos)
            vis_pos_list.append(v_pos)

        text_pos_ids = torch.stack(text_pos_list, dim=1).to(device)  # [3, B, S_text]
        vis_pos_ids = torch.stack(vis_pos_list, dim=1).to(device)  # [3, B, S_gen]

        return text_pos_ids, vis_pos_ids

    def reset_cache(self, cache_key: str | None = None):
        """Reset cached K/V from UND pathway.

        Args:
            cache_key: If provided, reset only the specified cache key.
                      If None, reset all caches.
        """
        if cache_key is None:
            # Reset all caches
            self.cached_kv = {}
            self.cached_gen_rope_inputs = {}
        else:
            # Reset specific cache
            if cache_key in self.cached_kv:
                del self.cached_kv[cache_key]
            if cache_key in self.cached_gen_rope_inputs:
                del self.cached_gen_rope_inputs[cache_key]

    def _ensure_cache_dicts(self):
        """Ensure cache dictionaries exist (for backwards compatibility)."""
        if not isinstance(self.cached_kv, dict):
            self.cached_kv = {}
        if not isinstance(self.cached_gen_rope_inputs, dict):
            self.cached_gen_rope_inputs = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        text_ids: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        fps: float | None = None,
        cache_key: str = "default",
        noisy_frame_mask: torch.Tensor | None = None,
        max_text_seq_len: int | None = None,
        sound_latents: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        action_domain_ids: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        action_fps: float | None = None,
        action_start_frame_offset: int = 1,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass for denoising.

        Args:
            hidden_states: [B, C, T, H, W] noisy latents
            encoder_hidden_states: Not used (text embedded in transformer)
            timestep: [B] diffusion timestep per sample
            text_ids: [B, S_text] tokenized text input
            text_mask: [B, S_text] attention mask for text (1=real, 0=pad)
            fps: video frame rate for temporal mRoPE scaling
            cache_key: Key for the UND K/V cache. Use different keys for
                conditional ("cond") and unconditional ("uncond") branches
                in CFG to avoid recomputing the cache every step.
            noisy_frame_mask: Optional [B, 1, T, 1, 1] mask where 1 marks
                noisy frames (timestep embedding applied) and 0 marks
                conditioned frames (clean context, embedding skipped).
                ``None`` means every frame is noisy (T2V / T2I).
            max_text_seq_len: Real text length already computed during
                tokenization. When omitted it is derived from ``text_mask``.
            action_latents: Optional [B, T_action, D_action] noisy action
                latents for action generation.
            action_domain_ids: [B] embodiment domain IDs (0=no-action default).
            action_noisy_mask: [B, T_action, 1] where 1=noisy, 0=conditioned;
                controls which action tokens receive the timestep embedding.
                ``None`` means all tokens are noisy.
            action_fps: Frame rate for action token temporal mRoPE scaling.
                Defaults to the video fps when None.
            action_start_frame_offset: Temporal offset applied to action
                position IDs relative to the video's media_offset (default 1).

        Returns:
            [B, C, T, H, W] velocity prediction, or a tuple
            (video_pred, ...) with extra tensors when action/sound are active.
        """
        if text_ids is None or text_mask is None:
            raise ValueError("Cosmos3 requires text_ids and text_mask to be passed")

        batch_size, C, T, H, W = hidden_states.shape
        Hp, Wp, _, _ = self._pad_to_patch_size(H, W)
        if max_text_seq_len is None:
            max_text_seq_len = int(text_mask.sum(dim=1).max().item())
        if max_text_seq_len < text_ids.shape[1]:
            text_ids = text_ids[:, :max_text_seq_len]
            text_mask = text_mask[:, :max_text_seq_len]

        sound_frames = 0
        if sound_latents is not None:
            if self.sp_size > 1:
                raise NotImplementedError(
                    "Cosmos3 sound generation does not support sequence parallelism yet"
                )
            sound_frames = sound_latents.shape[-1]

        action_frames = 0
        if action_latents is not None:
            if self.sp_size > 1:
                raise NotImplementedError(
                    "Cosmos3 action generation does not support sequence parallelism yet"
                )
            action_frames = action_latents.shape[1]
            if action_domain_ids is None:
                action_domain_ids = torch.zeros(
                    action_latents.shape[0],
                    dtype=torch.long,
                    device=action_latents.device,
                )

        # Check if sequence parallelism is enabled
        sequence_shard_enabled = self.sp_size > 1

        # Patchify and project to hidden dim
        hidden_gen, _ = self.proj_in(self.patchify(hidden_states, T, H, W))
        seq_len_orig = hidden_gen.shape[1]
        seq_shard_pad = 0

        # Per-token noisy mask follows the same pad/shard as hidden_gen, so
        # build it before the SP split.
        token_noisy_mask: torch.Tensor | None = None
        if noisy_frame_mask is not None:
            token_noisy_mask = (
                noisy_frame_mask[:, 0, :, 0, 0]
                .unsqueeze(-1)
                .expand(-1, -1, Hp * Wp)
                .reshape(batch_size, -1, 1)
                .to(hidden_gen.dtype)
            )

        # Shard sequence across GPUs if SP enabled
        if sequence_shard_enabled:
            if seq_len_orig % self.sp_size != 0:
                seq_shard_pad = self.sp_size - (seq_len_orig % self.sp_size)
                pad = torch.zeros(
                    (batch_size, seq_shard_pad, hidden_gen.shape[2]),
                    dtype=hidden_gen.dtype,
                    device=hidden_gen.device,
                )
                hidden_gen = torch.cat([hidden_gen, pad], dim=1)
                if token_noisy_mask is not None:
                    mask_pad = torch.zeros(
                        (batch_size, seq_shard_pad, 1),
                        dtype=token_noisy_mask.dtype,
                        device=token_noisy_mask.device,
                    )
                    token_noisy_mask = torch.cat([token_noisy_mask, mask_pad], dim=1)
            local_seq_len = hidden_gen.shape[1] // self.sp_size
            hidden_gen = hidden_gen.view(
                batch_size, self.sp_size, local_seq_len, hidden_gen.shape[2]
            )
            hidden_gen = hidden_gen[:, self.sp_rank, :, :]
            if token_noisy_mask is not None:
                token_noisy_mask = token_noisy_mask.view(
                    batch_size, self.sp_size, local_seq_len, 1
                )[:, self.sp_rank, :, :]

        # Add timestep embedding (computed in float32 for numerical stability, then cast back)
        time_embed = self.time_embedder(timestep.float())
        time_embed = time_embed.to(
            hidden_states.dtype
        )  # Cast to match hidden_gen dtype
        if token_noisy_mask is not None:
            hidden_gen = hidden_gen + time_embed.unsqueeze(1) * token_noisy_mask
        else:
            hidden_gen = hidden_gen + time_embed.unsqueeze(1)

        if action_latents is not None:
            hidden_action = self.action_proj_in(
                action_latents.to(hidden_gen.dtype), action_domain_ids
            )
            hidden_action = hidden_action + self.action_modality_embed.to(
                hidden_action.dtype
            )
            if action_noisy_mask is None:
                hidden_action = hidden_action + time_embed.unsqueeze(1)
            else:
                hidden_action = hidden_action + time_embed.unsqueeze(
                    1
                ) * action_noisy_mask.to(hidden_action.dtype)
            hidden_gen = torch.cat([hidden_gen, hidden_action], dim=1)

        if sound_latents is not None:
            packed_sound = sound_latents.permute(0, 2, 1).to(hidden_gen.dtype)
            hidden_sound, _ = self.audio_proj_in(packed_sound)
            hidden_sound = hidden_sound + self.audio_modality_embed.to(
                hidden_sound.dtype
            )
            hidden_sound = hidden_sound + time_embed.unsqueeze(1)
            hidden_gen = torch.cat([hidden_gen, hidden_sound], dim=1)

        self._ensure_cache_dicts()

        # Compute UND K/V cache for this cache_key if not already cached
        # This allows reusing the cache across denoising steps for the same text
        if (
            cache_key not in self.cached_kv
            or cache_key not in self.cached_gen_rope_inputs
        ):
            text_pos_ids, vis_pos_ids = self._compute_rope_position_ids(
                text_mask,
                T,
                Hp,
                Wp,
                fps,
                hidden_states.device,
                sound_frames=sound_frames,
                action_frames=action_frames,
                action_fps=action_fps if action_fps is not None else fps,
                action_start_frame_offset=action_start_frame_offset,
            )
            # UND K/V cache is kept FULL on all ranks (not sharded). Text
            # sequence is short, so memory impact is minimal, and the GEN
            # cross-attention needs the full K/V on every SP rank.
            self.cached_kv[cache_key] = self.language_model(
                text_ids, text_mask, text_pos_ids
            )
            if sequence_shard_enabled:
                if seq_shard_pad > 0:
                    pad_pos = vis_pos_ids[:, :, -1:].expand(-1, -1, seq_shard_pad)
                    vis_pos_ids = torch.cat([vis_pos_ids, pad_pos], dim=2)
                vis_pos_ids = vis_pos_ids.view(
                    3, batch_size, self.sp_size, local_seq_len
                )[:, :, self.sp_rank, :]
            self.cached_gen_rope_inputs[cache_key] = (
                self.language_model.rotary_emb.build_rope_cache_inputs(
                    vis_pos_ids, cache_dtype=hidden_gen.dtype
                )
            )

        cos_sin_gen, gen_rope_cache_positions = self.cached_gen_rope_inputs[cache_key]

        # Run GEN layers. `residual` is threaded so each layer's
        # input_layernorm and post_attention_layernorm can use the
        # fused add+rmsnorm path instead of separate add + norm kernels.
        cached_kv_for_key = self.cached_kv[cache_key]
        residual: torch.Tensor | None = None
        use_fused_qk_norm_rope = T > 1
        for i, layer in enumerate(self.gen_layers):
            k_und, v_und = cached_kv_for_key[i]
            hidden_gen, residual = layer(
                hidden_gen,
                k_und,
                v_und,
                cos_sin_gen,
                gen_rope_cache_positions,
                use_fused_qk_norm_rope,
                residual=residual,
            )

        # Collapse the trailing residual carry. RMSNorm and the linear
        # projection that follow are per-token, so we run them on the
        # local shard and only gather the (much smaller) patch-space
        # output. With patch_latent_dim ~= hidden_size / 21 for cosmos3,
        # this cuts the post-loop SP collective bandwidth ~21x.
        hidden_gen = hidden_gen + residual
        hidden_gen = self.norm_moe_gen(hidden_gen)

        extra_frames = action_frames + sound_frames
        if extra_frames > 0:
            s_video = hidden_gen.shape[1] - extra_frames
            video_hidden = hidden_gen[:, :s_video, :]
        else:
            video_hidden = hidden_gen
        output, _ = self.proj_out(video_hidden)

        if sequence_shard_enabled:
            output = sequence_model_parallel_all_gather(output, dim=1)
            if seq_shard_pad > 0:
                output = output[:, :seq_len_orig, :]

        video_pred = self.unpatchify(output, T, H, W)

        if extra_frames == 0:
            return video_pred

        extra_outputs: list[torch.Tensor] = []
        idx = s_video
        if action_frames > 0:
            action_hidden = hidden_gen[:, idx : idx + action_frames, :]
            extra_outputs.append(self.action_proj_out(action_hidden, action_domain_ids))
            idx += action_frames
        if sound_frames > 0:
            sound_hidden = hidden_gen[:, idx:, :]
            sound_output, _ = self.audio_proj_out(sound_hidden)
            extra_outputs.append(sound_output.permute(0, 2, 1).contiguous())

        return (video_pred, *extra_outputs)

    def preprocess_loaded_state_dict(
        self, iterator: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        # ModelOpt FP8 emits a 0-d per-tensor scale per source Linear. Where
        # sources fuse into a single MergedColumnParallelLinear (Q/K/V into
        # to_qkv, gate/up into gate_up_proj), the FP8 weights of each shard
        # are quantized against their own scale. Naively concatenating the
        # FP8 bytes and applying a single fused scale at runtime yields noise
        # (the K/V tiles get dequant'd with the wrong factor).
        #
        # Fix: per fused Linear, dequant each FP8 shard with its own scale,
        # pick max as the fused scale, requant each shard against the max,
        # then concat the requantized FP8 bytes. input_scale is shared across
        # shards (same activation tensor), so just take max — no requant
        # needed. The emitted fused tensors keep the full Q/K/V layout; the
        # column-parallel loader slices each logical shard for the local TP rank.
        mapping_fn = get_param_names_mapping(self.param_names_mapping)
        pending: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
        expected_count: dict[str, int] = {}

        def _try_emit(linear_target: str):
            groups = pending.get(linear_target, {})
            n = expected_count.get(linear_target)
            if n is None:
                return
            weights = groups.get("weight", {})
            w_scales = groups.get("weight_scale", {})
            i_scales = groups.get("input_scale", {})
            if len(weights) != n or len(w_scales) != n:
                return
            saw_input_scale = bool(i_scales)
            if saw_input_scale and len(i_scales) != n:
                return
            scales_t = torch.stack([w_scales[i].reshape(()) for i in range(n)])
            max_w_scale = scales_t.max()
            rescaled = []
            for i in range(n):
                w_fp8 = weights[i]
                original_scale = w_scales[i].reshape(()).to(torch.float32)
                w_dequant = w_fp8.to(torch.float32) * original_scale
                w_requant = (
                    (w_dequant / max_w_scale.to(torch.float32))
                    .clamp(-448.0, 448.0)
                    .to(torch.float8_e4m3fn)
                )
                rescaled.append(w_requant)
            merged_weight = torch.cat(rescaled, dim=0)
            pending.pop(linear_target, None)
            expected_count.pop(linear_target, None)
            yield linear_target + ".weight", merged_weight
            yield linear_target + ".weight_scale", max_w_scale
            if saw_input_scale:
                in_t = torch.stack([i_scales[i].reshape(()) for i in range(n)])
                yield linear_target + ".input_scale", in_t.max()

        for name, tensor in iterator:
            target_name, merge_index, num_to_merge = mapping_fn(name)
            if num_to_merge is None:
                yield target_name, tensor
                continue
            suffix = None
            for candidate in ("weight_scale", "input_scale", "weight"):
                if target_name.endswith("." + candidate):
                    suffix = candidate
                    break
            if suffix is None:
                yield name, tensor
                continue
            if suffix == "weight" and tensor.dtype != torch.float8_e4m3fn:
                yield name, tensor
                continue
            linear_target = target_name[: -(len(suffix) + 1)]
            pending.setdefault(linear_target, {}).setdefault(suffix, {})[
                merge_index
            ] = tensor
            expected_count[linear_target] = num_to_merge
            yield from _try_emit(linear_target)

    def post_load_weights(self, target_dtype: torch.dtype = torch.bfloat16) -> None:
        """Cast non-quantized parameters to their preferred dtypes and rebuild
        meta-device buffers.

        Time-embedder stays in float32 for numerical stability; embeddings and
        the VAE/LLM bridge linears go to ``target_dtype``. Quantized modules
        (e.g. FP8 from a ModelOpt export) are skipped — calling ``.to(dtype)``
        on them would cast their FP8 weights back to BF16/FP32 and break the
        quant kernels.

        Also re-materializes the RoPE ``inv_freq`` buffer, which can land on
        the meta device when the model is constructed under a meta context.
        """
        # Get the actual device from a loaded parameter
        device = next(self.parameters()).device

        # Recompute RoPE inv_freq buffer on the correct device
        # This is needed because model is created in meta device context
        rotary_emb = self.language_model.rotary_emb
        if rotary_emb.inv_freq.is_meta:
            dim = rotary_emb.head_dim
            rope_theta = 5000000.0  # From config
            inv_freq = 1.0 / (
                rope_theta
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)

        # Collect all quantized submodules so we can skip them during
        # dtype casts.  Calling .to(dtype) on a quantized layer would
        # cast FP8 weights back to BF16/FP32, breaking the quant kernels.
        quantized_modules: set[int] = {
            id(m)
            for m in self.modules()
            if hasattr(m, "quant_method")
            and not isinstance(m.quant_method, UnquantizedLinearMethod)
        }

        def _is_quantized(module: torch.nn.Module) -> bool:
            return id(module) in quantized_modules

        def _cast_direct(module: torch.nn.Module, dtype: torch.dtype) -> None:
            """Cast only the module's own parameters and buffers, not its
            children.  This avoids the recursive `.to()` which would cast
            quantized (FP8) weights back to BF16/FP32."""
            for key, param in module._parameters.items():
                if param is not None:
                    module._parameters[key] = torch.nn.Parameter(
                        param.data.to(dtype=dtype), requires_grad=False
                    )
            for key, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[key] = buf.to(dtype=dtype)

        # Time embedder should stay in float32 for numerical stability.
        # Cast only non-quantized submodules' own params (non-recursive).
        for module in self.time_embedder.modules():
            if not _is_quantized(module):
                _cast_direct(module, torch.float32)

        # Ensure embeddings and projections are in target dtype
        self.language_model.embed_tokens.to(target_dtype)
        for module in self.proj_in.modules():
            if not _is_quantized(module):
                _cast_direct(module, target_dtype)
        for module in self.proj_out.modules():
            if not _is_quantized(module):
                _cast_direct(module, target_dtype)

        # Convert RMSNorm layers to target dtype
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.to(target_dtype)


EntryClass = Cosmos3OmniTransformer
