# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import Gemma4VisionConfig

from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL
from sglang.srt.layers.clippable_linear import (
    ClippableGateUpParallelLinear,
    ClippableQKVParallelLinear,
    ClippableRowParallelLinear,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix, get_device_capability, is_cuda, is_hip

# ---------------------------------------------------------------------------
# 2-D Multidimensional RoPE (matches HF Gemma4RotaryEmbedding for vision)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class Gemma4VisionRotaryEmbedding(nn.Module):
    """Compute 2-D multidimensional RoPE cos/sin for patch positions."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta: float = config.rope_parameters["rope_theta"]

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, patch_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, hidden] – only used for device/dtype.
            patch_positions: [batch, num_patches, 2] – (x, y) coordinates.
        Returns:
            (cos, sin) each of shape [batch, num_patches, head_dim].
        """
        ndim = patch_positions.shape[-1]  # 2
        head_dim_per_dim = self.head_dim // ndim

        all_embs = []
        for d in range(ndim):
            dim_inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(
                        0, head_dim_per_dim, 2, device=x.device, dtype=torch.float
                    )
                    / head_dim_per_dim
                )
            )
            dim_inv_freq_expanded = dim_inv_freq[None, :, None].expand(
                patch_positions.shape[0], -1, 1
            )
            dim_positions = patch_positions[:, :, d].float()
            dim_positions_expanded = dim_positions[:, None, :]

            dim_freqs = (dim_inv_freq_expanded @ dim_positions_expanded).transpose(1, 2)
            dim_emb = torch.cat((dim_freqs, dim_freqs), dim=-1)
            all_embs.append(dim_emb)

        emb = torch.cat(all_embs, dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


def _apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply 2-D RoPE to x of shape [batch*seq, heads, head_dim].

    cos/sin have shape [batch, seq, head_dim]. We split along head_dim into
    ndim=2 parts and apply standard rotary to each independently.
    """
    ndim = 2
    chunk_size = x.shape[-1] // ndim
    x_parts = x.split(chunk_size, dim=-1)
    cos_parts = cos.split(chunk_size, dim=-1)
    sin_parts = sin.split(chunk_size, dim=-1)
    y_parts = [
        _apply_rotary(x_parts[k], cos_parts[k], sin_parts[k]) for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


# ---------------------------------------------------------------------------
# Vision Attention (TP-sharded, fused QKV)
# ---------------------------------------------------------------------------


class Gemma4VisionAttention(nn.Module):
    """Multi-head attention for the Gemma 4 vision encoder.

    QKV uses a fused ``ClippableQKVParallelLinear`` for efficient matmul with
    per-projection clip bounds.  Output projection uses ``ClippableLinear``.
    """

    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.head_dim = config.head_dim

        tp_size = get_attention_tp_size()
        self.num_heads_per_partition = config.num_attention_heads // tp_size
        self.num_kv_heads_per_partition = config.num_key_value_heads // tp_size

        self.qkv = ClippableQKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=config.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.o_proj = ClippableRowParallelLinear(
            input_size=config.num_attention_heads * config.head_dim,
            output_size=config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=False
        )

        backend = self._select_backend()
        self.qkv_backend = QKV_BACKEND_IMPL[backend](
            head_dim=config.head_dim,
            num_heads=self.num_heads_per_partition,
            num_kv_heads=self.num_kv_heads_per_partition,
            dropout=0.0,
            flatten_batch=True,
            softmax_in_single_precision=False,
            softmax_scale=1.0,
        )

    @staticmethod
    def _select_backend() -> str:
        """Mirror VisionAttention._determine_attention_backend for consistency."""
        from sglang.srt.server_args import get_global_server_args

        override = get_global_server_args().mm_attention_backend
        if override is not None:
            return override
        if is_cuda():
            major, _ = get_device_capability()
            if major == 9:
                from sglang.srt.utils import is_blackwell_supported

                if is_blackwell_supported():
                    return "triton_attn"
                return "fa3"
            return "triton_attn"
        if is_hip():
            # ROCm: use triton_attn to avoid SDPA flatten_batch issues
            # with multi-image/video inputs
            return "triton_attn"
        return "sdpa"

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q, k, v = self.qkv(hidden_states)

        q = q.reshape(bsz * seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.reshape(bsz * seq_len, self.num_kv_heads_per_partition, self.head_dim)
        v = v.reshape(bsz * seq_len, self.num_kv_heads_per_partition, self.head_dim)

        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        v = self.v_norm(v.reshape(-1, self.head_dim)).reshape(v.shape)

        cos_flat = cos.reshape(bsz * seq_len, 1, self.head_dim)
        sin_flat = sin.reshape(bsz * seq_len, 1, self.head_dim)
        q = _apply_multidimensional_rope(q, cos_flat, sin_flat)
        k = _apply_multidimensional_rope(k, cos_flat, sin_flat)

        if attention_mask is not None:
            attn_mask_4d = (
                attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)
            ).unsqueeze(1)
        else:
            attn_mask_4d = None

        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens=None,
            bsz=bsz,
            seq_len=seq_len,
            attention_mask=attn_mask_4d,
            softmax_scale=1.0,
        )

        output = rearrange(output, "(b s) h d -> b s (h d)", b=bsz)
        output = self.o_proj(output)
        return output


# ---------------------------------------------------------------------------
# Vision MLP (GatedGELU, TP-sharded)
# ---------------------------------------------------------------------------


class Gemma4VisionMLP(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                f"Gemma4VisionMLP expects hidden_activation='gelu_pytorch_tanh', "
                f"got {config.hidden_activation!r}"
            )
        self.gate_up = ClippableGateUpParallelLinear(
            input_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.down_proj = ClippableRowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x)
        x = F.gelu(gate, approximate="tanh") * up
        x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(
            config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Gemma4VisionMLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        eps = config.rms_norm_eps
        hs = config.hidden_size
        self.input_layernorm = Gemma4RMSNorm(hs, eps=eps)
        self.post_attention_layernorm = Gemma4RMSNorm(hs, eps=eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hs, eps=eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hs, eps=eps)

        self.register_buffer("layer_scalar", torch.ones(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# Vision Transformer (stack of encoder layers + RoPE)
# ---------------------------------------------------------------------------


class Gemma4VisionTransformer(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    config,
                    layer_idx=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: [batch, seq, hidden_size]
            attention_mask: [batch, seq] — True = valid token
            patch_positions: [batch, seq, 2]
        Returns:
            last_hidden_state: [batch, seq, hidden_size]
        """
        cos, sin = self.rotary_emb(inputs_embeds, patch_positions)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        return hidden_states


# ---------------------------------------------------------------------------
# Patch Embedder
# ---------------------------------------------------------------------------


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.position_embedding_size = config.position_embedding_size

        self.input_proj = nn.Linear(
            3 * self.patch_size**2, self.hidden_size, bias=False
        )
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(
        self, patch_positions: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        clamped_positions = patch_positions.clamp(min=0)
        one_hot = F.one_hot(clamped_positions, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1), 0.0, position_embeddings
        )
        return position_embeddings

    def _patch_projection(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Project pre-patchified pixels into model space.

        Args:
            pixel_values: [batch, num_patches, patch_pixels] — already patchified
                          by the image processor, values in [0, 1].
        """
        patches = 2 * (pixel_values - 0.5)
        return self.input_proj(patches.to(self.input_proj.weight.dtype))

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute patch embeddings with positional information.

        Args:
            pixel_values: [batch, num_patches, patch_pixels] — pre-patchified.
            pixel_position_ids: [batch, num_patches, 2] — (x, y) positions,
                                -1 for padding patches.
            padding_positions: [batch, num_patches] — True for padding patches.
        """
        hidden_states = self._patch_projection(pixel_values)
        position_embeddings = self._position_embeddings(
            pixel_position_ids, padding_positions
        )
        return hidden_states + position_embeddings


# ---------------------------------------------------------------------------
# Pooler
# ---------------------------------------------------------------------------


class Gemma4VisionPooler(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(
        self, x: torch.Tensor, patch_positions: torch.Tensor, length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = x.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {x.shape} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )
        clamped_positions = patch_positions.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]

        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2).to(x.dtype) @ x
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output, mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_positions: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (pooled_hidden_states, mask) where mask is True for valid tokens.
        """
        if output_length is None:
            raise ValueError("output_length is required for Gemma4VisionPooler")
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are patches"
                f" ({hidden_states.shape[1]}). Change the value of `num_soft_tokens` when processing."
            )
        length = output_length
        if isinstance(length, (list, tuple)):
            length = length[0]
        if hidden_states.shape[1] == length:
            mask = padding_positions
        else:
            hidden_states, mask = self._avg_pool_by_positions(
                hidden_states, patch_positions, length
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, mask


# ---------------------------------------------------------------------------
# Top-level Vision Encoder (patch_embedder → transformer → pooler)
# ---------------------------------------------------------------------------


class Gemma4VisionEncoder(nn.Module):
    """Drop-in replacement for HF ``Gemma4VisionEncoder`` with TP support."""

    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.pooling_kernel_size = config.pooling_kernel_size

        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionTransformer(
            config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.pooler = Gemma4VisionPooler(config)

        # Post-pooling standardization (normalizes vision tokens before projection)
        self.standardize = getattr(config, "standardize", False)
        if self.standardize:
            self.register_buffer("std_bias", torch.zeros(config.hidden_size))
            self.register_buffer("std_scale", torch.ones(config.hidden_size))

    @property
    def device(self) -> torch.device:
        return self.patch_embedder.input_proj.weight.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode pre-patchified pixel_values into soft tokens.

        Args:
            pixel_values: [batch, num_patches, patch_pixels] — pre-patchified
                          by the image processor.
            pixel_position_ids: [batch, num_patches, 2] — (x, y) positions,
                                -1 for padding patches.

        Returns:
            (hidden_states, pooler_mask) — hidden_states [batch, output_len, hidden],
            pooler_mask [batch, output_len] True = valid.
        """
        k2 = self.pooling_kernel_size * self.pooling_kernel_size
        output_length = pixel_values.shape[-2] // k2

        padding_positions = (pixel_position_ids == -1).all(dim=-1)

        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )

        last_hidden = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            patch_positions=pixel_position_ids,
        )

        pooled, pooler_mask = self.pooler(
            last_hidden,
            pixel_position_ids,
            padding_positions,
            output_length=output_length,
        )

        if self.standardize:
            pooled = (pooled - self.std_bias) * self.std_scale

        return pooled, pooler_mask
