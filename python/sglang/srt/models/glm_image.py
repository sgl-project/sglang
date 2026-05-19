# Copyright 2023-2024 SGLang Team
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

"""Inference-only GlmImage model compatible with HuggingFace weights."""

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import compute_cu_seqlens_from_grid_numpy
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_npu

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Vision encoder components
# --------------------------------------------------------------------------- #


class GlmImageVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        # self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
        # self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            # tp_size=self.tp_size,
            # tp_rank=self.tp_rank,
        )
        self.fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            # tp_size=self.tp_size,
            # tp_rank=self.tp_rank,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class GlmImageVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.patch_size, self.patch_size]
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class GlmImageVisionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.interpolated_method = "bilinear"

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths,
        image_shapes: torch.Tensor,
        h_coords: torch.Tensor,
        w_coords: torch.Tensor,
    ) -> torch.Tensor:
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        device = pos_embed_weight.device

        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device, dtype=torch.long)

        orig_size_sq = pos_embed_weight.shape[0]
        orig_size = int(orig_size_sq**0.5)
        pos_embed_2d = (
            pos_embed_weight.view(orig_size, orig_size, hidden_size)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

        target_h = torch.cat(
            [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
        ).to(device=device, dtype=torch.float32)
        target_w = torch.cat(
            [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
        ).to(device=device, dtype=torch.float32)

        h_coords = h_coords.to(device=device, dtype=torch.float32)
        w_coords = w_coords.to(device=device, dtype=torch.float32)
        norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
        norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

        grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

        interpolated_embed_fp32 = F.grid_sample(
            pos_embed_2d,
            grid,
            mode=self.interpolated_method,
            align_corners=False,
            padding_mode="border",
        )

        adapted_pos_embed_fp32 = (
            interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
        )
        adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(
            embeddings.device
        )

        embeddings = embeddings + adapted_pos_embed
        return embeddings


class GlmImageVisionBlock(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            proj_bias=config.attention_bias,
            qkv_bias=config.attention_bias,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_data_parallel=use_data_parallel,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )
        self.mlp = GlmImageVisionMLP(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # x shape: (S, B, H) where B=1
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(hidden_states, cu_seqlens=cu_seqlens)
        attn = rearrange(attn, "b s ... -> s b ...")
        x = x + attn

        hidden_states = self.norm2(x)
        mlp = self.mlp(hidden_states)
        x = x + mlp
        return x


class GlmImageVisionModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_merge_size = getattr(config, "spatial_merge_size", 1)
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        # No patch merger in GlmImage, output dim = hidden_size
        self.out_hidden_size = config.hidden_size

        self.embeddings = GlmImageVisionEmbeddings(config)
        self.patch_embed = GlmImageVisionPatchEmbed(config)

        self.blocks = nn.ModuleList(
            [
                GlmImageVisionBlock(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                    use_data_parallel=use_data_parallel,
                )
                for i in range(config.depth)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw):
        """Compute position coordinate IDs for position embedding interpolation."""
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(pixel_values)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_list = grid_thw.tolist()

        image_type_ids = self.rot_pos_emb(grid_thw_list)

        # Compute cu_seqlens using numpy for efficiency
        grid_thw_cpu = grid_thw if grid_thw.device.type == "cpu" else grid_thw.cpu()
        cu_seqlens = compute_cu_seqlens_from_grid_numpy(grid_thw_cpu)
        if not is_npu():
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
        else:
            cu_seqlens = cu_seqlens.to("cpu")

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        hidden_states = self.embeddings(
            hidden_states,
            seqlens,
            grid_thw,
            image_type_ids[:, 0].to(hidden_states.device),
            image_type_ids[:, 1].to(hidden_states.device),
        )

        # (S, H) -> (S, 1, H) for block processing
        hidden_states = hidden_states.unsqueeze(1)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens)

        # (S, 1, H) -> (S, H)
        return hidden_states.squeeze(1)


# --------------------------------------------------------------------------- #
# VQ-VAE
# --------------------------------------------------------------------------- #


class GlmImageVQVAE(nn.Module):
    """VQ-VAE module for encoding vision features into discrete tokens.

    Follows the HF transformers GlmImageVQVAE architecture:
    quant_conv (Conv2d) -> L2 normalize -> nearest codebook lookup -> indices
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim
        self.latent_channels = config.latent_channels

        # Codebook (quantize.embedding in HF)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Convolutions
        self.quant_conv = nn.Conv2d(self.latent_channels, self.embedding_dim, 1)
        self.post_quant_conv = nn.Conv2d(self.embedding_dim, self.latent_channels, 1)

        self.eval()  # frozen

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode spatial features to discrete codebook indices.

        Args:
            hidden_states: [B, latent_channels, H, W] spatial feature maps
        Returns:
            indices: [B*H*W] discrete codebook indices
        """
        conv_hidden = self.quant_conv(hidden_states)
        # Permute to [B, H, W, embed_dim] then flatten for distance computation
        z = conv_hidden.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        # L2 normalize
        z_flat = F.normalize(z_flat, p=2, dim=-1)
        codebook = F.normalize(self.embedding.weight, p=2, dim=-1)

        # Compute distances: (z - e)^2 = z^2 + e^2 - 2*z*e
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.matmul(z_flat, codebook.t())
        )
        indices = torch.argmin(distances, dim=1)
        return indices


# --------------------------------------------------------------------------- #
# Text model
# --------------------------------------------------------------------------- #


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_glm_image_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply GLM-Image rotary position embedding to query and key tensors.

    Args:
        q: Query tensor [num_tokens, num_heads, head_dim]
        k: Key tensor [num_tokens, num_kv_heads, head_dim]
        cos: Cosine values [num_tokens, rotary_dim]
        sin: Sine values [num_tokens, rotary_dim]

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as input
    """
    # cos/sin shape: [num_tokens, rotary_dim]
    # Need to unsqueeze for broadcasting with heads dimension
    cos = cos.unsqueeze(1)  # [num_tokens, 1, rotary_dim]
    sin = sin.unsqueeze(1)  # [num_tokens, 1, rotary_dim]

    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


class GlmImageRotaryEmbedding(nn.Module):
    """
    Custom Rotary Embedding for GLM-Image with M-RoPE support.

    GLM-Image uses a 3D position encoding (temporal, height, width) with
    M-RoPE sections [8, 12, 12]. This means:
    - First 8 dims use temporal positions
    - Next 12 dims use height positions
    - Next 12 dims use width positions
    - Pattern repeats for remaining dims

    Unlike vLLM's standard MRotaryEmbedding which uses cache-based lookup,
    this implementation computes cos/sin dynamically to handle arbitrary
    position values without cache size limitations.

    This follows the transformers reference implementation exactly:
    - inv_freq is expanded for matmul with position_ids
    - freqs = inv_freq @ position_ids (matrix multiplication)
    - apply_mrope interleaves frequency chunks from different dimensions
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        mrope_section: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # Compute rotary dimension
        self.rotary_dim = int(head_dim * partial_rotary_factor)

        # Default mrope_section for GLM-Image
        self.mrope_section = mrope_section if mrope_section is not None else [8, 12, 12]

        # Compute inverse frequencies
        # inv_freq shape: [rotary_dim // 2]
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Apply M-RoPE section interleaving.

        For mrope_section = [8, 12, 12]:
        - Split freqs into chunks of size [8, 12, 12, 8, 12, 12, ...]
        - Take chunk[i % 3] from each split (alternating T, H, W dimensions)
        - Concatenate back

        Args:
            freqs: Frequency tensor [3, num_tokens, rotary_dim // 2]

        Returns:
            Interleaved frequencies [num_tokens, rotary_dim // 2]
        """
        # freqs shape: [3, num_tokens, rotary_dim // 2]
        # Split along last dimension according to mrope_section
        chunks = freqs.split(self.mrope_section, dim=-1)

        # Take chunk[i % 3] from each split
        # chunks[i] has shape [3, num_tokens, section_size]
        # We select dimension 0 (T), 1 (H), or 2 (W) based on i % 3
        result = torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)

        return result

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key.

        Args:
            positions: Position IDs
                - Shape [num_tokens] for 1D positions (text-only)
                - Shape [3, num_tokens] for 3D M-RoPE positions (T, H, W)
            query: Query tensor [num_tokens, num_heads * head_dim]
            key: Key tensor [num_tokens, num_kv_heads * head_dim]

        Returns:
            Tuple of (rotated_query, rotated_key) with same shapes as input
        """
        # Get dimensions
        if positions.ndim == 1:
            num_tokens = positions.shape[0]
        else:
            num_tokens = positions.shape[1]

        device = positions.device
        dtype = query.dtype

        # Ensure inv_freq is on same device
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)

        if positions.ndim == 1:
            # 1D positions: expand to 3D with same values
            # Shape: [num_tokens] -> [3, num_tokens]
            positions_3d = positions.unsqueeze(0).expand(3, -1)
        else:
            # Already 3D: [3, num_tokens]
            positions_3d = positions

        # Follow reference implementation exactly:
        # Reference: inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, bs, -1, 1)
        # Reference: position_ids_expanded = position_ids[:, :, None, :].float()  # (3, bs, 1, positions)
        # Reference: freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        #
        # For vLLM (no batch dim):
        # inv_freq: [rotary_dim // 2]
        # positions_3d: [3, num_tokens]
        #
        # We want: freqs[i, j, k] = positions_3d[i, j] * inv_freq[k]
        # So: freqs = positions_3d[:, :, None] * inv_freq[None, None, :]
        # Shape: [3, num_tokens, 1] * [1, 1, rotary_dim // 2] = [3, num_tokens, rotary_dim // 2]

        # Compute frequencies using broadcasting (equivalent to matmul in reference)
        positions_expanded = positions_3d.unsqueeze(-1).float()  # [3, num_tokens, 1]
        inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, rotary_dim // 2]
        freqs = (
            positions_expanded * inv_freq_expanded
        )  # [3, num_tokens, rotary_dim // 2]

        # Apply M-RoPE interleaving
        # This selects different frequency dims from different position dims
        freqs = self._apply_mrope(freqs)  # [num_tokens, rotary_dim // 2]

        # Build cos/sin embeddings
        # Concatenate freqs with itself for full rotary_dim (real and imaginary parts)
        emb = torch.cat((freqs, freqs), dim=-1)  # [num_tokens, rotary_dim]
        cos = emb.cos().to(dtype)  # [num_tokens, rotary_dim]
        sin = emb.sin().to(dtype)  # [num_tokens, rotary_dim]

        # Reshape query and key for rotary application
        # query: [num_tokens, num_heads * head_dim] -> [num_tokens, num_heads, head_dim]
        query_shape = query.shape
        key_shape = key.shape

        query = query.view(num_tokens, -1, self.head_dim)
        key = key.view(num_tokens, -1, self.head_dim)

        # Apply rotary embeddings
        query, key = apply_glm_image_rotary_pos_emb(query, key, cos, sin)

        # Reshape back
        query = query.view(query_shape)
        key = key.view(key_shape)

        return query, key


class GlmImageTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 131072,
        quant_config: QuantizationConfig | None = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        partial_rotary_factor: float = 0.5,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=None,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
            partial_rotary_factor=partial_rotary_factor,
            is_neox_style=True,
        )

        self.rope_parameters = config.rope_parameters

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        rope_parameters = getattr(config, "rope_parameters", None)
        rope_theta = 10000.0
        partial_rotary_factor = 1.0
        mrope_section = [8, 12, 12]  # Default for GLM-Image

        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
            partial_rotary_factor = rope_parameters.get(
                "partial_rotary_factor", partial_rotary_factor
            )
            mrope_section = rope_parameters.get("mrope_section", mrope_section)

        self.rot2 = GlmImageRotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            mrope_section=mrope_section,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # q2, k2 = self.rotary_emb(positions, q, k)
        q, k = self.rot2(positions, q, k)
        # print(q2-q,k2-k)

        attn_output = self.attn(q, k, v, forward_batch)

        attn_output = self.o_proj(attn_output)

        return attn_output


class GlmImageRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        GlmImageRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class GlmImageTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(
            self.config, device
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [8, 12, 12])

    @staticmethod
    def compute_default_rope_parameters(
        config=None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    def forward(self, x, position_ids):
        # In contrast to other models, GLM-V has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions)

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            2, 3
        )
        freqs = self.apply_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_mrope(self, freqs, mrope_section):
        section = mrope_section
        chunks = freqs.split(section, dim=-1)
        result = torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)
        return result

    def load_weights(self, weights: Any) -> set[str]:
        # Copied from LlamaModel.load_weights but adapted
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def _load_with_shard_id(
            weight_loader, param, loaded_weight: torch.Tensor, shard_id
        ) -> None:

            try:
                weight_loader(param, loaded_weight, shard_id)
                return
            except (AssertionError, TypeError):
                pass

            # Fall back between common representations.
            if isinstance(shard_id, str):
                mapping = {"q": 0, "k": 1, "v": 2}
                if shard_id in mapping:
                    weight_loader(param, loaded_weight, mapping[shard_id])
                    return
                if shard_id.isdigit():
                    weight_loader(param, loaded_weight, int(shard_id))
                    return
            elif isinstance(shard_id, int):
                mapping = {0: "q", 1: "k", 2: "v"}
                if shard_id in mapping:
                    weight_loader(param, loaded_weight, mapping[shard_id])
                    return

            # Re-raise with a clearer message.
            raise TypeError(
                f"Unsupported shard_id={shard_id!r} for weight_loader={weight_loader} "
                f"(param={getattr(param, 'name', '<param>')})."
            )

        stacked_params_mapping = getattr(
            getattr(self.config, "arch_config", object()),
            "stacked_params_mapping",
            None,
        )
        if stacked_params_mapping is None:
            stacked_params_mapping = [
                # Fused QKV shards; downstream loaders may want "q/k/v" or 0/1/2.
                (".qkv_proj", ".q_proj", "q"),
                (".qkv_proj", ".k_proj", "k"),
                (".qkv_proj", ".v_proj", "v"),
                (".gate_up_proj", ".gate_proj", 0),
                (".gate_up_proj", ".up_proj", 1),
            ]

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # The config has stacked_params_mapping
            for (
                param_name,
                weight_name,
                shard_id,
            ) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                _load_with_shard_id(weight_loader, param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


class GlmImageTextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                "Glm-Image uses `silu` as the hidden activation "
                "function. Please set `hidden_activation` to "
                "`silu`."
            )
        self.activation_fn = SiluAndMul()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        gate_up, _ = self.gate_up_proj(x)

        x = self.activation_fn(gate_up)

        x, _ = self.down_proj(x)
        return x


class GlmImageTextDecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmImageTextAttention(
            layer_id=layer_id,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config,
                "num_key_value_heads",
                config.num_attention_heads,
            ),
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = GlmImageTextMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_self_attn_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states, _ = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            **kwargs,
        )

        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class GlmImageTextModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = GlmImageTextDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = None

        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers = nn.ModuleList(
            [
                GlmImageTextDecoderLayer(
                    layer_id=i,
                    config=config,
                    quant_config=self.quant_config,
                    prefix=add_prefix(f"layers.{i}", getattr(config, "prefix", "")),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmImageTextRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        forward_batch: ForwardBatch,
        position_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds
        # position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                # attention_mask=attention_mask,
                # position_ids=text_position_ids,
                # position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def get_input_embeddings(self):
        return self.embed_tokens


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #


class GlmImageForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vision_config = config.vision_config
        self.vq_config = config.vq_config
        self.text_config = config.text_config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        # Bridge rope_parameters -> rope_scaling so Glm4Model can pick it up
        if hasattr(self.text_config, "rope_parameters") and not getattr(
            self.text_config, "rope_scaling", None
        ):
            self.text_config.rope_scaling = self.text_config.rope_parameters

        # Vision encoder
        self.visual = GlmImageVisionModel(
            self.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        # VQ-VAE (small frozen module, no TP needed)
        self.vqvae = GlmImageVQVAE(self.vq_config)

        # Language model (reuse Glm4Model)
        self.model = GlmImageTextModel(
            self.text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # GLM-Image uses split-half (neox-style) rotary embedding, unlike GLM4
        # which uses interleaved rotation.  Glm4Attention defaults to
        # is_neox_style=False, so we override it here after construction.
        for layer in self.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                layer.self_attn.rotary_emb.is_neox_style = True

        # LogitsProcessor with vision_vocab_size
        vision_vocab_size = getattr(self.text_config, "vision_vocab_size", None)
        if vision_vocab_size is not None:
            logits_config = copy.copy(self.text_config)
            logits_config.vocab_size = vision_vocab_size
        else:
            logits_config = self.text_config

        # lm_head: maps hidden_size -> vision_vocab_size
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                logits_config.vocab_size,
                self.text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.is_mrope_enabled = (
            hasattr(self.text_config, "rope_scaling")
            and self.text_config.rope_scaling is not None
            and "mrope_section" in self.text_config.rope_scaling
        )

        self.logits_processor = LogitsProcessor(logits_config)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Run vision encoder -> VQ-VAE encode -> embed_tokens on discrete indices."""
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        # Vision encoder forward (with optional DP sharding)
        if self.use_data_parallel:
            vision_hidden = run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values,
                image_grid_thw.tolist(),
                rope_type="rope_3d",
            )
        else:
            vision_hidden = self.visual(pixel_values, grid_thw=image_grid_thw)

        # Split by image, reshape to spatial, run VQ-VAE encode, then embed
        hidden_size = vision_hidden.shape[-1]
        split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
        hidden_list = torch.split(vision_hidden, split_sizes, dim=0)

        embed_tokens = self.model.get_input_embeddings()
        all_embeds = []
        for idx, hs in enumerate(hidden_list):
            grid_t, grid_h, grid_w = image_grid_thw[idx].tolist()
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            # Reshape to spatial: [t, h, w, hidden] -> [t, hidden, h, w]
            hs = hs.view(grid_t, grid_h, grid_w, hidden_size)
            hs = hs.permute(0, 3, 1, 2).contiguous()
            # VQ-VAE encode: get discrete codebook indices
            indices = self.vqvae.encode(hs)
            # Embed via LLM embedding table
            embeds = embed_tokens(indices)
            all_embeds.append(embeds)

        return torch.cat(all_embeds, dim=0)

    def _get_decode_mrope_positions(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Look up pre-computed 2D spatial MRoPE positions during decode.

        Instead of using the default delta-based positions (which are sequential
        and identical across all 3 dims), this looks up the pre-computed 2D
        spatial positions stored in each request's MultimodalInputs.mrope_positions.
        This gives each generated image token the correct (temporal, height, width)
        coordinates based on its position in the target image grid.
        """
        batch_size = forward_batch.batch_size
        positions_list = []
        seq_lens = forward_batch.seq_lens_cpu

        for i in range(batch_size):
            mm_input = forward_batch.mm_inputs[i] if forward_batch.mm_inputs else None
            seq_len = int(seq_lens[i])

            if mm_input is not None and mm_input.mrope_positions is not None:
                stored = mm_input.mrope_positions
                idx = seq_len - 1
                if idx < stored.shape[1]:
                    # Look up the pre-computed 2D spatial position
                    pos = stored[:, idx : idx + 1]
                else:
                    # Beyond stored positions: fall back to sequential from last
                    last_max = stored[:, -1:].max().item()
                    overflow = idx - stored.shape[1] + 1
                    pos = torch.full((3, 1), last_max + overflow, dtype=torch.int64)
            else:
                pos = torch.full((3, 1), seq_len - 1, dtype=torch.int64)
            positions_list.append(pos)

        return torch.cat(positions_list, dim=1).to(
            device=forward_batch.input_ids.device, dtype=torch.int64
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
            # if forward_batch.forward_mode.is_decode():
            #    # Use pre-computed 2D spatial positions for image generation
            #    positions = self._get_decode_mrope_positions(forward_batch)
            # else:
            #    positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".up_proj", 1),
            (".gate_up_proj", ".gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Weight name mapping from HF checkpoint
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")
            if "model.visual." in name:
                name = name.replace("model.visual.", "visual.")
            if "model.vqmodel." in name:
                name = name.replace("model.vqmodel.", "vqvae.")
            if "vqvae.quantize.embedding" in name:
                name = name.replace("vqvae.quantize.embedding", "vqvae.embedding")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Vision uses fused QKV, skip stacked mapping
                if "visual" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # Map fused attn.qkv -> attn.qkv_proj for QKVParallelLinear
                    name = name.replace("attn.qkv.", "attn.qkv_proj.")

                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [GlmImageForConditionalGeneration]
