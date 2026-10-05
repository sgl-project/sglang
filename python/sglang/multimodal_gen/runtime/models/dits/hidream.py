"""HiDream-O1-Image: unified VLM diffusion model (modified Qwen3VL).

Ports the model from the official HiDream-O1-Image repository with NPU-optimized
changes: SDPA-based two-pass attention, SGLang distributed linears for TP,
and platform-agnostic device references.
"""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.hidream import HiDreamDitConfig
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum, current_platform


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    # cos/sin cast to q dtype (official)
    cos = cos.unsqueeze(unsqueeze_dim).to(q.dtype)
    sin = sin.unsqueeze(unsqueeze_dim).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vision-specific rotary embedding (float32 precision)."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = ((q * cos) + (rotate_half(q) * sin)).to(orig_q_dtype)
    k_embed = ((k * cos) + (rotate_half(k) * sin)).to(orig_k_dtype)
    return q_embed, k_embed


class HiDreamVisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act_fn = nn.GELU("tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class HiDreamVisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size,
            self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class HiDreamVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class HiDreamVisionPatchMerger(nn.Module):
    def __init__(self, hidden_size: int, out_hidden_size: int,
                 spatial_merge_size: int, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = hidden_size * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x)
        x = x.view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class HiDreamVisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.scaling = self.head_dim ** -0.5

    def forward(
        self, hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1).unsqueeze(0)  # [1, seq, heads, head_dim] -> [1, heads, seq, head_dim]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = q.split(lengths, dim=2)
        k_splits = k.split(lengths, dim=2)
        v_splits = v.split(lengths, dim=2)

        attn_outputs = []
        for q_c, k_c, v_c in zip(q_splits, k_splits, v_splits):
            out = F.scaled_dot_product_attention(
                q_c, k_c, v_c, is_causal=False, scale=self.scaling
            )
            attn_outputs.append(out)
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class HiDreamVisionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = HiDreamVisionAttention(hidden_size, num_heads)
        self.mlp = HiDreamVisionMLP(hidden_size, intermediate_size)

    def forward(
        self, hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class HiDreamVisionModel(nn.Module):
    """Full vision encoder with deepstack feature extraction."""

    def __init__(self, cfg):
        super().__init__()
        self.spatial_merge_size = cfg.vision_spatial_merge_size
        self.patch_size = cfg.vision_patch_size
        self.spatial_merge_unit = self.spatial_merge_size ** 2

        self.patch_embed = HiDreamVisionPatchEmbed(
            cfg.vision_patch_size, cfg.vision_temporal_patch_size,
            cfg.vision_in_channels, cfg.vision_hidden_size,
        )
        self.pos_embed = nn.Embedding(
            cfg.vision_num_position_embeddings, cfg.vision_hidden_size
        )
        self.num_grid_per_side = int(cfg.vision_num_position_embeddings ** 0.5)

        head_dim = cfg.vision_hidden_size // cfg.vision_num_heads
        self.rotary_pos_emb = HiDreamVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            HiDreamVisionBlock(cfg.vision_hidden_size, cfg.vision_num_heads,
                               cfg.vision_intermediate_size)
            for _ in range(cfg.vision_depth)
        ])
        self.merger = HiDreamVisionPatchMerger(
            cfg.vision_hidden_size, cfg.vision_out_hidden_size,
            cfg.vision_spatial_merge_size, use_postshuffle_norm=False,
        )
        self.deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList([
            HiDreamVisionPatchMerger(
                cfg.vision_hidden_size, cfg.vision_out_hidden_size,
                cfg.vision_spatial_merge_size, use_postshuffle_norm=True,
            )
            for _ in range(len(self.deepstack_visual_indexes))
        ])

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor, w_floor = h_idxs.int(), w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)
            dh, dw = h_idxs - h_floor, w_idxs - w_floor
            base_h, base_h_c = h_floor * self.num_grid_per_side, h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_c[None].T + w_floor[None]).flatten(),
                (base_h_c[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
        result = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pos_embed)
        return torch.cat(result)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.size(0)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                                position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    self.deepstack_merger_list[idx](hidden_states)
                )
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


class HiDreamTextRotaryEmbedding(nn.Module):
    """3D interleaved RoPE for the text decoder."""

    def __init__(self, head_dim: int, rope_theta: float, mrope_section: list[int],
                 max_position_embeddings: int):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section
        self.max_seq_len_cached = max_position_embeddings
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        """Reorganize from chunked [TTT...HHH...WWW] to interleaved [THTHWHTHW...TT]."""
        freqs_t = freqs[0]  # overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for the given 3D position_ids [3, batch, seq_len]."""
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        with torch.autocast(device_type=current_platform.device_type if hasattr(current_platform, 'device_type') else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos, sin


class HiDreamTextAttention(nn.Module):
    """GQA attention with q_norm/k_norm, TP-optimized Q/K/V/O."""

    def __init__(self, cfg, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim,
            bias=cfg.attention_bias, prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim,
            bias=cfg.attention_bias, prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim,
            bias=cfg.attention_bias, prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size,
            bias=cfg.attention_bias, prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q, _ = self.q_proj(hidden_states)
        q = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
        k, _ = self.k_proj(hidden_states)
        k = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
        v, _ = self.v_proj(hidden_states)
        v = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v


class HiDreamTextMLP(nn.Module):
    """SiLU gated MLP with TP-optimized projections."""

    def __init__(self, cfg, prefix: str = ""):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.intermediate_size,
            bias=False, prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.intermediate_size,
            bias=False, prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            cfg.intermediate_size, cfg.hidden_size,
            bias=False, prefix=f"{prefix}.down_proj",
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        out, _ = self.down_proj(self.act_fn(gate) * up)
        return out


class HiDreamTextDecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int, prefix: str = ""):
        super().__init__()
        self.self_attn = HiDreamTextAttention(cfg, layer_idx, prefix=f"{prefix}.self_attn")
        self.mlp = HiDreamTextMLP(cfg, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        q, k, v = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = self._attention(q, k, v, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states

    def _attention(self, q, k, v, attention_mask):
        """SDPA attention with optional 4D mask and GQA head repetition."""
        batch_size = q.shape[0]
        seq_len = q.shape[2]
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[3]
        
        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            scale=self.self_attn.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)
        out, _ = self.self_attn.o_proj(attn_output)
        return out


class HiDreamTextModel(nn.Module):
    """Text decoder with TP-optimized layers and DeepStack visual injection."""

    def __init__(self, cfg, prefix: str = ""):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size, prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList([
            HiDreamTextDecoderLayer(cfg, i, prefix=f"{prefix}.layers.{i}")
            for i in range(cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = HiDreamTextRotaryEmbedding(
            cfg.head_dim, cfg.rope_theta, list(cfg.mrope_section),
            cfg.max_position_embeddings,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local
        return hidden_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(position_ids)
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states, position_embeddings, attention_mask
            )
            if (deepstack_visual_embeds is not None
                    and visual_pos_masks is not None
                    and layer_idx < len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
        return self.norm(hidden_states)


class BottleneckPatchEmbed(nn.Module):
    """Two-layer linear with PCA bottleneck for noise patch embedding."""

    def __init__(self, patch_size: int, in_chans: int, pca_dim: int, embed_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(patch_size * patch_size * in_chans, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t * 1000, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class FinalLayer(nn.Module):
    """Linear projection from hidden to patch predictions."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class HiDreamForCausalLM(BaseDiT):
    """HiDream-O1-Image unified VLM diffusion model.

    Wraps vision encoder, text decoder, and diffusion heads into a single
    model that performs text-to-image and image-to-image generation via
    flow matching denoising.
    """

    _fsdp_shard_conditions = []
    param_names_mapping = {}
    _compile_conditions = []
    _supported_attention_backends = {
        AttentionBackendEnum.TORCH_SDPA,
    }

    def __init__(self, config: HiDreamDitConfig, hf_config: dict[str, Any], **kwargs):
        super().__init__(config=config, hf_config=hf_config, **kwargs)
        ac = self.config

        self.hidden_size = ac.hidden_size
        self.num_attention_heads = ac.num_attention_heads
        self.num_channels_latents = ac.num_channels_latents

        self.visual = HiDreamVisionModel(ac)

        self.language_model = HiDreamTextModel(ac, prefix="language_model")

        self.patch_size = ac.dit_patch_size
        self.in_channels = ac.in_channels
        self.t_embedder1 = TimestepEmbedder(ac.hidden_size, ac.frequency_embedding_size)
        self.x_embedder = BottleneckPatchEmbed(
            ac.dit_patch_size, ac.in_channels, ac.bottleneck_dim, ac.hidden_size,
        )
        self.final_layer2 = FinalLayer(ac.hidden_size, ac.dit_patch_size, ac.in_channels)
        self.tms_token_id = ac.tms_token_id

        self.image_token_id = ac.image_token_id
        self.video_token_id = ac.video_token_id
        self.vision_start_token_id = ac.vision_start_token_id

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.embed_tokens

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pixel_values = pixel_values.to(self.visual.patch_embed.proj.weight.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        special_image_mask = input_ids == self.image_token_id
        n_image_tokens = special_image_mask.sum()
        special_image_mask_3d = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            total_feat = sum(f.numel() for f in (image_features if isinstance(image_features, (list, tuple)) else [image_features]))
            if inputs_embeds[special_image_mask_3d].numel() != total_feat:
                raise ValueError(
                    f"Image features and tokens mismatch: tokens={n_image_tokens}, features={total_feat}"
                )
        return special_image_mask, None

    def _run_decoder_sdpa(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        token_types: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Two-pass SDPA attention: causal on AR tokens, full on all tokens."""
        text_model = self.language_model
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        position_embeddings = text_model.rotary_emb(position_ids)
        cos, sin = position_embeddings

        is_gen = token_types[0].bool()
        idx_ar = torch.nonzero(~is_gen, as_tuple=False).squeeze(-1)

        hidden_states = inputs_embeds
        batch_size, total_seq_len, _ = hidden_states.shape
        head_dim = text_model.layers[0].self_attn.head_dim
        dtype = hidden_states.dtype
        min_val = torch.finfo(dtype).min

        for layer_idx, decoder_layer in enumerate(text_model.layers):
            residual = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)
            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)
            attn = decoder_layer.self_attn

            q = attn.q_norm(attn.q_proj(normed)[0].view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(normed)[0].view(hidden_shape)).transpose(1, 2)
            v = attn.v_proj(normed)[0].view(hidden_shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            num_heads = q.shape[1]
            num_kv_heads = k.shape[1]
            if num_kv_heads < num_heads:
                repeat_factor = num_heads // num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)

            softmax_scale = attn.scaling

            if idx_ar.numel() > 0:
                q_ar = q[:, :, idx_ar].contiguous()
                k_ar = k[:, :, idx_ar].contiguous()
                v_ar = v[:, :, idx_ar].contiguous()
                out_ar = F.scaled_dot_product_attention(
                    q_ar, k_ar, v_ar, is_causal=True, scale=softmax_scale,
                )
            else:
                out_ar = None

            out_full = F.scaled_dot_product_attention(
                q, k, v, scale=softmax_scale,
            )

            if out_ar is not None:
                out_full = out_full.clone()
                out_full[:, :, idx_ar] = out_ar

            attn_output = out_full.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            attn_out, _ = attn.o_proj(attn_output)
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = residual + decoder_layer.mlp(
                decoder_layer.post_attention_layernorm(hidden_states)
            )

            if (deepstack_visual_embeds is not None
                    and visual_pos_masks is not None
                    and layer_idx < len(deepstack_visual_embeds)):
                hidden_states = text_model._deepstack_process(
                    hidden_states, visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        return text_model.norm(hidden_states)

    def _forward_generation(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        use_flash_attn: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for one denoising step; returns {'x_pred': patches}."""
        precomputed_image_embeds = kwargs.pop("precomputed_image_embeds", None)
        precomputed_deepstack = kwargs.pop("precomputed_deepstack_image_embeds", None)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        deepstack_image_embeds = None

        if pixel_values is not None:
            if precomputed_image_embeds is not None and precomputed_deepstack is not None:
                image_embeds = precomputed_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                deepstack_image_embeds = [
                    e.to(inputs_embeds.device, inputs_embeds.dtype)
                    for e in precomputed_deepstack
                ]
            else:
                image_embeds_list, deepstack_image_embeds = self.get_image_features(
                    pixel_values, image_grid_thw
                )
                image_embeds = torch.cat(image_embeds_list, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
            )

        visual_pos_masks = None
        if image_mask is not None:
            visual_pos_masks = image_mask[..., 0] if image_mask.dim() == 3 else image_mask

        if isinstance(timestep, list):
            timestep = torch.cat(timestep, dim=0)
        timestep = timestep.to(inputs_embeds.device)
        t_emb = self.t_embedder1(timestep)
        tms_mask = (input_ids == self.tms_token_id)
        tms_mask_3d = tms_mask.unsqueeze(-1).expand_as(inputs_embeds)
        t_emb_expanded = t_emb.unsqueeze(1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(tms_mask_3d, t_emb_expanded, inputs_embeds)

        if isinstance(vinputs, list):
            vinputs = torch.cat(vinputs, dim=0)
        vinputs = vinputs.to(inputs_embeds.device)
        vinputs_embedded = self.x_embedder(vinputs).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

        batch_size, total_seq_len, _ = inputs_embeds.shape

        if visual_pos_masks is not None:
            vinputs_seq_len = vinputs_embedded.shape[1]
            if visual_pos_masks.shape[0] != batch_size:
                visual_pos_masks = visual_pos_masks.expand(batch_size, -1)
            pad = torch.zeros(
                visual_pos_masks.shape[0], vinputs_seq_len,
                dtype=visual_pos_masks.dtype, device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, pad], dim=1)

        if isinstance(token_types, list):
            token_types = torch.cat(token_types, dim=0)
        token_types = token_types.to(inputs_embeds.device)
        if token_types.dim() == 1:
            token_types = token_types.unsqueeze(0)
        elif token_types.dim() == 2 and token_types.shape[-1] == 1:
            token_types = token_types.squeeze(-1).unsqueeze(0)
        if token_types.shape[0] == 1 and batch_size > 1:
            token_types = token_types.expand(batch_size, -1)

        if use_flash_attn:
            hidden_states = self._run_decoder_sdpa(
                inputs_embeds, position_ids, token_types,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_image_embeds,
            )
        else:
            dtype = inputs_embeds.dtype
            min_val = torch.finfo(dtype).min
            attn_masks = []
            for b in range(batch_size):
                causal = torch.full(
                    (total_seq_len, total_seq_len), min_val,
                    device=inputs_embeds.device, dtype=dtype,
                )
                causal = torch.triu(causal, diagonal=1)
                gen_positions = token_types[b].bool()
                causal[gen_positions, :] = 0
                attn_masks.append(causal)
            attention_mask_4d = torch.stack(attn_masks, dim=0).unsqueeze(1)

            hidden_states = self.language_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask_4d,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_image_embeds,
            )

        x_pred = self.final_layer2(hidden_states)
        out = {"x_pred": x_pred}
        if pixel_values is not None:
            # vision embeds cached across denoising steps
            out["cond_image_embeds"] = image_embeds
            out["cond_deepstack_image_embeds"] = deepstack_image_embeds
        return out

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        vinputs: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        token_types: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        use_flash_attn: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Main forward entry point. Dispatches to _forward_generation for diffusion."""
        if vinputs is not None:
            return self._forward_generation(
                input_ids=input_ids,
                position_ids=position_ids,
                vinputs=vinputs,
                timestep=timestep,
                token_types=token_types,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_flash_attn=use_flash_attn,
                **kwargs,
            )
        raise ValueError("HiDreamForCausalLM requires vinputs for generation mode")


EntryClass = HiDreamForCausalLM
