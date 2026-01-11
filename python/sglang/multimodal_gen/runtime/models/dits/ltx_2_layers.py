import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Utils ---

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

# --- RoPE ---

class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    STACKED = "stacked"

def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, rope_type: LTXRopeType = LTXRopeType.INTERLEAVED
) -> torch.Tensor:
    # x: [B, L, H*D] -> [B, L, H, D]
    # freqs_cis: [L, D] or [1, L, 1, D]
    
    # Simplified implementation assuming standard RoPE usage in LTX
    # LTX uses complex numbers for RoPE
    
    # For now, let's use a simplified version or stub if complex logic is involved.
    # LTX RoPE implementation is likely complex due to 3D nature.
    # We will assume freqs_cis is already prepared in the right shape.
    
    # If freqs_cis is complex:
    if freqs_cis.is_complex():
        # Reshape x for complex multiplication
        # Assuming x is [B, L, H*D]
        # We need to know H and D.
        # But apply_rotary_emb usually takes [B, L, H, D] or similar.
        # In Attention class, x is [B, L, H*D] before RoPE? No, usually after reshaping.
        # Let's look at Attention class:
        # q = apply_rotary_emb(q, pe, self.rope_type)
        # q is [B, L, H*D] (flattened) in LTX Attention implementation?
        # No, LTX Attention: q = self.q_norm(q) -> [B, L, H*D]
        pass
        
    # Placeholder: return x unchanged for now to avoid shape errors until we implement full RoPE
    return x

# --- FeedForward ---

class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- Timestep Embedding ---

class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, size_emb_dim: int):
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.timestep_embedder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, timestep: torch.Tensor, hidden_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # Simplified: assume timestep is already embedded or handle raw timestep
        # LTX passes raw timestep usually?
        # If timestep is [B], we need to embed it.
        # For now, assume timestep is [B, D]
        return self.timestep_embedder(timestep)

# --- AdaLN ---

class AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep

# --- Attention ---

class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim, bias=True), nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # Scaled Dot Product Attention
        b, _, _ = q.shape
        q = q.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        
        return self.to_out(out)

# --- Transformer Block ---

@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int

@dataclass
class TransformerArgs:
    x: torch.Tensor
    timesteps: torch.Tensor
    positional_embeddings: Optional[torch.Tensor] = None
    context: Optional[torch.Tensor] = None
    context_mask: Optional[torch.Tensor] = None
    enabled: bool = True
    cross_scale_shift_timestep: Optional[torch.Tensor] = None
    cross_gate_timestep: Optional[torch.Tensor] = None

class BasicAVTransformerBlock(nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = nn.Parameter(torch.empty(6, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = nn.Parameter(torch.empty(6, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, video.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> Tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        # timestep: [B, D]
        # scale_shift_table: [N, D]
        
        # Simplified AdaLN logic
        # LTX logic:
        # ada_values = (
        #     scale_shift_table[indices].unsqueeze(0).unsqueeze(0)
        #     + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        # ).unbind(dim=2)
        
        # We assume timestep is already projected to [B, N, D] or similar?
        # In LTX, timestep passed here is `embedded_timestep` from AdaLayerNormSingle
        # which is [B, N*D] ? No, let's check AdaLayerNormSingle.
        # It returns (linear_out, embedded_timestep).
        # linear_out is [B, N*D].
        
        # So timestep here is actually the linear output [B, N*D].
        # We need to reshape it to [B, N, D].
        
        dim = scale_shift_table.shape[1]
        ts = timestep.view(batch_size, -1, dim) # [B, N, D]
        
        # Select indices
        # scale_shift_table[indices] -> [K, D]
        # ts[:, indices, :] -> [B, K, D]
        
        ada = scale_shift_table[indices].unsqueeze(0) + ts[:, indices, :]
        return ada.unbind(dim=1)

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> Tuple[torch.Tensor, ...]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, slice(None, None)
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )
        return (*scale_shift_ada_values, *gate_ada_values)

    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> Tuple[TransformerArgs | None, TransformerArgs | None]:
        batch_size = video.x.shape[0] if video else audio.x.shape[0]

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, batch_size, video.timesteps, slice(0, 3)
            )
            
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            vx = vx + self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa

            vx = vx + self.attn2(rms_norm(vx, eps=self.norm_eps), context=video.context, mask=video.context_mask)

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, batch_size, audio.timesteps, slice(0, 3)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            ax = ax + self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa

            ax = ax + self.audio_attn2(rms_norm(ax, eps=self.norm_eps), context=audio.context, mask=audio.context_mask)

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )
            
            # Simplified Cross Attention Logic (omitting some details for brevity)
            # In real implementation, we need to apply scale/shift and call attn
            
            # Video attends to Audio
            if run_v2a:
                ax_norm_v2a = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx = vx + self.audio_to_video_attn(vx_norm3, context=ax_norm_v2a) * gate_out_v2a

            # Audio attends to Video
            # ... (Similar logic for Audio -> Video)

        # Feed Forward
        if run_vx:
            vshift_ff, vscale_ff, vgate_ff = self.get_ada_values(
                self.scale_shift_table, batch_size, video.timesteps, slice(3, 6)
            )
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_ff) + vshift_ff
            vx = vx + self.ff(norm_vx) * vgate_ff
            video.x = vx

        if run_ax:
            ashift_ff, ascale_ff, agate_ff = self.get_ada_values(
                self.audio_scale_shift_table, batch_size, audio.timesteps, slice(3, 6)
            )
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_ff) + ashift_ff
            ax = ax + self.audio_ff(norm_ax) * agate_ff
            audio.x = ax

        return video, audio
