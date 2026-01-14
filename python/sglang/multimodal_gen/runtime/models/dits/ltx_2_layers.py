from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

# --- Utils ---


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# --- RoPE ---


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    STACKED = "stacked"


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis, rope_type: LTXRopeType = LTXRopeType.INTERLEAVED
) -> torch.Tensor:
    if freqs_cis is None:
        return x

    if isinstance(freqs_cis, tuple) and len(freqs_cis) == 2:
        cos, sin = freqs_cis
    elif torch.is_tensor(freqs_cis) and freqs_cis.is_complex():
        cos = freqs_cis.real
        sin = freqs_cis.imag
    else:
        raise TypeError(
            "freqs_cis must be (cos, sin) tuple or a complex tensor, got "
            f"{type(freqs_cis)}"
        )

    if cos.ndim == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif cos.ndim == 3:
        # [B, L, D] -> [B, L, 1, D] for broadcasting across heads
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    elif cos.ndim != 4:
        raise ValueError(f"Unexpected RoPE tensor rank: cos.ndim={cos.ndim}")

    if x.ndim != 3:
        raise ValueError(
            f"Expected x to be rank-3 [B, L, H*D], got shape={tuple(x.shape)}"
        )

    b, l, hd = x.shape
    head_dim_half = int(cos.shape[-1])
    if head_dim_half <= 0:
        return x
    if (hd % (head_dim_half)) != 0:
        raise ValueError(
            f"RoPE dim mismatch: x last dim={hd}, rope half dim={head_dim_half}"
        )

    num_heads = hd // head_dim_half
    head_dim = head_dim_half

    x = x.view(b, l, num_heads, head_dim)
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)

    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)  # [B, L, H, D/2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)  # [B, L, H, D]

    out = x.float() * cos.float() + x_rotated.float() * sin.float()
    return out.to(dtype=x.dtype).view(b, l, num_heads * head_dim)


# --- FeedForward ---


class GELUApprox(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        self.net = nn.Sequential(
            GELUApprox(dim, inner_dim),
            nn.Identity(),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- Timestep Embedding ---


class PixArtAlphaTextProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
    ) -> None:
        super().__init__()
        if out_features is None:
            out_features = hidden_size

        self.linear_1 = nn.Linear(
            in_features=in_features, out_features=hidden_size, bias=True
        )
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(
            in_features=hidden_size, out_features=out_features, bias=True
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, size_emb_dim: int):
        super().__init__()
        self.outdim = size_emb_dim
        # Match ltx-core / diffusers weights:
        # - `time_proj` is a fixed sinusoidal projection (no parameters)
        # - `timestep_embedder` exposes `linear_1` / `linear_2`
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        if hidden_dtype is None:
            hidden_dtype = timestep.dtype

        if timestep.ndim != 1:
            timestep = timestep.view(-1)

        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        return timesteps_emb


# --- AdaLN ---


class AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            embedding_dim, embedding_coefficient * embedding_dim, bias=True
        )

    def forward(
        self,
        timestep: torch.Tensor,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if timestep.ndim != 1:
            timestep = timestep.view(-1)
        if batch_size is None:
            batch_size = timestep.shape[0]
        if timestep.shape[0] != batch_size:
            raise ValueError(
                f"timestep length {timestep.shape[0]} must equal batch_size {batch_size}"
            )
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

        # Match diffusers checkpoint naming.
        self.norm_q = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.norm_k = nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()
        )

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

        q = self.norm_q(q)
        k = self.norm_k(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        attn_mask = mask
        if attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = (1 - attn_mask.to(dtype=q.dtype)) * -1000000.0
            attn_mask = attn_mask[:, None, None, :]

        b, _, _ = q.shape
        q = q.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.dim_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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

            # Match diffusers checkpoint naming.
            self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(
                torch.empty(5, audio.dim)
            )
            self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(
                torch.empty(5, video.dim)
            )

        self.norm_eps = norm_eps

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> Tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        dim = scale_shift_table.shape[1]
        if timestep.ndim > 2:
            timestep = timestep.reshape(batch_size, -1)
        if timestep.ndim == 2 and timestep.shape[0] == batch_size:
            ts = timestep.view(batch_size, num_ada_params, dim)
        else:
            ts = timestep.view(batch_size, num_ada_params, dim)

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
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
            slice(None, None),
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
            slice(None, None),
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

            vshift_msa = vshift_msa[:, None, :]
            vscale_msa = vscale_msa[:, None, :]
            vgate_msa = vgate_msa[:, None, :]
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            vx = vx + self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa

            vx = vx + self.attn2(
                rms_norm(vx, eps=self.norm_eps),
                context=video.context,
                mask=video.context_mask,
            )

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, batch_size, audio.timesteps, slice(0, 3)
            )

            ashift_msa = ashift_msa[:, None, :]
            ascale_msa = ascale_msa[:, None, :]
            agate_msa = agate_msa[:, None, :]
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            ax = (
                ax
                + self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa
            )

            ax = ax + self.audio_attn2(
                rms_norm(ax, eps=self.norm_eps),
                context=audio.context,
                mask=audio.context_mask,
            )

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
                self.audio_a2v_cross_attn_scale_shift_table,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.video_a2v_cross_attn_scale_shift_table,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            scale_ca_audio_hidden_states_a2v = scale_ca_audio_hidden_states_a2v[
                :, None, :
            ]
            shift_ca_audio_hidden_states_a2v = shift_ca_audio_hidden_states_a2v[
                :, None, :
            ]
            scale_ca_audio_hidden_states_v2a = scale_ca_audio_hidden_states_v2a[
                :, None, :
            ]
            shift_ca_audio_hidden_states_v2a = shift_ca_audio_hidden_states_v2a[
                :, None, :
            ]
            gate_out_v2a = gate_out_v2a[:, None, :]

            scale_ca_video_hidden_states_a2v = scale_ca_video_hidden_states_a2v[
                :, None, :
            ]
            shift_ca_video_hidden_states_a2v = shift_ca_video_hidden_states_a2v[
                :, None, :
            ]
            scale_ca_video_hidden_states_v2a = scale_ca_video_hidden_states_v2a[
                :, None, :
            ]
            shift_ca_video_hidden_states_v2a = shift_ca_video_hidden_states_v2a[
                :, None, :
            ]
            gate_out_a2v = gate_out_a2v[:, None, :]

            # Simplified Cross Attention Logic (omitting some details for brevity)
            # In real implementation, we need to apply scale/shift and call attn

            if run_a2v:
                vx_scaled = (
                    vx_norm3 * (1 + scale_ca_video_hidden_states_a2v)
                    + shift_ca_video_hidden_states_a2v
                )
                ax_scaled = (
                    ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v)
                    + shift_ca_audio_hidden_states_a2v
                )
                vx = (
                    vx
                    + self.audio_to_video_attn(vx_scaled, context=ax_scaled)
                    * gate_out_a2v
                )

            if run_v2a:
                ax_scaled = (
                    ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a)
                    + shift_ca_audio_hidden_states_v2a
                )
                vx_scaled = (
                    vx_norm3 * (1 + scale_ca_video_hidden_states_v2a)
                    + shift_ca_video_hidden_states_v2a
                )
                ax = (
                    ax
                    + self.video_to_audio_attn(ax_scaled, context=vx_scaled)
                    * gate_out_v2a
                )

        # Feed Forward
        if run_vx:
            vshift_ff, vscale_ff, vgate_ff = self.get_ada_values(
                self.scale_shift_table, batch_size, video.timesteps, slice(3, 6)
            )
            vshift_ff = vshift_ff[:, None, :]
            vscale_ff = vscale_ff[:, None, :]
            vgate_ff = vgate_ff[:, None, :]
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_ff) + vshift_ff
            vx = vx + self.ff(norm_vx) * vgate_ff
            video.x = vx

        if run_ax:
            ashift_ff, ascale_ff, agate_ff = self.get_ada_values(
                self.audio_scale_shift_table, batch_size, audio.timesteps, slice(3, 6)
            )
            ashift_ff = ashift_ff[:, None, :]
            ascale_ff = ascale_ff[:, None, :]
            agate_ff = agate_ff[:, None, :]
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_ff) + ashift_ff
            ax = ax + self.audio_ff(norm_ax) * agate_ff
            audio.x = ax

        return video, audio
