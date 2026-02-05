# Copied and adapted from LTX-2 and WanVideo implementations.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2ArchConfig, LTX2Config
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_rank,
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import timestep_embedding
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def apply_interleaved_rotary_emb(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def apply_split_rotary_emb(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs
    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        b = x.shape[0]
        _, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(
            f"Expected x.shape[-1] to be even for split rotary, got {last}."
        )
    r = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, r).float()
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]
    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)
    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(b, t, -1)
    return out.to(dtype=x_dtype)


# ==============================================================================
# Layers and Embeddings
# ==============================================================================


class LTX2AudioVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: Tuple[int, ...] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.patch_size_t = int(patch_size_t)

        if rope_type not in ["interleaved", "split"]:
            raise ValueError(
                f"{rope_type=} not supported. Choose between 'interleaved' and 'split'."
            )
        self.rope_type = rope_type

        self.base_num_frames = int(base_num_frames)
        self.num_attention_heads = int(num_attention_heads)

        self.base_height = int(base_height)
        self.base_width = int(base_width)

        self.sampling_rate = int(sampling_rate)
        self.hop_length = int(hop_length)
        self.audio_latents_per_second = (
            float(self.sampling_rate) / float(self.hop_length) / float(scale_factors[0])
        )

        self.scale_factors = tuple(int(x) for x in scale_factors)
        self.theta = float(theta)
        self.causal_offset = int(causal_offset)

        self.modality = modality
        if self.modality not in ["video", "audio"]:
            raise ValueError(
                f"Modality {modality} is not supported. Supported modalities are `video` and `audio`."
            )
        self.double_precision = bool(double_precision)

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: float = 24.0,
        *,
        start_frame: int = 0,
    ) -> torch.Tensor:
        grid_f = torch.arange(
            start=int(start_frame),
            end=int(num_frames) + int(start_frame),
            step=self.patch_size_t,
            dtype=torch.float32,
            device=device,
        )
        grid_h = torch.arange(
            start=0,
            end=height,
            step=self.patch_size,
            dtype=torch.float32,
            device=device,
        )
        grid_w = torch.arange(
            start=0,
            end=width,
            step=self.patch_size,
            dtype=torch.float32,
            device=device,
        )
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)

        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = torch.tensor(
            patch_size, dtype=grid.dtype, device=grid.device
        )
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        latent_coords = torch.stack([grid, patch_ends], dim=-1)
        latent_coords = latent_coords.flatten(1, 3)
        latent_coords = latent_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        scale_tensor = torch.tensor(self.scale_factors, device=latent_coords.device)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
        pixel_coords[:, 0, ...] = (
            pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]
        ).clamp(min=0)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
        return pixel_coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        *,
        start_frame: int = 0,
    ) -> torch.Tensor:
        grid_f = torch.arange(
            start=int(start_frame),
            end=int(num_frames) + int(start_frame),
            step=self.patch_size_t,
            dtype=torch.float32,
            device=device,
        )

        audio_scale_factor = self.scale_factors[0]
        grid_start_mel = grid_f * audio_scale_factor
        grid_start_mel = (
            grid_start_mel + self.causal_offset - audio_scale_factor
        ).clip(min=0)
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate

        grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
        grid_end_mel = (grid_end_mel + self.causal_offset - audio_scale_factor).clip(
            min=0
        )
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate

        audio_coords = torch.stack([grid_start_s, grid_end_s], dim=-1)
        audio_coords = audio_coords.unsqueeze(0).expand(batch_size, -1, -1)
        audio_coords = audio_coords.unsqueeze(1)
        return audio_coords

    def prepare_coords(self, *args, **kwargs):
        if self.modality == "video":
            return self.prepare_video_coords(*args, **kwargs)
        return self.prepare_audio_coords(*args, **kwargs)

    def forward(
        self, coords: torch.Tensor, device: Optional[Union[str, torch.device]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or coords.device
        num_pos_dims = coords.shape[1]

        if coords.ndim == 4:
            coords_start, coords_end = coords.chunk(2, dim=-1)
            coords = (coords_start + coords_end) / 2.0
            coords = coords.squeeze(-1)

        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        else:
            max_positions = (self.base_num_frames,)

        grid = torch.stack(
            [coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1
        ).to(device)

        num_rope_elems = num_pos_dims * 2
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(
                start=0.0,
                end=1.0,
                steps=self.dim // num_rope_elems,
                dtype=freqs_dtype,
                device=device,
            ),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)

        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs
        freqs = freqs.transpose(-1, -2).flatten(2)

        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = torch.ones_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                sin_padding = torch.zeros_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)
        else:
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()

            if pad_size != 0:
                cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
                cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
                sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

            b = cos_freq.shape[0]
            t = cos_freq.shape[1]
            cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
            sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)
            cos_freqs = torch.swapaxes(cos_freq, 1, 2)
            sin_freqs = torch.swapaxes(sin_freq, 1, 2)

        return cos_freqs, sin_freqs


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=eps)


class LTX2TextProjection(nn.Module):
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

        self.linear_1 = ColumnParallelLinear(
            in_features, hidden_size, bias=True, gather_output=True
        )
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")

        self.linear_2 = ColumnParallelLinear(
            hidden_size, out_features, bias=True, gather_output=True
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class LTX2TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim: int, in_channels: int = 256) -> None:
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            in_channels, embedding_dim, bias=True, gather_output=True
        )
        self.linear_2 = ColumnParallelLinear(
            embedding_dim, embedding_dim, bias=True, gather_output=True
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_1(t_emb)
        x = F.silu(x)
        x, _ = self.linear_2(x)
        return x


class LTX2PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = LTX2TimestepEmbedder(embedding_dim, in_channels=256)

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        t = timestep.reshape(-1).to(dtype=torch.float32)
        t_emb = timestep_embedding(t, dim=256, max_period=10000, dtype=torch.float32)
        if hidden_dtype is not None:
            t_emb = t_emb.to(dtype=hidden_dtype)
        return self.timestep_embedder(t_emb)


class LTX2AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6) -> None:
        super().__init__()
        self.emb = LTX2PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim)
        self.silu = nn.SiLU()
        self.linear = ColumnParallelLinear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            bias=True,
            gather_output=True,
        )

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype).to(
            dtype=self.linear.weight.dtype
        )
        out, _ = self.linear(self.silu(embedded_timestep))
        return out, embedded_timestep


class LTX2TPRMSNormAcrossHeads(nn.Module):
    def __init__(
        self, full_hidden_size: int, local_hidden_size: int, eps: float
    ) -> None:
        super().__init__()
        self.full_hidden_size = full_hidden_size
        self.local_hidden_size = local_hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(local_hidden_size))

        tp_rank = get_tp_rank()

        def _weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            shard = loaded_weight.narrow(
                0, tp_rank * local_hidden_size, local_hidden_size
            )
            param.data.copy_(shard.to(dtype=param.dtype, device=param.device))

        setattr(self.weight, "weight_loader", _weight_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep track of the original dtype. We do the statistics in fp32 for
        # numerical stability, but cast the output back to the input dtype to
        orig_dtype = x.dtype
        if get_tp_world_size() == 1:
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
        else:
            local_sumsq = x.float().pow(2).sum(dim=-1, keepdim=True)
            global_sumsq = tensor_model_parallel_all_reduce(local_sumsq)
            var = global_sumsq / float(self.full_hidden_size)

        inv_rms_fp32 = torch.rsqrt(var + self.eps)
        y = (x.float() * inv_rms_fp32).to(dtype=orig_dtype)
        return y * self.weight.to(dtype=orig_dtype)


class LTX2Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        qk_norm: bool = True,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.query_dim = int(query_dim)
        self.context_dim = int(query_dim if context_dim is None else context_dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.inner_dim = self.heads * self.dim_head
        self.norm_eps = float(norm_eps)
        self.qk_norm = bool(qk_norm)

        tp_size = get_tp_world_size()
        if tp_size <= 0:
            raise ValueError(f"Invalid {tp_size=}. Expected tp_size >= 1.")
        if self.heads % tp_size != 0:
            raise ValueError(
                f"LTX2Attention requires heads divisible by tp_size, got "
                f"{self.heads=} {tp_size=}."
            )
        if self.inner_dim % tp_size != 0:
            # This should follow from heads % tp_size, but keep explicit for clarity.
            raise ValueError(
                f"LTX2Attention requires inner_dim divisible by tp_size, got "
                f"{self.inner_dim=} {tp_size=}."
            )
        self.local_heads = self.heads // tp_size

        self.to_q = ColumnParallelLinear(
            self.query_dim, self.inner_dim, bias=True, gather_output=False
        )
        self.to_k = ColumnParallelLinear(
            self.context_dim, self.inner_dim, bias=True, gather_output=False
        )
        self.to_v = ColumnParallelLinear(
            self.context_dim, self.inner_dim, bias=True, gather_output=False
        )

        self.q_norm: nn.Module | None = None
        self.k_norm: nn.Module | None = None
        if self.qk_norm:
            if tp_size == 1:
                self.q_norm = torch.nn.RMSNorm(self.inner_dim, eps=self.norm_eps)
                self.k_norm = torch.nn.RMSNorm(self.inner_dim, eps=self.norm_eps)
            else:
                self.q_norm = LTX2TPRMSNormAcrossHeads(
                    full_hidden_size=self.inner_dim,
                    local_hidden_size=self.inner_dim // tp_size,
                    eps=self.norm_eps,
                )
                self.k_norm = LTX2TPRMSNormAcrossHeads(
                    full_hidden_size=self.inner_dim,
                    local_hidden_size=self.inner_dim // tp_size,
                    eps=self.norm_eps,
                )

        self.to_out = nn.Sequential(
            RowParallelLinear(
                self.inner_dim, self.query_dim, bias=True, input_is_parallel=True
            ),
            nn.Identity(),
        )

        self.attn = USPAttention(
            num_heads=self.local_heads,
            head_size=self.dim_head,
            num_kv_heads=self.local_heads,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        q, _ = self.to_q(x)
        context_ = x if context is None else context
        k, _ = self.to_k(context_)
        v, _ = self.to_v(context_)

        if self.qk_norm:
            assert self.q_norm is not None and self.k_norm is not None
            q = self.q_norm(q)
            k = self.k_norm(k)

        if pe is not None:
            cos, sin = pe
            k_cos, k_sin = pe if k_pe is None else k_pe
            tp_size = get_tp_world_size()
            if tp_size > 1:
                tp_rank = get_tp_rank()
                cos, sin = self._slice_rope_for_tp(
                    cos, sin, tp_rank=tp_rank, tp_size=tp_size
                )
                k_cos, k_sin = self._slice_rope_for_tp(
                    k_cos, k_sin, tp_rank=tp_rank, tp_size=tp_size
                )
            if cos.dim() == 3:
                q = apply_interleaved_rotary_emb(q, (cos, sin))
                k = apply_interleaved_rotary_emb(k, (k_cos, k_sin))
            else:
                q = apply_split_rotary_emb(q, (cos, sin))
                k = apply_split_rotary_emb(k, (k_cos, k_sin))

        q = q.view(*q.shape[:-1], self.local_heads, self.dim_head)
        k = k.view(*k.shape[:-1], self.local_heads, self.dim_head)
        v = v.view(*v.shape[:-1], self.local_heads, self.dim_head)

        if mask is not None:
            # Fallback to SDPA for masked attention
            q_ = q.transpose(1, 2)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)

            if torch.is_floating_point(mask):
                m = mask
                if m.dim() == 2:
                    m = m[:, None, None, :]
                elif m.dim() == 3:
                    m = m[:, None, :, :]
                sdpa_mask = m.to(dtype=q_.dtype, device=q_.device)
            else:
                m = mask.to(dtype=q_.dtype, device=q_.device)
                if m.dim() == 2:
                    m = m[:, None, None, :]
                elif m.dim() == 3:
                    m = m[:, None, :, :]
                sdpa_mask = (m - 1.0) * torch.finfo(q_.dtype).max

            out = torch.nn.functional.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
            ).transpose(1, 2)
        else:
            out = self.attn(q, k, v)

        out = out.flatten(2)
        out, _ = self.to_out[0](out)
        return out

    def _slice_rope_for_tp(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice RoPE tensors to the local TP shard.

        - split-rope: cos/sin are shaped [B, H, T, R] (head-major), slice by heads.
        - interleaved-rope: cos/sin are shaped [B, T, D], where D matches the projected
          feature dimension and is sharded by TP.
        """
        if cos.ndim == 4:
            # [B, H, T, R]
            start = tp_rank * self.local_heads
            end = start + self.local_heads
            return cos[:, start:end, :, :], sin[:, start:end, :, :]
        elif cos.ndim == 3:
            # [B, T, D]
            d = cos.shape[-1]
            if d % tp_size != 0:
                raise ValueError(
                    f"RoPE dim must be divisible by tp_size, got {d=} {tp_size=}."
                )
            local_d = d // tp_size
            start = tp_rank * local_d
            end = start + local_d
            return cos[:, :, start:end], sin[:, :, start:end]
        raise ValueError(f"Unexpected RoPE tensor rank: {cos.ndim}. Expected 3 or 4.")


class LTX2FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4) -> None:
        super().__init__()
        if dim_out is None:
            dim_out = dim
        inner_dim = int(dim * mult)

        self.proj_in = ColumnParallelLinear(
            dim, inner_dim, bias=True, gather_output=True
        )
        self.act = nn.GELU(approximate="tanh")
        self.proj_out = ColumnParallelLinear(
            inner_dim, dim_out, bias=True, gather_output=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.proj_in(x)
        x = self.act(x)
        x, _ = self.proj_out(x)
        return x


class LTX2TransformerBlock(nn.Module):
    def __init__(
        self,
        idx: int,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_dim: int,
        audio_num_attention_heads: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps

        # 1. Self-Attention (video and audio)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1",
        )
        self.audio_attn1 = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.audio_attn1",
        )

        # 2. Prompt Cross-Attention
        self.attn2 = LTX2Attention(
            query_dim=dim,
            context_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn2",
        )
        self.audio_attn2 = LTX2Attention(
            query_dim=audio_dim,
            context_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.audio_attn2",
        )

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        self.audio_to_video_attn = LTX2Attention(
            query_dim=dim,
            context_dim=audio_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.audio_to_video_attn",
        )
        self.video_to_audio_attn = LTX2Attention(
            query_dim=audio_dim,
            context_dim=dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.video_to_audio_attn",
        )

        # 4. Feedforward layers
        self.ff = LTX2FeedForward(dim, dim_out=dim)
        self.audio_ff = LTX2FeedForward(audio_dim, dim_out=audio_dim)

        # 5. Modulation Parameters
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(
            torch.randn(6, audio_dim) / audio_dim**0.5
        )
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, dim))
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(
            torch.randn(5, audio_dim)
        )

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = int(scale_shift_table.shape[0])
        ada_values = (
            scale_shift_table[indices]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[
                :, :, indices, :
            ]
        ).unbind(dim=2)
        return [t.squeeze(2) for t in ada_values]

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
        temb_ca_scale_shift: torch.Tensor,
        temb_ca_audio_scale_shift: torch.Tensor,
        temb_ca_gate: torch.Tensor,
        temb_ca_audio_gate: torch.Tensor,
        video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ca_video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ca_audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        a2v_cross_attention_mask: Optional[torch.Tensor] = None,
        v2a_cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = hidden_states.size(0)

        # 1. Video and Audio Self-Attention
        vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
            self.scale_shift_table, batch_size, temb, slice(0, 3)
        )
        norm_hidden_states = (
            rms_norm(hidden_states, self.norm_eps) * (1 + vscale_msa) + vshift_msa
        )
        attn_hidden_states = self.attn1(norm_hidden_states, pe=video_rotary_emb)
        hidden_states = hidden_states + attn_hidden_states * vgate_msa

        ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, batch_size, temb_audio, slice(0, 3)
        )
        norm_audio_hidden_states = (
            rms_norm(audio_hidden_states, self.norm_eps) * (1 + ascale_msa) + ashift_msa
        )
        attn_audio_hidden_states = self.audio_attn1(
            norm_audio_hidden_states, pe=audio_rotary_emb
        )
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states * agate_msa

        # 2. Prompt Cross-Attention
        norm_hidden_states = rms_norm(hidden_states, self.norm_eps)
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            context=encoder_hidden_states,
            mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_audio_hidden_states = rms_norm(audio_hidden_states, self.norm_eps)
        attn_audio_hidden_states = self.audio_attn2(
            norm_audio_hidden_states,
            context=audio_encoder_hidden_states,
            mask=audio_encoder_attention_mask,
        )
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states

        # 3. Audio-to-Video and Video-to-Audio Cross-Attention
        norm_hidden_states = rms_norm(hidden_states, self.norm_eps)
        norm_audio_hidden_states = rms_norm(audio_hidden_states, self.norm_eps)

        # Compute combined ada params
        video_per_layer_ca_scale_shift = self.video_a2v_cross_attn_scale_shift_table[
            :4, :
        ]
        video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[4:, :]

        video_ca_scale_shift_table = (
            video_per_layer_ca_scale_shift[None, None, :, :].to(
                dtype=temb_ca_scale_shift.dtype, device=temb_ca_scale_shift.device
            )
            + temb_ca_scale_shift.reshape(
                batch_size, temb_ca_scale_shift.shape[1], 4, -1
            )
        ).unbind(dim=2)
        video_ca_gate = (
            video_per_layer_ca_gate[None, None, :, :].to(
                dtype=temb_ca_gate.dtype, device=temb_ca_gate.device
            )
            + temb_ca_gate.reshape(batch_size, temb_ca_gate.shape[1], 1, -1)
        ).unbind(dim=2)

        (
            video_a2v_ca_scale,
            video_a2v_ca_shift,
            video_v2a_ca_scale,
            video_v2a_ca_shift,
        ) = [t.squeeze(2) for t in video_ca_scale_shift_table]
        a2v_gate = video_ca_gate[0].squeeze(2)

        audio_per_layer_ca_scale_shift = self.audio_a2v_cross_attn_scale_shift_table[
            :4, :
        ]
        audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[4:, :]

        audio_ca_scale_shift_table = (
            audio_per_layer_ca_scale_shift[None, None, :, :].to(
                dtype=temb_ca_audio_scale_shift.dtype,
                device=temb_ca_audio_scale_shift.device,
            )
            + temb_ca_audio_scale_shift.reshape(
                batch_size, temb_ca_audio_scale_shift.shape[1], 4, -1
            )
        ).unbind(dim=2)
        audio_ca_gate = (
            audio_per_layer_ca_gate[None, None, :, :].to(
                dtype=temb_ca_audio_gate.dtype, device=temb_ca_audio_gate.device
            )
            + temb_ca_audio_gate.reshape(batch_size, temb_ca_audio_gate.shape[1], 1, -1)
        ).unbind(dim=2)

        (
            audio_a2v_ca_scale,
            audio_a2v_ca_shift,
            audio_v2a_ca_scale,
            audio_v2a_ca_shift,
        ) = [t.squeeze(2) for t in audio_ca_scale_shift_table]
        v2a_gate = audio_ca_gate[0].squeeze(2)

        # A2V
        mod_norm_hidden_states = (
            norm_hidden_states * (1 + video_a2v_ca_scale) + video_a2v_ca_shift
        )
        mod_norm_audio_hidden_states = (
            norm_audio_hidden_states * (1 + audio_a2v_ca_scale) + audio_a2v_ca_shift
        )

        a2v_attn_hidden_states = self.audio_to_video_attn(
            mod_norm_hidden_states,
            context=mod_norm_audio_hidden_states,
            pe=ca_video_rotary_emb,
            k_pe=ca_audio_rotary_emb,
            mask=a2v_cross_attention_mask,
        )
        hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states

        # V2A
        mod_norm_hidden_states = (
            norm_hidden_states * (1 + video_v2a_ca_scale) + video_v2a_ca_shift
        )
        mod_norm_audio_hidden_states = (
            norm_audio_hidden_states * (1 + audio_v2a_ca_scale) + audio_v2a_ca_shift
        )

        v2a_attn_hidden_states = self.video_to_audio_attn(
            mod_norm_audio_hidden_states,
            context=mod_norm_hidden_states,
            pe=ca_audio_rotary_emb,
            k_pe=ca_video_rotary_emb,
            mask=v2a_cross_attention_mask,
        )
        audio_hidden_states = audio_hidden_states + v2a_gate * v2a_attn_hidden_states

        # 4. Feedforward
        vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
            self.scale_shift_table, batch_size, temb, slice(3, None)
        )
        norm_hidden_states = (
            rms_norm(hidden_states, self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
        )
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * vgate_mlp

        ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
            self.audio_scale_shift_table, batch_size, temb_audio, slice(3, None)
        )
        norm_audio_hidden_states = (
            rms_norm(audio_hidden_states, self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
        )
        audio_ff_output = self.audio_ff(norm_audio_hidden_states)
        audio_hidden_states = audio_hidden_states + audio_ff_output * agate_mlp

        return hidden_states, audio_hidden_states


class LTX2VideoTransformer3DModel(CachableDiT, OffloadableDiTMixin):
    _fsdp_shard_conditions = LTX2ArchConfig()._fsdp_shard_conditions
    _compile_conditions = LTX2ArchConfig()._compile_conditions
    _supported_attention_backends = LTX2ArchConfig()._supported_attention_backends
    param_names_mapping = LTX2ArchConfig().param_names_mapping
    reverse_param_names_mapping = LTX2ArchConfig().reverse_param_names_mapping
    lora_param_names_mapping = LTX2ArchConfig().lora_param_names_mapping

    def _validate_tp_config(self, *, arch: LTX2ArchConfig, tp_size: int) -> None:
        """Validate TP-related dimension constraints (fail-fast)."""
        if tp_size < 1:
            raise ValueError(f"Invalid tp_size={tp_size}. Expected tp_size >= 1.")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "video hidden_size must be divisible by num_attention_heads, got "
                f"{self.hidden_size=} {self.num_attention_heads=}."
            )
        if self.audio_hidden_size % self.audio_num_attention_heads != 0:
            raise ValueError(
                "audio_hidden_size must be divisible by audio_num_attention_heads, got "
                f"{self.audio_hidden_size=} {self.audio_num_attention_heads=}."
            )

        if tp_size == 1:
            return

        if self.num_attention_heads % tp_size != 0:
            raise ValueError(
                "num_attention_heads must be divisible by tp_size, got "
                f"{self.num_attention_heads=} {tp_size=}."
            )
        if self.audio_num_attention_heads % tp_size != 0:
            raise ValueError(
                "audio_num_attention_heads must be divisible by tp_size, got "
                f"{self.audio_num_attention_heads=} {tp_size=}."
            )
        if self.hidden_size % tp_size != 0:
            raise ValueError(
                "hidden_size must be divisible by tp_size for TP-sharded projections, got "
                f"{self.hidden_size=} {tp_size=}."
            )
        if self.audio_hidden_size % tp_size != 0:
            raise ValueError(
                "audio_hidden_size must be divisible by tp_size for TP-sharded projections, got "
                f"{self.audio_hidden_size=} {tp_size=}."
            )
        if int(arch.out_channels) % tp_size != 0:
            raise ValueError(
                "out_channels must be divisible by tp_size for TP-sharded output projection, got "
                f"{arch.out_channels=} {tp_size=}."
            )
        if int(arch.audio_out_channels) % tp_size != 0:
            raise ValueError(
                "audio_out_channels must be divisible by tp_size for TP-sharded output projection, got "
                f"{arch.audio_out_channels=} {tp_size=}."
            )

    def __init__(self, config: LTX2Config, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.audio_hidden_size = arch.audio_hidden_size
        self.audio_num_attention_heads = arch.audio_num_attention_heads
        self.norm_eps = arch.norm_eps

        tp_size = get_tp_world_size()
        self._validate_tp_config(arch=arch, tp_size=tp_size)

        # 1. Patchification input projections
        # Matches LTX2Config().param_names_mapping
        self.patchify_proj = ColumnParallelLinear(
            arch.in_channels, self.hidden_size, bias=True, gather_output=True
        )
        self.audio_patchify_proj = ColumnParallelLinear(
            arch.audio_in_channels,
            self.audio_hidden_size,
            bias=True,
            gather_output=True,
        )

        # 2. Prompt embeddings
        self.caption_projection = LTX2TextProjection(
            in_features=arch.caption_channels, hidden_size=self.hidden_size
        )
        self.audio_caption_projection = LTX2TextProjection(
            in_features=arch.caption_channels, hidden_size=self.audio_hidden_size
        )

        # 3. Timestep Modulation Params and Embedding
        self.adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size, embedding_coefficient=6
        )
        self.audio_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size, embedding_coefficient=6
        )

        # Global Cross Attention Modulation Parameters
        self.av_ca_video_scale_shift_adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size, embedding_coefficient=4
        )
        self.av_ca_a2v_gate_adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size, embedding_coefficient=1
        )
        self.av_ca_audio_scale_shift_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size, embedding_coefficient=4
        )
        self.av_ca_v2a_gate_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size, embedding_coefficient=1
        )

        # Output Layer Scale/Shift Modulation parameters
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.hidden_size) / self.hidden_size**0.5
        )
        self.audio_scale_shift_table = nn.Parameter(
            torch.randn(2, self.audio_hidden_size) / self.audio_hidden_size**0.5
        )

        hf_patch_size = int(hf_config.get("patch_size", 1))
        hf_patch_size_t = int(hf_config.get("patch_size_t", 1))
        self.patch_size = (hf_patch_size_t, hf_patch_size, hf_patch_size)

        hf_audio_patch_size = int(hf_config.get("audio_patch_size", 1))
        hf_audio_patch_size_t = int(hf_config.get("audio_patch_size_t", 1))

        rope_type = (
            arch.rope_type.value
            if hasattr(arch.rope_type, "value")
            else str(arch.rope_type)
        )
        rope_double_precision = bool(getattr(arch, "double_precision_rope", True))
        causal_offset = int(hf_config.get("causal_offset", 1))

        pos_embed_max_pos = int(arch.positional_embedding_max_pos[0])
        base_height = int(arch.positional_embedding_max_pos[1])
        base_width = int(arch.positional_embedding_max_pos[2])

        audio_pos_embed_max_pos = int(arch.audio_positional_embedding_max_pos[0])

        self.video_scale_factors = (8, 32, 32)
        self.audio_scale_factors = (4,)

        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.hidden_size,
            patch_size=hf_patch_size,
            patch_size_t=hf_patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=self.video_scale_factors,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.num_attention_heads,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.audio_hidden_size,
            patch_size=hf_audio_patch_size,
            patch_size_t=hf_audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=16000,
            hop_length=160,
            scale_factors=self.audio_scale_factors,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.audio_num_attention_heads,
        )

        cross_attn_pos_embed_max_pos = max(pos_embed_max_pos, audio_pos_embed_max_pos)
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=int(arch.audio_cross_attention_dim),
            patch_size=hf_patch_size,
            patch_size_t=hf_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.num_attention_heads,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=int(arch.audio_cross_attention_dim),
            patch_size=hf_audio_patch_size,
            patch_size_t=hf_audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=16000,
            hop_length=160,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.audio_num_attention_heads,
        )

        self.cross_pe_max_pos = cross_attn_pos_embed_max_pos

        # 5. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                LTX2TransformerBlock(
                    idx=idx,
                    dim=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.hidden_size // self.num_attention_heads,
                    cross_attention_dim=arch.cross_attention_dim,
                    audio_dim=self.audio_hidden_size,
                    audio_num_attention_heads=self.audio_num_attention_heads,
                    audio_attention_head_dim=self.audio_hidden_size
                    // self.audio_num_attention_heads,
                    audio_cross_attention_dim=arch.audio_cross_attention_dim,
                    norm_eps=self.norm_eps,
                    qk_norm=True,  # Always True in LTX2
                    supported_attention_backends=self._supported_attention_backends,
                    prefix=config.prefix,
                )
                for idx in range(arch.num_layers)
            ]
        )

        # 6. Output layers
        self.norm_out = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps, elementwise_affine=False
        )
        self.proj_out = ColumnParallelLinear(
            self.hidden_size, arch.out_channels, bias=True, gather_output=True
        )

        self.audio_norm_out = nn.LayerNorm(
            self.audio_hidden_size, eps=self.norm_eps, elementwise_affine=False
        )
        self.audio_proj_out = ColumnParallelLinear(
            self.audio_hidden_size,
            arch.audio_out_channels,
            bias=True,
            gather_output=True,
        )

        self.out_channels_raw = arch.out_channels // (
            self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        )
        self.audio_out_channels = arch.audio_out_channels
        self.timestep_scale_multiplier = arch.timestep_scale_multiplier
        self.av_ca_timestep_scale_multiplier = arch.av_ca_timestep_scale_multiplier

        self.layer_names = ["transformer_blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 24.0,
        audio_num_frames: Optional[int] = None,
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:

        batch_size = hidden_states.size(0)
        audio_timestep = audio_timestep if audio_timestep is not None else timestep

        if num_frames is None or height is None or width is None:
            raise ValueError(
                "num_frames/height/width must be provided for RoPE coordinate generation."
            )
        if audio_num_frames is None:
            raise ValueError(
                "audio_num_frames must be provided for RoPE coordinate generation."
            )

        if video_coords is None:
            # Wan-style SP-RoPE: when SP is enabled, each rank runs on its local
            # time shard but RoPE positions must be offset to global time.
            #
            # We assume equal time sharding across SP ranks.
            if model_parallel_is_initialized():
                sp_world_size = get_sp_world_size()
                sp_rank = get_sp_parallel_rank()
            else:
                sp_world_size = 1
                sp_rank = 0

            video_shift = int(sp_rank) * int(num_frames) if sp_world_size > 1 else 0
            video_coords = self.rope.prepare_video_coords(
                batch_size=batch_size,
                num_frames=num_frames,
                height=height,
                width=width,
                device=hidden_states.device,
                fps=fps,
                start_frame=video_shift,
            )
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(
                batch_size=batch_size,
                num_frames=audio_num_frames,
                device=audio_hidden_states.device,
            )

        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
        audio_rotary_emb = self.audio_rope(
            audio_coords, device=audio_hidden_states.device
        )
        ca_video_rotary_emb = self.cross_attn_rope(
            video_coords[:, 0:1, :], device=hidden_states.device
        )
        ca_audio_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_states.device
        )

        # 2. Patchify input projections
        hidden_states, _ = self.patchify_proj(hidden_states)
        audio_hidden_states, _ = self.audio_patchify_proj(audio_hidden_states)

        # 3. Prepare timestep embeddings
        # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
        temb, embedded_timestep = self.adaln_single(
            timestep.flatten(),
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.size(-1)
        )

        temb_audio, audio_embedded_timestep = self.audio_adaln_single(
            audio_timestep.flatten()
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(
            batch_size, -1, audio_embedded_timestep.size(-1)
        )

        # 3.2. Prepare global modality cross attention modulation parameters
        ts_ca_mult = (
            self.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier
        )

        temb_ca_scale_shift, _ = self.av_ca_video_scale_shift_adaln_single(
            timestep.flatten()
        )
        temb_ca_scale_shift = temb_ca_scale_shift.view(
            batch_size, -1, temb_ca_scale_shift.shape[-1]
        )

        temb_ca_gate, _ = self.av_ca_a2v_gate_adaln_single(
            timestep.flatten() * ts_ca_mult
        )
        temb_ca_gate = temb_ca_gate.view(batch_size, -1, temb_ca_gate.shape[-1])

        temb_ca_audio_scale_shift, _ = self.av_ca_audio_scale_shift_adaln_single(
            audio_timestep.flatten()
        )
        temb_ca_audio_scale_shift = temb_ca_audio_scale_shift.view(
            batch_size, -1, temb_ca_audio_scale_shift.shape[-1]
        )

        temb_ca_audio_gate, _ = self.av_ca_v2a_gate_adaln_single(
            audio_timestep.flatten() * ts_ca_mult
        )
        temb_ca_audio_gate = temb_ca_audio_gate.view(
            batch_size, -1, temb_ca_audio_gate.shape[-1]
        )

        # 4. Prepare prompt embeddings
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        audio_encoder_hidden_states = self.audio_caption_projection(
            audio_encoder_hidden_states
        )

        # 5. Run blocks
        for block in self.transformer_blocks:
            hidden_states, audio_hidden_states = block(
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                # Keep the first 4 args positional to stay compatible with cache-dit's
                # LTX2 adapter, which treats `audio_hidden_states` as `encoder_hidden_states`
                # under ForwardPattern.Pattern_0.
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=temb_ca_scale_shift,
                temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
                temb_ca_gate=temb_ca_gate,
                temb_ca_audio_gate=temb_ca_audio_gate,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=ca_video_rotary_emb,
                ca_audio_rotary_emb=ca_audio_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
            )

        # 6. Output layers
        # Video
        scale_shift_values = self.scale_shift_table[None, None].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        ) + embedded_timestep[:, :, None].to(dtype=hidden_states.dtype)
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states, _ = self.proj_out(hidden_states)

        # Audio
        audio_scale_shift_values = self.audio_scale_shift_table[None, None].to(
            device=audio_hidden_states.device, dtype=audio_hidden_states.dtype
        ) + audio_embedded_timestep[:, :, None].to(dtype=audio_hidden_states.dtype)
        audio_shift, audio_scale = (
            audio_scale_shift_values[:, :, 0],
            audio_scale_shift_values[:, :, 1],
        )
        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_hidden_states, _ = self.audio_proj_out(audio_hidden_states)

        # Unpatchify if requested (default True for pipeline compatibility)
        return_latents = kwargs.get("return_latents", True)

        if return_latents:
            # Unpatchify Video
            # [B, N, C_out_raw*patch_vol] -> [B, C_out_raw, T, H, W]
            # Requires num_frames, height, width to be known
            if num_frames is not None and height is not None and width is not None:
                p_t, p_h, p_w = self.patch_size
                post_t, post_h, post_w = num_frames // p_t, height // p_h, width // p_w
                b = batch_size
                hidden_states = hidden_states.reshape(
                    b, post_t, post_h, post_w, self.out_channels_raw, p_t, p_h, p_w
                )
                hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
                    b, self.out_channels_raw, num_frames, height, width
                )

            # Unpatchify Audio
            # [B, N, C_out] -> [B, C_out, T] (or 4D/5D)
            if audio_num_frames is not None:
                b = batch_size
                # simple reshape for 1D patch
                audio_hidden_states = audio_hidden_states.permute(0, 2, 1)  # [B, C, T]

        return hidden_states, audio_hidden_states


# Backward-compatible alias (older internal name).
LTXModel = LTX2VideoTransformer3DModel
EntryClass = LTX2VideoTransformer3DModel
