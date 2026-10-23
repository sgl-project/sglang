# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Tencent-Hunyuan/Hy-Nomas
# Original PatchEmbed: Copyright 2020 Ross Wightman.
"""Nomas DiT building blocks, RoPE tables and tensor/Ulysses parallel helpers.

The blocks kept here differ in numerics from the shared SGLang layers. TP uses
the framework's parallel linear layers. Ulysses SP is model-specific: Nomas
attention is a FlexAttention block mask over its own segment KV cache, which
USPAttention cannot express, so the head/sequence exchange around attention is
done here with the shared collectives.
"""

import math

import torch
from einops import rearrange
from msgspec import Struct
from torch import nn
from torch.nn import functional as F

from sglang.kernels.ops.diffusion import modulate_scale_shift, residual_gate_add
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_tp_rank,
    get_tp_world_size,
    get_ulysses_ctx,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.visual_embedding import ModulateProjection


def attention(q, k, v, attn_mask):
    # Match upstream unpad/FA2/pad: padded query rows have zero attention.
    result = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask[:, None, None, :],
    )
    return result.transpose(1, 2).flatten(2) * attn_mask[:, :, None]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(
            x
        ) * self.weight


def modulate(x, shift, scale):
    return modulate_scale_shift(x, scale, shift)


def apply_gate(x, gate):
    return x * gate.unsqueeze(1)


def gated_residual(residual, update, gate):
    # The shared kernel preserves the multiply's storage-dtype rounding before
    # the residual addition, and falls back for mixed dtypes (camera residuals).
    return residual_gate_add(residual, update, gate.unsqueeze(1))


class PatchEmbed(nn.Module):
    """Conv3d patchify; [B, C, T, H, W] -> [B, T*H*W, D]."""

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of fractional timesteps, (*t.shape, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t.flatten()[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.view(*t.shape, dim)


class TimestepEmbedder(nn.Module):
    def __init__(
        self, hidden_size, act_layer, frequency_embedding_size=256, max_period=10000
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            act_layer(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        return self.mlp(t_freq.type(self.mlp[0].weight.dtype))


class FinalLayerPerBlock(nn.Module):
    """adaLN final layer; ``c`` is the per-request timestep vector [B, D]."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels)
        self.adaLN_modulation = ModulateProjection(
            hidden_size, factor=2, act_layer="silu"
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size, heads_num):
        super().__init__()
        self.heads_num = heads_num
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(hidden_size, 4 * hidden_size, act_type="silu")
        self.adaLN_modulation = ModulateProjection(
            hidden_size, factor=2, act_layer="silu"
        )

    def forward(self, x, c, attn_mask):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        qkv = self.self_attn_qkv(self.norm1(x))
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        x = x + apply_gate(self.self_attn_proj(attention(q, k, v, attn_mask)), gate_msa)
        return x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)


class IndividualTokenRefiner(nn.Module):
    def __init__(self, hidden_size, heads_num, depth):
        super().__init__()
        self.blocks = nn.ModuleList(
            [IndividualTokenRefinerBlock(hidden_size, heads_num) for _ in range(depth)]
        )

    def forward(self, x, c, mask):
        mask = mask.clone().bool()
        mask[:, 0] = True  # avoid an all-masked row producing NaN
        for block in self.blocks:
            x = block(x, c, mask)
        return x


def _to_tuple(x, dim):
    if isinstance(x, int):
        return (x,) * dim
    if len(x) != dim:
        raise ValueError(f"Expected length {dim} or int, but got {x}")
    return x


def get_meshgrid_nd(start, *args, dim=2, device=None):
    """[dim, *sizes] grid: ``(num)`` counts from 0, ``(start, stop)`` steps by 1."""
    if len(args) == 0:
        num = _to_tuple(start, dim)
        start, stop = (0,) * dim, num
    elif len(args) == 1:
        start, stop = _to_tuple(start, dim), _to_tuple(args[0], dim)
        num = [stop[i] - start[i] for i in range(dim)]
    else:
        raise ValueError(f"len(args) should be 0 or 1, but got {len(args)}")
    # torch.linspace(a, b, n+1)[:n] is upstream's np.linspace(endpoint=False).
    axes = [
        torch.linspace(
            start[i], stop[i], num[i] + 1, dtype=torch.float32, device=device
        )[: num[i]]
        for i in range(dim)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(xq, xk, freqs_cis):
    """Rotate [B, S, H, D] queries and keys with (cos, sin) tables of shape [S, D]."""
    cos, sin = freqs_cis
    if cos.shape != (xq.shape[1], xq.shape[-1]):
        raise ValueError(
            f"freqs_cis shape {cos.shape} does not match x shape {xq.shape}"
        )
    shape = [d if i in (1, xq.ndim - 1) else 1 for i, d in enumerate(xq.shape)]
    cos, sin = cos.view(*shape).to(xq.device), sin.view(*shape).to(xq.device)
    xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
    xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    return xq_out, xk_out


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0):
    """(cos, sin) of shape [S, dim] for float positions ``pos`` [S]."""
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)
    )
    freqs = torch.outer(pos, freqs)
    return freqs.cos().repeat_interleave(2, dim=1), freqs.sin().repeat_interleave(
        2, dim=1
    )


def get_nd_rotary_pos_embed(rope_dim_list, start, *args, theta=10000.0, device=None):
    """n-D RoPE tables (cos, sin) of shape [prod(sizes), sum(rope_dim_list)]."""
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list), device=device)
    embs = [
        get_1d_rotary_pos_embed(dim, grid[i].reshape(-1), theta)
        for i, dim in enumerate(rope_dim_list)
    ]
    return torch.cat([e[0] for e in embs], dim=1), torch.cat(
        [e[1] for e in embs], dim=1
    )


def get_rotary_pos_embed_with_offset(
    rope_sizes, time_offset, rope_dim_list, rope_theta
):
    """Tables for a segment whose first frame sits at ``time_offset`` of the clip."""
    tt, th, tw = rope_sizes
    return get_nd_rotary_pos_embed(
        rope_dim_list, (time_offset, 0, 0), (time_offset + tt, th, tw), theta=rope_theta
    )


class ParallelInfo(Struct, frozen=True):
    tp_size: int = 1
    tp_rank: int = 0
    sp_size: int = 1  # Ulysses degree; ring attention is not supported
    sp_rank: int = 0

    @property
    def head_shards(self) -> int:
        return self.tp_size * self.sp_size


def current_parallel_info() -> ParallelInfo:
    if not model_parallel_is_initialized():
        return ParallelInfo()
    if get_ring_parallel_world_size() > 1:
        raise ValueError("Nomas supports Ulysses sequence parallelism only, not ring")
    sp_size, sp_rank = get_ulysses_ctx()
    return ParallelInfo(get_tp_world_size(), get_tp_rank(), sp_size, sp_rank)


def seq_to_heads(x: torch.Tensor, info: ParallelInfo) -> torch.Tensor:
    """[1, S_local, H, D] -> [1, S_global, H/U, D] (Ulysses input exchange)."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_to_all_4D(x, scatter_dim=2, gather_dim=1)


def heads_to_seq(x: torch.Tensor, info: ParallelInfo) -> torch.Tensor:
    """[1, S_global, H/U, D] -> [1, S_local, H, D] (Ulysses output exchange)."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_to_all_4D(x, scatter_dim=1, gather_dim=2)


def gather_heads(x: torch.Tensor, info: ParallelInfo) -> torch.Tensor:
    """[1, S, H/U, D] -> [1, S, H, D] for the replicated text stream."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_gather(x, dim=2)


def gather_seq(x: torch.Tensor, info: ParallelInfo) -> torch.Tensor:
    """[1, S_local, C] -> [1, S_global, C]."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_gather(x, dim=1)


def local_token_range(num_tokens: int, info: ParallelInfo) -> tuple[int, int]:
    if num_tokens % info.sp_size:
        raise ValueError(
            f"{num_tokens} video tokens are not divisible by the sequence-parallel degree {info.sp_size}"
        )
    local = num_tokens // info.sp_size
    return info.sp_rank * local, (info.sp_rank + 1) * local
