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
from functools import lru_cache

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


def _fused_ok(x):
    return (
        x.is_cuda and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
    )


def _compile_exact(fn):
    # emulate_precision_casts keeps every intermediate rounding (BF16 storage
    # casts, separate FP32 multiply/add) so the fused kernel is bit-identical
    # to the eager sequence it replaces. Dynamic shapes cover segment lengths.
    return torch.compile(
        fn, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True}
    )


def _modulate_to(x, shift, scale, dtype):
    return (x * (1 + scale[:, None]) + shift[:, None]).to(dtype)


@lru_cache(maxsize=1)
def _compiled_modulate_to():
    return _compile_exact(_modulate_to)


def modulate_for_linear(x, shift, scale):
    """``modulate`` followed by the cast autocast would apply in the next linear.

    Under BF16 autocast the FP32 LayerNorm output is modulated in FP32, then each
    following linear casts it to BF16 again. Casting once, fused with the
    modulation, yields the same values without the repeated casts.
    """
    if not (
        x.dtype == torch.float32
        and x.is_cuda
        and torch.is_autocast_enabled(x.device.type)
    ):
        return modulate(x, shift, scale)
    dtype = torch.get_autocast_dtype(x.device.type)
    if _fused_ok(x) and scale.dtype == shift.dtype:
        with torch.autocast(x.device.type, enabled=False):
            return _compiled_modulate_to()(x, shift, scale, dtype)
    return modulate(x, shift, scale).to(dtype)


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


def timestep_embedding(t, dim=256, max_period=10000):
    """Sinusoidal embedding of fractional timesteps, (*t.shape, dim); dim is even."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t.flatten()[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding.view(*t.shape, dim)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t):
        return self.mlp(timestep_embedding(t).type(self.mlp[0].weight.dtype))


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


def get_meshgrid_nd(start, stop, device=None):
    """[len(start), *sizes] grid stepping by 1 from ``start`` up to ``stop``."""
    # torch.linspace(a, b, n+1)[:n] is upstream's np.linspace(endpoint=False).
    axes = [
        torch.linspace(a, b, b - a + 1, dtype=torch.float32, device=device)[: b - a]
        for a, b in zip(start, stop)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def _rotate(x, cos, sin):
    return (x.float() * cos + rotate_half(x.float()) * sin).type_as(x)


@lru_cache(maxsize=1)
def _compiled_rotate():
    return _compile_exact(_rotate)


def apply_rotary_emb_single(x, freqs_cis):
    """Rotate one [B, S, H, D] tensor with (cos, sin) tables of shape [S, D]."""
    cos, sin = _rope_tables(x, freqs_cis)
    if _fused_ok(x):
        # One kernel instead of the FP32 copy/negate/stack/multiply/add/cast chain.
        with torch.autocast(x.device.type, enabled=False):
            return _compiled_rotate()(x, cos, sin)
    return _rotate(x, cos, sin)


def _rope_tables(x, freqs_cis):
    cos, sin = freqs_cis
    if cos.shape != (x.shape[1], x.shape[-1]):
        raise ValueError(
            f"freqs_cis shape {cos.shape} does not match x shape {x.shape}"
        )
    shape = [d if i in (1, x.ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return cos.view(*shape).to(x.device), sin.view(*shape).to(x.device)


def _norm_rotate(x, mean_square, eps, weight, cos, sin, dtype):
    normed = ((x.float() * torch.rsqrt(mean_square + eps)).type_as(x) * weight).to(
        dtype
    )
    return _rotate(normed, cos, sin)


@lru_cache(maxsize=1)
def _compiled_norm_rotate():
    return _compile_exact(_norm_rotate)


def rms_norm_rotate(norm, x, freqs_cis, dtype):
    """``apply_rotary_emb_single(norm(x).to(dtype), freqs_cis)`` with one fused kernel.

    The mean-square reduction stays in ATen so its summation order is unchanged;
    the normalisation, weight, cast and rotation are fused with the same rounding.
    """
    if not _fused_ok(x):
        return apply_rotary_emb_single(norm(x).to(dtype), freqs_cis)
    cos, sin = _rope_tables(x, freqs_cis)
    mean_square = x.float().pow(2).mean(-1, keepdim=True)
    with torch.autocast(x.device.type, enabled=False):
        return _compiled_norm_rotate()(
            x, mean_square, norm.eps, norm.weight, cos, sin, dtype
        )


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


def get_nd_rotary_pos_embed(rope_dim_list, start, stop, theta, device=None):
    """n-D RoPE tables (cos, sin) of shape [prod(stop - start), sum(rope_dim_list)]."""
    grid = get_meshgrid_nd(start, stop, device=device)
    embs = [
        get_1d_rotary_pos_embed(dim, grid[i].reshape(-1), theta)
        for i, dim in enumerate(rope_dim_list)
    ]
    return torch.cat([e[0] for e in embs], dim=1), torch.cat(
        [e[1] for e in embs], dim=1
    )


class ParallelInfo(Struct, frozen=True):
    tp_size: int = 1
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
    return ParallelInfo(get_tp_world_size(), sp_size, sp_rank)


def ulysses_all_to_all(x, info: ParallelInfo, scatter_dim: int, gather_dim: int):
    """Ulysses exchange: (2, 1) local tokens -> local heads, (1, 2) back."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_to_all_4D(
        x, scatter_dim=scatter_dim, gather_dim=gather_dim
    )


def sp_all_gather(x, info: ParallelInfo, dim: int):
    """Gather heads (dim 2, replicated text) or tokens (dim 1, final output)."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_gather(x, dim=dim)


def local_token_range(num_tokens: int, info: ParallelInfo) -> tuple[int, int]:
    if num_tokens % info.sp_size:
        raise ValueError(
            f"{num_tokens} video tokens are not divisible by the sequence-parallel degree {info.sp_size}"
        )
    local = num_tokens // info.sp_size
    return info.sp_rank * local, (info.sp_rank + 1) * local
