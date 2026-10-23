# SPDX-License-Identifier: Apache-2.0
# Adapted from Hy-Nomas (Apache-2.0):
# https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/98a28976f23552bb2e2cf7941986e9cd10a3118a/hymm/models/dit/modules/posemb_layers.py
"""Nomas RoPE tables: real (cos, sin), heads-last layout; arithmetic order matches upstream."""

import torch


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
