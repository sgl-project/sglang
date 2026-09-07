"""Hyper-connections: the residual stream is hc_mult parallel copies. Each sublayer
sits between hc_pre (collapse the copies) and hc_post (expand back, mixing the
residual through a doubly-stochastic comb matrix)."""

import torch
import torch.nn.functional as F


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
):
    """mixes [..., (2 + hc) * hc] -> pre [..., hc], post [..., hc], comb [..., hc, hc].
    The flattened row holds pre, post, then the comb matrix consecutively."""
    hc = hc_mult
    lead = mixes.shape[:-1]
    flat = mixes.reshape(-1, (2 + hc) * hc).float()
    scale = hc_scale.float()
    base = hc_base.float()

    pre = torch.sigmoid(flat[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(flat[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])

    comb = (flat[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)
    comb = torch.exp(comb - comb.amax(dim=2, keepdim=True))
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)

    return pre.reshape(*lead, hc), post.reshape(*lead, hc), comb.reshape(*lead, hc, hc)


def hc_mixes(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    norm_eps: float,
):
    """x [b, s, hc, d], hc_fn [(2 + hc) * hc, hc * d] fp32. One rms statistic per token
    over the whole flattened hc * d stream, applied after the projection."""
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """[b, s, hc, d] x [b, s, hc] -> [b, s, d]"""
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)


def hc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
):
    """x [b, s, d], residual [b, s, hc, d], post [b, s, hc], comb [b, s, hc, hc] -> [b, s, hc, d]"""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """One-hot on copy 0: the first block reads the embedding copy unmixed."""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix
