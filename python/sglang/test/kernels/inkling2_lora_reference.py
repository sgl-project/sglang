"""Torch references for sglang.kernels.ops.lora.moe.inkling2_routed (same keyword API).

Rows outside [offsets[0], offsets[-1]) come back zero where the kernels leave them unset.
"""

import torch


def _row_experts(offsets: torch.Tensor, rows: int) -> torch.Tensor:
    """Local expert of each row, -1 outside this rank's rows."""
    expert = torch.full((rows,), -1, dtype=torch.long)
    for e in range(offsets.numel() - 1):
        expert[int(offsets[e]) : int(offsets[e + 1])] = e
    return expert


def _row_scale(nvfp4_scales, expert: torch.Tensor) -> torch.Tensor:
    x_amax, w_amax, w_denominator = nvfp4_scales
    return x_amax.float() * w_amax.float()[expert] / (1536.0 * w_denominator.float())


def capped_swiglu(gate, up, gamma, gate_cap, up_cap):
    sigmoid = torch.sigmoid(gate)
    if gate_cap > 0:
        gate = gate_cap * torch.tanh(gate / gate_cap)
    if up_cap > 0:
        up = up_cap * torch.tanh(up / up_cap)
    return gate * sigmoid * up * gamma


def routed_gate_up_delta(
    *, h, b, acc, dst2src, offsets, top_k, global_experts, nvfp4_scales=None
):
    rank = b.shape[2]
    n = b.shape[1] // 2
    expert = _row_experts(offsets, acc.shape[0])
    for r in torch.nonzero(expert >= 0).flatten().tolist():
        e, token = int(expert[r]), int(dst2src[r]) // top_k
        gate = b[e, :n].float() @ h[token, :rank].float()
        up = b[e, n:].float() @ h[token, rank:].float()
        if nvfp4_scales is not None:
            scale = _row_scale(nvfp4_scales, expert)[r]
            inverse = 1.0 / scale if scale > 0 else 0.0
            gate, up = gate * inverse, up * inverse
        acc[r, 0::2] = (acc[r, 0::2].float() + gate).to(acc.dtype)
        acc[r, 1::2] = (acc[r, 1::2].float() + up).to(acc.dtype)


def routed_down_inner(
    *, acc, gammas, a, offsets, gate_cap, up_cap, global_experts, nvfp4_scales=None
):
    rank = a.shape[1]
    expert = _row_experts(offsets, acc.shape[0])
    inner = torch.zeros((acc.shape[0], rank), dtype=torch.float32)
    for r in torch.nonzero(expert >= 0).flatten().tolist():
        e = int(expert[r])
        gate, up = acc[r, 0::2].float(), acc[r, 1::2].float()
        if nvfp4_scales is not None:
            scale = _row_scale(nvfp4_scales, expert)[r]
            gate = (gate * scale).bfloat16().float()
            up = (up * scale).bfloat16().float()
        act = capped_swiglu(gate, up, gammas[r].float(), gate_cap, up_cap).bfloat16()
        inner[r] = a[e].float() @ act.float()
    return inner


def routed_token_rank(*, inner, src2dst, offsets, mask, top_k):
    tokens = src2dst.numel() // top_k
    lo, hi = int(offsets[0]), int(offsets[-1])
    out = torch.zeros((tokens, inner.shape[1]), dtype=torch.float32)
    for t in range(tokens):
        for k in range(top_k):
            row = int(src2dst[t * top_k + k])
            if lo <= row < hi:
                out[t] += inner[row].float()
        out[t] *= mask[t].float()
    return out


def unweighted_post_reorder(x, src2dst, top_k):
    rows = x[src2dst.long()].unflatten(0, (-1, top_k))
    return rows.float().sum(1).to(x.dtype)
