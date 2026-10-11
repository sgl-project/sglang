"""MiMo vision sinks bias the first real key of each packed segment.

Unlike a text/GPT-OSS sink, this key contributes its value. Starting with
ordinary windowed attention O and its natural-log LSE, the exact correction is
    beta = expm1(sink) * exp(q @ k0 * scale - lse)
    O' = (O + beta * v0) / (1 + beta).
Only queries that can see key zero are corrected. CUDA uses a small in-place
kernel over those prefixes; no token-sized gather buffers or host syncs are
needed. The CPU implementation and segmented SDPA are correctness references.
"""

import torch
import torch.nn.functional as F


def apply_first_key_sink(q, k, v, output, lse, sinks, cu_seqlens, window_size, scale):
    """Correct an inference-only FA3/FA4 output in place; LSE has shape [H, T]."""
    if lse.shape != (q.shape[1], q.shape[0]):
        raise ValueError(f"Unexpected vision LSE shape: {tuple(lse.shape)}")
    if q.shape[1] % k.shape[1] or sinks.numel() != q.shape[1]:
        raise ValueError("Vision sink heads must match Q heads and a valid GQA ratio")
    if q.shape[0] == 0:
        return output
    if q.is_cuda:
        from sglang.kernels.ops.attention.vision_sink import first_key_sink_inplace

        first_key_sink_inplace(
            q, k, v, output, lse, sinks, cu_seqlens, window_size[0], scale
        )
        return output

    # Keep fp64 available for an independent high-precision test oracle.
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    groups = q.shape[1] // k.shape[1]
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        stop = end if window_size[0] < 0 else min(end, start + window_size[0] + 1)
        if stop <= start:
            continue
        k0 = k[start].repeat_interleave(groups, dim=0).to(dtype)
        v0 = v[start].repeat_interleave(groups, dim=0).to(dtype)
        logits0 = (q[start:stop].to(dtype) * k0).sum(-1) * scale
        beta = (
            sinks.to(dtype).expm1() * (logits0 - lse[:, start:stop].T.to(dtype)).exp()
        )
        corrected = (output[start:stop].to(dtype) + beta[..., None] * v0) / (
            1 + beta[..., None]
        )
        output[start:stop].copy_(corrected.to(output.dtype))
    return output


def first_key_sdpa(
    q,
    k,
    v,
    cu_seqlens,
    *,
    sinks=None,
    window_size=(-1, -1),
    scale=None,
    attention_mask=None,
    softmax_in_single_precision=False,
    dropout=0.0,
):
    """Slow, per-segment MiMo reference, including full-attention GQA layers."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    groups = q.shape[1] // k.shape[1]
    if q.shape[1] % k.shape[1]:
        raise ValueError("Invalid vision GQA ratio")
    output = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        if start == end:
            continue
        qc = q[start:end].transpose(0, 1).unsqueeze(0)
        kc = k[start:end].repeat_interleave(groups, dim=1).transpose(0, 1).unsqueeze(0)
        vc = v[start:end].repeat_interleave(groups, dim=1).transpose(0, 1).unsqueeze(0)
        length = end - start
        # Additive masks preserve a masked-out key0: -inf + sink is still -inf.
        bias = q.new_zeros((1, q.shape[1], length, length))
        if attention_mask is not None:
            mask = (
                attention_mask[..., start:end]
                if attention_mask.shape[-1] != 1
                else attention_mask
            )
            if mask.shape[-2] != 1:
                mask = mask[..., start:end, :]
            mask = mask.to(q.device)
            if mask.dtype == torch.bool:
                bias.masked_fill_(~mask, float("-inf"))
            else:
                bias = bias + mask.to(q.dtype)
        positions = torch.arange(length, device=q.device)
        distance = positions[:, None] - positions[None, :]
        if window_size[0] >= 0:
            bias.masked_fill_(distance > window_size[0], float("-inf"))
        if window_size[1] >= 0:
            bias.masked_fill_(distance < -window_size[1], float("-inf"))
        if sinks is not None:
            bias[..., 0] += sinks.to(q.dtype).view(1, -1, 1)
        if softmax_in_single_precision:
            weights = torch.softmax(
                qc.float() @ kc.float().transpose(-1, -2) * scale + bias.float(), dim=-1
            )
            weights = torch.nan_to_num(weights).to(q.dtype)
            out = F.dropout(weights, p=dropout, training=False) @ vc
        else:
            out = F.scaled_dot_product_attention(
                qc, kc, vc, attn_mask=bias, scale=scale, dropout_p=dropout
            )
        output[start:end] = out.squeeze(0).transpose(0, 1)
    return output
