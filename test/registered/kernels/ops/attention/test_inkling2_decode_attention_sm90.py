"""decode_partials (Hopper paged decode with the compact relative bias) must match plain attention over each sequence's
own keys for any KV page size. Slots are scattered across the cache, as batched serving allocates them: at page size 1
consecutive keys sit on unrelated pages, so a tile must look up each key's page."""

import pytest
import torch

from sglang.kernels.ops.inkling2.decode_attention_sm90 import decode_partials
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HEADS_Q, HEADS_K, D, EXTENT = 8, 1, 128, 64


def _combine(o, m, l):
    """Merge split partials (unnormalized o, row max m, row sum l) into the attention output."""
    w = torch.exp(m - m.max(0, keepdim=True).values)
    return (w[..., None] * o).sum(0) / (w * l).sum(0)[..., None]


def _reference(q, keys, values, bias, scale, qlen, window):
    """q (qlen, heads_q, d) for the last qlen positions of one sequence; keys/values (seqlen, heads_k, d)."""
    seqlen = keys.shape[0]
    group = HEADS_Q // HEADS_K
    k = keys.float().repeat_interleave(group, 1)
    v = values.float().repeat_interleave(group, 1)
    s = torch.einsum("qhd,khd->qhk", q.float(), k) * scale
    dist = (seqlen - qlen + torch.arange(qlen, device=q.device))[
        :, None
    ] - torch.arange(seqlen, device=q.device)
    keep = dist >= 0
    if window is not None:
        keep &= dist <= window
    near = (dist < EXTENT) & keep
    s = s + torch.where(
        near[:, None, :],
        bias.float().gather(
            -1, dist.clamp(0, EXTENT - 1)[:, None, :].expand(-1, HEADS_Q, -1)
        ),
        0,
    )
    s = s.masked_fill(~keep[:, None, :], float("-inf"))
    return torch.einsum("qhk,khd->qhd", s.softmax(-1), v)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="needs sm90",
)
@pytest.mark.parametrize("page_size", [1, 16, 64, 128])
@pytest.mark.parametrize("qlen, window", [(1, None), (1, 100), (3, None)])
def test_decode_partials_matches_attention_for_scattered_pages(page_size, qlen, window):
    torch.manual_seed(0)
    dev, scale = "cuda", D**-0.5
    seqlens = [700, 129, 3, 300]
    pages_per_seq = max(-(-n // page_size) for n in seqlens)
    total_pages = pages_per_seq * len(seqlens) + 8
    k_cache = torch.randn(
        total_pages, page_size, HEADS_K, D, device=dev, dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    # Every sequence gets pages scattered across the whole cache.
    page_table = torch.randperm(total_pages, device=dev)[
        : len(seqlens) * pages_per_seq
    ].view(len(seqlens), -1)
    page_table = page_table.to(torch.int32)
    q = torch.randn(len(seqlens) * qlen, HEADS_Q, D, device=dev, dtype=torch.bfloat16)
    bias = torch.randn(
        len(seqlens) * qlen, HEADS_Q, EXTENT, device=dev, dtype=torch.bfloat16
    )
    seqused_k = torch.tensor(seqlens, device=dev, dtype=torch.int32)

    o, m, l = decode_partials(
        q, k_cache, v_cache, page_table, seqused_k, bias, scale, window_size_left=window
    )
    out = _combine(o, m, l)[:, 0]

    for b, n in enumerate(seqlens):
        slots = page_table[b].long().repeat_interleave(
            page_size
        ) * page_size + torch.arange(page_size, device=dev).repeat(pages_per_seq)
        keys = k_cache.view(-1, HEADS_K, D)[slots[:n]]
        values = v_cache.view(-1, HEADS_K, D)[slots[:n]]
        rows = slice(b * qlen, (b + 1) * qlen)
        want = _reference(q[rows], keys, values, bias[rows], scale, qlen, window)
        torch.testing.assert_close(out[rows], want, atol=2e-2, rtol=2e-2)
