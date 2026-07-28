"""Unit tests for flash_decode_with_gqa_share_sparse (sparse GQA attention).

Tests the Triton sparse GQA kernel against a PyTorch reference that computes
attention only on the topk blocks via standard softmax, covering GQA ratios,
sink tokens, paged KV (randperm), variable seq_lens, and edge cases.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)

DEVICE = "cuda"
RTOL = 5e-3
ATOL = 5e-3


def pytorch_sparse_gqa_reference(
    q,
    sink,
    k_cache,
    v_cache,
    req_to_token,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
):
    """PyTorch reference: gather topk block tokens, then batched attention."""
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    gqa_group_size = num_q_heads // num_kv_heads
    topk = topk_idx.shape[2]
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # Build per-batch token positions from topk block indices, padded to uniform length
    max_tokens = topk * block_size
    all_slots = torch.zeros(batch_size, max_tokens, dtype=torch.long, device=q.device)
    mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=q.device)

    for b in range(batch_size):
        sl = seq_lens[b].item()
        offset = 0
        for t in range(topk):
            bi = topk_idx[0, b, t].item()
            if bi < 0:
                continue
            start = bi * block_size
            end = min(start + block_size, sl)
            n = end - start
            positions = torch.arange(start, end, device=q.device)
            all_slots[b, offset : offset + n] = req_to_token[b, positions].long()
            mask[b, offset : offset + n] = True
            offset += n

    # Gather K/V: [BS, max_tokens, num_kv_heads, hd] -> [BS, num_q_heads, max_tokens, hd]
    k = k_cache[all_slots].float()  # [BS, max_tokens, num_kv_heads, hd]
    v = v_cache[all_slots].float()
    k = k.permute(0, 2, 1, 3).repeat_interleave(
        gqa_group_size, dim=1
    )  # [BS, num_q_heads, max_tokens, hd]
    v = v.permute(0, 2, 1, 3).repeat_interleave(gqa_group_size, dim=1)

    # QK: [BS, num_q_heads, 1, hd] @ [BS, num_q_heads, hd, max_tokens] -> [BS, num_q_heads, max_tokens]
    qk = (q.float().unsqueeze(2) @ k.transpose(-1, -2)).squeeze(2) * sm_scale
    # Mask invalid positions
    qk = qk.masked_fill(~mask.unsqueeze(1), float("-inf"))

    if sink is not None:
        sink_score = (q.float() * sink.float().unsqueeze(0)).sum(
            dim=-1, keepdim=True
        ) * sm_scale  # [BS, num_q_heads, 1]
        qk = torch.cat([sink_score, qk], dim=-1)  # [BS, num_q_heads, 1+max_tokens]
        attn = torch.softmax(qk, dim=-1)
        o = (attn[:, :, 1:].unsqueeze(2) @ v).squeeze(2)  # [BS, num_q_heads, hd]
    else:
        attn = torch.softmax(qk, dim=-1)
        o = (attn.unsqueeze(2) @ v).squeeze(2)

    return o


def build_inputs(
    batch_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    seq_lens_list,
    block_size,
    topk,
    with_sink=False,
    paged=True,
    dtype=torch.bfloat16,
):
    max_kv_len = max(seq_lens_list)
    max_slots = batch_size * max_kv_len

    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    req_to_token = torch.zeros(batch_size, max_kv_len, dtype=torch.int32, device=DEVICE)
    slot_ids = torch.zeros(batch_size, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)

    for i in range(batch_size):
        base = i * max_kv_len
        slot_ids[i] = i
        if paged:
            req_to_token[i, :max_kv_len] = (
                torch.randperm(max_kv_len, device=DEVICE) + base
            ).to(torch.int32)
        else:
            req_to_token[i, :max_kv_len] = torch.arange(
                base, base + max_kv_len, device=DEVICE
            ).to(torch.int32)

    num_blocks_list = [(sl + block_size - 1) // block_size for sl in seq_lens_list]
    actual_topk = min(topk, min(num_blocks_list))
    topk_idx = torch.zeros(
        num_kv_heads, batch_size, topk, dtype=torch.int32, device=DEVICE
    )
    for kh in range(num_kv_heads):
        for b in range(batch_size):
            nb = num_blocks_list[b]
            ak = min(topk, nb)
            perm = torch.randperm(nb, device=DEVICE)[:ak]
            topk_idx[kh, b, :ak] = perm.to(torch.int32)
            if ak < topk:
                topk_idx[kh, b, ak:] = -1

    sink = (
        torch.randn(num_q_heads, head_dim, dtype=dtype, device=DEVICE)
        if with_sink
        else None
    )

    return q, sink, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx


def _case(bs, nqh, nkh, hd, blk, tk, sink, seq_pat, paged=True):
    tag = (
        f"bs{bs}_gqa{nqh}:{nkh}_hd{hd}_blk{blk}_topk{tk}"
        f"_{'sink' if sink else 'nosink'}_{seq_pat}"
        f"{'_paged' if paged else '_contig'}"
    )
    return pytest.param(bs, nqh, nkh, hd, blk, tk, sink, seq_pat, paged, id=tag)


def make_seq_lens(pattern, batch_size, block_size):
    if pattern == "aligned":
        return [1024] * batch_size
    elif pattern == "unaligned":
        base = [513, 1023, 257, 769]
        return (base * ((batch_size + len(base) - 1) // len(base)))[:batch_size]
    elif pattern == "short":
        return [block_size * 2] * batch_size
    elif pattern == "mixed":
        base = [block_size, block_size * 4, block_size * 16, block_size * 2]
        return (base * ((batch_size + len(base) - 1) // len(base)))[:batch_size]
    elif pattern == "long":
        return [block_size * 512] * batch_size


CASES = [
    # Baseline: GQA 8:1
    _case(1, 8, 1, 128, 64, 16, False, "aligned"),
    _case(2, 8, 1, 128, 64, 16, False, "aligned"),
    _case(4, 8, 1, 128, 64, 32, False, "aligned"),
    # Unaligned seq_lens
    _case(4, 8, 1, 128, 64, 16, False, "unaligned"),
    # Short sequences (topk ~ num_blocks)
    _case(2, 8, 1, 128, 64, 16, False, "short"),
    # topk=1
    _case(2, 8, 1, 128, 64, 1, False, "aligned"),
    # topk=32 (production config)
    _case(2, 8, 1, 128, 64, 32, False, "aligned"),
    # With sink
    _case(2, 8, 1, 128, 64, 16, True, "aligned"),
    _case(4, 8, 1, 128, 64, 16, True, "unaligned"),
    # GQA 32:8
    _case(2, 32, 8, 128, 64, 16, False, "aligned"),
    # GQA 16:1 (production)
    _case(2, 16, 1, 128, 64, 32, False, "aligned"),
    # head_dim=64
    _case(2, 8, 4, 64, 64, 16, False, "aligned"),
    # block_size=32
    _case(2, 8, 1, 128, 32, 16, False, "aligned"),
    # Large BS
    _case(32, 8, 1, 128, 64, 16, False, "aligned"),
    _case(128, 8, 1, 128, 64, 32, False, "short"),
    # Contiguous (non-paged) KV
    _case(2, 8, 1, 128, 64, 16, False, "aligned", paged=False),
    # Mixed seq_lens in batch
    _case(4, 8, 1, 128, 64, 8, False, "mixed"),
    # Long sequence
    _case(1, 8, 1, 128, 64, 32, False, "long"),
]


@pytest.mark.parametrize("bs,nqh,nkh,hd,blk,tk,with_sink,seq_pat,paged", CASES)
def test_sparse_gqa_vs_reference(bs, nqh, nkh, hd, blk, tk, with_sink, seq_pat, paged):
    """Kernel output must match PyTorch reference (attend only to topk blocks)."""
    torch.manual_seed(42)
    seq_lens_list = make_seq_lens(seq_pat, bs, blk)

    q, sink, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx = (
        build_inputs(
            bs,
            nqh,
            nkh,
            hd,
            seq_lens_list,
            blk,
            tk,
            with_sink=with_sink,
            paged=paged,
        )
    )

    o_kernel = flash_decode_with_gqa_share_sparse(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        blk,
        topk_idx,
    )
    o_ref = pytorch_sparse_gqa_reference(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        blk,
        topk_idx,
    )
    o_ref = o_ref.to(o_kernel.dtype)

    assert torch.allclose(
        o_kernel.float(), o_ref.float(), rtol=RTOL, atol=ATOL
    ), f"max abs diff {(o_kernel.float() - o_ref.float()).abs().max().item():.4e}"


@pytest.mark.parametrize(
    "bs,nqh,nkh,hd,blk,tk,with_sink,seq_pat,paged",
    [
        _case(2, 8, 1, 128, 64, 32, False, "short"),
    ],
)
def test_sparse_gqa_topk_exceeds_blocks(
    bs, nqh, nkh, hd, blk, tk, with_sink, seq_pat, paged
):
    """topk > num_blocks: kernel should handle gracefully (some topk_idx = -1)."""
    torch.manual_seed(42)
    seq_lens_list = [blk] * bs

    q, sink, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx = (
        build_inputs(
            bs,
            nqh,
            nkh,
            hd,
            seq_lens_list,
            blk,
            tk,
            with_sink=with_sink,
            paged=paged,
        )
    )

    o_kernel = flash_decode_with_gqa_share_sparse(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        blk,
        topk_idx,
    )
    o_ref = pytorch_sparse_gqa_reference(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        blk,
        topk_idx,
    )
    o_ref = o_ref.to(o_kernel.dtype)

    assert torch.allclose(
        o_kernel.float(), o_ref.float(), rtol=RTOL, atol=ATOL
    ), f"max abs diff {(o_kernel.float() - o_ref.float()).abs().max().item():.4e}"


@pytest.mark.parametrize(
    "bs,nqh,nkh,hd,blk,tk,with_sink,seq_pat,paged",
    [
        _case(4, 8, 1, 128, 64, 16, False, "aligned"),
    ],
)
def test_sparse_gqa_deterministic(bs, nqh, nkh, hd, blk, tk, with_sink, seq_pat, paged):
    """Two calls with same inputs must produce identical outputs."""
    torch.manual_seed(42)
    seq_lens_list = make_seq_lens(seq_pat, bs, blk)

    q, sink, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx = (
        build_inputs(
            bs,
            nqh,
            nkh,
            hd,
            seq_lens_list,
            blk,
            tk,
            with_sink=with_sink,
            paged=paged,
        )
    )

    o1 = flash_decode_with_gqa_share_sparse(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        blk,
        topk_idx,
    )
    o2 = flash_decode_with_gqa_share_sparse(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        blk,
        topk_idx,
    )

    assert torch.equal(
        o1, o2
    ), f"non-deterministic: max diff {(o1.float() - o2.float()).abs().max().item():.4e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
