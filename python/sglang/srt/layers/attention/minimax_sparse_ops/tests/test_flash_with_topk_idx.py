import sys

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.minimax_sparse_ops.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)

DEVICE = "cuda"
RTOL_VS_REF = 5e-3
ATOL_VS_REF = 5e-3


# ---------------------------------------------------------------------------
# Reference & helpers
# ---------------------------------------------------------------------------


def pytorch_reference(
    q,
    sink,
    k_cache,
    v_cache,
    req_to_token,
    seq_lens,
    slot_ids,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    sm_scale=None,
    score_type="max",
):
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    gqa_group_size = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    max_sl = seq_lens.max().item()

    # Gather K/V for all batches: [BS, max_sl, kv_heads, hd] -> [BS, q_heads, max_sl, hd]
    all_slots = req_to_token[:, :max_sl].long()
    k = (
        k_cache[all_slots]
        .float()
        .permute(0, 2, 1, 3)
        .repeat_interleave(gqa_group_size, dim=1)
    )
    v = (
        v_cache[all_slots]
        .float()
        .permute(0, 2, 1, 3)
        .repeat_interleave(gqa_group_size, dim=1)
    )

    # Batched QK: [BS, q_heads, max_sl]
    qk = (q.float().unsqueeze(2) @ k.transpose(-1, -2)).squeeze(2) * sm_scale
    seq_mask = torch.arange(max_sl, device=q.device).unsqueeze(0) < seq_lens.unsqueeze(
        1
    )
    qk = qk.masked_fill(~seq_mask.unsqueeze(1), float("-inf"))

    # Attention output (full attention on all tokens)
    if sink is not None:
        sink_score = (q.float() * sink.float().unsqueeze(0)).sum(
            dim=-1, keepdim=True
        ) * sm_scale
        attn = torch.softmax(torch.cat([sink_score, qk], dim=-1), dim=-1)
        o = (attn[:, :, 1:].unsqueeze(2) @ v).squeeze(2)
    else:
        attn = torch.softmax(qk, dim=-1)
        o = (attn.unsqueeze(2) @ v).squeeze(2)

    # Block scores + topk
    topk_idx = torch.full(
        (num_q_heads, batch_size, topk), -1, dtype=torch.int32, device=q.device
    )
    max_num_blocks = (max_sl + block_size - 1) // block_size
    padded_len = max_num_blocks * block_size
    qk_padded = torch.full(
        (batch_size, num_q_heads, padded_len), float("-inf"), device=q.device
    )
    qk_padded[:, :, :max_sl] = qk
    block_scores = qk_padded.reshape(
        batch_size, num_q_heads, max_num_blocks, block_size
    )
    if score_type == "max":
        block_scores = block_scores.max(dim=-1).values
    else:
        bmax = block_scores.max(dim=-1, keepdim=True).values
        block_scores = bmax.squeeze(-1) + torch.log(
            torch.sum(torch.exp(block_scores - bmax), dim=-1)
        )
        block_scores = torch.where(block_scores.isnan(), float("-inf"), block_scores)

    for b in range(batch_size):
        sl = seq_lens[b].item()
        num_blocks = (sl + block_size - 1) // block_size
        bs_b = block_scores[b, :, :num_blocks].clone()
        if init_blocks > 0:
            bs_b[:, :init_blocks] = 1e30
        if local_blocks > 0:
            local_start = max(0, num_blocks - local_blocks)
            bs_b[:, local_start:num_blocks] = 1e29
        actual_topk = min(topk, num_blocks)
        _, tidx = bs_b.topk(actual_topk, dim=-1)
        topk_idx[:, b, :actual_topk] = tidx.to(torch.int32)

    return o, topk_idx


def build_inputs(
    batch_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    seq_lens_list,
    max_kv_len=None,
    with_sink=False,
    dtype=torch.bfloat16,
):
    if max_kv_len is None:
        max_kv_len = max(seq_lens_list)
    max_slots = batch_size * max_kv_len
    k_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=DEVICE)
    req_to_token = torch.zeros(batch_size, max_kv_len, dtype=torch.int32, device=DEVICE)
    slot_ids = torch.zeros(batch_size, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    for i in range(batch_size):
        base = i * max_kv_len
        slot_ids[i] = i
        req_to_token[i, :max_kv_len] = torch.arange(
            base, base + max_kv_len, device=DEVICE
        )
    sink = (
        torch.randn(num_q_heads, head_dim, dtype=dtype, device=DEVICE)
        if with_sink
        else None
    )
    return q, sink, k_cache, v_cache, req_to_token, seq_lens, max_kv_len, slot_ids


def make_seq_lens(pattern, batch_size, block_size):
    if pattern == "aligned":
        sl = [1024] * batch_size
    elif pattern == "unaligned":
        base = [513, 1023, 257, 769]
        sl = (base * ((batch_size + len(base) - 1) // len(base)))[:batch_size]
    elif pattern == "mixed":
        base = [64, 2048, 512, 128]
        sl = (base * ((batch_size + len(base) - 1) // len(base)))[:batch_size]
    elif pattern == "few_blocks":
        base = [block_size, block_size * 2]
        sl = (base * ((batch_size + len(base) - 1) // len(base)))[:batch_size]
    elif pattern == "long":
        sl = [524288] * batch_size
    return sl, max(sl)


# ---------------------------------------------------------------------------
# Test cases: compact set covering all code paths + long sequence.
# ---------------------------------------------------------------------------


def _case(bs, nqh, nkh, hd, blk, sink, seq_pat, ib, lb, tk):
    tag = (
        f"bs{bs}_gqa{nqh}:{nkh}_hd{hd}_blk{blk}"
        f"_{'sink' if sink else 'nosink'}_{seq_pat}"
        f"_init{ib}_local{lb}_topk{tk}"
    )
    return pytest.param(bs, nqh, nkh, hd, blk, sink, seq_pat, ib, lb, tk, id=tag)


# fmt: off
CASES = [
    # -- Core code paths --
    _case(2, 8, 1, 128, 64, False, "aligned",    0, 0, 16),  # baseline
    _case(2, 8, 1, 128, 64, False, "unaligned",  2, 4, 16),  # unaligned + init+local
    _case(2, 8, 1, 128, 32, False, "mixed",      0, 0, 16),  # block_size=32 + mixed
    _case(2, 8, 1, 128, 64, True,  "aligned",    0, 0, 16),  # sink
    _case(8, 8, 1, 128, 64, True,  "unaligned",  2, 4, 16),  # sink + init+local
    _case(8, 8, 1, 128, 64, False, "aligned",    0, 0, 1),   # topk=1
    _case(32, 8, 1, 128, 64, False, "few_blocks", 0, 0, 32),  # topk > num_blocks
    _case(32, 32, 8, 128, 64, False, "aligned",   0, 0, 16),  # GQA 32:8
    _case(128, 8, 4, 64,  64, False, "aligned",    0, 0, 16),  # head_dim=64
    _case(128, 8, 4, 128, 64, False, "few_blocks", 0, 0, 1),   # single batch, 1 block, topk=1
    # -- Long sequence (512k) --
    _case(1, 8, 1, 128, 64, False, "long",       0, 0, 16),  # 512k baseline
    _case(1, 8, 1, 128, 64, False, "long",       2, 4, 16),  # 512k + init+local
    _case(1, 8, 1, 128, 64, True,  "long",       0, 0, 16),  # 512k + sink
]
# fmt: on


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score_type", ["max", "lse"])
@pytest.mark.parametrize(
    "bs,nqh,nkh,hd,blk,with_sink,seq_pat,ib,lb,tk",
    CASES,
)
def test_flash_decode_with_topk_idx(
    bs, nqh, nkh, hd, blk, with_sink, seq_pat, ib, lb, tk, score_type
):
    torch.manual_seed(42)
    seq_lens, mkl = make_seq_lens(seq_pat, bs, blk)

    q, sink, k_cache, v_cache, req_to_token, seq_lens_t, mkl, slot_ids = build_inputs(
        bs,
        nqh,
        nkh,
        hd,
        seq_lens,
        max_kv_len=mkl,
        with_sink=with_sink,
    )

    o_new, topk_new, _ = flash_decode_with_topk_idx(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens_t,
        mkl,
        slot_ids,
        blk,
        tk,
        ib,
        lb,
        score_type=score_type,
    )
    o_ref, topk_ref = pytorch_reference(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens_t,
        slot_ids,
        blk,
        tk,
        ib,
        lb,
        score_type=score_type,
    )
    o_ref = o_ref.to(o_new.dtype)

    # --- attention output vs reference ---
    assert torch.allclose(
        o_new.float(), o_ref.float(), rtol=RTOL_VS_REF, atol=ATOL_VS_REF
    ), f"vs ref max abs diff {(o_new.float() - o_ref.float()).abs().max().item():.4e}"

    # --- topk set match vs reference ---
    for h in range(nqh):
        for b in range(bs):
            sl = seq_lens[b]
            num_blocks = (sl + blk - 1) // blk
            actual_k = min(tk, num_blocks)
            set_new = set(topk_new[h, b, :actual_k].tolist())
            set_ref = set(topk_ref[h, b, :actual_k].tolist())
            assert (
                set_new == set_ref
            ), f"topk mismatch at h={h} b={b}: kernel={set_new} ref={set_ref}"

    # --- topk sentinel: invalid positions must be -1 ---
    for b in range(bs):
        sl = seq_lens[b]
        num_blocks = (sl + blk - 1) // blk
        actual_k = min(tk, num_blocks)
        if actual_k < tk:
            invalid = topk_new[:, b, actual_k:]
            assert (
                invalid == -1
            ).all(), f"sentinel fail at b={b}: expected -1, got {invalid[invalid != -1].tolist()}"


@pytest.mark.parametrize("score_type", ["max", "lse"])
@pytest.mark.parametrize(
    "bs,nqh,nkh,hd,blk,with_sink,seq_pat,ib,lb,tk",
    CASES,
)
def test_flash_decode_score_only(
    bs, nqh, nkh, hd, blk, with_sink, seq_pat, ib, lb, tk, score_type
):
    torch.manual_seed(42)
    seq_lens, mkl = make_seq_lens(seq_pat, bs, blk)

    q, sink, k_cache, v_cache, req_to_token, seq_lens_t, mkl, slot_ids = build_inputs(
        bs,
        nqh,
        nkh,
        hd,
        seq_lens,
        max_kv_len=mkl,
        with_sink=with_sink,
    )

    o_new, topk_new, _ = flash_decode_with_topk_idx(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens_t,
        mkl,
        slot_ids,
        blk,
        tk,
        ib,
        lb,
        disable_index_value=True,
        score_type=score_type,
    )
    _, topk_ref = pytorch_reference(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens_t,
        slot_ids,
        blk,
        tk,
        ib,
        lb,
        score_type=score_type,
    )

    assert o_new is None, "expected None output when disable_index_value=True"

    for h in range(nqh):
        for b in range(bs):
            sl = seq_lens[b]
            num_blocks = (sl + blk - 1) // blk
            actual_k = min(tk, num_blocks)
            set_new = set(topk_new[h, b, :actual_k].tolist())
            set_ref = set(topk_ref[h, b, :actual_k].tolist())
            assert (
                set_new == set_ref
            ), f"topk mismatch at h={h} b={b}: kernel={set_new} ref={set_ref}"

    for b in range(bs):
        sl = seq_lens[b]
        num_blocks = (sl + blk - 1) // blk
        actual_k = min(tk, num_blocks)
        if actual_k < tk:
            invalid = topk_new[:, b, actual_k:]
            assert (
                invalid == -1
            ).all(), f"sentinel fail at b={b}: expected -1, got {invalid[invalid != -1].tolist()}"


def test_flash_decode_jit_topk_trivial_rows_skip_score_writes():
    torch.manual_seed(123)
    bs, nqh, nkh, hd, blk, tk = 4, 8, 1, 128, 64, 32
    seq_lens, mkl = make_seq_lens("few_blocks", bs, blk)

    q, sink, k_cache, v_cache, req_to_token, seq_lens_t, mkl, slot_ids = build_inputs(
        bs,
        nqh,
        nkh,
        hd,
        seq_lens,
        max_kv_len=mkl,
    )

    with envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.override(True):
        o_new, topk_new, real_seq_lens = flash_decode_with_topk_idx(
            q,
            sink,
            k_cache,
            v_cache,
            req_to_token,
            seq_lens_t,
            mkl,
            slot_ids,
            blk,
            tk,
            0,
            0,
        )
    o_ref, topk_ref = pytorch_reference(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens_t,
        slot_ids,
        blk,
        tk,
        0,
        0,
    )

    assert real_seq_lens is None
    assert torch.allclose(
        o_new.float(), o_ref.float(), rtol=RTOL_VS_REF, atol=ATOL_VS_REF
    )
    for h in range(nqh):
        for b in range(bs):
            sl = seq_lens[b]
            num_blocks = (sl + blk - 1) // blk
            actual_k = min(tk, num_blocks)
            assert set(topk_new[h, b, :actual_k].tolist()) == set(
                topk_ref[h, b, :actual_k].tolist()
            )
            assert (topk_new[h, b, actual_k:] == -1).all()


def test_flash_decode_dense_page_table_trivial_rows_skip_score_writes():
    torch.manual_seed(321)
    bs, nqh, nkh, hd, blk, tk, page_size = 3, 4, 1, 128, 64, 32, 1
    seq_lens, mkl = make_seq_lens("few_blocks", bs, blk)

    q, sink, k_cache, v_cache, req_to_token, seq_lens_t, mkl, slot_ids = build_inputs(
        bs,
        nqh,
        nkh,
        hd,
        seq_lens,
        max_kv_len=mkl,
    )

    o_new, page_table, real_seq_lens = flash_decode_with_topk_idx(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens_t,
        mkl,
        slot_ids,
        blk,
        tk,
        0,
        0,
        use_dense_main_attn=True,
        page_size=page_size,
    )
    o_ref, _ = pytorch_reference(
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens_t,
        slot_ids,
        blk,
        tk,
        0,
        0,
    )

    assert torch.allclose(
        o_new.float(), o_ref.float(), rtol=RTOL_VS_REF, atol=ATOL_VS_REF
    )
    assert page_table.shape == (bs * nqh, tk * blk // page_size)
    assert torch.equal(real_seq_lens.cpu(), seq_lens_t.repeat_interleave(nqh).cpu())

    page_table_cpu = page_table.cpu()
    req_to_token_cpu = req_to_token.cpu()
    for b, seq_len in enumerate(seq_lens):
        valid_pages = seq_len // page_size
        for h in range(nqh):
            row = b * nqh + h
            expected = req_to_token_cpu[b, :seq_len:page_size] // page_size * nqh + h
            assert torch.equal(page_table_cpu[row, :valid_pages], expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
