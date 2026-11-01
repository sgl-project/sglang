"""DFLASH sampled verify on each row's top-k support, from the target's vocab shards.

With a top-k, only the top-k tokens of a verify row carry target mass. The dense
selector verify still all-gathers the fp32 logits of every row, softcaps them, runs
a radix top-k, scatters p and the selector's q into dense vocab-sized tensors and
scans the vocabulary in the chain sampler. Here instead:

1. ``pack_shard_topk``: each TP rank packs the top-``W`` (``W >= max top_k``) of
   its raw logits shard as int32 (value bits, global id) pairs, which the caller
   all-gathers rank-major;
2. ``merge_shard_topk``: softcap the ``world * W`` candidates as
   ``gathered_softcap_logits`` does, sort by (value desc, id asc), keep the first
   ``max_top_k`` and the row argmax (lowest id among maxima, as ``torch.argmax``);
3. ``sparse_chain_speculative_sampling``: the accept rule and residual resample of
   ``chain_speculative_sampling_triton`` with p the softmax over the row's kept
   top-k and q looked up in the selector's candidates.

Off the support p = 0, so the support holds the whole residual. The final CDF is
walked in token-id order, the order the dense kernel scans the vocabulary, so both
pick the same token for the same coin up to fp32 summation order. The accept
kernel adapts the top-k chain sampler of sgl-project/sglang#41283 to an exactly-k
top-k and a sparse q.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

try:
    from flashinfer import top_k as _flashinfer_top_k
except ImportError:
    _flashinfer_top_k = None


def shard_topk_width(max_top_k: int) -> int:
    """Per-shard top-k width. The 16 extra slots keep logits tied with the k-th one,
    common with bf16 logits; the merge keeps the lowest ids among ties, a valid
    exactly-k truncation whose tied ids may differ from torch.topk's."""
    return triton.next_power_of_2(max_top_k + 16)


@triton.jit
def _pack_shard_topk_kernel(
    vals_ptr,
    ids_ptr,
    out_ptr,
    vals_row_stride,
    ids_row_stride,
    col_offset,
    W: tl.constexpr,
):
    # out[row] = [float bits of the W values | W global ids], int32.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, W)
    v = tl.load(vals_ptr + row * vals_row_stride + offs)
    i = tl.load(ids_ptr + row * ids_row_stride + offs).to(tl.int32) + col_offset
    base = out_ptr + row * (2 * W)
    tl.store(base + offs, v.to(tl.int32, bitcast=True))
    tl.store(base + W + offs, i)


@triton.jit
def _merge_support_kernel(
    gathered_ptr,  # [world, rows, 2W] int32
    sup_vals_ptr,  # [rows, K_BLOCK] fp32, softcapped, sorted desc
    sup_ids_ptr,  # [rows, K_BLOCK] int32
    argmax_ptr,  # [rows] int64
    num_rows,
    softcap,
    W: tl.constexpr,
    WORLD: tl.constexpr,
    N_BLOCK: tl.constexpr,  # next_power_of_2(WORLD * W)
    K_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N_BLOCK)
    valid = offs < WORLD * W
    r = (offs // W).to(tl.int64)
    c = (offs % W).to(tl.int64)
    base = gathered_ptr + (r * num_rows + row) * (2 * W)
    bits = tl.load(base + c, mask=valid, other=0)
    ids = tl.load(base + W + c, mask=valid, other=0)
    x = bits.to(tl.float32, bitcast=True)
    # gathered_softcap_logits' formula, so kept values match the dense row bit for bit.
    x = x / softcap
    x = libdevice.tanh(x)
    x = x * softcap
    # Padding lanes sort last: -inf below every finite logit.
    x = tl.where(valid, x, float("-inf"))
    ids = tl.where(valid, ids, 0x7FFFFFFF)
    # Order-preserving signed int32 of the float, then (value desc, id asc) in one
    # int64 key: the low half holds 0x7FFFFFFF - id, which never touches the sign.
    xb = x.to(tl.int32, bitcast=True)
    ordered = tl.where(xb >= 0, xb, xb ^ 0x7FFFFFFF)
    key = (ordered.to(tl.int64) << 32) | (0x7FFFFFFF - ids).to(tl.int64)
    key = tl.sort(key, descending=True)
    out_ids = (0x7FFFFFFF - (key & 0xFFFFFFFF)).to(tl.int32)
    hi = (key >> 32).to(tl.int32)
    out_bits = tl.where(hi >= 0, hi, hi ^ 0x7FFFFFFF)
    out_vals = out_bits.to(tl.float32, bitcast=True)
    keep = offs < K_BLOCK
    tl.store(sup_vals_ptr + row * K_BLOCK + offs, out_vals, mask=keep)
    tl.store(sup_ids_ptr + row * K_BLOCK + offs, out_ids, mask=keep)
    first = tl.sum(tl.where(offs == 0, out_ids, 0))
    tl.store(argmax_ptr + row, first.to(tl.int64))


@triton.jit
def _support_probs(
    sup_vals_ptr,
    row,
    offs,
    kept,
    temperature,
    K_BLOCK: tl.constexpr,
):
    # softmax over the kept top-k of (logit / temperature), the dense path's
    # masked F.softmax(topk_logits) restricted to the support.
    z = tl.load(sup_vals_ptr + row * K_BLOCK + offs)
    z = z / temperature
    z = tl.where(kept, z, float("-inf"))
    e = tl.where(kept, tl.exp(z - tl.max(z, axis=0)), 0.0)
    return e / tl.sum(e, axis=0)


@triton.jit
def _sparse_chain_accept_kernel(
    sup_vals_ptr,  # [bs * NUM_SLOTS, K_BLOCK] fp32
    sup_ids_ptr,  # [bs * NUM_SLOTS, K_BLOCK] int32
    temps_ptr,  # [bs, 1] fp32
    temps_stride,
    top_ks_ptr,  # [bs] int
    candidates_ptr,  # [bs, NUM_SLOTS] int64
    cand_stride,
    q_ids_ptr,  # [bs, NUM_SLOTS - 1, KS] int64 (selector candidate ids)
    q_ids_stride_b,
    q_ids_stride_s,
    q_vals_ptr,  # [bs, NUM_SLOTS - 1, KS] fp32 (selector q)
    q_vals_stride_b,
    q_vals_stride_s,
    uni_ptr,  # [bs, NUM_SLOTS - 1] fp32
    uni_stride,
    uni_final_ptr,  # [bs] fp32
    correct_len_ptr,  # [bs] int32
    bonus_ptr,  # [bs] int64
    max_top_k,
    vocab_size,
    NUM_SLOTS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    KS: tl.constexpr,
    KS_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, K_BLOCK)
    ks = tl.arange(0, KS_BLOCK)
    ks_mask = ks < KS
    temperature = tl.load(temps_ptr + pid * temps_stride).to(tl.float32)
    top_k = tl.load(top_ks_ptr + pid).to(tl.int32)
    top_k = tl.minimum(tl.maximum(top_k, 1), vocab_size)
    top_k = tl.minimum(top_k, max_top_k)
    kept = offs < top_k

    num_correct = 0
    cur_row = 0
    step = 1
    go = 1
    while (step < NUM_SLOTS) and (go == 1):
        draft_token = tl.load(candidates_ptr + pid * cand_stride + step).to(tl.int32)
        row = pid * NUM_SLOTS + cur_row
        ids = tl.load(sup_ids_ptr + row * K_BLOCK + offs)
        probs = _support_probs(sup_vals_ptr, row, offs, kept, temperature, K_BLOCK)
        p = tl.sum(tl.where(kept & (ids == draft_token), probs, 0.0), axis=0)
        qi = tl.load(
            q_ids_ptr + pid * q_ids_stride_b + cur_row * q_ids_stride_s + ks,
            mask=ks_mask,
            other=-1,
        ).to(tl.int32)
        qv = tl.load(
            q_vals_ptr + pid * q_vals_stride_b + cur_row * q_vals_stride_s + ks,
            mask=ks_mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.sum(tl.where(qi == draft_token, qv, 0.0), axis=0)
        coin = tl.load(uni_ptr + pid * uni_stride + (step - 1))
        q_is_prob = (q > 0.0) & (q <= 1.0)
        if q_is_prob & (coin * q < p):
            num_correct += 1
            cur_row = step
            step += 1
        else:
            go = 0

    tl.store(correct_len_ptr + pid, num_correct.to(tl.int32))

    row = pid * NUM_SLOTS + cur_row
    ids = tl.load(sup_ids_ptr + row * K_BLOCK + offs)
    p_val = _support_probs(sup_vals_ptr, row, offs, kept, temperature, K_BLOCK)
    if go == 1:
        # Every draft accepted: sample the bonus from p of the last row.
        val = p_val
    else:
        # q at each support id: the selector's candidate ids are distinct per
        # row, so the matching entry is the scattered dense q.
        qi = tl.load(
            q_ids_ptr + pid * q_ids_stride_b + cur_row * q_ids_stride_s + ks,
            mask=ks_mask,
            other=-1,
        ).to(tl.int32)
        qv = tl.load(
            q_vals_ptr + pid * q_vals_stride_b + cur_row * q_vals_stride_s + ks,
            mask=ks_mask,
            other=0.0,
        ).to(tl.float32)
        match = ids[:, None] == qi[None, :]
        q_sup = tl.sum(tl.where(match, qv[None, :], 0.0), axis=1)
        q_sup = tl.where((q_sup >= 0.0) & (q_sup <= 1.0), q_sup, 0.0)
        diff = p_val - q_sup
        val = tl.where(diff > 0.0, diff, 0.0)
    val = tl.where(kept, val, 0.0)
    norm_sum = tl.sum(val, axis=0)
    # A degenerate residual (q >= p on the whole support) falls back to p, where the
    # dense kernel emits token vocab_size - 1.
    if norm_sum <= 0.0:
        val = tl.where(kept, p_val, 0.0)
        norm_sum = tl.sum(val, axis=0)

    # CDF in token-id order, as the dense kernel scans the vocabulary:
    # sort (id << 32 | value bits); val >= 0, so its bits stay in the low half.
    sid = tl.where(kept, ids, 2147483647).to(tl.int64)
    keys = tl.sort((sid << 32) | val.to(tl.int32, bitcast=True).to(tl.int64))
    sorted_ids = (keys >> 32).to(tl.int32)
    sorted_val = (keys & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
    cdf = tl.cumsum(sorted_val, axis=0)
    target_u = tl.load(uni_final_ptr + pid) * norm_sum
    final_token = tl.min(tl.where(cdf > target_u, sorted_ids, 2147483647), axis=0)
    if final_token == 2147483647:
        # Rounding left target_u at the top of the CDF: the last token with mass.
        final_token = tl.max(tl.where(sorted_val > 0.0, sorted_ids, -1), axis=0)
    tl.store(bonus_ptr + pid, final_token.to(tl.int64))


def pack_shard_topk(
    logits_shard: torch.Tensor, *, max_top_k: int, col_offset: int
) -> torch.Tensor:
    """``[rows, 2W]`` int32 of the top ``shard_topk_width(max_top_k)`` raw logits of
    a contiguous fp32 vocab shard: their float bits, then their global ids."""
    assert logits_shard.dtype == torch.float32 and logits_shard.is_contiguous()
    rows = logits_shard.shape[0]
    w = shard_topk_width(max_top_k)
    if _flashinfer_top_k is not None:
        vals, ids = _flashinfer_top_k(logits_shard, w, sorted=False, deterministic=True)
    else:
        vals, ids = torch.topk(logits_shard, w, dim=-1, sorted=False)
    out = torch.empty((rows, 2 * w), dtype=torch.int32, device=logits_shard.device)
    _pack_shard_topk_kernel[(rows,)](
        vals,
        ids,
        out,
        vals.stride(0),
        ids.stride(0),
        col_offset,
        W=w,
    )
    return out


def merge_shard_topk(
    gathered: torch.Tensor, *, world: int, softcap: float, max_top_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-``max_top_k`` support of the full softcapped rows from the rank-major
    all-gather ``[world * rows, 2W]`` of ``pack_shard_topk``.

    Returns ``(sup_vals [rows, K] fp32 sorted desc, sup_ids [rows, K] int32,
    argmax [rows] int64)``; columns ``>= max_top_k`` are padding.
    """
    rows = gathered.shape[0] // world
    w = gathered.shape[1] // 2
    k_block = max(16, triton.next_power_of_2(max_top_k))
    sup_vals = torch.empty((rows, k_block), dtype=torch.float32, device=gathered.device)
    sup_ids = torch.empty((rows, k_block), dtype=torch.int32, device=gathered.device)
    argmax = torch.empty((rows,), dtype=torch.int64, device=gathered.device)
    _merge_support_kernel[(rows,)](
        gathered,
        sup_vals,
        sup_ids,
        argmax,
        rows,
        softcap,
        W=w,
        WORLD=world,
        N_BLOCK=triton.next_power_of_2(world * w),
        K_BLOCK=k_block,
    )
    return sup_vals, sup_ids, argmax


def sparse_chain_speculative_sampling(
    *,
    sup_vals: torch.Tensor,
    sup_ids: torch.Tensor,
    candidates: torch.Tensor,
    q_ids: torch.Tensor,
    q_vals: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    max_top_k: int,
    vocab_size: int,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chain rejection sample on the support: ``(correct_len int32 [bs], bonus int64 [bs])``."""
    bs, num_slots = candidates.shape
    k_block = sup_vals.shape[-1]
    ks = q_ids.shape[-1]
    assert sup_vals.is_contiguous() and sup_ids.is_contiguous()
    assert q_ids.stride(-1) == 1 and q_vals.stride(-1) == 1
    assert uniform_samples.stride(-1) == 1
    temps = temperatures.reshape(bs, -1)
    correct_len = torch.empty((bs,), dtype=torch.int32, device=candidates.device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=candidates.device)
    _sparse_chain_accept_kernel[(bs,)](
        sup_vals,
        sup_ids,
        temps,
        temps.stride(0),
        top_ks,
        candidates,
        candidates.stride(0),
        q_ids,
        q_ids.stride(0),
        q_ids.stride(1),
        q_vals,
        q_vals.stride(0),
        q_vals.stride(1),
        uniform_samples,
        uniform_samples.stride(0),
        uniform_samples_final,
        correct_len,
        bonus,
        max_top_k,
        vocab_size,
        NUM_SLOTS=num_slots,
        K_BLOCK=k_block,
        KS=ks,
        KS_BLOCK=triton.next_power_of_2(ks),
    )
    return correct_len, bonus
