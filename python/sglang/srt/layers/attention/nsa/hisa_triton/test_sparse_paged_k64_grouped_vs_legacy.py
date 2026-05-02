"""K=64 sparse_paged: grouped (new unified path) vs legacy kernel
(deleted from kernels.py but inlined here as the reference).

Both are triton; same math, same fp8 accumulation order at G=1 K=64
(grouped reduces to legacy when GEMM_TILE = paged_block_size). Should
match BIT-EXACT (max_abs == 0).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    sparse_paged_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa_triton.test_e2e_simulated import (
    BLOCK_TOPK_FORMULA, make_decode_inputs,
)


@triton.jit
def _legacy_k64_kernel(
    Q_ptr, KvCacheFp8_ptr, KvCacheFp32_ptr, TopK_ptr, Logits_ptr, W_ptr,
    ContextLens_ptr, BlockTables_ptr,
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    stride_kv8_p, stride_kv8_b,
    stride_kv32_p, stride_kv32_b,
    stride_topk_b, stride_topk_s, stride_topk_n,
    stride_logits_b, stride_logits_s, stride_logits_n,
    stride_w_b, stride_w_s, stride_w_h,
    stride_bt_b, stride_bt_mb,
    max_blocks, num_phys,
    PAGED_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SUBS_PER_TOPK: tl.constexpr,
):
    b = tl.program_id(0)
    seq_i = tl.program_id(1)
    subblock_idx = tl.program_id(2)
    n_i = subblock_idx // SUBS_PER_TOPK
    sub_i = subblock_idx % SUBS_PER_TOPK

    topk_block_id = tl.load(
        TopK_ptr + b * stride_topk_b + seq_i * stride_topk_s + n_i * stride_topk_n,
    ).to(tl.int32)
    block_s_i = topk_block_id * KV_BLOCK_SIZE + sub_i * PAGED_BLOCK_SIZE
    logical_page = block_s_i // PAGED_BLOCK_SIZE
    valid_page = (logical_page >= 0) & (logical_page < max_blocks)
    phys = tl.load(
        BlockTables_ptr + b * stride_bt_b + logical_page * stride_bt_mb,
        mask=valid_page, other=0,
    ).to(tl.int32)
    valid = valid_page & (phys >= 0) & (phys < num_phys)
    phys = tl.where(valid, phys, 0)

    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q = tl.load(
        Q_ptr + b * stride_q_b + seq_i * stride_q_s
        + h_offs[:, None] * stride_q_h + d_offs[None, :] * stride_q_d
    )
    w = tl.load(W_ptr + b * stride_w_b + seq_i * stride_w_s + h_offs * stride_w_h)

    bn_offs = tl.arange(0, PAGED_BLOCK_SIZE)
    k_byte_offs = bn_offs[:, None] * DIM + d_offs[None, :]
    k = tl.load(
        KvCacheFp8_ptr + phys * stride_kv8_p + k_byte_offs * stride_kv8_b
    )
    SCALE_OFFSET: tl.constexpr = PAGED_BLOCK_SIZE * DIM // 4
    k_scale = tl.load(
        KvCacheFp32_ptr + phys * stride_kv32_p
        + (SCALE_OFFSET + bn_offs) * stride_kv32_b
    )
    s = tl.dot(k, q.trans(1, 0), out_dtype=tl.float32)
    s = s * k_scale[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]
    logits = tl.sum(s, axis=1)

    context_len = tl.load(ContextLens_ptr + b)
    k_i = block_s_i + bn_offs
    pos_valid = (k_i >= 0) & (k_i < context_len) & valid
    logits = tl.where(pos_valid, logits, float("-inf"))
    out_cols = n_i * KV_BLOCK_SIZE + sub_i * PAGED_BLOCK_SIZE + bn_offs
    tl.store(
        Logits_ptr + b * stride_logits_b + seq_i * stride_logits_s
        + out_cols * stride_logits_n,
        logits,
    )


def legacy_k64_call(q_fp8, kv_cache_fp8, topk_idx, weights, context_lens, block_tables):
    B, seq_len, H, D = q_fp8.shape
    topk = int(topk_idx.shape[-1])
    num_phys, paged_block_size, _, _ = kv_cache_fp8.shape
    max_blocks = block_tables.shape[-1]
    K = 64
    if weights.ndim == 2:
        weights = weights.view(B, seq_len, H)
    kv_flat = kv_cache_fp8.view(num_phys, -1)
    kv8 = kv_flat.view(torch.float8_e4m3fn)
    kv32 = kv_flat.view(torch.float32)
    logits = torch.empty(
        (B, seq_len, topk * K), device=q_fp8.device, dtype=torch.float32,
    )
    SUBS_PER_TOPK = 1  # K // paged_block_size at K==paged
    grid = (B, seq_len, topk * SUBS_PER_TOPK)
    _legacy_k64_kernel[grid](
        q_fp8, kv8, kv32, topk_idx, logits, weights, context_lens, block_tables,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2), q_fp8.stride(3),
        kv8.stride(0), kv8.stride(1),
        kv32.stride(0), kv32.stride(1),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        logits.stride(0), logits.stride(1), logits.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        max_blocks, num_phys,
        PAGED_BLOCK_SIZE=paged_block_size,
        KV_BLOCK_SIZE=K,
        HEADS=H, DIM=D,
        SUBS_PER_TOPK=SUBS_PER_TOPK,
    )
    return logits


def main():
    K = 64
    block_topk = BLOCK_TOPK_FORMULA // K
    print("=" * 80)
    print(f"K=64 sparse_paged: grouped (new) vs legacy (deleted) — bit-equal expected")
    print("=" * 80)
    print(f"{'B':>3} {'ctx':>5} {'topk':>5} | {'max_abs':>10} {'rel':>10} {'eq':>5}  status")
    print("-" * 80)

    for B in (1, 4, 32):
        for ctx in (8 * 1024, 65 * 1024):
            inputs = make_decode_inputs(K, B, ctx)
            q, kv, _, _, weights, cl, bt = inputs
            torch.manual_seed(2)
            n_pool = (ctx + K - 1) // K
            topk_idx = torch.randint(
                0, n_pool, (B, 1, block_topk),
                device="cuda", dtype=torch.int64,
            )

            ref = legacy_k64_call(q, kv, topk_idx, weights, cl, bt)
            out = sparse_paged_mqa_triton(
                q_fp8=q, kv_cache_fp8=kv,
                topk_block_index=topk_idx, kv_block_size=K,
                weights=weights, context_lens=cl, block_tables=bt,
            )

            # NaN-tolerant equality: random uint8 bytes interpreted as f32
            # scales sometimes produce NaN (~1/256). Both kernels see the
            # same K_scale → same NaN at same positions, but torch.equal
            # returns False per IEEE. Real check: NaN-count match, -inf
            # count match, finite values bit-equal.
            ref_nan = torch.isnan(ref); out_nan = torch.isnan(out)
            ref_neg = torch.isneginf(ref); out_neg = torch.isneginf(out)
            ref_pos = torch.isposinf(ref); out_pos = torch.isposinf(out)
            nan_match = torch.equal(ref_nan, out_nan)
            neg_match = torch.equal(ref_neg, out_neg)
            pos_match = torch.equal(ref_pos, out_pos)
            both = torch.isfinite(ref) & torch.isfinite(out)
            if both.any():
                d = (ref[both] - out[both]).abs()
                max_abs = d.max().item()
                scale = ref[both].abs().max().item()
                rel = max_abs / max(scale, 1e-6)
            else:
                max_abs, rel = 0.0, 0.0
            bit_eq = nan_match and neg_match and pos_match and max_abs == 0.0
            status = "OK" if bit_eq else "FAIL"
            print(f"{B:>3} {ctx//1024:>3}K {block_topk:>5} | "
                  f"{max_abs:>10.4f} {rel:>10.2e} {str(bit_eq):>5}  [{status}]")
            del q, kv, weights, cl, bt, topk_idx, ref, out
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
