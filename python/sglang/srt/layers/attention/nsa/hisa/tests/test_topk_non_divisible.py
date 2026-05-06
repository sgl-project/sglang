"""Method-B parity test: non-divisible topk vs padded-divisible topk.

After the in-kernel masking change in `_block_sparse_mqa_grouped_kernel` and
`_sparse_paged_mqa_grouped_kernel`, both grouped kernels accept any topk
(no longer require ``topk % GROUP_SIZE == 0``).

Strategy: for each kernel, run two configs and verify byte-equal logits on
the overlapping prefix:

  A) topk = T_nd  (non-divisible, e.g. T_nd = T_div - 1)
  B) topk = T_div (divisible, T_div padded above T_nd)

Build B's topk_block_index by concatenating A's index + arbitrary extra
columns. Compare ``out_A[:, : T_nd*K]`` vs ``out_B[:, : T_nd*K]`` — must be
identical (the extra columns in B are not in A's output).

Also runs a no-op size sanity (topk=1, smallest non-divisible) and the
boundary case ``topk = T_div - 1``.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    block_sparse_mqa_triton,
    sparse_paged_mqa_triton,
)


DEVICE = torch.device("cuda")
H, D = 64, 128


# =============================================================================
# Prefill block_sparse_mqa_triton — ragged
# =============================================================================

def make_ragged_inputs(seq_q, seq_kv, K, topk):
    torch.manual_seed(0)
    q = torch.randn(seq_q, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(seq_kv, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_scale = 0.05 + 0.02 * torch.rand(seq_kv, device=DEVICE, dtype=torch.float32)
    weights = torch.randn(seq_q, H, device=DEVICE, dtype=torch.float32)
    cu_ks = torch.zeros(seq_q, device=DEVICE, dtype=torch.int32)
    cu_ke = torch.linspace(seq_kv // 4, seq_kv, seq_q, device=DEVICE).to(torch.int32)
    num_blocks = seq_kv // K
    topk_idx = torch.randint(
        0, num_blocks, (seq_q, topk), device=DEVICE, dtype=torch.int64,
    )
    return q, k_fp8, k_scale, topk_idx, weights, cu_ks, cu_ke


def prefill_run(K, topk_target, topk_full):
    """Run with topk=topk_target via slicing topk_idx[:, :topk_target]; full
    column count for B is topk_full. Both share the same data prefix.
    """
    seq_q = 256
    seq_kv = 16384
    q, k_fp8, ks, topk_idx_full, w, cu_ks, cu_ke = make_ragged_inputs(
        seq_q, seq_kv, K, topk_full,
    )
    topk_idx_target = topk_idx_full[:, :topk_target].contiguous()

    out_target = block_sparse_mqa_triton(
        q_fp8=q, k_fp8=k_fp8, k_scale=ks,
        topk_block_index=topk_idx_target,
        kv_block_size=K, weights=w,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )
    out_full = block_sparse_mqa_triton(
        q_fp8=q, k_fp8=k_fp8, k_scale=ks,
        topk_block_index=topk_idx_full,
        kv_block_size=K, weights=w,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )
    return out_target, out_full[:, : topk_target * K].contiguous()


def _byte_equal_nan_aware(a: torch.Tensor, b: torch.Tensor) -> bool:
    """torch.equal treats NaN != NaN. We want NaN-positions == NaN-positions
    AND finite values bit-equal. (Random uint8 → f32 scale view in decode test
    can produce NaN K-scale → NaN GEMM result; both A and B see the same NaN
    so the kernel is correct.)
    """
    if a.shape != b.shape:
        return False
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    if not torch.equal(nan_a, nan_b):
        return False
    finite = ~nan_a
    return torch.equal(a[finite], b[finite])


def test_prefill_nondivisible():
    print("=" * 80)
    print("Prefill block_sparse_mqa_triton — non-divisible topk parity")
    print("=" * 80)
    print(f"{'K':>4} {'GROUP':>5} {'topk':>5} {'topk_full':>9} | shape | byte_eq | max|Δ|")
    print("-" * 80)

    cfgs = [
        # (K, GROUP_SIZE = 256/K, topk values)
        (8,  32, [1, 31, 33, 65, 100, 1023, 2047]),  # mix of small + non-aligned
        (16, 16, [1, 15, 17, 33, 100, 1023, 2047]),
        (32,  8, [1,  7,  9, 17, 100, 1023, 2047]),
        (64,  4, [1,  3,  5,  9, 100, 1023, 2047]),
    ]
    fail = 0
    for K, GROUP_SIZE, topk_list in cfgs:
        for topk in topk_list:
            # Pick topk_full as next-multiple-of-GROUP-above-topk.
            topk_full = ((topk + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
            if topk_full == topk:
                topk_full = topk + GROUP_SIZE  # force a non-trivial pad
            try:
                out_t, out_f = prefill_run(K, topk, topk_full)
                shape_ok = (out_t.shape == out_f.shape == (256, topk * K))
                byte_eq = _byte_equal_nan_aware(out_t, out_f)
                # NaN-safe diff (both have -inf masks; equal means same finite + same inf)
                if not byte_eq:
                    finite = torch.isfinite(out_t) & torch.isfinite(out_f)
                    if finite.any():
                        max_diff = (out_t[finite] - out_f[finite]).abs().max().item()
                    else:
                        max_diff = 0.0
                else:
                    max_diff = 0.0
                ok = shape_ok and byte_eq
                fail += int(not ok)
                print(
                    f"{K:>4} {GROUP_SIZE:>5} {topk:>5} {topk_full:>9} |"
                    f" {str(shape_ok):>5} | {str(byte_eq):>7} | {max_diff:>10.3e}"
                )
            except Exception as e:
                fail += 1
                print(
                    f"{K:>4} {GROUP_SIZE:>5} {topk:>5} {topk_full:>9} | "
                    f"ERROR: {type(e).__name__}: {str(e)[:50]}"
                )
    return fail


# =============================================================================
# Decode sparse_paged_mqa_triton — paged
# =============================================================================

def make_paged_inputs(B, seq_kv, K, topk, paged_block_size=64):
    """Generate paged inputs: kv_cache, block_tables, context_lens, topk_idx."""
    torch.manual_seed(1)
    num_blocks = (seq_kv + paged_block_size - 1) // paged_block_size
    num_phys = num_blocks * 2  # some margin

    q = torch.randn(B, 1, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    # kv_cache_fp8: [num_phys, paged_block_size, 1, D+4] uint8
    kv_cache = torch.randint(
        0, 256, (num_phys, paged_block_size, 1, D + 4),
        device=DEVICE, dtype=torch.uint8,
    )
    weights = torch.randn(B, 1, H, device=DEVICE, dtype=torch.float32)
    context_lens = torch.full((B,), seq_kv, device=DEVICE, dtype=torch.int32)

    block_tables = torch.arange(
        num_blocks, device=DEVICE, dtype=torch.int32,
    )[None, :].expand(B, -1).contiguous()

    num_kv_blocks = seq_kv // K
    topk_idx = torch.randint(
        0, num_kv_blocks, (B, 1, topk), device=DEVICE, dtype=torch.int64,
    )
    return q, kv_cache, weights, context_lens, block_tables, topk_idx


def decode_run(K, topk_target, topk_full):
    B = 8
    seq_kv = 8192
    q, kv, w, ctx, bt, topk_idx_full = make_paged_inputs(B, seq_kv, K, topk_full)
    topk_idx_target = topk_idx_full[:, :, :topk_target].contiguous()

    out_target = sparse_paged_mqa_triton(
        q_fp8=q, kv_cache_fp8=kv,
        topk_block_index=topk_idx_target,
        kv_block_size=K, weights=w,
        context_lens=ctx, block_tables=bt,
    )
    out_full = sparse_paged_mqa_triton(
        q_fp8=q, kv_cache_fp8=kv,
        topk_block_index=topk_idx_full,
        kv_block_size=K, weights=w,
        context_lens=ctx, block_tables=bt,
    )
    return out_target, out_full[..., : topk_target * K].contiguous()


def test_decode_nondivisible():
    print("=" * 80)
    print("Decode sparse_paged_mqa_triton — non-divisible topk parity")
    print("=" * 80)
    print(f"{'K':>4} {'GROUP':>5} {'topk':>5} {'topk_full':>9} | shape | byte_eq | max|Δ|")
    print("-" * 80)

    cfgs = [
        # (K, GROUP_SIZE = 64/K, topk values)
        (8, 8, [1, 7, 9, 15, 100, 511, 1023]),
        (16, 4, [1, 3, 5, 9,  100, 511, 1023]),
        (32, 2, [1, 3, 5, 7,  100, 511, 1023]),
    ]
    fail = 0
    for K, GROUP_SIZE, topk_list in cfgs:
        for topk in topk_list:
            topk_full = ((topk + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
            if topk_full == topk:
                topk_full = topk + GROUP_SIZE
            try:
                out_t, out_f = decode_run(K, topk, topk_full)
                shape_ok = (out_t.shape == out_f.shape == (8, 1, topk * K))
                byte_eq = _byte_equal_nan_aware(out_t, out_f)
                if not byte_eq:
                    finite = torch.isfinite(out_t) & torch.isfinite(out_f)
                    max_diff = (out_t[finite] - out_f[finite]).abs().max().item() if finite.any() else 0.0
                else:
                    max_diff = 0.0
                ok = shape_ok and byte_eq
                fail += int(not ok)
                print(
                    f"{K:>4} {GROUP_SIZE:>5} {topk:>5} {topk_full:>9} |"
                    f" {str(shape_ok):>5} | {str(byte_eq):>7} | {max_diff:>10.3e}"
                )
            except Exception as e:
                fail += 1
                print(
                    f"{K:>4} {GROUP_SIZE:>5} {topk:>5} {topk_full:>9} | "
                    f"ERROR: {type(e).__name__}: {str(e)[:50]}"
                )
    return fail


def main():
    f1 = test_prefill_nondivisible()
    print()
    f2 = test_decode_nondivisible()
    print()
    print("=" * 80)
    total_fail = f1 + f2
    print(f"TOTAL FAILURES: {total_fail}  (prefill={f1}, decode={f2})")
    print("=" * 80)
    if total_fail == 0:
        print("ALL_OK")


if __name__ == "__main__":
    main()
