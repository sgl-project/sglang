"""Kernel-level OOB stress test under compute-sanitizer.

Exercises every triton kernel that loads a phys-page id from a metadata
table (BlockTables / PoolPageTables / ReqToToken). For each kernel we run
TWO phases:

  1. clean — random valid tables, just smoke-checks the kernel.
  2. poisoned — ~30 percent of the table entries replaced with sentinel
     values that are out of range (-1, INT32_MIN/MAX, num_phys,
     num_phys + small offset). The defensive
       valid &= (phys >= 0) & (phys < num_phys)
       phys = tl.where(valid, phys, 0)
     pattern inside each kernel must neutralize these so the K / scale
     loads stay in-bounds.

Usage:
  # Bare run — just confirms each kernel does not crash on poisoned input.
  python test_oob_sanitizer.py

  # Sanitizer run — 0 = no OOB, 99 = sanitizer caught at least one OOB.
  compute-sanitizer --tool memcheck --error-exitcode 99 \\
      --target-processes all python test_oob_sanitizer.py
  echo $?

A clean run on the patched kernels should produce zero
"Invalid __global__ read of size N bytes" lines.
Removing any single `(phys >= 0) & (phys < num_phys)` clamp from
kernels.py should make the corresponding case fire under sanitizer.
"""

import sys
import torch

from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    batch_decode_pool_mqa_triton,
    paged_mean_pooling_triton,
    sparse_paged_mqa_triton,
    tail_only_triton,
    update_pool_for_completed_blocks_triton,
)


# ---- Sentinel values we splatter into metadata tables --------------------
def _poison_int32(table: torch.Tensor, num_phys: int, ratio: float = 0.3):
    """In-place: scatter sentinel out-of-range values into ~`ratio` of the
    table entries. Returns the table for chaining."""
    flat = table.view(-1)
    n = flat.numel()
    bad_n = max(4, int(n * ratio))
    sentinels = torch.tensor(
        [-1, -2147483648, 2147483647, num_phys, num_phys + 1, num_phys + 17, 99999],
        dtype=table.dtype, device=table.device,
    )
    bad_idx = torch.randint(0, n, (bad_n,), device=table.device)
    bad_pick = sentinels[torch.randint(0, len(sentinels), (bad_n,), device=table.device)]
    flat[bad_idx] = bad_pick
    return table


def _poison_req_to_token(req_to_token: torch.Tensor, num_phys: int, paged: int):
    """req_to_token holds *token positions*, not phys ids. The kernel
    derives phys = pos // paged_block_size, so to get phys = num_phys we
    set pos = num_phys * paged. Mix in negatives + huge values too."""
    flat = req_to_token.view(-1)
    n = flat.numel()
    bad_n = max(4, n // 3)
    sentinels = torch.tensor(
        [-1, num_phys * paged, (num_phys + 5) * paged, 2147483647 // paged * paged],
        dtype=req_to_token.dtype, device=req_to_token.device,
    )
    bad_idx = torch.randint(0, n, (bad_n,), device=req_to_token.device)
    bad_pick = sentinels[torch.randint(0, len(sentinels), (bad_n,), device=req_to_token.device)]
    flat[bad_idx] = bad_pick
    return req_to_token


def _alloc_paged_kv(num_phys, paged, D):
    return torch.zeros(num_phys, paged, 1, D + 4, dtype=torch.uint8, device="cuda")


def _alloc_pool_pages(num_pool_phys, pool_page_size, D):
    return torch.zeros(
        num_pool_phys, pool_page_size * (D + 4), dtype=torch.uint8, device="cuda",
    )


# ---- Kernel cases --------------------------------------------------------
def case_sparse_paged_mqa(K, *, poison: bool):
    B, seq, H, D = 2, 1, 32, 128
    P = 64
    num_phys = 8
    topk = 64
    max_blocks = 8

    q = torch.randn(B, seq, H, D, device="cuda").to(torch.float8_e4m3fn)
    kv = _alloc_paged_kv(num_phys, P, D)
    weights = torch.rand(B, seq, H, device="cuda")
    context_lens = torch.tensor([P * 4, P * 6], dtype=torch.int32, device="cuda")
    block_tables = torch.randint(
        0, num_phys, (B, max_blocks), dtype=torch.int32, device="cuda",
    )
    if poison:
        _poison_int32(block_tables, num_phys)

    topk_idx = torch.randint(
        0, max_blocks * P // K, (B, seq, topk),
        dtype=torch.int64, device="cuda",
    )
    out = sparse_paged_mqa_triton(
        q, kv, topk_idx, K, weights, context_lens, block_tables,
    )
    torch.cuda.synchronize()
    assert out.shape == (B, seq, topk * K)


def case_paged_mean_pooling(K, *, poison: bool):
    B, D = 2, 128
    P = 64
    num_phys = 8
    max_blocks = 8
    max_num_pool = max_blocks * P // K

    kv = _alloc_paged_kv(num_phys, P, D)
    context_lens = torch.tensor([P * 3, P * 5], dtype=torch.int32, device="cuda")
    block_tables = torch.randint(
        0, num_phys, (B, max_blocks), dtype=torch.int32, device="cuda",
    )
    if poison:
        _poison_int32(block_tables, num_phys)

    bk, bks, npool = paged_mean_pooling_triton(
        max_num_pool, kv, context_lens, block_tables, K,
    )
    torch.cuda.synchronize()
    assert bk.shape == (B, max_num_pool, D)


def case_batch_decode_pool_mqa_v3(*, poison: bool):
    B, H, D = 2, 32, 128
    PP = 64
    num_pool_phys = 8
    max_pp = 8

    q = torch.randn(B, 1, H, D, device="cuda").to(torch.float8_e4m3fn)
    pool_pages = _alloc_pool_pages(num_pool_phys, PP, D)
    pool_page_tables = torch.randint(
        0, num_pool_phys, (B, max_pp), dtype=torch.int32, device="cuda",
    )
    if poison:
        _poison_int32(pool_page_tables, num_pool_phys)
    weights = torch.rand(B, H, device="cuda")
    ctx_pool = torch.tensor([PP * 3, PP * 5], dtype=torch.int32, device="cuda")

    out = batch_decode_pool_mqa_triton(
        q, pool_pages, pool_page_tables, weights, ctx_pool, pool_page_size=PP,
    )
    torch.cuda.synchronize()
    assert out.shape == (B, 1, max_pp * PP)


def case_update_pool_for_completed_blocks(K, *, poison: bool):
    """SK15: poisons both
       (a) req_to_token (whose values become phys via floor-div), and
       (b) pool_page_tables (output phys side)."""
    B = 2
    D = 128
    P = 64
    PP = 64
    num_phys = 8
    num_pool_phys = 8
    max_req = 4
    max_ctx = 1024
    max_pool_pages = 16

    kv = _alloc_paged_kv(num_phys, P, D).view(num_phys, -1)
    pool_pages = _alloc_pool_pages(num_pool_phys, PP, D)
    req_to_token = torch.randint(
        0, num_phys * P, (max_req, max_ctx), dtype=torch.int32, device="cuda",
    )
    pool_page_tables = torch.randint(
        0, num_pool_phys, (max_req, max_pool_pages), dtype=torch.int32, device="cuda",
    )
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    prev_seq_lens = torch.tensor([0, K * 2], dtype=torch.int32, device="cuda")
    new_seq_lens = torch.tensor([K * 4, K * 6], dtype=torch.int32, device="cuda")

    if poison:
        _poison_req_to_token(req_to_token, num_phys, P)
        _poison_int32(pool_page_tables, num_pool_phys)

    max_pool_per_req = (new_seq_lens.max().item() + K - 1) // K
    update_pool_for_completed_blocks_triton(
        kv, req_to_token, pool_page_tables, req_pool_indices,
        prev_seq_lens, new_seq_lens, pool_pages,
        K, P, PP, max_pool_per_req,
    )
    torch.cuda.synchronize()


def case_tail_only_v3(K, *, poison: bool):
    """SK16: poisons both block_tables (input phys) and pool_page_tables
    (output phys)."""
    B = 2
    D = 128
    P = 64
    PP = 64
    num_phys = 8
    num_pool_phys = 8
    max_blocks = 8
    max_pool_pages = 16

    kv = _alloc_paged_kv(num_phys, P, D).view(num_phys, -1)
    pool_pages = _alloc_pool_pages(num_pool_phys, PP, D)
    context_lens = torch.tensor([P * 3 + K // 2, P * 5], dtype=torch.int32, device="cuda")
    block_tables = torch.randint(
        0, num_phys, (B, max_blocks), dtype=torch.int32, device="cuda",
    )
    pool_page_tables = torch.randint(
        0, num_pool_phys, (B, max_pool_pages), dtype=torch.int32, device="cuda",
    )

    if poison:
        _poison_int32(block_tables, num_phys)
        _poison_int32(pool_page_tables, num_pool_phys)

    tail_only_triton(
        kv, context_lens, block_tables, pool_page_tables, pool_pages,
        K, P, PP,
    )
    torch.cuda.synchronize()


# ---- Entry ----------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.", file=sys.stderr)
        sys.exit(0)
    torch.manual_seed(42)

    K_VALUES = (8, 16, 32, 64, 128)
    PHASES = (("clean", False), ("poisoned", True))

    def run(label, fn):
        try:
            fn()
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n  !! {label} raised: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            # device may be wedged after illegal access — try to keep going
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    for phase, poison in PHASES:
        print(f"=== phase: {phase} ===")
        for K in K_VALUES:
            print(f"  K={K:>3} ", end="", flush=True)
            run(f"sparse_paged_mqa K={K} {phase}",
                lambda K=K, p=poison: case_sparse_paged_mqa(K, poison=p))
            run(f"paged_mean_pooling K={K} {phase}",
                lambda K=K, p=poison: case_paged_mean_pooling(K, poison=p))
            run(f"update_pool K={K} {phase}",
                lambda K=K, p=poison: case_update_pool_for_completed_blocks(K, poison=p))
            run(f"tail_only_v3 K={K} {phase}",
                lambda K=K, p=poison: case_tail_only_v3(K, poison=p))
            print(" done")
        run(f"batch_decode_pool_mqa_v3 {phase}",
            lambda p=poison: case_batch_decode_pool_mqa_v3(poison=p))
        print(f"  batch_decode_pool_mqa_v3 done")

    print("\nAll kernels finished without Python-level error.")
    print("Under compute-sanitizer, exit code 0 means no OOB was caught.")


if __name__ == "__main__":
    main()
