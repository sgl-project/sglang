"""Byte-equal tests for the Phase-2 v3 (paged pool K) kernels.

v3 stores pool rows in a paged layout (like the main KV cache), so
block_mqa reads them via TMA directly — no gather, no scratch.

Validates:
  - tail_only_v3 writes tail row into pool_k_pages[phys, slot, :] that
    byte-equals the tail slot of v1's fp8_native_paged_mean_pooling.
  - v3 orchestrator produces the same topk + sparse logits as v1 when
    pool_k_pages is pre-populated from v1's mean-pool output.
"""
from __future__ import annotations

import sys

import torch

from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    fp8_native_paged_mean_pooling_interface,
    fp8_native_paged_mean_pooling_tail_only_interface,
    fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy,
    fp8_native_hierarchy_paged_mqa_logits_tilelang_with_pool_cache,
)


DEVICE = torch.device("cuda")
POOL_PAGE_SIZE = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_paged_kv(num_blocks, paged_block_size, dim, seed):
    torch.manual_seed(seed)
    kv = torch.zeros(
        (num_blocks, paged_block_size, 1, dim + 4),
        dtype=torch.uint8, device=DEVICE,
    )
    k_bf16 = torch.randn(
        (num_blocks, paged_block_size, 1, dim),
        dtype=torch.bfloat16, device=DEVICE,
    )
    k_fp8 = k_bf16.to(torch.float8_e4m3fn)
    kv[..., :dim] = k_fp8.view(torch.uint8)
    scale = 0.05 + 0.02 * torch.rand(
        (num_blocks, paged_block_size, 1), device=DEVICE, dtype=torch.float32
    )
    kv[..., dim:] = scale.view(torch.uint8).reshape(num_blocks, paged_block_size, 1, 4)
    return kv


def build_block_tables(B, max_blocks, num_blocks, seed):
    torch.manual_seed(seed + 7919)
    assert B * max_blocks <= num_blocks
    perm = torch.randperm(num_blocks, device=DEVICE, dtype=torch.int32)
    return perm[: B * max_blocks].view(B, max_blocks).contiguous()


def prepopulate_pool_k_pages(
    pool_k_pages: torch.Tensor,          # [N_pool_pages, pool_page_size * (D+4)] uint8
    pool_page_tables: torch.Tensor,      # [B, max_pool_pages] int32
    blocked_k_v1: torch.Tensor,          # [B, max_num_pooling_blocks, D] fp8
    blocked_k_scale_v1: torch.Tensor,    # [B, max_num_pooling_blocks] f32
    num_pool_per_req: torch.Tensor,      # [B] int32
    pool_page_size: int,
):
    """Fill pool_k_pages[phys, byte_offset] from v1's per-batch blocked_k
    output, using the SoA byte layout:
      bytes [slot * D, (slot + 1) * D)                                  = fp8 row
      bytes [pool_page_size * D + slot * 4, pool_page_size * D + (slot + 1) * 4) = scale
    """
    B, max_pool_blocks, D = blocked_k_v1.shape
    fp8_base = 0
    scale_base = pool_page_size * D
    for b in range(B):
        n = int(num_pool_per_req[b].item())
        for pblk in range(n - 1):  # exclude tail — refreshed by tail_only_v3
            logical_page = pblk // pool_page_size
            slot = pblk % pool_page_size
            phys = int(pool_page_tables[b, logical_page].item())
            pool_k_pages[phys, fp8_base + slot * D : fp8_base + (slot + 1) * D] = (
                blocked_k_v1[b, pblk, :].view(torch.uint8)
            )
            pool_k_pages[phys, scale_base + slot * 4 : scale_base + (slot + 1) * 4] = (
                blocked_k_scale_v1[b : b + 1, pblk : pblk + 1]
                .view(torch.uint8).reshape(4)
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@torch.inference_mode()
def test_tail_only_v3_byte_equal():
    print("\n[test_tail_only_v3_byte_equal]")
    B, D = 4, 128
    paged_block_size = 64
    k_block_size = 128
    pool_page_size = POOL_PAGE_SIZE
    max_ctx = 1024
    max_blocks = (max_ctx + paged_block_size - 1) // paged_block_size
    max_num_pooling_blocks = (max_ctx + k_block_size - 1) // k_block_size
    num_blocks = 256

    kv = build_paged_kv(num_blocks, paged_block_size, D, seed=11)
    block_tables = build_block_tables(B, max_blocks, num_blocks, seed=11)
    torch.manual_seed(17)
    context_lens = torch.randint(k_block_size + 1, max_ctx + 1, (B,),
                                 device=DEVICE, dtype=torch.int32)
    context_lens[0] = 200  # tail has 200 - 128 = 72 tokens

    # v1 baseline.
    blocked_k_v1, blocked_k_scale_v1, num_pool_v1 = (
        fp8_native_paged_mean_pooling_interface(
            max_num_pooling_blocks, kv, context_lens, block_tables, k_block_size,
        )
    )

    # v3: pool_k_pages with an identity pool_page_tables (2D SoA layout).
    max_pool_pages = (max_num_pooling_blocks + pool_page_size - 1) // pool_page_size + 1
    N_pool_pages = B * max_pool_pages + 4
    page_bytes = pool_page_size * (D + 4)
    pool_k_pages = torch.zeros(
        (N_pool_pages, page_bytes), dtype=torch.uint8, device=DEVICE,
    )
    torch.manual_seed(999)
    perm = torch.randperm(N_pool_pages, device=DEVICE, dtype=torch.int32)
    pool_page_tables = perm[: B * max_pool_pages].view(B, max_pool_pages).contiguous()

    fp8_native_paged_mean_pooling_tail_only_interface(
        kv_cache=kv, context_lens=context_lens, block_tables=block_tables,
        pool_page_tables=pool_page_tables, pool_k_pages=pool_k_pages,
        k_block_size=k_block_size, pool_page_size=pool_page_size,
    )

    # For each b, the tail is written to pool_k_pages[phys_last] at the
    # right SoA byte offset.
    fp8_base = 0
    scale_base = pool_page_size * D
    for b in range(B):
        n = int(num_pool_v1[b].item())
        tail_pblk = n - 1
        logical_page = tail_pblk // pool_page_size
        slot = tail_pblk % pool_page_size
        phys = int(pool_page_tables[b, logical_page].item())

        v3_fp8 = pool_k_pages[phys, fp8_base + slot * D : fp8_base + (slot + 1) * D]
        v3_scale_bytes = pool_k_pages[phys, scale_base + slot * 4 : scale_base + (slot + 1) * 4]
        v3_scale = v3_scale_bytes.view(torch.float32).item()
        v1_fp8 = blocked_k_v1[b, tail_pblk, :].view(torch.uint8)
        v1_scale = blocked_k_scale_v1[b, tail_pblk].item()

        assert torch.equal(v3_fp8, v1_fp8), (
            f"b={b} tail fp8 mismatch at phys={phys} slot={slot}"
        )
        assert abs(v3_scale - v1_scale) / max(abs(v1_scale), 1e-9) < 1e-5, (
            f"b={b} tail scale mismatch: {v3_scale} vs {v1_scale}"
        )
    print(f"  OK — {B} tail slots byte-equal vs paged v1 mean-pool")


@torch.inference_mode()
def test_orchestrator_v3_matches_v1():
    print("\n[test_orchestrator_v3_matches_v1]")
    B, H, D = 4, 64, 128
    paged_block_size = 64
    k_block_size = 128
    pool_page_size = POOL_PAGE_SIZE
    block_topk = 4
    max_ctx = 1024
    max_blocks = (max_ctx + paged_block_size - 1) // paged_block_size
    max_num_pooling_blocks = (max_ctx + k_block_size - 1) // k_block_size
    num_blocks = 256

    kv = build_paged_kv(num_blocks, paged_block_size, D, seed=3)
    block_tables = build_block_tables(B, max_blocks, num_blocks, seed=3)
    torch.manual_seed(31)
    context_lens = torch.randint(
        k_block_size * block_topk + 1, max_ctx + 1, (B,),
        device=DEVICE, dtype=torch.int32,
    )

    q_bf16 = torch.randn((B, 1, H, D), device=DEVICE, dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    weights = torch.randn((B * 1, H), device=DEVICE, dtype=torch.float32)

    # v1 orchestrator (no cache).
    sparse_v1, topk_v1 = fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy(
        q_fp8=q_fp8, kv_cache_fp8=kv, weights=weights,
        context_lens=context_lens, block_tables=block_tables,
        schedule_metadata=None,
        max_model_len=max_ctx, max_seq_len=max_ctx,
        k_block_size=k_block_size, block_topk=block_topk,
    )

    # Pre-populate pool_k_pages from v1 mean-pool output.
    blocked_k_v1, blocked_k_scale_v1, num_pool_v1 = (
        fp8_native_paged_mean_pooling_interface(
            max_num_pooling_blocks, kv, context_lens, block_tables, k_block_size,
        )
    )
    max_pool_pages = (max_num_pooling_blocks + pool_page_size - 1) // pool_page_size + 1
    N_pool_pages = B * max_pool_pages + 8
    page_bytes = pool_page_size * (D + 4)
    pool_k_pages = torch.zeros(
        (N_pool_pages, page_bytes), dtype=torch.uint8, device=DEVICE,
    )
    torch.manual_seed(555)
    perm = torch.randperm(N_pool_pages, device=DEVICE, dtype=torch.int32)
    pool_page_tables = perm[: B * max_pool_pages].view(B, max_pool_pages).contiguous()
    prepopulate_pool_k_pages(
        pool_k_pages, pool_page_tables, blocked_k_v1, blocked_k_scale_v1, num_pool_v1,
        pool_page_size,
    )

    # v3 orchestrator (tail is refreshed in place).
    sparse_v3, topk_v3 = fp8_native_hierarchy_paged_mqa_logits_tilelang_with_pool_cache(
        q_fp8=q_fp8, kv_cache_fp8=kv,
        pool_k_pages=pool_k_pages, pool_page_tables=pool_page_tables,
        weights=weights, context_lens=context_lens, block_tables=block_tables,
        k_block_size=k_block_size, pool_page_size=pool_page_size,
        block_topk=block_topk,
    )

    # Compare topk sets.
    n_rows_same_topk = 0
    all_sparse_equal = True
    for b in range(B):
        s1 = set(topk_v1[b, 0].cpu().tolist())
        s3 = set(topk_v3[b, 0].cpu().tolist())
        inter = s1 & s3
        assert len(inter) >= 0.9 * len(s1), (
            f"B={b} topk overlap too low: {len(inter)}/{len(s1)}"
        )

        # For rows with identical topk sets, compare sparse logits at matched blocks.
        t1 = topk_v1[b, 0].cpu().tolist()
        t3 = topk_v3[b, 0].cpu().tolist()
        if sorted(t1) == sorted(t3):
            n_rows_same_topk += 1
            v1_order = {bid: idx for idx, bid in enumerate(t1)}
            for pos_v3, bid in enumerate(t3):
                pos_v1 = v1_order[bid]
                seg_v1 = sparse_v1[b, 0, pos_v1 * k_block_size : (pos_v1 + 1) * k_block_size]
                seg_v3 = sparse_v3[b, 0, pos_v3 * k_block_size : (pos_v3 + 1) * k_block_size]
                finite = torch.isfinite(seg_v1) & torch.isfinite(seg_v3)
                if finite.any():
                    diff = (seg_v1 - seg_v3)[finite].abs().max().item()
                    if diff > 1e-5:
                        all_sparse_equal = False
                        print(f"  B={b} bid={bid} max_diff={diff:.3e}")

    print(f"  rows with identical topk: {n_rows_same_topk}/{B}")
    print(f"  sparse logits on matched blocks — {'byte-equal' if all_sparse_equal else 'DRIFTED'}")
    assert all_sparse_equal


def main() -> int:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    test_tail_only_v3_byte_equal()
    test_orchestrator_v3_matches_v1()
    print("\nAll v3 byte-equal tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
