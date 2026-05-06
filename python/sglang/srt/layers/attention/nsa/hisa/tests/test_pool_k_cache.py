"""Unit tests for hisa/pool_k_cache.py (v3 — paged pool K).

Tests:
  1. HisaPoolPageAllocator alloc/free
  2. HisaReqToPoolPagePool write/get
  3. alloc_pool_pages_for_extend (single chunk + chunked prefill)
  4. alloc_pool_pages_for_decode (crossing vs not crossing page boundary)
  5. free_req_pool_pages
  6. update_pool_for_completed_blocks correctness (v3 — writes to paged layout)
"""
from __future__ import annotations

import sys
import traceback

import torch

from sglang.srt.layers.attention.nsa.hisa.tilelang_legacy import (
    fp8_native_paged_mean_pooling_interface,
)
from sglang.srt.layers.attention.nsa.hisa.pool_k_cache import (
    HisaNSATokenToKVPool,
    HisaPoolPageAllocator,
    HisaReqToPoolPagePool,
)


DEVICE = torch.device("cuda")
POOL_PAGE_SIZE = 64


def make_mini_pool(
    num_pages: int = 16,                # main KV cache pages
    page_size: int = 64,
    index_head_dim: int = 128,
    layer_num: int = 2,
    k_block_size: int = 128,
    max_req: int = 8,
    pool_page_size: int = POOL_PAGE_SIZE,
    num_pool_pages_global: int = 16,
    max_pool_pages_per_req: int = 8,
):
    pool = HisaNSATokenToKVPool.__new__(HisaNSATokenToKVPool)
    pool.size = num_pages * page_size
    pool.page_size = page_size
    pool.index_head_dim = index_head_dim
    pool.layer_num = layer_num
    pool.start_layer = 0
    pool.device = DEVICE
    pool.k_block_size = k_block_size
    pool.pool_page_size = pool_page_size
    pool.num_pool_pages_global = num_pool_pages_global
    pool.max_pool_pages_per_req = max_pool_pages_per_req

    pool.index_k_with_scale_buffer = [
        torch.zeros(
            (num_pages, page_size * (index_head_dim + 4)),
            dtype=torch.uint8, device=DEVICE,
        )
        for _ in range(layer_num)
    ]
    D = index_head_dim
    page_bytes = pool_page_size * (D + 4)
    pool.pool_k_pages = [
        torch.zeros(
            (num_pool_pages_global, page_bytes),
            dtype=torch.uint8, device=DEVICE,
        )
        for _ in range(layer_num)
    ]
    pool.pool_page_allocator = HisaPoolPageAllocator(num_pool_pages_global, DEVICE)
    pool.req_to_pool_page = HisaReqToPoolPagePool(max_req, max_pool_pages_per_req, DEVICE)
    pool._scratch_prev_lens_i32 = torch.zeros(max_req, dtype=torch.int32, device=DEVICE)
    pool._scratch_new_lens_i32 = torch.zeros(max_req, dtype=torch.int32, device=DEVICE)
    pool._pool_watermark_i32 = torch.zeros(
        (layer_num, max_req), dtype=torch.int32, device=DEVICE,
    )
    return pool


def populate_main_buffer(pool, layer_id, req_to_token, req_idx, start_pos, k_bf16):
    N, D = k_bf16.shape
    assert D == pool.index_head_dim
    positions = req_to_token[req_idx, start_pos:start_pos + N]

    abs_max = k_bf16.abs().max(dim=-1).values.float().clamp(min=1e-6)
    scale_f32 = abs_max / 448.0
    scaled = k_bf16.float() / scale_f32.unsqueeze(-1)
    k_fp8 = scaled.to(torch.float8_e4m3fn)

    buf = pool.index_k_with_scale_buffer[layer_id]
    for n in range(N):
        pos = int(positions[n].item())
        page = pos // pool.page_size
        slot = pos % pool.page_size
        buf[page, slot * D : (slot + 1) * D] = k_fp8[n].view(torch.uint8)
        scale_offset = pool.page_size * D + slot * 4
        buf[page, scale_offset : scale_offset + 4] = (
            scale_f32[n:n + 1].view(torch.uint8).squeeze(0)
        )
    return k_fp8, scale_f32


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_allocator_basic():
    alloc = HisaPoolPageAllocator(num_pages=16, device=DEVICE)
    assert alloc.available_size() == 16
    a = alloc.alloc(5)
    assert a.tolist() == [0, 1, 2, 3, 4]
    b = alloc.alloc(3)
    assert b.tolist() == [5, 6, 7]
    alloc.free(a)
    assert alloc.available_size() == 13
    c = alloc.alloc(2)
    assert c.tolist() == [8, 9]
    assert alloc.alloc(100) is None
    print("    allocator_basic OK")


def test_req_to_pool_page_mapping():
    rm = HisaReqToPoolPagePool(size=4, max_pool_pages_per_req=8, device=DEVICE)
    ids = torch.tensor([10, 20, 30], dtype=torch.int32, device=DEVICE)
    rm.write(2, 0, ids)
    assert rm.get_ids(2).tolist() == [10, 20, 30]
    assert rm.num_pages(2) == 3
    rm.write(2, 3, torch.tensor([40], dtype=torch.int32, device=DEVICE))
    assert rm.get_ids(2).tolist() == [10, 20, 30, 40]
    rm.free(2)
    assert rm.num_pages(2) == 0
    print("    req_to_pool_page_mapping OK")


def test_alloc_for_extend_single_chunk():
    # K=128, pool_page_size=64 → tokens_per_page = 8192. Requests up to
    # 8192 tokens fit in 1 pool page.
    pool = make_mini_pool(num_pages=32, k_block_size=128)
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[0], prefix_lens=[0], seq_lens=[4096],
    )
    ids = pool.req_to_pool_page.get_ids(0)
    assert ids.numel() == 1, f"expected 1 page, got {ids.numel()}"

    # Longer request that spans 2 pool pages.
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[1], prefix_lens=[0], seq_lens=[10000],
    )
    ids2 = pool.req_to_pool_page.get_ids(1)
    assert ids2.numel() == 2, f"expected 2 pages, got {ids2.numel()}"
    print(f"    alloc_for_extend: 1 + 2 pages OK")


def test_alloc_for_decode_crossing_page():
    # K=128, pool_page_size=64 → 1 pool page covers 8192 real tokens.
    # Test: decode crosses from 8192 → 8193. Should alloc new page.
    pool = make_mini_pool(num_pages=256, k_block_size=128)
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[0], prefix_lens=[0], seq_lens=[8192],
    )
    assert pool.req_to_pool_page.num_pages(0) == 1

    # Decode 8192 → 8193 crosses pool-page boundary.
    pool.alloc_pool_pages_for_decode(
        req_pool_indices=[0], seq_lens_after_decode=[8193],
    )
    assert pool.req_to_pool_page.num_pages(0) == 2

    # Decode 8193 → 8194 (no crossing).
    pool.alloc_pool_pages_for_decode(
        req_pool_indices=[0], seq_lens_after_decode=[8194],
    )
    assert pool.req_to_pool_page.num_pages(0) == 2
    print("    alloc_for_decode_crossing_page OK: 1 → 2 pages")


def test_alloc_for_extend_prefix_cache_hit():
    """Regression test for the cache-hit stale-page bug (A4a).

    Scenario: req 0 allocates 3 pool pages, finishes, frees. Then enough
    other requests run to circle the FIFO free list back to req 0's
    pages — those page IDs become LIVE under different ownership. Now
    req 0's slot is reused by a NEW request that arrives with a
    prefix-cache hit (prefix_lens > 0). The new alloc must allocate
    FRESH pool pages covering the prefix range — not skip the prefix
    alloc and leave ``req_to_pool_page[0, 0..prev_pages]`` pointing at
    page IDs that are NOW LIVE under another request, which the
    "always pool from 0" mitigation would then corrupt.
    """
    # Pool sized so that consecutive non-cache-hit allocs drain the
    # fresh pages, forcing the FIFO allocator to recycle req 0's freed
    # page IDs to a different request.
    pool = make_mini_pool(num_pages=64, k_block_size=128, max_req=8,
                          num_pool_pages_global=9)
    # req 0: alloc 3 → [0,1,2], free.
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[0], prefix_lens=[0], seq_lens=[20000],
    )
    ids_r0 = pool.req_to_pool_page.get_ids(0).clone()
    assert ids_r0.numel() == 3 and ids_r0.tolist() == [0, 1, 2]
    pool.free_req_pool_pages(0)

    # Drain the 6 fresh pages so allocator's free list cycles back
    # to ids_r0 = [0, 1, 2] for the next allocation.
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[1], prefix_lens=[0], seq_lens=[20000],
    )
    ids_r1 = pool.req_to_pool_page.get_ids(1).clone()  # [3, 4, 5]
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[2], prefix_lens=[0], seq_lens=[20000],
    )
    ids_r2 = pool.req_to_pool_page.get_ids(2).clone()  # [6, 7, 8]

    # req 3 grabs page 0 (formerly req 0's slot 0). It is now LIVE.
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[3], prefix_lens=[0], seq_lens=[100],
    )
    ids_r3 = pool.req_to_pool_page.get_ids(3).clone()  # [0]
    assert 0 in ids_r3.tolist(), (
        f"setup failure: page 0 not recycled to req 3, got {ids_r3.tolist()}"
    )

    # Slot 0 in req_to_pool_page still holds the stale [0, 1, 2] from
    # the freed req 0 — and page 0 is now ALIVE under req 3.
    stale_row = pool.req_to_pool_page.req_to_pool_page[0, :3].tolist()
    assert stale_row == [0, 1, 2]

    # Now: req 0 slot reused with a CACHE HIT. Buggy code would compute
    # prev_pages=ceil(4096/8192)=1, alloc only 1 page for the new range,
    # and leave req_to_pool_page[0, 0] = 0 — aliasing req 3's live page.
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[0], prefix_lens=[4096], seq_lens=[10000],
    )
    ids_r0_v2 = pool.req_to_pool_page.get_ids(0).clone()
    # Fix expectation: 2 pool pages covering the full [0, 10000) range.
    assert ids_r0_v2.numel() == 2, f"expected 2 pages, got {ids_r0_v2.numel()}"

    # Critical: slot 0 of req 0's table must NOT alias any LIVE page of
    # any other request. With the bug, slot 0 would still be 0 (== r3's
    # live page).
    live_pages = set(ids_r1.tolist() + ids_r2.tolist() + ids_r3.tolist())
    r0_v2_pages = set(ids_r0_v2.tolist())
    overlap = r0_v2_pages & live_pages
    assert not overlap, (
        f"FRESH alloc must not collide with live pages: "
        f"r0_v2={ids_r0_v2.tolist()} live={sorted(live_pages)} overlap={overlap}"
    )
    # And the actual stored slot 0 must hold one of r0_v2's IDs (fresh).
    slot0 = int(pool.req_to_pool_page.req_to_pool_page[0, 0].item())
    assert slot0 in r0_v2_pages, (
        f"slot 0 = {slot0} not in fresh alloc {sorted(r0_v2_pages)}"
    )
    assert slot0 not in live_pages, (
        f"slot 0 = {slot0} aliases live page (would corrupt that req)"
    )
    print(
        f"    cache-hit fresh-alloc OK: r0_v2={ids_r0_v2.tolist()}, "
        f"slot 0 = {slot0} disjoint from live={sorted(live_pages)}"
    )


def test_pool_watermark_advance_and_reset():
    """A4b: watermark advances on extend, resets to 0 on free."""
    pool = make_mini_pool(num_pages=64, k_block_size=128, max_req=4,
                          layer_num=2, num_pool_pages_global=8)
    K = pool.k_block_size  # 128
    # Initial state: all watermarks zero.
    assert pool._pool_watermark_i32.eq(0).all()

    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[0, 1], prefix_lens=[0, 0], seq_lens=[10000, 5000],
    )
    # Simulate extend store hook for layer 0.
    new_seq = torch.tensor([10000, 5000], dtype=torch.int32, device=DEVICE)
    req_idx = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    pool.advance_pool_watermark(layer_id=0, req_pool_indices=req_idx,
                                new_seq_lens=new_seq)
    # Watermark = floor(seq, K). 10000 // 128 * 128 = 9984; 5000 // 128 * 128 = 4992.
    wm = pool._pool_watermark_i32
    assert int(wm[0, 0]) == 9984, f"got {int(wm[0, 0])}"
    assert int(wm[0, 1]) == 4992, f"got {int(wm[0, 1])}"
    # Layer 1 untouched.
    assert int(wm[1, 0]) == 0
    assert int(wm[1, 1]) == 0

    # Read-back via load_extend_prev_seq_lens_from_watermark.
    out_prev = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    pool.load_extend_prev_seq_lens_from_watermark(
        layer_id=0, req_pool_indices=req_idx, out_prev=out_prev,
    )
    assert out_prev.tolist() == [9984, 4992]

    # Free req 0 → watermark for req 0 resets across all layers.
    pool.free_req_pool_pages(0)
    assert int(wm[0, 0]) == 0
    assert int(wm[1, 0]) == 0
    # req 1 untouched.
    assert int(wm[0, 1]) == 4992
    print("    pool_watermark_advance_and_reset OK")


def test_free_req_pool_pages():
    pool = make_mini_pool(num_pages=32, k_block_size=128)
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[0], prefix_lens=[0], seq_lens=[20000],
    )
    expected_pages = 3  # ceildiv(20000, 8192) = 3
    assert pool.req_to_pool_page.num_pages(0) == expected_pages
    before = pool.pool_page_allocator.available_size()
    pool.free_req_pool_pages(0)
    after = pool.pool_page_allocator.available_size()
    assert pool.req_to_pool_page.num_pages(0) == 0
    assert after - before == expected_pages
    print(f"    free_req_pool_pages OK: {expected_pages} pages returned")


def test_update_pool_for_completed_blocks_v3():
    """Populate main buffer for req 0 with known K, complete 2 pool blocks,
    call update_pool_for_completed_blocks, verify pool_k_pages content
    byte-equals the paged v1 reference.
    """
    torch.manual_seed(0)
    k_block_size = 128
    pool_page_size = POOL_PAGE_SIZE
    num_pages = 8
    max_ctx = 256
    D = 128
    pool = make_mini_pool(
        num_pages=num_pages, page_size=64, index_head_dim=D,
        k_block_size=k_block_size, layer_num=1,
        max_req=4, num_pool_pages_global=16, max_pool_pages_per_req=2,
    )
    layer_id = 0

    # req_to_token: req 0 uses pages 0..3 (logical positions 0..255).
    req_to_token = torch.zeros((4, max_ctx), dtype=torch.int32, device=DEVICE)
    req_to_token[0, :max_ctx] = torch.arange(0, max_ctx, dtype=torch.int32, device=DEVICE)

    # Allocate pool pages for req 0 (256 tokens → 2 pool blocks, all fit in 1 page).
    pool.alloc_pool_pages_for_extend(
        req_pool_indices=[0], prefix_lens=[0], seq_lens=[256],
    )
    assert pool.req_to_pool_page.num_pages(0) == 1
    page_ids = pool.req_to_pool_page.get_ids(0).tolist()
    print(f"    (setup) pool_page_ids = {page_ids}")

    # Write 256 tokens into main buffer.
    k_bf16 = torch.randn(256, D, device=DEVICE, dtype=torch.bfloat16)
    populate_main_buffer(pool, layer_id, req_to_token, 0, 0, k_bf16)

    # Reference via paged v1 mean-pool.
    kv_for_paged = pool.index_k_with_scale_buffer[layer_id].view(
        num_pages, pool.page_size, 1, D + 4,
    )
    block_tables = torch.arange(0, 4, dtype=torch.int32, device=DEVICE).unsqueeze(0).contiguous()
    ctx = torch.tensor([256], dtype=torch.int32, device=DEVICE)
    ref_fp8, ref_scale, _ = fp8_native_paged_mean_pooling_interface(
        max_num_pooling_blocks=2, kv_cache=kv_for_paged,
        context_lens=ctx, block_tables=block_tables,
        k_block_size=k_block_size,
    )
    ref_fp8 = ref_fp8[0]      # [2, D] fp8
    ref_scale = ref_scale[0]  # [2] f32

    # Call v3 update.
    pool.update_pool_for_completed_blocks(
        layer_id=layer_id,
        req_to_token=req_to_token,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        prev_seq_lens=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        new_seq_lens=torch.tensor([256], dtype=torch.int32, device=DEVICE),
        max_pool_per_req_grid=2,
    )

    # Verify pool_k_pages[phys_page, slot_bytes] byte-equal vs ref.
    buf = pool.pool_k_pages[layer_id]
    fp8_base = 0
    scale_base = pool_page_size * D
    for pblk in range(2):
        logical_page = pblk // pool_page_size
        slot = pblk % pool_page_size
        phys = int(pool.req_to_pool_page.req_to_pool_page[0, logical_page].item())
        got_fp8 = buf[phys, fp8_base + slot * D : fp8_base + (slot + 1) * D]
        got_scale_bytes = buf[phys, scale_base + slot * 4 : scale_base + (slot + 1) * 4]
        got_scale = got_scale_bytes.view(torch.float32).item()

        assert torch.equal(got_fp8, ref_fp8[pblk].view(torch.uint8)), (
            f"pool_block {pblk} fp8 bytes mismatch at phys={phys} slot={slot}"
        )
        assert abs(got_scale - ref_scale[pblk].item()) / max(abs(ref_scale[pblk].item()), 1e-9) < 1e-5, (
            f"pool_block {pblk} scale mismatch"
        )
    print(f"    update_pool_for_completed_blocks: 2 blocks byte-equal vs paged v1")


TESTS = [
    ("allocator_basic", test_allocator_basic),
    ("req_to_pool_page_mapping", test_req_to_pool_page_mapping),
    ("alloc_for_extend_single_chunk", test_alloc_for_extend_single_chunk),
    ("alloc_for_extend_prefix_cache_hit", test_alloc_for_extend_prefix_cache_hit),
    ("pool_watermark_advance_and_reset", test_pool_watermark_advance_and_reset),
    ("alloc_for_decode_crossing_page", test_alloc_for_decode_crossing_page),
    ("free_req_pool_pages", test_free_req_pool_pages),
    ("update_pool_for_completed_blocks_v3", test_update_pool_for_completed_blocks_v3),
]


def main() -> int:
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    n_pass = n_fail = 0
    for name, fn in TESTS:
        try:
            print(f"[RUN ] {name}")
            fn()
            print(f"[PASS] {name}")
            n_pass += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed (of {len(TESTS)})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
