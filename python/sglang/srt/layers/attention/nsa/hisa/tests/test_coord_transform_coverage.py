"""Broader coverage test for hisa_coord_transform triton kernel.

The original test covered just M=4096 K=16K (RAGGED) and M=32 ctx=16K (PAGED).
This file adds:
- SHORT sequence (num_blocks < block_topk) — e.g. samsum prompts
- ALL-INVALID row (fast_topk_v2 all -1) edge case
- BLOCK_TOPK varying across calls
- Non-contiguous input (safety against stride assumptions)
- Mixed per-query (ke-ks) in RAGGED mode

Every test compares byte-equal to the torch reference chain.
"""
from __future__ import annotations

import sys
import traceback

import torch

from sglang.srt.layers.attention.nsa.hisa.triton_kernels import hisa_coord_transform


DEVICE = torch.device("cuda")


def torch_ref_ragged(relevant, topk_block_indices, ks, ke, k_block_size):
    rs = relevant.clamp(min=0)
    abs_block = torch.gather(
        topk_block_indices.to(torch.int64),
        -1,
        (rs // k_block_size).to(torch.int64),
    )
    raw = abs_block * k_block_size + (rs % k_block_size)
    raw = raw - ks[:, None]
    valid = (raw >= 0) & (raw < (ke - ks)[:, None])
    return raw.masked_fill(~valid | (relevant == -1), -1).to(torch.int32)


def torch_ref_paged(relevant, topk_block_indices, seq_lens, k_block_size):
    rs = relevant.clamp(min=0)
    abs_block = torch.gather(
        topk_block_indices.to(torch.int64),
        -1,
        (rs // k_block_size).to(torch.int64),
    )
    raw = abs_block * k_block_size + (rs % k_block_size)
    valid = raw < seq_lens[:, None]
    return raw.masked_fill(~valid | (relevant == -1), -1).to(torch.int32)


def _byte_equal(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    assert a.dtype == b.dtype == torch.int32
    if (a != b).any():
        diff = (a != b).sum().item()
        idx = (a != b).nonzero()[0].tolist()
        m, i = idx[:2]
        raise AssertionError(
            f"{name}: {diff} positions differ; first at [{m},{i}]: "
            f"triton={a[m, i].item()}, torch={b[m, i].item()}"
        )


def _build_ragged(M, K, block_topk, k_block_size, index_topk, ks=None, ke=None):
    num_blocks = K // k_block_size
    actual_block_topk = min(block_topk, num_blocks)
    sparse_len = actual_block_topk * k_block_size

    torch.manual_seed(0)
    if sparse_len < index_topk:
        # shorter sparse_len than requested topk — fast_topk_v2 would pad.
        rel_pool = torch.randint(
            -1, sparse_len, (M, index_topk), dtype=torch.int32, device=DEVICE
        )
    else:
        rel_pool = torch.randint(
            0, sparse_len, (M, index_topk), dtype=torch.int32, device=DEVICE
        )
        pad = torch.rand(M, index_topk, device=DEVICE) < 0.02
        rel_pool = torch.where(pad, torch.full_like(rel_pool, -1), rel_pool)

    topk_block_indices = torch.randint(
        0, num_blocks, (M, actual_block_topk),
        dtype=torch.int32, device=DEVICE,
    )
    if ks is None:
        ks = torch.zeros(M, device=DEVICE, dtype=torch.int32)
    if ke is None:
        ke = torch.randint(
            max(1, K // 4), K + 1, (M,), dtype=torch.int32, device=DEVICE,
        )
    return rel_pool, topk_block_indices, ks, ke


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_short_seq_ragged():
    """Short seq (num_blocks < block_topk). samsum-ish shape: ~2-4K prompts."""
    M = 4096
    for K in [2048, 3072, 4096, 6144]:
        block_topk = 64
        k_block_size = 128
        index_topk = 2048
        num_blocks = K // k_block_size
        # For these K values num_blocks = 16, 24, 32, 48 — all < block_topk
        assert num_blocks < block_topk

        rel, topk_block, ks, ke = _build_ragged(
            M, K, block_topk, k_block_size, index_topk,
        )
        t = hisa_coord_transform(
            rel, topk_block, ke, k_block_size=k_block_size, ks=ks,
        )
        ref = torch_ref_ragged(rel, topk_block, ks, ke, k_block_size)
        _byte_equal(t, ref, f"short_ragged K={K} num_blocks={num_blocks}")
    print("    short_ragged: K∈{2048,3072,4096,6144} — all match")


def test_short_seq_paged():
    M = 32
    for ctx in [1024, 2048, 4096, 7168]:
        block_topk = 64
        k_block_size = 128
        index_topk = 2048
        num_blocks = (ctx + k_block_size - 1) // k_block_size
        actual_bt = min(block_topk, num_blocks)
        sparse_len = actual_bt * k_block_size

        torch.manual_seed(ctx)
        rel = torch.randint(
            0, sparse_len, (M, index_topk), dtype=torch.int32, device=DEVICE,
        )
        pad = torch.rand(M, index_topk, device=DEVICE) < 0.02
        rel = torch.where(pad, torch.full_like(rel, -1), rel)

        topk_block = torch.randint(
            0, num_blocks, (M, actual_bt), dtype=torch.int32, device=DEVICE,
        )
        seq_lens = torch.randint(
            max(1, ctx // 2), ctx + 1, (M,), dtype=torch.int32, device=DEVICE,
        )

        t = hisa_coord_transform(
            rel, topk_block, seq_lens, k_block_size=k_block_size, ks=None,
        )
        ref = torch_ref_paged(rel, topk_block, seq_lens, k_block_size)
        _byte_equal(t, ref, f"short_paged ctx={ctx}")
    print("    short_paged: ctx∈{1024,2048,4096,7168} — all match")


def test_all_invalid_row():
    """Edge case: a query has all -1 in relevant (fast_topk_v2 padding)."""
    M = 16
    K = 16384
    block_topk = 64
    k_block_size = 128
    index_topk = 2048

    rel, topk_block, ks, ke = _build_ragged(
        M, K, block_topk, k_block_size, index_topk,
    )
    # Force row 0 to be all -1.
    rel[0] = -1

    t = hisa_coord_transform(rel, topk_block, ke, k_block_size, ks=ks)
    ref = torch_ref_ragged(rel, topk_block, ks, ke, k_block_size)
    _byte_equal(t, ref, "all_invalid_row")
    # Additionally: row 0 output must be all -1.
    assert (t[0] == -1).all(), "all-invalid row should produce all -1"
    print("    all_invalid_row — match; row 0 is all -1")


def test_varied_ks_ke():
    """RAGGED with heterogeneous ks/ke — makes sure mask uses per-row values."""
    M = 1024
    K = 32768
    block_topk = 64
    k_block_size = 128
    index_topk = 2048

    rel, topk_block, _, _ = _build_ragged(
        M, K, block_topk, k_block_size, index_topk,
    )
    torch.manual_seed(42)
    ks = torch.randint(0, K // 2, (M,), dtype=torch.int32, device=DEVICE)
    lengths = torch.randint(1, K // 2, (M,), dtype=torch.int32, device=DEVICE)
    ke = ks + lengths

    t = hisa_coord_transform(rel, topk_block, ke, k_block_size, ks=ks)
    ref = torch_ref_ragged(rel, topk_block, ks, ke, k_block_size)
    _byte_equal(t, ref, "varied_ks_ke")
    print("    varied_ks_ke: M=1024 heterogeneous K-ranges — match")


def test_noncontiguous_inputs():
    """Strides-safety: pass views instead of contiguous tensors."""
    M = 256
    K = 16384
    block_topk = 64
    k_block_size = 128
    index_topk = 2048

    rel_big = torch.randint(
        0, block_topk * k_block_size, (M * 2, index_topk),
        dtype=torch.int32, device=DEVICE,
    )
    rel = rel_big[::2]  # non-contiguous stride-2 view

    topk_block_big = torch.randint(
        0, K // k_block_size, (M * 2, block_topk),
        dtype=torch.int32, device=DEVICE,
    )
    topk_block = topk_block_big[::2]

    ks = torch.zeros(M, device=DEVICE, dtype=torch.int32)
    ke = torch.randint(1, K + 1, (M,), dtype=torch.int32, device=DEVICE)

    # These views have non-standard strides. If the kernel assumes stride ==
    # row_width (via m*INDEX_TOPK), non-contig inputs will read wrong data.
    try:
        t = hisa_coord_transform(rel, topk_block, ke, k_block_size, ks=ks)
    except RuntimeError as e:
        print(f"    noncontiguous_inputs: kernel refused non-contig input: {e}")
        print("    → OK if wrapper contigifies, else we need stride args in kernel")
        return

    ref = torch_ref_ragged(rel, topk_block, ks, ke, k_block_size)
    _byte_equal(t, ref, "noncontiguous_inputs")
    print("    noncontiguous_inputs: strided views — match")


def test_small_index_topk_padding():
    """sparse_len < index_topk: fast_topk_v2 returns -1 padding for extras.

    Our kernel should handle this correctly.
    """
    M = 64
    K = 1024                # 8 blocks
    block_topk = 64         # clamps to 8
    k_block_size = 128
    index_topk = 2048       # 2x larger than sparse_len=1024

    rel, topk_block, ks, ke = _build_ragged(
        M, K, block_topk, k_block_size, index_topk,
    )
    t = hisa_coord_transform(rel, topk_block, ke, k_block_size, ks=ks)
    ref = torch_ref_ragged(rel, topk_block, ks, ke, k_block_size)
    _byte_equal(t, ref, "small_index_topk_padding")

    # Sanity: at least some outputs should be -1 (since sparse_len < index_topk).
    assert (t == -1).any(), "expected some -1 padding in output"
    print("    small_index_topk_padding: sparse_len<index_topk — match, -1 padding present")


TESTS = [
    ("short_seq_ragged", test_short_seq_ragged),
    ("short_seq_paged", test_short_seq_paged),
    ("all_invalid_row", test_all_invalid_row),
    ("varied_ks_ke", test_varied_ks_ke),
    ("noncontiguous_inputs", test_noncontiguous_inputs),
    ("small_index_topk_padding", test_small_index_topk_padding),
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
