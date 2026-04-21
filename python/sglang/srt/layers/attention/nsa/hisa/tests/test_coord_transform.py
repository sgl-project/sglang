"""Correctness + perf test for the fused hisa coord_transform triton kernel.

Checks that ``hisa_coord_transform`` matches the torch reference chain (the
code it replaces in :mod:`hierarchy_indexer`) and reports the perf / memory
savings.

Reference chain (RAGGED prefill):
    rs = relevant.clamp(min=0)
    abs_block = gather(topk_block_indices, rs // k_block_size)
    raw = abs_block * k_block_size + (rs % k_block_size) - ks
    valid = (raw >= 0) & (raw < ke - ks)
    final = raw.masked_fill(~valid | (relevant == -1), -1).to(int32)

PAGED decode: no ks subtract; ``valid = raw < seq_len``.
"""
from __future__ import annotations

import statistics
import sys
import traceback

import torch

from sglang.srt.layers.attention.nsa.hisa.triton_kernel import hisa_coord_transform


DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# Torch reference (what the triton kernel replaces)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def bench_fn(fn, warmups=5, iters=20):
    torch.cuda.synchronize()
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def peak_memory_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    _ = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    return peak / 1024 / 1024


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def build_inputs_ragged(M, K, block_topk, k_block_size, index_topk):
    """Synthesize inputs mimicking what the HisaIndexer would pass."""
    num_blocks = K // k_block_size
    actual_block_topk = min(block_topk, num_blocks)
    sparse_len = actual_block_topk * k_block_size

    relevant = torch.randint(
        0, sparse_len, (M, index_topk), dtype=torch.int32, device=DEVICE,
    )
    # Mix in some -1 (simulating fast_topk_v2 padding).
    pad_mask = torch.rand(M, index_topk, device=DEVICE) < 0.02
    relevant = torch.where(pad_mask, torch.full_like(relevant, -1), relevant)

    # Random distinct block indices per query (well, random ints).
    topk_block_indices = torch.randint(
        0, num_blocks, (M, block_topk), dtype=torch.int32, device=DEVICE,
    )
    # Causal-ish ks/ke for prefill.
    ks = torch.zeros(M, device=DEVICE, dtype=torch.int32)
    ke = torch.randint(
        max(1, K // 4), K + 1, (M,), dtype=torch.int32, device=DEVICE,
    )
    return relevant, topk_block_indices, ks, ke, actual_block_topk, sparse_len


def test_ragged_correctness():
    """RAGGED prefill: triton output must byte-equal torch ref."""
    torch.manual_seed(0)
    K = 16384
    M = 4096
    block_topk = 64
    k_block_size = 128
    index_topk = 2048

    relevant, topk_block_indices, ks, ke, actual_bt, sparse_len = build_inputs_ragged(
        M, K, block_topk, k_block_size, index_topk,
    )

    triton_out = hisa_coord_transform(
        relevant, topk_block_indices, ke,
        k_block_size=k_block_size, ks=ks,
    )
    torch_out = torch_ref_ragged(relevant, topk_block_indices, ks, ke, k_block_size)

    assert triton_out.shape == torch_out.shape
    assert triton_out.dtype == torch_out.dtype == torch.int32

    match = (triton_out == torch_out).all().item()
    if not match:
        diff = (triton_out != torch_out).sum().item()
        first_mismatch = (triton_out != torch_out).nonzero()[0].tolist()
        m, i = first_mismatch
        print(f"    MISMATCH: {diff} positions differ; first at [{m},{i}]")
        print(f"    triton={triton_out[m, i].item()}, torch={torch_out[m, i].item()}")
        print(f"    relevant[{m},{i}]={relevant[m, i].item()}, "
              f"topk_block[{m}][:4]={topk_block_indices[m, :4].tolist()}")
    assert match, "triton coord_transform disagrees with torch reference (RAGGED)"
    print(f"    RAGGED correctness: M={M} K={K} actual_block_topk={actual_bt} — match")


def test_paged_correctness():
    """PAGED decode: ks=None variant."""
    torch.manual_seed(1)
    M = 32
    ctx = 16384
    block_topk = 64
    k_block_size = 128
    index_topk = 2048

    num_blocks = ctx // k_block_size
    sparse_len = min(block_topk, num_blocks) * k_block_size

    relevant = torch.randint(
        0, sparse_len, (M, index_topk), dtype=torch.int32, device=DEVICE,
    )
    pad_mask = torch.rand(M, index_topk, device=DEVICE) < 0.02
    relevant = torch.where(pad_mask, torch.full_like(relevant, -1), relevant)

    topk_block_indices = torch.randint(
        0, num_blocks, (M, block_topk), dtype=torch.int32, device=DEVICE,
    )
    # Vary seq_lens per request.
    seq_lens = torch.randint(ctx // 2, ctx + 1, (M,), dtype=torch.int32, device=DEVICE)

    triton_out = hisa_coord_transform(
        relevant, topk_block_indices, seq_lens,
        k_block_size=k_block_size, ks=None,
    )
    torch_out = torch_ref_paged(relevant, topk_block_indices, seq_lens, k_block_size)

    assert (triton_out == torch_out).all().item(), "PAGED mismatch"
    print(f"    PAGED correctness: M={M} ctx={ctx} — match")


def test_perf_and_memory():
    """Perf: triton kernel vs torch chain at a production-realistic shape."""
    torch.manual_seed(2)
    M = 8192            # typical chunked-prefill Q
    K = 65536           # full context
    block_topk = 64
    k_block_size = 128
    index_topk = 2048

    relevant, topk_block_indices, ks, ke, _, _ = build_inputs_ragged(
        M, K, block_topk, k_block_size, index_topk,
    )

    # Warm up both.
    _ = hisa_coord_transform(relevant, topk_block_indices, ke, k_block_size, ks=ks)
    _ = torch_ref_ragged(relevant, topk_block_indices, ks, ke, k_block_size)

    t_triton = bench_fn(
        lambda: hisa_coord_transform(
            relevant, topk_block_indices, ke, k_block_size, ks=ks
        )
    )
    t_torch = bench_fn(
        lambda: torch_ref_ragged(relevant, topk_block_indices, ks, ke, k_block_size)
    )

    peak_triton = peak_memory_mb(
        lambda: hisa_coord_transform(
            relevant, topk_block_indices, ke, k_block_size, ks=ks
        )
    )
    peak_torch = peak_memory_mb(
        lambda: torch_ref_ragged(relevant, topk_block_indices, ks, ke, k_block_size)
    )

    print(f"    Perf (M={M}, K={K}, index_topk={index_topk}):")
    print(f"      torch chain:   {t_torch:7.3f} ms   peak alloc {peak_torch:7.1f} MB")
    print(f"      triton fused:  {t_triton:7.3f} ms   peak alloc {peak_triton:7.1f} MB")
    print(f"      speedup:       {t_torch / t_triton:7.2f}x")
    print(f"      mem saving:    {peak_torch - peak_triton:7.1f} MB "
          f"({100 * (1 - peak_triton / peak_torch):4.1f}%)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("ragged_correctness", test_ragged_correctness),
    ("paged_correctness", test_paged_correctness),
    ("perf_and_memory", test_perf_and_memory),
]


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
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
