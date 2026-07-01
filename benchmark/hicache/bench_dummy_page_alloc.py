"""Microbenchmark for the HiCache non-zero-copy prefetch dummy-page allocation.

HF3FS / EIC non-zero-copy prefetch prepares one dummy read buffer per page and per
batch (all pages of a batch are alive at once), e.g.

    [host_pool.get_dummy_flat_data_page() for _ in range(page_num)]

(storage/hf3fs/storage_hf3fs.py, storage/eic/eic_storage.py). The buffer is fully
overwritten by the storage read before use, so any zero-fill is wasted.

This bench reproduces that per-batch pattern and compares:

    zeros : torch.zeros(shape, pin_memory=True)   # current — alloc + memset
    empty : torch.empty(shape, pin_memory=True)   # proposed — alloc only
    reuse : one preallocated (page_num, *shape) pinned buffer, one view per slot

It reports the STEADY-STATE per-batch time (after warmup), which is what production
sees. The gap between `empty` and `reuse` reflects how much PyTorch's pinned host
caching allocator already reuses freed blocks (i.e. how little a hand-written
reusable bounce buffer would add on top of the simple zeros->empty change).

Notes / caveats:
  - Measures MHA-shaped bf16 pages only, not every changed pool type.
  - Measures buffer-prep cost only, not end-to-end storage prefetch latency.

Run:  python benchmark/hicache/bench_dummy_page_alloc.py
"""

from __future__ import annotations

import time

import torch

DTYPE = torch.bfloat16

# (label, layer_num, page_size, kv_heads, head_dim, page_num)
CONFIGS = [
    ("Qwen2.5-7B  ps=64", 28, 64, 4, 128, 64),
    ("Qwen2.5-7B  ps=16", 28, 16, 4, 128, 64),
]
WARMUP = 3  # unmeasured batches to reach pinned-allocator steady state
MEASURE = 10  # measured batches, averaged


def page_shape(layer_num, page_size, kv_heads, head_dim):
    # MHA host-pool dummy page: (2, layer_num, page_size, head_num, head_dim)
    return (2, layer_num, page_size, kv_heads, head_dim)


def _alloc_batch(shape, page_num, mode, prealloc):
    if mode == "reuse":
        return [prealloc[i] for i in range(page_num)]
    if mode == "zeros":
        return [
            torch.zeros(shape, dtype=DTYPE, pin_memory=True) for _ in range(page_num)
        ]
    return [torch.empty(shape, dtype=DTYPE, pin_memory=True) for _ in range(page_num)]


def _touch(pages):
    # Emulate the storage read landing in the buffer so nothing is optimized away
    # and the buffers stay alive for the whole batch (like the real list-comp).
    for p in pages:
        p.view(-1)[0] = 1


def bench_steady_ms(shape, page_num, mode, prealloc=None):
    for _ in range(WARMUP):
        pages = _alloc_batch(shape, page_num, mode, prealloc)
        _touch(pages)
        del pages
    times = []
    for _ in range(MEASURE):
        t0 = time.perf_counter()
        pages = _alloc_batch(shape, page_num, mode, prealloc)
        _touch(pages)
        times.append((time.perf_counter() - t0) * 1e3)  # ms
        del pages
    return sum(times) / len(times)


def main():
    print(
        f"torch {torch.__version__}  (pinned host caching allocator active by default)\n"
    )
    header = (
        f"{'config':<20} {'page MB':>8} {'batch MB':>9} {'mode':>6} "
        f"{'steady ms':>11} {'steady/page us':>15}"
    )
    print(header)
    print("-" * len(header))
    for label, layer_num, page_size, kv_heads, head_dim, page_num in CONFIGS:
        shape = page_shape(layer_num, page_size, kv_heads, head_dim)
        numel = 1
        for s in shape:
            numel *= s
        page_mb = numel * 2 / 1e6
        for mode in ("zeros", "empty", "reuse"):
            prealloc = (
                torch.empty((page_num,) + shape, dtype=DTYPE, pin_memory=True)
                if mode == "reuse"
                else None
            )
            steady = bench_steady_ms(shape, page_num, mode, prealloc)
            print(
                f"{label:<20} {page_mb:>8.2f} {page_mb*page_num:>9.0f} {mode:>6} "
                f"{steady:>11.3f} {steady/page_num*1e3:>15.1f}"
            )
            del prealloc
        print()


if __name__ == "__main__":
    main()
