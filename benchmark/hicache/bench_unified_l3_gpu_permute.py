"""Who should do the unified-layout permute for head-partitioned L3 chunks?

When the fleet grid splits the kv-head axis, a head-subgroup chunk is not
contiguous in any host pool layout, so the L3 adapter currently stages it
through a pinned buffer with a CPU `copy_` (`gather_unified_chunks` /
`scatter_unified_chunks`).

But the permute does not have to happen on the CPU. The bytes already cross
PCIe once for the L2 write-back / load, and those transfers are DMA-bound, not
compute-bound - so the GPU can gather the head slice on the way out and the
permute rides along nearly free. This benchmark prices both.

    python3 benchmark/hicache/bench_unified_l3_gpu_permute.py

Requires a GPU. CPU-only correctness and the staged-path costs live in
test/registered/unit/mem_cache/test_hicache_unified_layout_perf.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

LAYERS = 80
LOCAL_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 64
PAGES = 64
HEAD_GROUP = 2  # the split that forces the copy; 1 = finest grid
DTYPE = torch.bfloat16


def cuda_time(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def host_time(fn, warmup: int = 1, iters: int = 3) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@dataclass
class Rig:
    tokens: int
    total_bytes: int
    head_ranges: list
    chunk_elems: int
    gpu: torch.Tensor
    host_pool: torch.Tensor
    staging: torch.Tensor
    gpu_tmp: torch.Tensor


def build() -> Rig:
    tokens = PAGES * PAGE_SIZE
    total = 2 * LAYERS * tokens * LOCAL_KV_HEADS * HEAD_DIM * DTYPE.itemsize
    return Rig(
        tokens=tokens,
        total_bytes=total,
        head_ranges=[(i, i + HEAD_GROUP) for i in range(0, LOCAL_KV_HEADS, HEAD_GROUP)],
        chunk_elems=LAYERS * tokens * HEAD_GROUP * HEAD_DIM,
        # device pool: per-layer (token, head, dim), K and V
        gpu=torch.zeros(
            (2, LAYERS, tokens, LOCAL_KV_HEADS, HEAD_DIM), dtype=DTYPE, device="cuda"
        ),
        # host pool: page_first_direct
        host_pool=torch.zeros(
            (2, PAGES, LAYERS, PAGE_SIZE, LOCAL_KV_HEADS, HEAD_DIM), dtype=DTYPE
        ),
        # the adapter's pinned, store-registered staging buffer
        staging=torch.empty(total, dtype=torch.uint8, pin_memory=True),
        gpu_tmp=torch.empty(
            (LAYERS, tokens, HEAD_GROUP, HEAD_DIM), dtype=DTYPE, device="cuda"
        ),
    )


def cpu_gather(r: Rig) -> None:
    """Today's save path: host pool -> pinned staging, on the backup thread."""
    cursor = 0
    nbytes = LAYERS * PAGE_SIZE * HEAD_GROUP * HEAD_DIM * DTYPE.itemsize
    for page in range(PAGES):
        view = r.host_pool[:, page]
        for h0, h1 in r.head_ranges:
            for kv in range(2):
                src = view[kv, :, :, h0:h1]
                r.staging[cursor : cursor + nbytes].view(DTYPE).view(src.shape).copy_(
                    src
                )
                cursor += nbytes


def cpu_scatter(r: Rig) -> None:
    """Today's load path: pinned staging -> host pool, on the prefetch thread."""
    cursor = 0
    nbytes = LAYERS * PAGE_SIZE * HEAD_GROUP * HEAD_DIM * DTYPE.itemsize
    for page in range(PAGES):
        view = r.host_pool[:, page]
        for h0, h1 in r.head_ranges:
            for kv in range(2):
                dst = view[kv, :, :, h0:h1]
                dst.copy_(
                    r.staging[cursor : cursor + nbytes].view(DTYPE).view(dst.shape)
                )
                cursor += nbytes


def gpu_gather(r: Rig) -> None:
    """Proposed save path: device pool -> pinned staging, GPU does the permute."""
    cursor = 0
    for h0, h1 in r.head_ranges:
        for kv in range(2):
            src = r.gpu[kv, :, :, h0:h1, :]
            dst = (
                r.staging[
                    cursor * DTYPE.itemsize : (cursor + r.chunk_elems) * DTYPE.itemsize
                ]
                .view(DTYPE)
                .view(src.shape)
            )
            dst.copy_(src)
            cursor += r.chunk_elems


def gpu_scatter(r: Rig) -> None:
    """Proposed load path: pinned staging -> device pool, GPU does the permute."""
    cursor = 0
    for h0, h1 in r.head_ranges:
        for kv in range(2):
            src = (
                r.staging[
                    cursor * DTYPE.itemsize : (cursor + r.chunk_elems) * DTYPE.itemsize
                ]
                .view(DTYPE)
                .view(LAYERS, r.tokens, HEAD_GROUP, HEAD_DIM)
            )
            r.gpu[kv, :, :, h0:h1, :].copy_(src)
            cursor += r.chunk_elems


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a GPU")
    r = build()
    gb = r.total_bytes / 1e9
    flat_h = torch.empty(r.total_bytes // DTYPE.itemsize, dtype=DTYPE, pin_memory=True)
    flat_d = torch.empty_like(flat_h, device="cuda")

    d2h_ceil = cuda_time(lambda: flat_h.copy_(flat_d))
    h2d_ceil = cuda_time(lambda: flat_d.copy_(flat_h))
    cpu_g = host_time(lambda: cpu_gather(r))
    cpu_s = host_time(lambda: cpu_scatter(r))
    gpu_g = cuda_time(lambda: gpu_gather(r))
    gpu_s = cuda_time(lambda: gpu_scatter(r))

    print(
        f"{gb * 1e3:.0f} MB   layers={LAYERS} local_kv_heads={LOCAL_KV_HEADS} "
        f"head_dim={HEAD_DIM} page={PAGE_SIZE} pages={PAGES} "
        f"head_group={HEAD_GROUP} ({torch.cuda.get_device_name()})"
    )
    print(f"\n{'stage':<44}{'ms':>9}{'GB/s':>9}")
    print(f"{'-' * 62}")
    print(
        f"{'contiguous D2H (no-split ceiling)':<44}{d2h_ceil * 1e3:9.2f}{gb / d2h_ceil:9.1f}"
    )
    print(
        f"{'contiguous H2D (no-split ceiling)':<44}{h2d_ceil * 1e3:9.2f}{gb / h2d_ceil:9.1f}"
    )
    print(
        f"{'CPU gather   host pool -> staging  (today)':<44}{cpu_g * 1e3:9.2f}{gb / cpu_g:9.1f}"
    )
    print(
        f"{'CPU scatter  staging -> host pool  (today)':<44}{cpu_s * 1e3:9.2f}{gb / cpu_s:9.1f}"
    )
    print(
        f"{'GPU gather   device pool -> staging':<44}{gpu_g * 1e3:9.2f}{gb / gpu_g:9.1f}"
    )
    print(
        f"{'GPU scatter  staging -> device pool':<44}{gpu_s * 1e3:9.2f}{gb / gpu_s:9.1f}"
    )

    print(f"\n{'end-to-end, head-partitioned chunks':<44}{'ms':>9}{'speedup':>9}")
    print(f"{'-' * 62}")
    save_today = d2h_ceil + cpu_g
    save_prop = d2h_ceil + gpu_g
    load_today = cpu_s + h2d_ceil
    load_prop = gpu_s
    print(
        f"{'save  today   D2H to L2 + CPU gather':<44}{save_today * 1e3:9.2f}{'1.00x':>9}"
    )
    print(
        f"{'save  GPU     D2H to L2 + GPU gather':<44}{save_prop * 1e3:9.2f}"
        f"{save_today / save_prop:8.2f}x"
    )
    print(
        f"{'save  GPU, L3-only (no L2 copy)':<44}{gpu_g * 1e3:9.2f}"
        f"{save_today / gpu_g:8.2f}x"
    )
    print(f"{'load  today   CPU scatter + H2D':<44}{load_today * 1e3:9.2f}{'1.00x':>9}")
    print(
        f"{'load  GPU     staging -> device pool':<44}{load_prop * 1e3:9.2f}"
        f"{load_today / load_prop:8.2f}x"
    )
    print(
        "\nThe permute costs the GPU path only the gap against the contiguous\n"
        "ceiling; it removes the CPU copy entirely. Caveat: a GPU-driven L3\n"
        "write needs the device copy to still exist, i.e. the L3 admission\n"
        "decision fused into write-back - see ANALYSIS_unified_l3_zero_copy.md."
    )


if __name__ == "__main__":
    main()
