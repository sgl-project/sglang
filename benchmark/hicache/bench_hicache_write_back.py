from __future__ import annotations

import csv
import platform
from dataclasses import dataclass
from pathlib import Path

import torch
import triton.testing
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer_lf_pf,
    transfer_kv_all_layer_mla_lf_pf,
)

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
    transfer_hicache_all_layer_mla_staged_lf_pf,
    transfer_hicache_all_layer_staged_lf_pf,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "benchmark" / "hicache" / "profiles"
CSV_PATH = OUT_DIR / "writeback_compare.csv"
MD_PATH = ROOT / "benchmark" / "hicache" / "HICACHE_WRITEBACK_COMPARISON.md"

DTYPE = torch.bfloat16
PAGE_SIZE = 64
NUM_LAYERS_LIST = [16, 24, 32, 40, 48, 56, 64, 72, 80]
BATCH_PAGES = [4, 8, 16, 32, 64]
ELEMENT_DIMS = [256, 512, 1024, 2048]
TOTAL_PAGES = max(128, max(BATCH_PAGES) * 2)
WARMUP = 5
REP = 25


@dataclass(frozen=True)
class BenchRow:
    mode: str
    num_layers: int
    element_dim: int
    batch_pages: int
    item_bytes: int
    baseline_gib_s: float
    staged_gib_s: float
    speedup: float


def _make_page_starts(total_pages: int, num_pages: int, page_size: int) -> torch.Tensor:
    order = torch.randperm(total_pages, device="cuda", dtype=torch.int64)
    return order[:num_pages] * page_size


def _make_token_indices(
    page_starts: torch.Tensor, page_size: int, *, device: str
) -> torch.Tensor:
    arange = torch.arange(page_size, device=device, dtype=torch.int64)
    return (page_starts.to(device=device)[:, None] + arange).reshape(-1)


def _to_gib_per_s(moved_bytes: int, ms: float) -> float:
    return moved_bytes / (ms * 1e-3) / (1024**3)


def _bench(fn) -> float:
    ms = triton.testing.do_bench(fn, warmup=WARMUP, rep=REP)
    return float(ms)


def _validate_mha_correctness(
    baseline,
    staged,
    dst_k_baseline: torch.Tensor,
    dst_v_baseline: torch.Tensor,
    dst_k_staged: torch.Tensor,
    dst_v_staged: torch.Tensor,
) -> None:
    dst_k_baseline.zero_()
    dst_v_baseline.zero_()
    dst_k_staged.zero_()
    dst_v_staged.zero_()
    baseline()
    staged()
    torch.cuda.synchronize()
    torch.testing.assert_close(dst_k_staged, dst_k_baseline)
    torch.testing.assert_close(dst_v_staged, dst_v_baseline)


def _validate_mla_correctness(
    baseline,
    staged,
    dst_baseline: torch.Tensor,
    dst_staged: torch.Tensor,
) -> None:
    dst_baseline.zero_()
    dst_staged.zero_()
    baseline()
    staged()
    torch.cuda.synchronize()
    torch.testing.assert_close(dst_staged, dst_baseline)


def _bench_mha(num_layers: int, element_dim: int, batch_pages: int) -> BenchRow:
    item_bytes = element_dim * torch.tensor([], dtype=DTYPE).element_size()
    assert can_use_hicache_jit_kernel(element_size=item_bytes)

    total_tokens = TOTAL_PAGES * PAGE_SIZE
    chunk_tokens = batch_pages * PAGE_SIZE
    src_page_starts = _make_page_starts(TOTAL_PAGES, batch_pages, PAGE_SIZE)
    dst_page_starts = _make_page_starts(TOTAL_PAGES, batch_pages, PAGE_SIZE)
    src_indices = _make_token_indices(src_page_starts, PAGE_SIZE, device="cuda")
    dst_indices = _make_token_indices(dst_page_starts, PAGE_SIZE, device="cuda")

    src_k_layers = [
        torch.randn(total_tokens, element_dim, dtype=DTYPE, device="cuda")
        for _ in range(num_layers)
    ]
    src_v_layers = [
        torch.randn(total_tokens, element_dim, dtype=DTYPE, device="cuda")
        for _ in range(num_layers)
    ]
    src_k_ptrs = torch.tensor(
        [x.data_ptr() for x in src_k_layers], dtype=torch.uint64, device="cuda"
    )
    src_v_ptrs = torch.tensor(
        [x.data_ptr() for x in src_v_layers], dtype=torch.uint64, device="cuda"
    )

    dst_k_baseline = torch.empty(
        total_tokens, num_layers, element_dim, dtype=DTYPE, pin_memory=True
    )
    dst_v_baseline = torch.empty_like(dst_k_baseline, pin_memory=True)
    dst_k_staged = torch.empty_like(dst_k_baseline, pin_memory=True)
    dst_v_staged = torch.empty_like(dst_v_baseline, pin_memory=True)
    staging_k = torch.empty(
        chunk_tokens, num_layers, element_dim, dtype=DTYPE, device="cuda"
    )
    staging_v = torch.empty_like(staging_k)

    baseline = lambda: transfer_kv_all_layer_lf_pf(
        src_k_layers=src_k_ptrs,
        dst_k=dst_k_baseline,
        src_v_layers=src_v_ptrs,
        dst_v=dst_v_baseline,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_bytes,
        dst_layout_dim=item_bytes * num_layers,
        num_layers=num_layers,
    )
    staged = lambda: transfer_hicache_all_layer_staged_lf_pf(
        k_ptr_src=src_k_ptrs,
        v_ptr_src=src_v_ptrs,
        src_indices=src_indices,
        dst_indices=dst_indices,
        staging_k=staging_k,
        staging_v=staging_v,
        dst_k=dst_k_staged,
        dst_v=dst_v_staged,
        page_size=PAGE_SIZE,
    )

    _validate_mha_correctness(
        baseline,
        staged,
        dst_k_baseline,
        dst_v_baseline,
        dst_k_staged,
        dst_v_staged,
    )

    moved_bytes = chunk_tokens * num_layers * item_bytes * 2
    baseline_ms = _bench(baseline)
    staged_ms = _bench(staged)
    baseline_bw = _to_gib_per_s(moved_bytes, baseline_ms)
    staged_bw = _to_gib_per_s(moved_bytes, staged_ms)
    return BenchRow(
        mode="MHA",
        num_layers=num_layers,
        element_dim=element_dim,
        batch_pages=batch_pages,
        item_bytes=item_bytes,
        baseline_gib_s=baseline_bw,
        staged_gib_s=staged_bw,
        speedup=staged_bw / baseline_bw,
    )


def _bench_mla(num_layers: int, element_dim: int, batch_pages: int) -> BenchRow:
    item_bytes = element_dim * torch.tensor([], dtype=DTYPE).element_size()
    assert can_use_hicache_jit_kernel(element_size=item_bytes)

    total_tokens = TOTAL_PAGES * PAGE_SIZE
    chunk_tokens = batch_pages * PAGE_SIZE
    src_page_starts = _make_page_starts(TOTAL_PAGES, batch_pages, PAGE_SIZE)
    dst_page_starts = _make_page_starts(TOTAL_PAGES, batch_pages, PAGE_SIZE)
    src_indices = _make_token_indices(src_page_starts, PAGE_SIZE, device="cuda")
    dst_indices = _make_token_indices(dst_page_starts, PAGE_SIZE, device="cuda")

    src_layers = [
        torch.randn(total_tokens, element_dim, dtype=DTYPE, device="cuda")
        for _ in range(num_layers)
    ]
    src_ptrs = torch.tensor(
        [x.data_ptr() for x in src_layers], dtype=torch.uint64, device="cuda"
    )
    dst_baseline = torch.empty(
        total_tokens, num_layers, element_dim, dtype=DTYPE, pin_memory=True
    )
    dst_staged = torch.empty_like(dst_baseline, pin_memory=True)
    staging = torch.empty(
        chunk_tokens, num_layers, element_dim, dtype=DTYPE, device="cuda"
    )

    baseline = lambda: transfer_kv_all_layer_mla_lf_pf(
        src_layers=src_ptrs,
        dst=dst_baseline,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_bytes,
        dst_layout_dim=item_bytes * num_layers,
        num_layers=num_layers,
    )
    staged = lambda: transfer_hicache_all_layer_mla_staged_lf_pf(
        ptr_src=src_ptrs,
        src_indices=src_indices,
        dst_indices=dst_indices,
        staging=staging,
        dst=dst_staged,
        page_size=PAGE_SIZE,
    )

    _validate_mla_correctness(
        baseline,
        staged,
        dst_baseline,
        dst_staged,
    )

    moved_bytes = chunk_tokens * num_layers * item_bytes
    baseline_ms = _bench(baseline)
    staged_ms = _bench(staged)
    baseline_bw = _to_gib_per_s(moved_bytes, baseline_ms)
    staged_bw = _to_gib_per_s(moved_bytes, staged_ms)
    return BenchRow(
        mode="MLA",
        num_layers=num_layers,
        element_dim=element_dim,
        batch_pages=batch_pages,
        item_bytes=item_bytes,
        baseline_gib_s=baseline_bw,
        staged_gib_s=staged_bw,
        speedup=staged_bw / baseline_bw,
    )


def _write_csv(rows: list[BenchRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "num_layers",
                "element_dim",
                "item_bytes",
                "batch_pages",
                "baseline_gib_s",
                "staged_gib_s",
                "speedup",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.mode,
                    row.num_layers,
                    row.element_dim,
                    row.item_bytes,
                    row.batch_pages,
                    f"{row.baseline_gib_s:.6f}",
                    f"{row.staged_gib_s:.6f}",
                    f"{row.speedup:.4f}",
                ]
            )


def _table(rows: list[BenchRow], mode: str) -> str:
    lines = [
        "| num_layers | element_dim | item_bytes | batch_pages | `sgl_kernel *_lf_pf` GiB/s | `jit *_staged_lf_pf` GiB/s | speedup |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.mode != mode:
            continue
        lines.append(
            f"| {row.num_layers} | {row.element_dim} | {row.item_bytes} | {row.batch_pages} | "
            f"{row.baseline_gib_s:.3f} | {row.staged_gib_s:.3f} | {row.speedup:.2f}x |"
        )
    return "\n".join(lines)


def _summarize(rows: list[BenchRow], mode: str) -> tuple[BenchRow, BenchRow]:
    filtered = [row for row in rows if row.mode == mode]
    best = max(filtered, key=lambda row: row.speedup)
    worst = min(filtered, key=lambda row: row.speedup)
    return best, worst


def _write_report_en(rows: list[BenchRow]) -> None:
    gpu_name = torch.cuda.get_device_name(0)
    mha_best, mha_worst = _summarize(rows, "MHA")
    mla_best, mla_worst = _summarize(rows, "MLA")
    md = f"""# HiCache Writeback Benchmark Comparison

Environment:

- GPU: `{gpu_name}`
- Platform: `{platform.platform()}`
- DType: `{DTYPE}`
- `page_size`: `{PAGE_SIZE}`
- `num_layers` sweep: `{NUM_LAYERS_LIST}`
- `total_pages`: `{TOTAL_PAGES}`
- `batch_pages` sweep: `{BATCH_PAGES}`
- `element_dim` sweep: `{ELEMENT_DIMS}`
- `src_indices` / `dst_indices`: CUDA token indices
- Timing: `triton.testing.do_bench(warmup={WARMUP}, rep={REP})`

Comparison target:

- MHA: `sgl_kernel.transfer_kv_all_layer_lf_pf` vs `sglang.jit_kernel.hicache.transfer_hicache_all_layer_staged_lf_pf`
- MLA: `sgl_kernel.transfer_kv_all_layer_mla_lf_pf` vs `sglang.jit_kernel.hicache.transfer_hicache_all_layer_mla_staged_lf_pf`

Metric:

- Effective writeback bandwidth in `GiB/s`
- `speedup = staged_jit / sgl_kernel_lf_pf`

## MHA

{_table(rows, "MHA")}

## MLA

{_table(rows, "MLA")}

## Takeaways

- The staged JIT path is compared directly against the installed `sgl_kernel` LF->PF kernels on the same host-destination writeback workload.
- Best MHA staged case: `num_layers={mha_best.num_layers}`, `element_dim={mha_best.element_dim}`, `batch_pages={mha_best.batch_pages}`, `speedup={mha_best.speedup:.2f}x`.
- Weakest MHA staged case: `num_layers={mha_worst.num_layers}`, `element_dim={mha_worst.element_dim}`, `batch_pages={mha_worst.batch_pages}`, `speedup={mha_worst.speedup:.2f}x`.
- Best MLA staged case: `num_layers={mla_best.num_layers}`, `element_dim={mla_best.element_dim}`, `batch_pages={mla_best.batch_pages}`, `speedup={mla_best.speedup:.2f}x`.
- Weakest MLA staged case: `num_layers={mla_worst.num_layers}`, `element_dim={mla_worst.element_dim}`, `batch_pages={mla_worst.batch_pages}`, `speedup={mla_worst.speedup:.2f}x`.
- MHA bandwidth uses combined payload bytes for `K + V`.
- MLA bandwidth uses single-buffer payload bytes.
- Raw data is also available in `benchmark/hicache/profiles/writeback_compare.csv`.
"""
    MD_PATH.write_text(md)


def main() -> None:
    torch.manual_seed(0)
    rows: list[BenchRow] = []
    for num_layers in NUM_LAYERS_LIST:
        for element_dim in ELEMENT_DIMS:
            for batch_pages in BATCH_PAGES:
                rows.append(_bench_mha(num_layers, element_dim, batch_pages))
                rows.append(_bench_mla(num_layers, element_dim, batch_pages))
    _write_csv(rows)
    _write_report_en(rows)
    for row in rows:
        print(
            row.mode,
            "num_layers=",
            row.num_layers,
            "element_dim=",
            row.element_dim,
            "batch_pages=",
            row.batch_pages,
            "baseline_gib_s=",
            f"{row.baseline_gib_s:.3f}",
            "staged_gib_s=",
            f"{row.staged_gib_s:.3f}",
            "speedup=",
            f"{row.speedup:.2f}x",
        )
    print(f"wrote {CSV_PATH}")
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    main()
