#!/usr/bin/env python3
"""Probe selective FP8 KV dequantization for BF16 DSA sparse prefill.

This benchmark compares the production candidate against both the current
path and a framework ``torch.unique`` oracle:

* current full-prefix paged dequantization;
* selective dequantization without deduplication;
* allocation-free full-prefix dequantization (to separate allocator effects);
* ``torch.unique`` physical-slot deduplication oracle;
* dense-epoch/prefix-scan physical-slot deduplication candidate;
* isolated and combined selected-row dequantization timings.

The JSON/CSV output retains isolated stages so an apparent end-to-end win can
be rejected if it comes only from avoiding a framework allocation.
"""

import argparse
import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, relative_path: str):
    path = _REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dequant_module = _load_module(
    "dsa_dequant_k_cache",
    "python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py",
)
_case_module = _load_module(
    "dsa_selective_kv_benchmark_utils",
    "benchmark/kernels/attention/dsa_selective_kv_benchmark_utils.py",
)
_selective_runtime_module = _load_module(
    "dsa_selective_kv_runtime",
    "python/sglang/srt/layers/attention/dsa/selective_kv_dequant.py",
)
_selective_kernel_module = _load_module(
    "dsa_selective_kv_kernel",
    "python/sglang/kernels/ops/attention/dsa/selective_kv_dequant.py",
)

dequantize_k_cache_paged = _dequant_module.dequantize_k_cache_paged
SyntheticDSACase = _case_module.SyntheticDSACase
build_benchmark_record = _case_module.build_benchmark_record
make_synthetic_dsa_indices = _case_module.make_synthetic_dsa_indices
SelectiveKVWorkspace = _selective_runtime_module.SelectiveKVWorkspace
prepare_dense_epoch_selection = _selective_runtime_module.prepare_dense_epoch_selection
deduplicate_kv_slots_dense_epoch = (
    _selective_kernel_module.deduplicate_kv_slots_dense_epoch
)

DIM_QUANT = 656
DEFAULT_CASES = (
    SyntheticDSACase(8192, 1, 2048, 1.0, seed=101),
    SyntheticDSACase(32768, 4, 2048, 0.50, seed=102),
    SyntheticDSACase(65536, 16, 2048, 0.125, seed=103),
    SyntheticDSACase(
        65536,
        16,
        2048,
        0.125,
        physical_unique_ratio=0.50,
        seed=104,
    ),
)


def build_quantized_kv_pool(pool_tokens: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(7)
    nope = (
        torch.randn(pool_tokens, 512, generator=generator, device=device) * 0.25
    ).to(torch.float8_e4m3fn)
    scales = (
        torch.rand(pool_tokens, 4, generator=generator, device=device) * 0.05 + 1e-3
    ).to(torch.float32)
    rope = torch.randn(pool_tokens, 64, generator=generator, device=device).to(
        torch.bfloat16
    )

    raw = torch.empty(pool_tokens, DIM_QUANT, dtype=torch.uint8, device=device)
    raw[:, :512] = nope.view(torch.uint8)
    raw[:, 512:528] = scales.view(torch.uint8)
    raw[:, 528:] = rope.view(torch.uint8)
    return raw.view(torch.float8_e4m3fn).view(pool_tokens, 1, DIM_QUANT)


def bench_us(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _dedup_remap(
    page_table: torch.Tensor, flat_topk: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    physical_occurrences = page_table[flat_topk.long()]
    physical_slots, inverse = torch.unique(
        physical_occurrences.reshape(-1), sorted=True, return_inverse=True
    )
    return physical_slots, inverse.to(torch.int32).reshape(flat_topk.shape)


def run_case(
    case: SyntheticDSACase,
    pool: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    git_sha: str,
) -> dict:
    cpu_inputs = make_synthetic_dsa_indices(case)
    page_table = cpu_inputs.page_table_1_flattened.to(pool.device)
    flat_topk = cpu_inputs.flat_topk_indices.to(pool.device)
    physical_occurrences = page_table[flat_topk.long()]
    physical_slots, remapped = _dedup_remap(page_table, flat_topk)
    workspace = SelectiveKVWorkspace(pool.device)
    dense_selection = prepare_dense_epoch_selection(
        page_table,
        flat_topk,
        num_pool_rows=pool.shape[0],
        workspace=workspace,
        deduplicate_fn=deduplicate_kv_slots_dense_epoch,
    )
    full_out = torch.empty(
        (page_table.numel(), 1, 576), dtype=torch.bfloat16, device=pool.device
    )
    no_dedup_out = torch.empty(
        (physical_occurrences.numel(), 1, 576),
        dtype=torch.bfloat16,
        device=pool.device,
    )
    dense_out = workspace.get_bf16(dense_selection.capacity)

    # Validate both selective variants against exactly the rows read from the
    # current full-prefix BF16 workspace.
    full = dequantize_k_cache_paged(pool, page_table)
    expected = full.view(case.prefix_rows, -1)[flat_topk.long()]
    no_dedup = dequantize_k_cache_paged(pool, physical_occurrences.reshape(-1))
    no_dedup = no_dedup.view(*flat_topk.shape, -1)
    selected = dequantize_k_cache_paged(pool, physical_slots).view(
        physical_slots.numel(), -1
    )
    reconstructed = selected[remapped.long()]
    dense_selected = dequantize_k_cache_paged(
        pool,
        dense_selection.selected_physical_slots,
        out=dense_out,
        num_valid_rows=dense_selection.num_unique,
    ).view(dense_selection.capacity, -1)
    dense_reconstructed = dense_selected[dense_selection.remapped_topk.long()]
    torch.testing.assert_close(no_dedup, expected, rtol=0, atol=0)
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
    torch.testing.assert_close(dense_reconstructed, expected, rtol=0, atol=0)
    assert int(dense_selection.num_unique.item()) == physical_slots.numel()
    torch.testing.assert_close(
        dense_selection.selected_physical_slots[: physical_slots.numel()],
        physical_slots.to(torch.int32),
    )
    del (
        full,
        expected,
        no_dedup,
        selected,
        reconstructed,
        dense_selected,
        dense_reconstructed,
    )

    def full_dequant():
        return dequantize_k_cache_paged(pool, page_table)

    def full_dequant_cached():
        return dequantize_k_cache_paged(pool, page_table, out=full_out)

    def selective_no_dedup():
        return dequantize_k_cache_paged(
            pool, physical_occurrences.reshape(-1), out=no_dedup_out
        )

    def oracle_dedup_remap():
        return _dedup_remap(page_table, flat_topk)

    def dense_dedup_remap():
        return prepare_dense_epoch_selection(
            page_table,
            flat_topk,
            num_pool_rows=pool.shape[0],
            workspace=workspace,
            deduplicate_fn=deduplicate_kv_slots_dense_epoch,
        )

    def oracle_selected_dequant():
        return dequantize_k_cache_paged(pool, physical_slots)

    def dense_selected_dequant():
        return dequantize_k_cache_paged(
            pool,
            dense_selection.selected_physical_slots,
            out=dense_out,
            num_valid_rows=dense_selection.num_unique,
        )

    def oracle_selective_total():
        selected_slots, selected_remap = _dedup_remap(page_table, flat_topk)
        return dequantize_k_cache_paged(pool, selected_slots), selected_remap

    def dense_selective_total():
        selection = prepare_dense_epoch_selection(
            page_table,
            flat_topk,
            num_pool_rows=pool.shape[0],
            workspace=workspace,
            deduplicate_fn=deduplicate_kv_slots_dense_epoch,
        )
        return (
            dequantize_k_cache_paged(
                pool,
                selection.selected_physical_slots,
                out=dense_out,
                num_valid_rows=selection.num_unique,
            ),
            selection.remapped_topk,
        )

    timings = {
        "full_dequant_us": bench_us(full_dequant, warmup=warmup, iterations=iterations),
        "full_dequant_cached_us": bench_us(
            full_dequant_cached, warmup=warmup, iterations=iterations
        ),
        "selective_no_dedup_us": bench_us(
            selective_no_dedup, warmup=warmup, iterations=iterations
        ),
        "oracle_dedup_remap_us": bench_us(
            oracle_dedup_remap, warmup=warmup, iterations=iterations
        ),
        "dense_dedup_remap_us": bench_us(
            dense_dedup_remap, warmup=warmup, iterations=iterations
        ),
        "oracle_selected_dequant_us": bench_us(
            oracle_selected_dequant, warmup=warmup, iterations=iterations
        ),
        "dense_selected_dequant_us": bench_us(
            dense_selected_dequant, warmup=warmup, iterations=iterations
        ),
        "oracle_selective_total_us": bench_us(
            oracle_selective_total, warmup=warmup, iterations=iterations
        ),
        "dense_selective_total_us": bench_us(
            dense_selective_total, warmup=warmup, iterations=iterations
        ),
    }
    return build_benchmark_record(
        case=case,
        inputs=cpu_inputs,
        timings_us=timings,
        device_name=torch.cuda.get_device_name(pool.device),
        git_sha=git_sha,
        pool_rows=pool.shape[0],
    )


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
    ).strip()


def _write_artifacts(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dsa_selective_kv_dequant.json"
    csv_path = output_dir / "dsa_selective_kv_dequant.csv"
    json_path.write_text(json.dumps(records, indent=2) + "\n")
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=_case_module.BENCHMARK_RESULT_COLUMNS
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--case-index",
        type=int,
        action="append",
        help="run only the selected zero-based DEFAULT_CASES entry; repeatable",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this benchmark requires one SM90 GPU")
    device = torch.device(args.device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 9:
        raise SystemExit(
            f"this DSA benchmark targets SM90/H20; got compute capability {major}.{minor}"
        )
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("--warmup and --iterations must both be positive")

    cases = DEFAULT_CASES
    if args.case_index is not None:
        invalid = [index for index in args.case_index if not 0 <= index < len(cases)]
        if invalid:
            raise SystemExit(
                f"--case-index must be in [0, {len(cases) - 1}], got {invalid}"
            )
        cases = tuple(DEFAULT_CASES[index] for index in args.case_index)

    torch.cuda.set_device(device)
    max_prefix = max(case.prefix_rows for case in cases)
    pool = build_quantized_kv_pool(max_prefix, device)
    git_sha = _git_sha()
    records = []
    for case in cases:
        record = run_case(
            case,
            pool,
            warmup=args.warmup,
            iterations=args.iterations,
            git_sha=git_sha,
        )
        records.append(record)
        print(json.dumps(record, sort_keys=False))

    if args.output_dir is not None:
        _write_artifacts(records, args.output_dir)


if __name__ == "__main__":
    main()
