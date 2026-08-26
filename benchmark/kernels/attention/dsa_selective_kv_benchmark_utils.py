from dataclasses import dataclass
from typing import Mapping

import torch

KV_BYTES_PER_ROW = 656
BF16_BYTES_PER_ROW = 576 * 2
DENSE_SCAN_BLOCK_SIZE = 256

TIMING_COLUMNS = (
    "full_dequant_us",
    "full_dequant_cached_us",
    "selective_no_dedup_us",
    "oracle_dedup_remap_us",
    "dense_dedup_remap_us",
    "oracle_selected_dequant_us",
    "dense_selected_dequant_us",
    "oracle_selective_total_us",
    "dense_selective_total_us",
)

BENCHMARK_RESULT_COLUMNS = (
    "device_name",
    "git_sha",
    "prefix_rows",
    "query_tokens",
    "topk",
    "requested_logical_unique_ratio",
    "requested_physical_unique_ratio",
    "valid_topk_entries",
    "logical_unique_rows",
    "physical_unique_rows",
    "logical_unique_ratio",
    "physical_unique_ratio",
    "pool_rows",
    "selection_capacity_rows",
    "full_kv_traffic_bytes",
    "active_selective_kv_traffic_bytes",
    "full_bf16_workspace_bytes",
    "fixed_selective_bf16_workspace_bytes",
    "dense_metadata_workspace_bytes",
    "estimated_dense_metadata_traffic_bytes",
    *TIMING_COLUMNS,
    "dense_speedup_vs_current",
    "dense_speedup_vs_cached_full",
)


@dataclass(frozen=True)
class SyntheticDSACase:
    prefix_rows: int
    query_tokens: int
    topk: int
    unique_ratio: float
    physical_unique_ratio: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("prefix_rows", "query_tokens", "topk"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("unique_ratio", "physical_unique_ratio"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")


@dataclass(frozen=True)
class SyntheticDSAInputs:
    page_table_1_flattened: torch.Tensor
    flat_topk_indices: torch.Tensor
    logical_unique_rows: int
    physical_unique_rows: int


def make_synthetic_dsa_indices(case: SyntheticDSACase) -> SyntheticDSAInputs:
    """Create deterministic DSA index geometry without requiring CUDA.

    ``unique_ratio`` controls duplicate logical top-k occurrences.  The
    independent ``physical_unique_ratio`` aliases selected logical positions in
    the page table, modeling radix-prefix sharing between requests.
    """
    occurrences = case.query_tokens * case.topk
    logical_unique_rows = min(
        case.prefix_rows,
        occurrences,
        max(1, round(occurrences * case.unique_ratio)),
    )
    physical_unique_rows = min(
        logical_unique_rows,
        max(1, round(logical_unique_rows * case.physical_unique_ratio)),
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)
    selected_logical = torch.randperm(
        case.prefix_rows, generator=generator, dtype=torch.int64
    )[:logical_unique_rows]

    # Repeat every selected row before shuffling, which guarantees that the
    # generated tensor has exactly logical_unique_rows distinct values.
    occurrence_indices = torch.arange(occurrences, dtype=torch.int64)
    topk_flat = selected_logical[occurrence_indices % logical_unique_rows]
    topk_flat = topk_flat[
        torch.randperm(occurrences, generator=generator, dtype=torch.int64)
    ]

    # Unselected logical positions remain one-to-one.  Only selected positions
    # are deliberately aliased, so the reported physical ratio describes the
    # rows attention can actually read.
    page_table = torch.arange(case.prefix_rows, dtype=torch.int32)
    physical_ids = torch.arange(physical_unique_rows, dtype=torch.int32)
    alias_assignment = physical_ids[
        torch.arange(logical_unique_rows, dtype=torch.int64) % physical_unique_rows
    ]
    page_table[selected_logical] = alias_assignment

    return SyntheticDSAInputs(
        page_table_1_flattened=page_table,
        flat_topk_indices=topk_flat.to(torch.int32).reshape(
            case.query_tokens, case.topk
        ),
        logical_unique_rows=logical_unique_rows,
        physical_unique_rows=physical_unique_rows,
    )


def build_benchmark_record(
    *,
    case: SyntheticDSACase,
    inputs: SyntheticDSAInputs,
    timings_us: Mapping[str, float],
    device_name: str,
    git_sha: str,
    pool_rows: int,
) -> dict:
    """Return a stable row suitable for both JSON and CSV serialization."""
    missing = [column for column in TIMING_COLUMNS if column not in timings_us]
    if missing:
        raise ValueError(f"missing timing columns: {missing}")
    for column in TIMING_COLUMNS:
        value = timings_us[column]
        if value < 0:
            raise ValueError(f"{column} must be non-negative, got {value}")
    dense_selective_total_us = float(timings_us["dense_selective_total_us"])
    if dense_selective_total_us == 0:
        raise ValueError("dense_selective_total_us must be positive")
    if not isinstance(pool_rows, int) or isinstance(pool_rows, bool) or pool_rows <= 0:
        raise ValueError(f"pool_rows must be a positive integer, got {pool_rows!r}")

    valid_entries = case.query_tokens * case.topk
    selection_capacity = min(case.prefix_rows, valid_entries)
    num_scan_blocks = (pool_rows + DENSE_SCAN_BLOCK_SIZE - 1) // DENSE_SCAN_BLOCK_SIZE
    record = {
        "device_name": device_name,
        "git_sha": git_sha,
        "prefix_rows": case.prefix_rows,
        "query_tokens": case.query_tokens,
        "topk": case.topk,
        "requested_logical_unique_ratio": case.unique_ratio,
        "requested_physical_unique_ratio": case.physical_unique_ratio,
        "valid_topk_entries": valid_entries,
        "logical_unique_rows": inputs.logical_unique_rows,
        "physical_unique_rows": inputs.physical_unique_rows,
        "logical_unique_ratio": inputs.logical_unique_rows / valid_entries,
        "physical_unique_ratio": inputs.physical_unique_rows / valid_entries,
        "pool_rows": pool_rows,
        "selection_capacity_rows": selection_capacity,
        "full_kv_traffic_bytes": case.prefix_rows
        * (KV_BYTES_PER_ROW + BF16_BYTES_PER_ROW),
        "active_selective_kv_traffic_bytes": inputs.physical_unique_rows
        * (KV_BYTES_PER_ROW + BF16_BYTES_PER_ROW),
        "full_bf16_workspace_bytes": case.prefix_rows * BF16_BYTES_PER_ROW,
        "fixed_selective_bf16_workspace_bytes": selection_capacity * BF16_BYTES_PER_ROW,
        # int64 slot_epoch + int32 slot_to_compact + selected physical IDs +
        # remapped occurrences + block offsets + int64 epoch + int32 count.
        "dense_metadata_workspace_bytes": pool_rows * (8 + 4)
        + selection_capacity * 4
        + valid_entries * 4
        + num_scan_blocks * 4
        + 8
        + 4,
        # Two full int64 epoch scans plus a conservative charge for block
        # prefix-scan traffic and compact-row publication.
        "estimated_dense_metadata_traffic_bytes": pool_rows * 2 * 8
        + num_scan_blocks * 12
        + inputs.physical_unique_rows * 12,
    }
    record.update({column: float(timings_us[column]) for column in TIMING_COLUMNS})
    record["dense_speedup_vs_current"] = (
        float(timings_us["full_dequant_us"]) / dense_selective_total_us
    )
    record["dense_speedup_vs_cached_full"] = (
        float(timings_us["full_dequant_cached_us"]) / dense_selective_total_us
    )
    assert tuple(record) == BENCHMARK_RESULT_COLUMNS
    return record
