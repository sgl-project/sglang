#!/usr/bin/env python3
"""Compare Kimi-K3 CUDA-graph traces by rank, layer, stage, and token row."""

from __future__ import annotations

import argparse
import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path


_ROW_RE = re.compile(
    r"K3_GRAPH_ROW_TRACE rank=(?P<rank>\d+) replay_id=(?P<replay>\d+) "
    r"mode=(?P<mode>\S+) "
    r"capture_bs=(?P<bs>\d+) capture_num_tokens=(?P<num_tokens>\d+) "
    r"layer_id=(?P<layer>-?\d+) stage=(?P<stage>\S+) "
    r"row_kind=(?P<row_kind>\S+) .*?"
    r"row_indices=(?P<rows>\[[^\]]*\]) "
    r"row_abs=(?P<abs>\[[^\]]*\]) row_sum=(?P<sum>\[[^\]]*\]) "
    r"row_sq=(?P<sq>\[[^\]]*\]) row_min=(?P<min>\[[^\]]*\]) "
    r"row_max=(?P<max>\[[^\]]*\])"
)

_EXACT_RE = re.compile(
    r"K3_GRAPH_EXACT_ROW_TRACE rank=(?P<rank>\d+) replay_id=(?P<replay>\d+) "
    r"mode=(?P<mode>\S+) "
    r"capture_bs=(?P<bs>\d+) capture_num_tokens=(?P<num_tokens>\d+) "
    r"layer_id=(?P<layer>-?\d+) stage=(?P<stage>\S+) "
    r"row_kind=(?P<row_kind>\S+) .*?"
    r"row_indices=(?P<rows>\[[^\]]*\]) values=(?P<values>.*)$"
)

_MAP_RE = re.compile(
    r"K3_GRAPH_VERIFY_ROW_MAP rank=(?P<rank>\d+) replay_id=(?P<replay>\d+) "
    r"mode=(?P<mode>\S+) capture_bs=(?P<bs>\d+) "
    r"capture_num_tokens=(?P<num_tokens>\d+) .*? rows=(?P<rows>\[.*\])$"
)

_ACCEPT_RE = re.compile(
    r"K3_VERIFY_ACCEPT_ROW_TRACE rank=(?P<rank>\d+) "
    r"replay_id=(?P<replay>\d+) req_row=(?P<req_row>\d+) "
    r"req_pool_index=\d+ verify_len=(?P<verify_len>\d+) "
    r"verify_input_ids=(?P<verify_ids>\[[^\]]*\]) "
    r"correct_len=(?P<correct_len>\d+) commit_len=(?P<commit_len>\d+) "
    r"bonus_id=(?P<bonus_id>-?\d+) candidates=(?P<candidates>\[.*?\]) "
    r"accepted=(?P<accepted>\[.*\])$"
)

_STAGE_ORDER = {
    name: order
    for order, name in enumerate(
        (
            "layer_input_hidden",
            "layer_input_residual",
            "attention_input_hidden",
            "mamba_indices",
            "ssm_state_read",
            "conv_state_read",
            "kda_q",
            "kda_k",
            "kda_v",
            "kda_a",
            "kda_b",
            "kda_output",
            "attention_output",
            "attn_out",
            "attention_residual",
            "attention_residual_prefix",
            "moe_input",
            "moe_router_logits",
            "moe_topk_weights",
            "moe_topk_ids",
            "moe_routed_input",
            "moe_expert_output",
            "moe_routed_latent",
            "moe_routed_up",
            "moe_shared_output",
            "moe_final",
            "moe_output",
            "mlp_out",
            "layer_output_hidden",
            "layer_output_residual",
        )
    )
}


@dataclass(frozen=True)
class RowKey:
    rank: int
    replay_id: int
    mode: str
    capture_bs: int
    capture_num_tokens: int
    layer_id: int
    stage: str
    row_kind: str
    row: int


@dataclass(frozen=True)
class RowValue:
    kind: str
    values: tuple


@dataclass(frozen=True)
class MetadataKey:
    rank: int
    replay_id: int
    mode: str
    capture_bs: int
    capture_num_tokens: int
    compact_row: int


@dataclass(frozen=True)
class AcceptKey:
    rank: int
    replay_id: int
    req_row: int


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _parse(path: Path) -> dict[RowKey, RowValue]:
    rows: dict[RowKey, RowValue] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _ROW_RE.search(line)
            if match is not None:
                indices = ast.literal_eval(match["rows"])
                columns = [
                    ast.literal_eval(match[name])
                    for name in ("abs", "sum", "sq", "min", "max")
                ]
                if any(len(column) != len(indices) for column in columns):
                    raise ValueError(
                        f"malformed K3 row trace in {path}: {line.rstrip()}"
                    )
                for offset, row in enumerate(indices):
                    key = RowKey(
                        rank=int(match["rank"]),
                        replay_id=int(match["replay"]),
                        mode=match["mode"],
                        capture_bs=int(match["bs"]),
                        capture_num_tokens=int(match["num_tokens"]),
                        layer_id=int(match["layer"]),
                        stage=match["stage"],
                        row_kind=match["row_kind"],
                        row=int(row),
                    )
                    rows[key] = RowValue(
                        "stats",
                        tuple(float(column[offset]) for column in columns),
                    )
                continue

            match = _EXACT_RE.search(line)
            if match is None:
                continue
            indices = ast.literal_eval(match["rows"])
            values = ast.literal_eval(match["values"])
            if len(values) != len(indices):
                raise ValueError(
                    f"malformed K3 exact trace in {path}: {line.rstrip()}"
                )
            for offset, row in enumerate(indices):
                key = RowKey(
                    rank=int(match["rank"]),
                    replay_id=int(match["replay"]),
                    mode=match["mode"],
                    capture_bs=int(match["bs"]),
                    capture_num_tokens=int(match["num_tokens"]),
                    layer_id=int(match["layer"]),
                    stage=match["stage"],
                    row_kind=match["row_kind"],
                    row=int(row),
                )
                rows[key] = RowValue("exact", (_freeze(values[offset]),))
    return rows


def _parse_metadata(path: Path):
    metadata: dict[MetadataKey, tuple] = {}
    accepts: dict[AcceptKey, tuple] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _MAP_RE.search(line)
            if match is not None:
                for row in ast.literal_eval(match["rows"]):
                    key = MetadataKey(
                        rank=int(match["rank"]),
                        replay_id=int(match["replay"]),
                        mode=match["mode"],
                        capture_bs=int(match["bs"]),
                        capture_num_tokens=int(match["num_tokens"]),
                        compact_row=int(row["compact_row"]),
                    )
                    # req_pool_index is intentionally excluded because the
                    # allocator can choose a different slot between runs.
                    metadata[key] = (
                        int(row["dense_row"]),
                        int(row["req_row"]),
                        int(row["verify_step"]),
                        int(row["input_id"]),
                        int(row["position"]),
                        int(row["seq_len"]),
                    )
                continue

            match = _ACCEPT_RE.search(line)
            if match is None:
                continue
            key = AcceptKey(
                rank=int(match["rank"]),
                replay_id=int(match["replay"]),
                req_row=int(match["req_row"]),
            )
            accepts[key] = (
                int(match["verify_len"]),
                _freeze(ast.literal_eval(match["verify_ids"])),
                int(match["correct_len"]),
                int(match["commit_len"]),
                int(match["bonus_id"]),
                _freeze(ast.literal_eval(match["candidates"])),
                _freeze(ast.literal_eval(match["accepted"])),
            )
    return metadata, accepts


def _exact_differences(baseline: dict, current: dict):
    common = sorted(baseline.keys() & current.keys(), key=repr)
    differences = [
        (key, baseline[key], current[key])
        for key in common
        if baseline[key] != current[key]
    ]
    return (
        differences,
        sorted(baseline.keys() - current.keys(), key=repr),
        sorted(current.keys() - baseline.keys(), key=repr),
    )


def _sort_key(key: RowKey):
    return (
        key.replay_id,
        key.layer_id,
        _STAGE_ORDER.get(key.stage, len(_STAGE_ORDER)),
        key.stage,
        key.rank,
        key.row_kind,
        key.row,
    )


def _relative_error(current: float, baseline: float) -> float:
    if math.isnan(current) or math.isnan(baseline):
        return math.inf
    return abs(current - baseline) / max(abs(baseline), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    baseline = _parse(args.baseline)
    current = _parse(args.current)
    baseline_metadata, baseline_accepts = _parse_metadata(args.baseline)
    current_metadata, current_accepts = _parse_metadata(args.current)
    metadata_diff, metadata_missing_current, metadata_missing_baseline = (
        _exact_differences(baseline_metadata, current_metadata)
    )
    accept_diff, accept_missing_current, accept_missing_baseline = (
        _exact_differences(baseline_accepts, current_accepts)
    )
    common = sorted(
        baseline.keys() & current.keys(),
        key=_sort_key,
    )
    missing_current = sorted(baseline.keys() - current.keys(), key=_sort_key)
    missing_baseline = sorted(current.keys() - baseline.keys(), key=_sort_key)

    differences = []
    for key in common:
        base_record = baseline[key]
        current_record = current[key]
        if base_record.kind != current_record.kind:
            differences.append(
                (key, math.inf, (), base_record.values, current_record.values)
            )
            continue
        base_values = base_record.values
        current_values = current_record.values
        if base_record.kind == "exact":
            if base_values != current_values:
                differences.append(
                    (key, math.inf, (), base_values, current_values)
                )
            continue
        errors = tuple(
            _relative_error(cur, base)
            for cur, base in zip(current_values, base_values)
        )
        # abs_sum and square_sum are the stable magnitude checks. signed_sum,
        # min, and max remain in the output as diagnostics but do not alone
        # trigger a mismatch around zero.
        magnitude_error = max(errors[0], errors[2])
        if magnitude_error > args.threshold:
            differences.append((key, magnitude_error, errors, base_values, current_values))

    print(
        f"baseline_rows={len(baseline)} current_rows={len(current)} "
        f"common_rows={len(common)} mismatched_rows={len(differences)} "
        f"missing_current={len(missing_current)} missing_baseline={len(missing_baseline)}"
    )
    print(
        f"metadata_mismatches={len(metadata_diff)} "
        f"metadata_missing_current={len(metadata_missing_current)} "
        f"metadata_missing_baseline={len(metadata_missing_baseline)} "
        f"accept_mismatches={len(accept_diff)} "
        f"accept_missing_current={len(accept_missing_current)} "
        f"accept_missing_baseline={len(accept_missing_baseline)}"
    )
    if metadata_diff:
        key, base_value, current_value = metadata_diff[0]
        print(
            f"FIRST_METADATA_MISMATCH replay_id={key.replay_id} "
            f"rank={key.rank} compact_row={key.compact_row} "
            f"baseline={base_value} current={current_value}"
        )
    if accept_diff:
        key, base_value, current_value = accept_diff[0]
        print(
            f"FIRST_ACCEPT_MISMATCH replay_id={key.replay_id} "
            f"rank={key.rank} req_row={key.req_row} "
            f"baseline={base_value} current={current_value}"
        )
    if differences:
        first = differences[0]
        print(
            "FIRST_MISMATCH "
            f"replay_id={first[0].replay_id} layer={first[0].layer_id} "
            f"stage={first[0].stage} "
            f"rank={first[0].rank} row_kind={first[0].row_kind} "
            f"row={first[0].row} "
            f"magnitude_rel={first[1]:.6g}"
        )
    for key, magnitude_error, errors, base_values, current_values in differences[
        : args.limit
    ]:
        print(
            f"DIFF replay_id={key.replay_id} layer={key.layer_id} "
            f"stage={key.stage} rank={key.rank} "
            f"row_kind={key.row_kind} row={key.row} "
            f"magnitude_rel={magnitude_error:.6g} "
            f"all_rel={errors} baseline={base_values} current={current_values}"
        )
    for key in missing_current[: args.limit]:
        print(f"MISSING_CURRENT {key}")
    for key in missing_baseline[: args.limit]:
        print(f"MISSING_BASELINE {key}")
    has_difference = any(
        (
            differences,
            missing_current,
            missing_baseline,
            metadata_diff,
            metadata_missing_current,
            metadata_missing_baseline,
            accept_diff,
            accept_missing_current,
            accept_missing_baseline,
        )
    )
    return 1 if has_difference else 0


if __name__ == "__main__":
    raise SystemExit(main())
