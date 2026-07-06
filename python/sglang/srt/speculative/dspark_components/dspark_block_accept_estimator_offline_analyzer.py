from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import msgspec
import typer


class BlockEstimate(msgspec.Struct):
    lo: float
    hi: float
    category: str
    a_first: Optional[float]
    cap_trim_positive: bool
    true_block_accept: int


class LoadedRecords(msgspec.Struct):
    blocks: List[Dict[str, Any]]
    gathers: Dict[Tuple[str, int], List[List[Any]]]
    eos_terminated: set[Tuple[str, int]]


class PerRequestEstimate(msgspec.Struct):
    rid: str
    num_blocks: int
    mean_lo: float
    mean_hi: float
    mean_mid: float


def load_records(jsonl_path: Path) -> LoadedRecords:
    blocks: List[Dict[str, Any]] = []
    gathers: Dict[Tuple[str, int], List[List[Any]]] = defaultdict(list)
    eos_terminated: set[Tuple[str, int]] = set()
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            for block_fct in rec.get("eos_end", []):
                eos_terminated.add((rec["rid"], block_fct))
            if "cl" not in rec:
                continue
            blocks.append(rec)
            for entry in rec.get("pg", []):
                src_fct, offset, p_lp, draft_token, realized_token = entry
                gathers[(rec["rid"], src_fct)].append(
                    [offset, p_lp, draft_token, realized_token]
                )
    return LoadedRecords(blocks=blocks, gathers=gathers, eos_terminated=eos_terminated)


def evaluate_block(
    rec: Dict[str, Any],
    gathers: Dict[Tuple[str, int], List[List[Any]]],
    gamma: int,
    eos_terminated: set[Tuple[str, int]],
) -> BlockEstimate:
    cl = rec["cl"]
    window = rec["w"]
    cap_trim = rec.get("ct", 0)
    true_block_accept = cl + cap_trim + 1
    if "q_lp" not in rec:
        value = cl + 1.0
        return BlockEstimate(
            lo=value,
            hi=value,
            category="exact_in_window",
            a_first=None,
            cap_trim_positive=cap_trim >= 1,
            true_block_accept=true_block_accept,
        )

    q_lps = rec["q_lp"]
    entries = {e[0]: e for e in gathers.get((rec["rid"], rec["fct"]), [])}
    base = window + 1.0
    prod = 1.0
    lo_extra = 0.0
    category = "censored_resolved_exact"
    tail = 0.0
    a_first: Optional[float] = None

    for offset in range(window + 1, gamma + 1):
        entry = entries.get(offset)
        if entry is None:
            if (rec["rid"], rec["fct"]) in eos_terminated:
                category = "censored_eos"
                tail = 0.0
            else:
                category = "censored_at_end"
                tail = prod * (gamma - offset + 1)
            break
        _, p_lp, draft_token, realized_token = entry
        a = min(1.0, math.exp(p_lp - q_lps[offset - window - 1]))
        if a_first is None:
            a_first = a
        prod *= a
        lo_extra += prod
        if draft_token != realized_token:
            if offset < gamma:
                category = "censored_diverged"
                tail = prod * (gamma - offset)
            break

    return BlockEstimate(
        lo=base + lo_extra,
        hi=base + lo_extra + tail,
        category=category,
        a_first=a_first,
        cap_trim_positive=cap_trim >= 1,
        true_block_accept=true_block_accept,
    )


def per_request_estimates(
    blocks: List[Dict[str, Any]], results: List[BlockEstimate]
) -> List[PerRequestEstimate]:
    sums: Dict[str, List[float]] = {}
    for rec, result in zip(blocks, results):
        entry = sums.setdefault(rec["rid"], [0.0, 0.0, 0.0])
        entry[0] += result.lo
        entry[1] += result.hi
        entry[2] += 1
    per_request: List[PerRequestEstimate] = []
    for rid, (sum_lo, sum_hi, count) in sums.items():
        per_request.append(
            PerRequestEstimate(
                rid=rid,
                num_blocks=int(count),
                mean_lo=sum_lo / count,
                mean_hi=sum_hi / count,
                mean_mid=0.5 * (sum_lo + sum_hi) / count,
            )
        )
    return per_request


def evaluate_all(loaded: LoadedRecords, gamma: int) -> List[BlockEstimate]:
    return [
        evaluate_block(rec, loaded.gathers, gamma, loaded.eos_terminated)
        for rec in loaded.blocks
    ]


def analyze(jsonl_path: Path, *, gamma: int, arm: str) -> Dict[str, Any]:
    loaded = load_records(jsonl_path)
    return summarize(loaded.blocks, evaluate_all(loaded, gamma), arm=arm)


def summarize(
    blocks: List[Dict[str, Any]], results: List[BlockEstimate], *, arm: str
) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"arm": arm, "num_blocks": 0}

    mean_lo = sum(r.lo for r in results) / n
    mean_hi = sum(r.hi for r in results) / n
    categories: Dict[str, int] = defaultdict(int)
    for r in results:
        categories[r.category] += 1

    censored = [r for r in results if r.category != "exact_in_window"]
    widths = [r.hi - r.lo for r in censored]

    out: Dict[str, Any] = {
        "arm": arm,
        "num_blocks": n,
        "estimate_lo": mean_lo,
        "estimate_hi": mean_hi,
        "estimate_mid": (mean_lo + mean_hi) / 2,
        "mean_bracket_width_overall": (mean_hi - mean_lo),
        "mean_bracket_width_censored": (sum(widths) / len(widths) if widths else 0.0),
        "categories": dict(categories),
        "censored_fraction": len(censored) / n if n else 0.0,
        "mean_window_drafts": sum(b["w"] for b in blocks) / n,
        "mean_cl": sum(b["cl"] for b in blocks) / n,
    }

    a_pairs = [
        (r.a_first, r.cap_trim_positive) for r in censored if r.a_first is not None
    ]
    if a_pairs:
        out["mean_analytic_a_first"] = sum(a for a, _ in a_pairs) / len(a_pairs)
        out["empirical_cap_trim_positive_rate"] = sum(1 for _, c in a_pairs if c) / len(
            a_pairs
        )

    if arm == "cap-accept":
        out.update(
            _cap_accept_truth(
                results=results, censored=censored, mean_lo=mean_lo, mean_hi=mean_hi
            )
        )

    per_request = per_request_estimates(blocks, results)
    if per_request:
        out["per_request_summary"] = {
            "num_requests": len(per_request),
            "mean_of_request_mids": sum(p.mean_mid for p in per_request)
            / len(per_request),
            "mean_blocks_per_request": sum(p.num_blocks for p in per_request)
            / len(per_request),
        }

    return out


def _cap_accept_truth(
    *,
    results: List[BlockEstimate],
    censored: List[BlockEstimate],
    mean_lo: float,
    mean_hi: float,
) -> Dict[str, Any]:
    n = len(results)
    truths = [r.true_block_accept for r in results]
    true_mean = sum(truths) / n
    out: Dict[str, Any] = {
        "true_mean_block_accept": true_mean,
        "aggregate_truth_in_bracket": mean_lo - 1e-9 <= true_mean <= mean_hi + 1e-9,
        "aggregate_truth_error_vs_mid": (mean_lo + mean_hi) / 2 - true_mean,
    }
    if censored:
        censored_truth = sum(r.true_block_accept for r in censored) / len(censored)
        censored_lo = sum(r.lo for r in censored) / len(censored)
        censored_hi = sum(r.hi for r in censored) / len(censored)
        out["censored_true_mean"] = censored_truth
        out["censored_estimate_lo"] = censored_lo
        out["censored_estimate_hi"] = censored_hi
        out["censored_truth_in_bracket"] = (
            censored_lo - 1e-9 <= censored_truth <= censored_hi + 1e-9
        )
        out["calibration_bins"] = _calibration_bins(censored)
    return out


def _calibration_bins(
    censored: List[BlockEstimate], *, num_bins: int = 8
) -> List[Dict[str, Any]]:
    ordered = sorted(censored, key=lambda r: (r.lo + r.hi) / 2)
    bins: List[Dict[str, Any]] = []
    size = max(1, len(ordered) // num_bins)
    for start in range(0, len(ordered), size):
        chunk = ordered[start : start + size]
        bins.append(
            {
                "n": len(chunk),
                "mean_estimate_mid": sum((r.lo + r.hi) / 2 for r in chunk) / len(chunk),
                "mean_estimate_lo": sum(r.lo for r in chunk) / len(chunk),
                "mean_estimate_hi": sum(r.hi for r in chunk) / len(chunk),
                "mean_truth": sum(r.true_block_accept for r in chunk) / len(chunk),
            }
        )
    return bins


def analyze_driver_meta(meta_path: Path) -> Dict[str, Any]:
    total: Dict[str, float] = defaultdict(float)
    total_ct = 0.0
    n = 0
    with meta_path.open() as f:
        for line in f:
            rec = json.loads(line)
            meta = rec["meta_info"]
            ct = meta.get("spec_verify_ct", 0)
            if not ct:
                continue
            n += 1
            total_ct += ct
            for field in (
                "spec_accept_length",
                "spec_cap_length",
                "spec_block_accept_length",
            ):
                if meta.get(field) is not None:
                    total[field] += meta[field] * ct
    return {
        "num_requests_with_verify": n,
        "total_verify_ct": total_ct,
        **{f"{k}_step_weighted": v / total_ct for k, v in total.items() if total_ct},
    }


def main(
    recorder_jsonl: Annotated[Path, typer.Argument()],
    gamma: Annotated[int, typer.Argument()],
    arm: Annotated[str, typer.Argument()],
    driver_meta: Annotated[Optional[Path], typer.Option()] = None,
    output: Annotated[Optional[Path], typer.Option()] = None,
    per_request_output: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    loaded = load_records(recorder_jsonl)
    results = evaluate_all(loaded, gamma)
    result = summarize(loaded.blocks, results, arm=arm)
    if driver_meta is not None:
        result["driver_meta"] = analyze_driver_meta(driver_meta)

    text = json.dumps(result, indent=2)
    print(text)
    if output is not None:
        output.write_text(text + "\n")

    if per_request_output is not None:
        per_request = per_request_estimates(loaded.blocks, results)
        with per_request_output.open("w") as f:
            for entry in per_request:
                f.write(json.dumps(msgspec.to_builtins(entry)) + "\n")


if __name__ == "__main__":
    typer.run(main)
