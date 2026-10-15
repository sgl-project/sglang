"""Inspect saved code-edit outputs without starting a server or loading weights."""

import argparse
import json
from itertools import combinations
from pathlib import Path

from bench_code_edits import MODES, model_assets, sha256
from code_edit_report import compare_outputs, first_divergence


def read_json(path):
    return json.loads(path.read_text())


def inspect_results(directory, tokenizer=None, context_tokens=8):
    summary = read_json(directory / "summary.json")
    inputs = read_json(directory / "inputs.json")
    experiment = read_json(directory / "experiment.json")
    prompts = {row["task_id"]: row for row in inputs["requests"]}
    if len(prompts) != len(inputs["requests"]):
        raise ValueError("Duplicate task IDs in inputs.json")
    if summary.get("framework") != "sglang" or "baseline" not in summary["modes"]:
        raise ValueError("Expected a completed SGLang summary including baseline")

    trials = {}
    for mode, report in summary["modes"].items():
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        for trial in report["trials"]:
            repetition = trial["repetition"]
            if type(repetition) is not int or repetition < 1:
                raise ValueError("Invalid repetition")
            key = (mode, repetition)
            if key in trials:
                raise ValueError(f"Duplicate trial: {key}")
            path = directory / f"{mode}-{repetition}.requests.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            ids = [row["task_id"] for row in rows]
            if ids != trial["task_ids"] or len(ids) != len(set(ids)):
                raise ValueError(f"Task IDs/order differ from summary: {path}")
            if len(rows) != trial["num_requests"] or not rows:
                raise ValueError(f"Request count differs from summary: {path}")
            for row in rows:
                prompt = prompts[row["task_id"]]
                if row["prompt_token_ids"] != prompt["input_ids"]:
                    raise ValueError(f"Prompt IDs differ from saved inputs: {path}")
                if row.get("validation_errors"):
                    raise ValueError(f"Request validation failed: {path}")
            trials[key] = rows

    first_baseline_rep = min(rep for mode, rep in trials if mode == "baseline")
    modes = {}
    for mode, report in summary["modes"].items():
        cases, counts = [], []
        for trial in report["trials"]:
            rep = trial["repetition"]
            ref_rep = first_baseline_rep if mode == "baseline" else rep
            if trial["reference_baseline_repetition"] != ref_rep:
                raise ValueError(f"Wrong baseline reference: {mode}-{rep}")
            reference = trials["baseline", ref_rep]
            actual = trials[mode, rep]
            mismatches = compare_outputs(reference, actual)
            if mismatches != trial["mismatches"]:
                raise ValueError(
                    f"Saved summary disagrees with raw outputs: {mode}-{rep}"
                )
            counts.append({"repetition": rep, "mismatch_count": len(mismatches)})
            for left, right in zip(reference, actual):
                difference = first_divergence(left["output_ids"], right["output_ids"])
                if difference is None:
                    continue
                index = difference["token_index"]
                start, end = max(0, index - context_tokens), index + context_tokens + 1
                baseline_ids = left["output_ids"]
                actual_ids = right["output_ids"]
                case = {
                    "task_id": right["task_id"],
                    "repetition": rep,
                    "baseline_repetition": ref_rep,
                    **difference,
                    "input_tokens": len(prompts[right["task_id"]]["input_ids"]),
                    "baseline_output_tokens": len(baseline_ids),
                    "actual_output_tokens": len(actual_ids),
                    "window_start": start,
                    "common_prefix_tail_ids": baseline_ids[start:index],
                    "baseline_window_ids": baseline_ids[start:end],
                    "actual_window_ids": actual_ids[start:end],
                    "baseline_saved_text": left.get("text"),
                    "actual_saved_text": right.get("text"),
                }
                if tokenizer is not None:
                    for label, ids in (
                        ("common_prefix_tail", baseline_ids[start:index]),
                        ("baseline_window", baseline_ids[start:end]),
                        ("actual_window", actual_ids[start:end]),
                    ):
                        case[label + "_text"] = tokenizer.decode(
                            ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    for label in ("baseline", "actual"):
                        token = difference[label + "_token_id"]
                        case[label + "_token_piece"] = (
                            tokenizer.convert_ids_to_tokens(token)
                            if token is not None
                            else None
                        )
                cases.append(case)
        if len(cases) != report["mismatch_count"]:
            raise ValueError(
                f"Summary mismatch count disagrees with raw outputs: {mode}"
            )
        repetitions = sorted(rep for name, rep in trials if name == mode)
        repeatability = [
            {
                "repetition": rep,
                "mismatches": compare_outputs(
                    trials[mode, repetitions[0]], trials[mode, rep]
                ),
            }
            for rep in repetitions[1:]
        ]
        modes[mode] = {
            "request_repetition_pairs": sum(
                len(trials[mode, rep]) for rep in repetitions
            ),
            "mismatch_count": len(cases),
            "unique_mismatching_task_ids": list(
                dict.fromkeys(c["task_id"] for c in cases)
            ),
            "trials": counts,
            "repeatability": repeatability,
            "cases": cases,
        }

    cross_mode = []
    for left, right in combinations(modes, 2):
        repetitions = sorted(
            {rep for mode, rep in trials if mode == left}
            & {rep for mode, rep in trials if mode == right}
        )
        differences = [
            {"repetition": rep, **mismatch}
            for rep in repetitions
            for mismatch in compare_outputs(trials[left, rep], trials[right, rep])
        ]
        cross_mode.append(
            {
                "reference_mode": left,
                "actual_mode": right,
                "request_repetition_pairs": sum(
                    len(trials[left, rep]) for rep in repetitions
                ),
                "mismatch_count": len(differences),
                "mismatches": differences,
            }
        )
    return {
        "schema_version": 1,
        "directory": str(directory.resolve()),
        "benchmark_source_revision": experiment["source_revision"],
        "summary_sha256": sha256(directory / "summary.json"),
        "decoded_windows": tokenizer is not None,
        "modes": modes,
        "cross_mode": cross_mode,
        "notes": [
            "Counts are request/repetition pairs, not differing token counts.",
            "Baseline's first trial is compared with itself; it is not repeatability evidence.",
            "Token indices are zero-based output positions, excluding the prompt.",
            "Cross-mode baseline_token_id denotes the reference_mode's token.",
            "Decoded windows may split a multi-token UTF-8 character; token IDs are authoritative.",
            "Saved artifacts contain no target logits or verification trees; this report cannot prove the cause.",
        ],
    }


def print_report(report, mode=None, task_id=None, limit=3):
    print("Benchmark source:", report["benchmark_source_revision"])
    print(
        "Mode                              mismatches / pairs    unique tasks    repeat differences"
    )
    for name, result in report["modes"].items():
        repeats = result["repeatability"]
        repeat_count = (
            str(sum(len(r["mismatches"]) for r in repeats)) if repeats else "not tested"
        )
        print(
            f"{name:33} {result['mismatch_count']:5} / {result['request_repetition_pairs']:<5}"
            f" {len(result['unique_mismatching_task_ids']):15} {repeat_count:>21}"
        )
        print("  Per repetition:", result["trials"])
        print("  Mismatching task IDs:", result["unique_mismatching_task_ids"])
    print("\nPairwise output differences (same repetition):")
    for pair in report["cross_mode"]:
        print(
            f"  {pair['reference_mode']} vs {pair['actual_mode']}: "
            f"{pair['mismatch_count']}/{pair['request_repetition_pairs']}"
        )
    for name, result in report["modes"].items():
        if mode is not None and name != mode:
            continue
        cases = [
            c
            for c in result["cases"]
            if task_id is None or str(c["task_id"]) == task_id
        ]
        for case in cases[:limit]:
            print(
                f"\n{name}, repetition {case['repetition']}, task {case['task_id']}, "
                f"first differing output token {case['token_index']}"
            )
            print(
                "  Token IDs:", case["baseline_token_id"], "->", case["actual_token_id"]
            )
            for label in ("common_prefix_tail", "baseline_window", "actual_window"):
                print(
                    f"  {label}:", repr(case.get(label + "_text", case[label + "_ids"]))
                )
            if "baseline_token_piece" in case:
                print(
                    "  Token pieces:",
                    repr(case["baseline_token_piece"]),
                    "->",
                    repr(case["actual_token_piece"]),
                )
    print(
        "\nNo model inference performed. Target logits/trees were not saved; cause remains unproven."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Optional local tokenizer snapshot for decoded windows",
    )
    parser.add_argument(
        "--output", type=Path, help="New JSON file; never overwrites existing artifacts"
    )
    parser.add_argument("--mode", choices=MODES, help="Filter printed examples only")
    parser.add_argument("--task-id", help="Filter printed examples only")
    parser.add_argument(
        "--limit", type=int, default=3, help="Printed examples per mode"
    )
    parser.add_argument("--context-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 1 or args.context_tokens < 0:
        parser.error("limit must be positive and context-tokens nonnegative")
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    tokenizer = None
    if args.model_path is not None:
        inputs = read_json(args.directory / "inputs.json")
        if model_assets(args.model_path) != inputs["model_assets"]:
            parser.error("Model/tokenizer metadata differs from the saved input export")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True
        )
    report = inspect_results(args.directory, tokenizer, args.context_tokens)
    if args.mode is not None and args.mode not in report["modes"]:
        parser.error(f"Mode absent from this run: {args.mode}")
    if args.output is not None:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print_report(report, args.mode, args.task_id, args.limit)


if __name__ == "__main__":
    main()
