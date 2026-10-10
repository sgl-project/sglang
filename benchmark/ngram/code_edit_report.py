"""Aggregation and output checks for the serial code-edit experiment."""

import hashlib
import json
from statistics import mean, median, stdev


def distribution(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {
        "values": values,
        "median": median(values),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "stdev": stdev(values) if len(values) > 1 else None,
    }


def summarize_trial(mode, repetition, requests, window_s):
    tokens = sum(len(r["output_ids"]) for r in requests)
    generation_s = sum(r["generation_s"] for r in requests)
    workflow_s = sum(r["workflow_s"] for r in requests)
    verify = sum(r["meta_info"].get("spec_verify_ct", 0) for r in requests)
    correct = sum(r["meta_info"].get("spec_num_correct_drafts", 0) for r in requests)
    proposed = sum(r["meta_info"].get("spec_num_proposed_drafts", 0) for r in requests)
    return {
        "mode": mode,
        "repetition": repetition,
        "num_requests": len(requests),
        "task_ids": [r["task_id"] for r in requests],
        "output_tokens": tokens,
        "generation_s": generation_s,
        "workflow_s": workflow_s,
        "window_s": window_s,
        "generation_tok_s": tokens / generation_s,
        "workflow_tok_s": tokens / workflow_s,
        "window_tok_s": tokens / window_s,
        "mean_latency_ms": mean(r["generation_s"] for r in requests) * 1000,
        "mean_workflow_ms": mean(r["workflow_s"] for r in requests) * 1000,
        "mean_ttft_ms": mean(r["ttft_s"] for r in requests) * 1000,
        "mean_tpot_ms": mean(r["tpot_s"] for r in requests if r["tpot_s"] is not None)
        * 1000,
        "mean_setup_ms": mean(r["setup_s"] for r in requests) * 1000,
        "mean_corpus_load_ms": mean(r["corpus_load_s"] for r in requests) * 1000,
        "spec_verify_ct": verify,
        "spec_num_correct_drafts": correct,
        "spec_num_proposed_drafts": proposed,
        "acceptance_length": tokens / verify if verify else None,
        "draft_acceptance_rate": correct / proposed if proposed else None,
    }


def first_divergence(reference, actual):
    for index in range(max(len(reference), len(actual))):
        left = reference[index] if index < len(reference) else None
        right = actual[index] if index < len(actual) else None
        if left != right:
            return {
                "token_index": index,
                "baseline_token_id": left,
                "actual_token_id": right,
            }
    return None


def compare_outputs(reference, actual):
    if [r["task_id"] for r in reference] != [r["task_id"] for r in actual]:
        raise ValueError("Cannot pair outputs with different request IDs/order")
    mismatches = []
    for left, right in zip(reference, actual):
        divergence = first_divergence(left["output_ids"], right["output_ids"])
        if divergence is not None:
            mismatches.append({"task_id": right["task_id"], **divergence})
    return mismatches


def trt_reference(directory, num_requests):
    """Keep internally timed TRT results in a separate, normalized table.

    Old reports have no raw token IDs or input hash. Shape checks do not prove
    workload provenance or correctness; explicitly retain that limitation.
    """
    reports = {}
    for mode in ("baseline", "ngram", "sa3", "sa_longest"):
        paths = sorted(directory.glob(f"{mode}-*.report.json"))
        if not paths:
            raise ValueError(f"No TRT reports for {mode} in {directory}")
        reports[mode] = []
        for path in paths:
            report = json.loads(path.read_text())
            info = report["request_info"]
            if info["num_requests"] != num_requests or info["avg_output_length"] != 256:
                raise ValueError(
                    f"{path}: TRT request/output counts differ from this run"
                )
            reports[mode].append(
                {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "output_tok_s": report["performance"][
                        "system_output_throughput_tok_s"
                    ],
                    "tpot_ms": report["streaming_metrics"]["avg_tpot_ms"],
                }
            )
    baseline = median(r["output_tok_s"] for r in reports["baseline"])
    return {
        "timing_boundary": "TRT internal benchmark request window",
        "absolute_comparison_valid": False,
        "historical_workload_identity_verified": False,
        "notes": [
            "TRT internal timing and SGLang HTTP timing have different boundaries.",
            "Shape checks cannot verify the historical model, inputs, GPU, or scheduler settings.",
            "The token export verifies loader parity, not what an earlier run consumed.",
            "Existing TRT smoke has two baseline text mismatches and mixed overlap settings; investigate before correctness claims.",
        ],
        "modes": {
            mode: {
                "trials": records,
                "output_tok_s": distribution(r["output_tok_s"] for r in records),
                "tpot_ms": distribution(r["tpot_ms"] for r in records),
                "speedup_vs_own_baseline": median(r["output_tok_s"] for r in records)
                / baseline,
            }
            for mode, records in reports.items()
        },
    }


def build_summary(trials, trt_directory=None):
    baselines = {t["repetition"]: t for t in trials if t["mode"] == "baseline"}
    first_baseline = baselines[min(baselines)]
    metrics = (
        "generation_tok_s",
        "workflow_tok_s",
        "window_tok_s",
        "mean_latency_ms",
        "mean_workflow_ms",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_setup_ms",
        "mean_corpus_load_ms",
        "acceptance_length",
        "draft_acceptance_rate",
    )
    grouped = {}
    for trial in trials:
        baseline = baselines[trial["repetition"]]
        # Baseline repeatability uses repetition 1; other modes use paired baselines.
        reference = first_baseline if trial["mode"] == "baseline" else baseline
        mismatches = compare_outputs(reference["requests"], trial["requests"])
        entry = {k: v for k, v in trial.items() if k != "requests"}
        entry.update(
            {
                "generation_speedup": trial["generation_tok_s"]
                / baseline["generation_tok_s"],
                "workflow_speedup": trial["workflow_tok_s"]
                / baseline["workflow_tok_s"],
                "reference_baseline_repetition": reference["repetition"],
                "mismatches": mismatches,
            }
        )
        grouped.setdefault(trial["mode"], []).append(entry)
    modes = {}
    for mode, entries in grouped.items():
        mode_trials = [t for t in trials if t["mode"] == mode]
        first_trial = min(mode_trials, key=lambda t: t["repetition"])
        repeatability = [
            {
                "repetition": t["repetition"],
                "mismatches": compare_outputs(first_trial["requests"], t["requests"]),
            }
            for t in mode_trials
            if t is not first_trial
        ]
        modes[mode] = {
            **{metric: distribution(e[metric] for e in entries) for metric in metrics},
            "generation_speedup": distribution(
                e["generation_speedup"] for e in entries
            ),
            "workflow_speedup": distribution(e["workflow_speedup"] for e in entries),
            "mismatch_count": sum(len(e["mismatches"]) for e in entries),
            "repeatability": repeatability,
            "trials": entries,
        }
    result = {
        "schema_version": 1,
        "framework": "sglang",
        "modes": modes,
        "notes": [
            "Each trial uses serial requests, fixed output length, and cold request state.",
            "Generation throughput = output tokens / sum of HTTP generation latencies.",
            "Workflow throughput includes flush, corpus build/upload, isolation checks, generation, and removal.",
            "Window throughput also includes client result validation and JSONL writes between requests.",
            "Startup, warmup, and offline input-token verification are recorded separately and excluded.",
            "TPOT uses first-to-last token-bearing event time / (output tokens - 1); events may contain multiple tokens.",
            "Acceptance ratios use summed final request counters; proposed drafts count capacity including padding.",
            "SGLang completion/verify and TRT output/(decode_iteration+1) have different first/final-step conventions.",
            "Mismatches are token-level comparisons against the same repetition's baseline; baseline repeats use repetition 1.",
            "A single baseline trial has no repeatability evidence; fixed-length generation does not evaluate edit quality.",
            "SGLang static prompt SAM plus online Trie differs from TRT growing request-local history.",
        ],
    }
    if trt_directory is not None:
        result["trt_reference"] = trt_reference(
            trt_directory, first_baseline["num_requests"]
        )
    return result
