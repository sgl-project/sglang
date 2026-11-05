"""Offline analysis of diagnose_spec_opt_out_prefix.py artifacts.

No tolerance converts a token mismatch into a pass. Full-prefix checks gate
score comparisons; matching only the last input token is insufficient.
"""

import json
import sys
from pathlib import Path

import numpy as np


def score_summary(values, selected, reference):
    values, reference = np.asarray(values), np.asarray(reference)
    if values.ndim != 1 or values.shape != reference.shape or values.size < 2:
        raise ValueError("expected equal vocabulary vectors")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("non-finite logits")
    winners = np.flatnonzero(values == values.max()).tolist()
    top = np.partition(values, -2)[-2:]
    delta = values.astype(np.float64) - reference
    return dict(
        selected=selected,
        greedy_argmax_correct=selected == winners[0],
        tied_max_ids=winners,
        top_margin=float(top.max() - top.min()),
        max_abs_delta_from_fresh=float(np.abs(delta).max()),
        rms_delta_from_fresh=float(np.sqrt(np.mean(delta**2))),
    )


def same_prefix(request_ids, emitted_ids, prefix):
    history = list(request_ids) + list(emitted_ids)
    return len(history) >= len(prefix) and history[: len(prefix)] == prefix


def analyze(root):
    root = Path(root)
    inputs = json.loads((root / "inputs.json").read_text())
    prefixes = {int(k): v for k, v in inputs["prefixes"].items()}
    records = [
        json.loads(s) for s in (root / "captures.jsonl").read_text().splitlines()
    ]
    responses = {}
    for mode in ("target", "spec"):
        for case in (
            "warm_plain",
            "warm_probe",
            "fresh-0",
            "fresh-1",
            "one-decode-0",
            "one-decode-1",
        ):
            saved = json.loads((root / f"{mode}-{case}.json").read_text())
            rows = json.loads(saved["response_text"])
            for row in rows:
                assert row["output_ids"] == [
                    x[1] for x in row["meta_info"]["output_token_logprobs"]
                ]
            responses[f"{mode}-{case}"] = (
                saved["request"]["input_ids"],
                [r["output_ids"] for r in rows],
            )
    summary = []
    for record in records:
        name = record["name"]
        case = next(k for k in responses if name.startswith(k + "-"))
        index, prefix = next(
            (i, p) for i, p in prefixes.items() if len(p) - 1 == record["position"]
        )
        row = index if "warm_" in case else 0
        request, emitted = responses[case]
        valid = same_prefix(request[row], emitted[row], prefix)
        reference = next(
            r for r in records if r["name"].startswith(f"target-fresh-{index}-")
        )
        item = dict(case=case, position=record["position"], identical_prefix=valid)
        if valid:
            item.update(
                score_summary(
                    np.load(root / (name + ".npy")),
                    record["selected"],
                    np.load(root / (reference["name"] + ".npy")),
                )
            )
            offset = len(prefix) - len(request[row])
            item["selected_matches_emitted"] = (
                offset >= 0 and emitted[row][offset] == record["selected"]
            )
        summary.append(item)
    result = dict(
        logits=summary,
        unprobed_equals_probed={
            mode: responses[f"{mode}-warm_plain"][1]
            == responses[f"{mode}-warm_probe"][1]
            for mode in ("target", "spec")
        },
        warm_target_equals_spec=responses["target-warm_plain"][1]
        == responses["spec-warm_plain"][1],
    )
    (root / "validated-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), indent=2))
