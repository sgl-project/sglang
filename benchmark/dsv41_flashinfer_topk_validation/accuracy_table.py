"""Summarize sgl-eval results of the final-code e2e runs as Markdown.

    python3 accuracy_table.py <e2e root> <out dir> <run> [<run> ...]

Writes <out dir>/accuracy-summary.md (headline score +- standard error per eval, how
many RULER2 responses ran to the context limit, and every other run's difference from the
first run paired by prompt, +- the paired standard error), <out dir>/accuracy-details.md (the
RULER2 per-subtask table, then each run's run.txt and raw sgl-eval ``aggregate`` block)
and <out dir>/accuracy-stats.json (the same numbers for tools/build_pr_body.py).
Run folders are named ``r<N>-<config>``; tables show ``<config>``.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

EVALS = ("gsm8k", "ruler2")


def load(root, run):
    folder = os.path.join(root, run)
    with open(os.path.join(folder, "run.txt")) as f:
        run_txt = f.read().strip()
    metrics, rows = {}, {}
    for path in sorted(glob.glob(os.path.join(folder, "sgl_eval_*", "metrics.json"))):
        with open(path) as f:
            data = json.load(f)
        metrics[data["name"]] = data
        with open(os.path.join(os.path.dirname(path), "output-rs0.jsonl")) as f:
            rows[data["name"]] = [json.loads(line) for line in f if line.strip()]
    return run_txt, metrics, rows


def config(run):
    return run.split("-", 1)[1]


def stats(data, rows):
    """Mean and standard error of the per-response scores; checks the mean against sgl-eval's."""
    # GSM8K rows carry symbolic_correct; RULER2 rows carry is_correct (partial credit).
    field = "symbolic_correct" if data["name"] == "gsm8k" else "is_correct"
    scores = [float(row[field]) for row in rows]
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / (n - 1)
    if abs(mean - data["aggregate"]["score"]) > 1e-9 or n != data["num_examples"]:
        raise SystemExit(f"{data['name']}: per-response mean {mean} != aggregate score")
    capped = sum(row["finish_reason"] == "length" for row in rows)
    return mean, math.sqrt(var / n), n, capped


def scores_by_id(data, rows):
    field = "symbolic_correct" if data["name"] == "gsm8k" else "is_correct"
    return {row["id"]: float(row[field]) for row in rows}


def paired(loaded, name):
    """Each later run's mean difference from the first run over the same prompts, and its
    standard error from the per-prompt differences."""
    runs = list(loaded)
    ref_metrics, ref_rows = loaded[runs[0]][1], loaded[runs[0]][2]
    ref = scores_by_id(ref_metrics[name], ref_rows[name])
    out = {}
    for run in runs[1:]:
        _, metrics, rows = loaded[run]
        other = scores_by_id(metrics[name], rows[name])
        if set(other) != set(ref):
            raise SystemExit(f"{run} {name}: prompt ids differ from {runs[0]}")
        diffs = [other[i] - ref[i] for i in ref]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        out[config(run)] = {"diff": mean, "se": math.sqrt(var / n), "n": n, "differ": sum(d != 0 for d in diffs)}
    return out


def summary(loaded):
    lines = [
        "| config | GSM8K (%) | RULER2 64K (%) | RULER2 responses cut at the 1M-token context |",
        "|---|---:|---:|---:|",
    ]
    for run, (_, metrics, rows) in loaded.items():
        cells = [f"`{config(run)}`"]
        capped_cell = ""
        for name in EVALS:
            if name not in metrics:
                cells.append("")
                continue
            mean, se, n, capped = stats(metrics[name], rows[name])
            cells.append(f"{100 * mean:.2f} ± {100 * se:.2f}")
            if name == "ruler2":
                capped_cell = f"{capped} / {n}"
        cells.append(capped_cell)
        lines.append("| " + " | ".join(cells) + " |")
    runs = list(loaded)
    ref = config(runs[0])
    lines += [
        "",
        f"| config | GSM8K minus `{ref}`, paired (points) | RULER2 minus `{ref}`, paired (points) |",
        "|---|---:|---:|",
    ]
    by_eval = {name: paired(loaded, name) for name in EVALS}
    for run in runs[1:]:
        cells = [f"`{config(run)}`"]
        for name in EVALS:
            p = by_eval[name][config(run)]
            cells.append(f"{100 * p['diff']:+.2f} ± {100 * p['se']:.2f} ({p['differ']} of {p['n']} prompts scored differently)")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def details(loaded):
    out = []
    keys = sorted(
        {
            key
            for _, metrics, _ in loaded.values()
            for key in metrics.get("ruler2", {}).get("aggregate", {})
            if key.startswith("task.")
        }
    )
    out.append("| RULER2 subtask | " + " | ".join(f"`{config(run)}`" for run in loaded) + " |")
    out.append("|---|" + "---:|" * len(loaded))
    for key in keys:
        values = []
        for _, metrics, _ in loaded.values():
            value = metrics.get("ruler2", {}).get("aggregate", {}).get(key)
            values.append("" if value is None else f"{100 * value:.2f}")
        out.append(f"| {key.removeprefix('task.')} | " + " | ".join(values) + " |")

    gens = {json.dumps(m["gen"], sort_keys=True) for _, metrics, _ in loaded.values() for m in metrics.values()}
    shared_gen = len(gens) == 1
    if shared_gen:
        out.append("")
        out.append(f"Sampling for every run and eval: `{gens.pop()}`")
    for run, (run_txt, metrics, _) in loaded.items():
        out.append("")
        out.append(f"`{config(run)}` (`{run}`):")
        out.append("")
        out.append("```text")
        out.append(run_txt)
        for name in EVALS:
            if name not in metrics:
                continue
            data = metrics[name]
            gen = "" if shared_gen else f" gen={json.dumps(data['gen'], sort_keys=True)}"
            out.append(
                f"{name}: sgl_eval_version={data['sgl_eval_version']} num_examples={data['num_examples']} "
                f"num_threads={data['num_threads']} latency_seconds={data['latency_seconds']:.1f}{gen}"
            )
            out.append(json.dumps(data["aggregate"], sort_keys=True))
        out.append("```")
    return "\n".join(out)


def stats_json(loaded):
    out = {}
    by_eval = {name: paired(loaded, name) for name in EVALS}
    for run, (_, metrics, rows) in loaded.items():
        out[config(run)] = {}
        for name in EVALS:
            if name in metrics:
                mean, se, n, capped = stats(metrics[name], rows[name])
                out[config(run)][name] = {"mean": mean, "se": se, "n": n, "capped": capped}
                if config(run) in by_eval[name]:
                    out[config(run)][name]["paired"] = by_eval[name][config(run)]
    return out


def main():
    root, out_dir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    loaded = {run: load(root, run) for run in runs}
    with open(os.path.join(out_dir, "accuracy-stats.json"), "w") as f:
        json.dump(stats_json(loaded), f, indent=1)
    with open(os.path.join(out_dir, "accuracy-summary.md"), "w") as f:
        f.write(summary(loaded) + "\n")
    with open(os.path.join(out_dir, "accuracy-details.md"), "w") as f:
        f.write(details(loaded) + "\n")
    print(summary(loaded))


if __name__ == "__main__":
    main()
