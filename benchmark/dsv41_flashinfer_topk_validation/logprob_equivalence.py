"""Greedy logprob comparison of DeepSeek-V4.1-Flash servers that differ only in the
index-layer top-k backend, on the RULER2 64K prompts the accuracy runs used.

    python3 logprob_equivalence.py collect --url http://127.0.0.1:30000 --data-dir <ruler2 cache> \
        --per-task 20 --max-new-tokens 32 --out <label>.jsonl
    python3 logprob_equivalence.py compare --out <report>.md <label_a>.jsonl:<label_b>.jsonl ...

collect: each RULER2 row's assembled ``question`` (a completion-style prompt; the *_basic
ones end right before the bare answer) goes to /generate as raw text, one request at a time
so every server sees the same batch shapes, greedy, returning the logprob of every generated
token and the top-20 logprobs at every step. --ignore-eos makes every prompt decode
--max-new-tokens tokens (without it most prompts stop at an EOS first token). Servers run with --disable-radix-cache so a repeated prompt
is prefilled again.

compare: for each pair, over the prompts both files hold: first-token top-1 agreement,
|delta logprob| of a's first token, KL(a || b) on the first step's shared top-20 support
(renormalized; a lower bound on the full KL), how many prompts generate identical tokens
and the mean identical prefix, |delta logprob| over the identical prefix, and each side's
needle recall (fraction of expected values in the generated text) over the *_basic
subtasks, whose raw prompts end right before the bare answer.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time
import urllib.request

TOP_N = 20


def post(url, payload, timeout=1800):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def load_rows(data_dir, per_task):
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "test.jsonl"))):
        task = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for line in list(f)[:per_task]:
                row = json.loads(line)
                rows.append(
                    {
                        "id": f"{task}-{row['index']}",
                        "task": task,
                        "question": row["question"],
                        "expected": row["expected_answer"],
                    }
                )
    return rows


def collect(args):
    rows = load_rows(args.data_dir, args.per_task)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    with open(args.out, "a") as out:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            start = time.time()
            result = post(
                f"{args.url}/generate",
                {
                    "text": row["question"],
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": args.max_new_tokens,
                        "ignore_eos": args.ignore_eos,
                    },
                    "return_logprob": True,
                    "top_logprobs_num": TOP_N,
                    "logprob_start_len": -1,
                },
            )
            meta = result["meta_info"]
            record = {
                "id": row["id"],
                "task": row["task"],
                "expected": row["expected"],
                "prompt_tokens": meta["prompt_tokens"],
                "text": result["text"],
                "tokens": [t[1] for t in meta["output_token_logprobs"]],
                "logprobs": [t[0] for t in meta["output_token_logprobs"]],
                "top": [[[t[0], t[1]] for t in step] for step in meta["output_top_logprobs"]],
                "seconds": time.time() - start,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"{i + 1}/{len(rows)} {row['id']} prompt={meta['prompt_tokens']} {record['seconds']:.1f}s", flush=True)


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def expected_values(record):
    expected = record["expected"] if isinstance(record["expected"], list) else [record["expected"]]
    return [str(e) for e in expected]


def answers_in_raw_completion(record):
    """The *_basic subtasks' raw prompts end right before the bare answer (a number or a
    document index); the harder subtasks' formats assume the chat template."""
    return record["task"].endswith("_basic")


def recall(record):
    expected = expected_values(record)
    return sum(e.lower() in record["text"].lower() for e in expected) / len(expected)


def first_step(a, b):
    top_a = {tok: lp for lp, tok in a["top"][0]}
    top_b = {tok: lp for lp, tok in b["top"][0]}
    shared = [tok for tok in top_a if tok in top_b]
    tok = a["tokens"][0]
    delta = abs(a["logprobs"][0] - top_b[tok]) if tok in top_b else float("inf")
    if not shared:
        return a["tokens"][0] == b["tokens"][0], delta, float("inf")
    za = math.log(sum(math.exp(top_a[t]) for t in shared))
    zb = math.log(sum(math.exp(top_b[t]) for t in shared))
    kl = sum(math.exp(top_a[t] - za) * ((top_a[t] - za) - (top_b[t] - zb)) for t in shared)
    return a["tokens"][0] == b["tokens"][0], delta, max(kl, 0.0)


def label(path):
    """``logprob/<config>/pass1.jsonl`` -> ``<config>-pass1``."""
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return f"{folder}-{os.path.basename(path).removesuffix('.jsonl')}"


def compare_pair(path_a, path_b):
    a_all, b_all = load(path_a), load(path_b)
    ids = [i for i in a_all if i in b_all]
    short = [i for i in ids if answers_in_raw_completion(a_all[i])]
    agree_first, deltas, kls, identical, prefix, step_deltas = 0, [], [], 0, [], []
    for i in ids:
        a, b = a_all[i], b_all[i]
        same, delta, kl = first_step(a, b)
        agree_first += same
        deltas.append(delta)
        kls.append(kl)
        n = 0
        while n < min(len(a["tokens"]), len(b["tokens"])) and a["tokens"][n] == b["tokens"][n]:
            step_deltas.append(abs(a["logprobs"][n] - b["logprobs"][n]))
            n += 1
        prefix.append(n)
        identical += n == len(a["tokens"]) == len(b["tokens"])

    def q(values, p):
        values = sorted(values)
        return values[min(len(values) - 1, int(p * len(values)))] if values else float("nan")

    return {
        "a": label(path_a),
        "b": label(path_b),
        "prompts": len(ids),
        "first_top1_agree": agree_first,
        "first_dlogp_median": statistics.median(deltas),
        "first_dlogp_max": max(deltas),
        "first_kl_mean": statistics.fmean(kls),
        "first_kl_max": max(kls),
        "identical_outputs": identical,
        "prefix_mean": statistics.fmean(prefix),
        "step_dlogp_median": statistics.median(step_deltas) if step_deltas else float("nan"),
        "step_dlogp_p99": q(step_deltas, 0.99),
        "step_dlogp_max": max(step_deltas) if step_deltas else float("nan"),
        "recall_prompts": len(short),
        "recall_a": statistics.fmean(recall(a_all[i]) for i in short) if short else float("nan"),
        "recall_b": statistics.fmean(recall(b_all[i]) for i in short) if short else float("nan"),
    }


def compare(args):
    rows = [compare_pair(*pair.split(":")) for pair in args.pairs]
    header = (
        "| a | b | prompts | first token top-1 agree | first-token abs dlogp median / max | "
        "first-token KL mean / max | identical outputs | mean identical prefix | "
        "abs dlogp on identical prefix median / p99 / max | *_basic needle recall a / b (prompts) |"
    )
    lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| `{r['a']}` | `{r['b']}` | {r['prompts']} | {r['first_top1_agree']} | "
            f"{r['first_dlogp_median']:.2e} / {r['first_dlogp_max']:.2e} | "
            f"{r['first_kl_mean']:.2e} / {r['first_kl_max']:.2e} | {r['identical_outputs']} | "
            f"{r['prefix_mean']:.1f} | {r['step_dlogp_median']:.2e} / {r['step_dlogp_p99']:.2e} / "
            f"{r['step_dlogp_max']:.2e} | {r['recall_a']:.3f} / {r['recall_b']:.3f} ({r['recall_prompts']}) |"
        )
    text = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(text + "\n\n" + json.dumps(rows, indent=1) + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", required=True)
    c.add_argument("--data-dir", required=True)
    c.add_argument("--per-task", type=int, default=20)
    c.add_argument("--max-new-tokens", type=int, default=32)
    c.add_argument("--ignore-eos", action="store_true", help="decode max-new-tokens tokens on every prompt")
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--out", required=True)
    m.add_argument("pairs", nargs="+")
    args = parser.parse_args()
    {"collect": collect, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
