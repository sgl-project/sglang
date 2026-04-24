"""Score a SuperGPQA trace JSONL against its meta sidecar.

Inputs:
    --trace  results/supergpqa/mc{mc}_{variant}.jsonl   (bench_serving output)
    --meta   prompts/supergpqa.meta.jsonl               (from prepare_prompts_supergpqa.py)

Matches trace.generated_texts[i] against meta[i].answer_letter.  The two
files are parallel by index because bench_serving preserves input order
and the prep script wrote prompts.jsonl + meta.jsonl in lockstep.

Emits a .scores.json with overall accuracy plus per-discipline,
per-field, and per-difficulty breakdowns.  Failed requests
(trace.errors[i] set) are counted as incorrect.

Qwen3 thinking mode: responses may begin with `<think>...</think>`
blocks containing candidate letters, scratchwork, and discarded
answers.  We strip those blocks before letter extraction so the
cascade lands on the *final* answer, not an intermediate guess.

Letter extraction cascade (first match wins):
    1. Exact 'Answer: X' on its own line (matches our system-instruction).
    2. Last occurrence of 'answer is X' / '\\boxed{X}'.
    3. Last standalone capital letter A–J surrounded by punctuation.
    4. Fallback: last A–J letter anywhere in the response.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

OPTION_LETTERS = "ABCDEFGHIJ"

_RE_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_RE_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*[:\-]\s*([A-J])\b")
_RE_ANSWER_IS = re.compile(r"(?i)answer\s+is\s*\(?\*?\s*([A-J])\b")
_RE_BOXED = re.compile(r"\\boxed\s*\{\s*([A-J])\s*\}")
_RE_FENCED_LETTER = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")


def strip_thinking(response: str) -> str:
    """Remove Qwen3 <think>...</think> blocks before letter extraction.

    If the response has an open <think> with no matching </think> (because
    max_tokens cut off mid-reasoning), we return "" — a truncated reasoning
    block has no real answer, and scoring the thinking would pick up a
    scratch-letter the model later rejected.
    """
    if not response:
        return ""
    cleaned = _RE_THINK_BLOCK.sub("", response)
    if "<think>" in cleaned and "</think>" not in cleaned:
        return ""
    return cleaned.strip()


def extract_letter(response: str, n_options: int) -> Optional[str]:
    if not response:
        return None
    valid = set(OPTION_LETTERS[:n_options])

    for regex in (_RE_ANSWER_LINE, _RE_ANSWER_IS, _RE_BOXED):
        matches = regex.findall(response)
        for hit in reversed(matches):
            if hit in valid:
                return hit

    tail = response[-200:]
    for hit in reversed(_RE_FENCED_LETTER.findall(tail)):
        if hit in valid:
            return hit

    for hit in reversed(_RE_FENCED_LETTER.findall(response)):
        if hit in valid:
            return hit
    return None


def _load_trace(path: Path) -> dict:
    with open(path) as f:
        text = f.read().lstrip()
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


def _load_meta(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True,
                    help="Path to the bench_serving output JSONL "
                         "(results/supergpqa/mc*_*.jsonl).")
    ap.add_argument("--meta", type=Path, required=True,
                    help="Path to prompts/supergpqa.meta.jsonl.")
    ap.add_argument("--out", type=Path,
                    help="Output path for .scores.json. Defaults to "
                         "the trace file with .jsonl → .scores.json.")
    ap.add_argument("--per-doc", action="store_true",
                    help="Include per-document predictions in the output "
                         "(small overhead; useful for debugging).")
    args = ap.parse_args()

    out = args.out or args.trace.with_suffix(".scores.json")

    trace = _load_trace(args.trace)
    meta = _load_meta(args.meta)

    generated = trace.get("generated_texts", [])
    errors = trace.get("errors", [None] * len(generated))
    input_lens = trace.get("input_lens", [])
    output_lens = trace.get("output_lens", [])

    n_trace = len(generated)
    n_meta = len(meta)
    n = min(n_trace, n_meta)
    if n_trace != n_meta:
        print(
            f"WARNING: trace has {n_trace} rows, meta has {n_meta}. "
            f"Scoring the first {n} rows only."
        )

    by_discipline: dict[str, list[int]] = defaultdict(list)
    by_field: dict[str, list[int]] = defaultdict(list)
    by_difficulty: dict[str, list[int]] = defaultdict(list)
    by_calculation: dict[str, list[int]] = defaultdict(list)

    n_correct = 0
    n_failed = 0
    n_unparsed = 0
    records: list[dict] = []

    for i in range(n):
        m = meta[i]
        resp = generated[i] or ""
        err = errors[i] if i < len(errors) else None
        gold = m["answer_letter"]
        n_opt = m.get("n_options", 4)

        if err:
            pred = None
            correct = 0
            n_failed += 1
        else:
            pred = extract_letter(strip_thinking(resp), n_opt)
            if pred is None:
                n_unparsed += 1
                correct = 0
            else:
                correct = int(pred == gold)
        n_correct += correct

        by_discipline[m.get("discipline") or ""].append(correct)
        by_field[m.get("field") or ""].append(correct)
        by_difficulty[m.get("difficulty") or ""].append(correct)
        by_calculation[str(m.get("is_calculation"))].append(correct)

        records.append({
            "row_index": m.get("row_index"),
            "doc_id": m.get("doc_id"),
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "error": err,
            "input_len": input_lens[i] if i < len(input_lens) else None,
            "output_len": output_lens[i] if i < len(output_lens) else None,
        })

    def _summary(buckets: dict[str, list[int]]) -> dict:
        return {
            k: {"n": len(v), "acc": (sum(v) / len(v)) if v else 0.0}
            for k, v in sorted(buckets.items())
        }

    scores = {
        "task": "supergpqa",
        "trace": str(args.trace),
        "meta": str(args.meta),
        "n_total": n,
        "n_correct": n_correct,
        "n_failed": n_failed,
        "n_unparsed": n_unparsed,
        "accuracy": n_correct / n if n else 0.0,
        "accuracy_by_discipline": _summary(by_discipline),
        "accuracy_by_field": _summary(by_field),
        "accuracy_by_difficulty": _summary(by_difficulty),
        "accuracy_by_is_calculation": _summary(by_calculation),
    }
    if args.per_doc:
        scores["per_doc"] = records

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(
        f"[{args.trace.name}] n={n} correct={n_correct} failed={n_failed} "
        f"unparsed={n_unparsed} acc={scores['accuracy']:.4f} → {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
