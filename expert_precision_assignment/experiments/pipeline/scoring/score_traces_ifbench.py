"""Score an IFBench trace JSONL against its meta sidecar.

Inputs (paths relative to experiments/):
    --trace    data/results/ifbench/mc{mc}_{variant}.jsonl  (bench_serving output)
    --meta     pipeline/prompt/ifbench.meta.jsonl           (from prepare_prompts_ifbench.py)
    --vendored pipeline/scoring/vendored/ifbench            (AllenAI verifiers)

Matches trace.generated_texts[i] against meta[i] by index (bench_serving
preserves input order).  Calls AllenAI's official verifiers to compute
the four standard IFBench metrics:

    prompt_level_strict_acc   fraction of prompts where ALL constraints pass
    inst_level_strict_acc     fraction of individual constraints that pass
    prompt_level_loose_acc    strict, but with 8 response transformations tried
    inst_level_loose_acc      per-instruction variant of loose

"loose" scoring tries the original response plus 7 transformations
(strip first line, strip last line, strip both, strip `*` markers, and
combinations) — a pass in any variant counts as a pass.

VENDORING (one-time, before running this script; paths relative to experiments/):

    mkdir -p pipeline/scoring/vendored
    cd pipeline/scoring/vendored
    git clone https://github.com/allenai/IFBench.git ifbench_repo
    cd ifbench_repo && git checkout cb932e352a505306ad0115272211df14bb8f628f
    cd ../ && ln -sfn ifbench_repo/instructions.py .
    ln -sfn ifbench_repo/instructions_registry.py .
    ln -sfn ifbench_repo/instructions_util.py .
    ln -sfn ifbench_repo/evaluation_lib.py .
    cd ../../../..
    pip install absl-py langdetect nltk immutabledict spacy emoji syllapy
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    python -m spacy download en_core_web_sm

Then run:
    python pipeline/scoring/score_traces_ifbench.py \
        --trace data/results/ifbench/mc128_thr128.jsonl \
        --meta pipeline/prompt/ifbench.meta.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_VENDORED = THIS_DIR / "vendored" / "ifbench"

_RE_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(response: str) -> str:
    """Remove Qwen3 <think>...</think> blocks before constraint verification.

    Qwen3 in thinking mode emits the model's reasoning between <think> and
    </think> tags before the actual answer.  AllenAI's IFBench verifiers
    measure constraints on the WHOLE response, so leaving the reasoning in
    would falsify "exactly N words", "ends with phrase X", "no markdown",
    etc.  We strip the reasoning and only score the post-think answer.

    If the response has an open <think> with no matching </think> (because
    max_tokens cut off mid-reasoning), drop everything up to a sane fallback
    so we don't accidentally score the partial reasoning.
    """
    if not response:
        return ""
    cleaned = _RE_THINK_BLOCK.sub("", response)
    if "<think>" in cleaned and "</think>" not in cleaned:
        return ""
    return cleaned.strip()


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


def _import_ifbench(vendored_dir: Path):
    if not vendored_dir.exists():
        raise FileNotFoundError(
            f"Vendored IFBench dir not found: {vendored_dir}\n"
            f"See the header of this script for vendoring instructions."
        )
    sys.path.insert(0, str(vendored_dir))
    try:
        from evaluation_lib import (
            InputExample,
            test_instruction_following_strict,
            test_instruction_following_loose,
        )
    except ImportError as e:
        raise ImportError(
            f"Failed to import vendored IFBench modules from {vendored_dir}. "
            f"Ensure instructions.py, instructions_registry.py, "
            f"instructions_util.py, evaluation_lib.py are present and all "
            f"pip dependencies are installed. Original error: {e}"
        ) from e
    return InputExample, test_instruction_following_strict, test_instruction_following_loose


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--vendored", type=Path, default=DEFAULT_VENDORED,
                    help="Path to the AllenAI IFBench vendored Python modules.")
    ap.add_argument("--out", type=Path,
                    help="Defaults to the trace file with .jsonl → .scores.json.")
    ap.add_argument("--per-doc", action="store_true",
                    help="Include per-document pass/fail records in output.")
    args = ap.parse_args()

    out = args.out or args.trace.with_suffix(".scores.json")

    InputExample, score_strict, score_loose = _import_ifbench(args.vendored)

    trace = _load_trace(args.trace)
    meta = _load_meta(args.meta)

    generated = trace.get("generated_texts", [])
    errors = trace.get("errors", [None] * len(generated))

    n_trace = len(generated)
    n_meta = len(meta)
    n = min(n_trace, n_meta)
    if n_trace != n_meta:
        print(
            f"WARNING: trace has {n_trace} rows, meta has {n_meta}. "
            f"Scoring the first {n} rows only."
        )

    prompt_strict_pass = 0
    prompt_loose_pass = 0
    inst_strict_total = 0
    inst_loose_total = 0
    inst_strict_pass = 0
    inst_loose_pass = 0
    n_failed = 0
    records: list[dict] = []

    for i in range(n):
        m = meta[i]
        resp = generated[i] or ""
        err = errors[i] if i < len(errors) else None
        instr_ids = list(m.get("instruction_id_list") or [])
        if not instr_ids:
            continue

        if err:
            n_failed += 1
            inst_strict_total += len(instr_ids)
            inst_loose_total += len(instr_ids)
            records.append({
                "row_index": m.get("row_index"),
                "key": m.get("key"),
                "error": err,
                "prompt_strict": False,
                "prompt_loose": False,
                "inst_strict": [False] * len(instr_ids),
                "inst_loose": [False] * len(instr_ids),
            })
            continue

        inp = InputExample(
            key=m.get("key"),
            instruction_id_list=instr_ids,
            prompt=m["prompt"],
            kwargs=list(m.get("kwargs") or [{} for _ in instr_ids]),
        )
        p2r = {m["prompt"]: strip_thinking(resp)}

        try:
            out_strict = score_strict(inp, p2r)
            out_loose = score_loose(inp, p2r)
        except Exception as e:
            n_failed += 1
            inst_strict_total += len(instr_ids)
            inst_loose_total += len(instr_ids)
            records.append({
                "row_index": m.get("row_index"),
                "key": m.get("key"),
                "error": f"scorer_exception: {type(e).__name__}: {e}",
                "prompt_strict": False,
                "prompt_loose": False,
                "inst_strict": [False] * len(instr_ids),
                "inst_loose": [False] * len(instr_ids),
            })
            continue

        s_list = list(out_strict.follow_instruction_list)
        l_list = list(out_loose.follow_instruction_list)
        p_strict = bool(out_strict.follow_all_instructions)
        p_loose = bool(out_loose.follow_all_instructions)

        inst_strict_total += len(s_list)
        inst_loose_total += len(l_list)
        inst_strict_pass += sum(1 for b in s_list if b)
        inst_loose_pass += sum(1 for b in l_list if b)
        prompt_strict_pass += int(p_strict)
        prompt_loose_pass += int(p_loose)

        records.append({
            "row_index": m.get("row_index"),
            "key": m.get("key"),
            "error": None,
            "prompt_strict": p_strict,
            "prompt_loose": p_loose,
            "inst_strict": [bool(b) for b in s_list],
            "inst_loose": [bool(b) for b in l_list],
        })

    scores: dict[str, Any] = {
        "task": "ifbench",
        "trace": str(args.trace),
        "meta": str(args.meta),
        "n_total": n,
        "n_failed": n_failed,
        "prompt_level_strict_acc": (prompt_strict_pass / n) if n else 0.0,
        "prompt_level_loose_acc": (prompt_loose_pass / n) if n else 0.0,
        "inst_level_strict_acc": (
            (inst_strict_pass / inst_strict_total) if inst_strict_total else 0.0
        ),
        "inst_level_loose_acc": (
            (inst_loose_pass / inst_loose_total) if inst_loose_total else 0.0
        ),
        "counts": {
            "prompts_total": n,
            "prompts_strict_pass": prompt_strict_pass,
            "prompts_loose_pass": prompt_loose_pass,
            "instructions_total_strict": inst_strict_total,
            "instructions_total_loose": inst_loose_total,
            "instructions_strict_pass": inst_strict_pass,
            "instructions_loose_pass": inst_loose_pass,
        },
    }
    if args.per_doc:
        scores["per_doc"] = records

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(
        f"[{args.trace.name}] n={n} failed={n_failed} "
        f"prompt_strict={scores['prompt_level_strict_acc']:.4f} "
        f"prompt_loose={scores['prompt_level_loose_acc']:.4f} "
        f"inst_strict={scores['inst_level_strict_acc']:.4f} "
        f"inst_loose={scores['inst_level_loose_acc']:.4f} → {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
