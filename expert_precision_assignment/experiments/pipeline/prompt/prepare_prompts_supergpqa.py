"""Prepare SuperGPQA prompts for the openai-dataset sweep path.

Reads m-a-p/SuperGPQA from HuggingFace (26,529 MCQs, 4–10 options per
question, across 13 disciplines / 72 fields / 285 subfields) and emits
two parallel JSONLs:

    prompts/supergpqa.jsonl       ← openai-chat format, bench_serving input
    prompts/supergpqa.meta.jsonl  ← per-row ground truth + metadata, same index

The two files have exactly the same number of lines in exactly the same
order, so scoring works by zipping them with the trace JSONL's
`generated_texts` array (bench_serving preserves input order).

Prompts go through a seeded shuffle before writing so that a
calibration pass on the first N prompts covers diverse disciplines.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROFILE_DIR = THIS_DIR.parent
if str(PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_DIR))

HF_REPO = "m-a-p/SuperGPQA"
PREFERRED_SPLITS = ("test", "train", "validation")

SYSTEM_INSTRUCTION = (
    "You are answering a graduate-level multiple-choice question. "
    "Think briefly, then output a single line in the form "
    "'Answer: X' where X is the letter of the correct option."
)

MAX_TOKENS = 8192
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
PRESENCE_PENALTY = 1.5
FREQUENCY_PENALTY = 0.5
ENABLE_THINKING = True

OPTION_LETTERS = "ABCDEFGHIJ"


def _resolve_split(repo, config, preferred, get_splits_fn):
    available = get_splits_fn(repo, config_name=config) if config else get_splits_fn(repo)
    for s in preferred:
        if s in available:
            return s
    if not available:
        raise RuntimeError(f"No splits in {repo} (config={config})")
    return available[0]


def _format_question(question: str, options: list[str]) -> str:
    lines = [question.rstrip(), ""]
    for letter, opt in zip(OPTION_LETTERS, options):
        lines.append(f"{letter}. {opt}")
    lines.append("")
    lines.append("Answer with a single letter in the form 'Answer: X'.")
    return "\n".join(lines)


def _load_recipe_defaults(recipe_path: Path | None) -> tuple[dict, str]:
    """Return (argparse defaults dict, output stem).

    Without --recipe: output goes to `prompts/supergpqa.jsonl` (backward compat).
    With --recipe: output goes to `prompts/<task>_<variant>.jsonl` and sampling
    defaults come from `recipe.sampling.*`, `--limit` from `recipe.dataset.limit`,
    `--seed` from `recipe.dataset.seed`.
    """
    if recipe_path is None:
        return {}, "supergpqa"
    from recipe import load_recipe, compound_name, sampling_kwargs
    recipe = load_recipe(recipe_path)
    if recipe["task"] != "supergpqa":
        raise SystemExit(
            f"Recipe task is '{recipe['task']}', expected 'supergpqa'. "
            f"Use prepare_prompts_{recipe['task']}.py instead."
        )
    sm = sampling_kwargs(recipe)
    ds = recipe.get("dataset") or {}
    return {
        "max_tokens": sm.get("max_tokens", MAX_TOKENS),
        "temperature": sm.get("temperature", TEMPERATURE),
        "top_p": sm.get("top_p", TOP_P),
        "top_k": sm.get("top_k", TOP_K),
        "presence_penalty": sm.get("presence_penalty", PRESENCE_PENALTY),
        "frequency_penalty": sm.get("frequency_penalty", FREQUENCY_PENALTY),
        "enable_thinking": sm.get("enable_thinking", ENABLE_THINKING),
        "limit": ds.get("limit", 0),
        "seed": ds.get("seed", 1234),
    }, compound_name(recipe)


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--recipe", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    dflt, out_stem = _load_recipe_defaults(pre_args.recipe)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, default=None,
                    help="Path to a recipe YAML. Seeds all sampling defaults "
                         "and switches the output file to "
                         "`prompts/<task>_<variant>.jsonl`.")
    ap.add_argument("--limit", type=int, default=dflt.get("limit", 0),
                    help="Cap on # of prompts (0 = all 26,529).")
    ap.add_argument("--seed", type=int, default=dflt.get("seed", 1234),
                    help="Shuffle seed for deterministic calib-subset coverage.")
    ap.add_argument("--split", default=None,
                    help="HF split name. Default: auto-detect (first of "
                         f"{PREFERRED_SPLITS} that exists).")
    ap.add_argument("--max_tokens", type=int,
                    default=dflt.get("max_tokens", MAX_TOKENS),
                    help="Output budget. 8192 fits Qwen3 thinking + answer.")
    ap.add_argument("--temperature", type=float,
                    default=dflt.get("temperature", TEMPERATURE),
                    help="Qwen3 thinking-mode recommended: 0.6 (NOT 0 — "
                         "greedy + thinking causes self-correction loops).")
    ap.add_argument("--top_p", type=float, default=dflt.get("top_p", TOP_P),
                    help="Qwen3 thinking-mode recommended: 0.95.")
    ap.add_argument("--top_k", type=int, default=dflt.get("top_k", TOP_K),
                    help="Qwen3 thinking-mode recommended: 20.")
    ap.add_argument("--presence_penalty", type=float,
                    default=dflt.get("presence_penalty", PRESENCE_PENALTY),
                    help="Qwen3 docs recommend 1.5 to break repetition loops.")
    ap.add_argument("--frequency_penalty", type=float,
                    default=dflt.get("frequency_penalty", FREQUENCY_PENALTY),
                    help="Scales-with-count repetition penalty.")
    ap.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction,
                    default=dflt.get("enable_thinking", ENABLE_THINKING),
                    help="Toggle Qwen3 thinking mode. Pass --no-enable_thinking "
                         "for non-thinking mode; in that case lower --max_tokens "
                         "(e.g. 1024). Scoring strips <think>...</think> either way.")
    args = ap.parse_args()

    out_jsonl = THIS_DIR / f"{out_stem}.jsonl"
    out_meta = THIS_DIR / f"{out_stem}.meta.jsonl"

    from datasets import get_dataset_split_names, load_dataset
    split = args.split or _resolve_split(HF_REPO, None, PREFERRED_SPLITS,
                                         get_dataset_split_names)
    print(f"Using split: {split}")
    ds = load_dataset(HF_REPO, split=split)

    rows = list(range(len(ds)))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    n_prompts = 0
    n_meta = 0
    with open(out_jsonl, "w") as fp, open(out_meta, "w") as fm:
        for idx in rows:
            doc = ds[idx]
            question = doc["question"]
            options = list(doc["options"])
            answer_letter = doc["answer_letter"]

            if answer_letter not in OPTION_LETTERS[: len(options)]:
                continue

            user_msg = _format_question(question, options)

            prompt_record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "presence_penalty": args.presence_penalty,
                "frequency_penalty": args.frequency_penalty,
                "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
            }
            meta_record = {
                "doc_id": doc.get("uuid") or f"row_{idx}",
                "row_index": idx,
                "answer_letter": answer_letter,
                "answer_text": doc.get("answer"),
                "n_options": len(options),
                "discipline": doc.get("discipline"),
                "field": doc.get("field"),
                "subfield": doc.get("subfield"),
                "difficulty": doc.get("difficulty"),
                "is_calculation": doc.get("is_calculation"),
            }
            fp.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
            fm.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
            n_prompts += 1
            n_meta += 1

    assert n_prompts == n_meta
    print(f"Wrote {n_prompts} prompts → {out_jsonl}")
    print(f"Wrote {n_meta} metadata rows → {out_meta}")
    print(
        f"  max_tokens={args.max_tokens} temperature={args.temperature} "
        f"top_p={args.top_p} top_k={args.top_k} "
        f"pres_pen={args.presence_penalty} freq_pen={args.frequency_penalty} "
        f"thinking={args.enable_thinking} seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
