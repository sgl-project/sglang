"""Prepare AllenAI IFBench prompts for the openai-dataset sweep path.

Reads allenai/IFBench_test from HuggingFace (300 test prompts with 58 OOD
instruction-following constraints, distinct from Google's IFEval) and
emits two parallel JSONLs:

    prompts/ifbench.jsonl       ← openai-chat format, bench_serving input
    prompts/ifbench.meta.jsonl  ← per-row prompt + constraint list + kwargs

Scoring is offline via scoring/score_traces_ifbench.py, which needs the
vendored AllenAI verifiers at scoring/vendored/ifbench/ — see that
script's header for the one-shot `git clone` command.

No shuffle: IFBench is only 300 prompts, calibration can just use the
first N.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROFILE_DIR = THIS_DIR.parent
if str(PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_DIR))

HF_REPO = "allenai/IFBench_test"
PREFERRED_SPLITS = ("test", "train", "validation")

MAX_TOKENS = 8192
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
PRESENCE_PENALTY = 1.5
FREQUENCY_PENALTY = 0.5
ENABLE_THINKING = True


def _resolve_split(repo, config, preferred, get_splits_fn):
    available = get_splits_fn(repo, config_name=config) if config else get_splits_fn(repo)
    for s in preferred:
        if s in available:
            return s
    if not available:
        raise RuntimeError(f"No splits in {repo} (config={config})")
    return available[0]


def _load_recipe_defaults(recipe_path: Path | None) -> tuple[dict, str]:
    """Return (argparse defaults dict, output stem).

    If no --recipe is given, defaults come from this script's constants and
    output goes to `prompts/ifbench.jsonl` (backward compat).  With a recipe,
    defaults come from `recipe.sampling.*` and output goes to
    `prompts/<task>_<variant>.jsonl`.
    """
    if recipe_path is None:
        return {}, "ifbench"
    from recipe import load_recipe, compound_name, sampling_kwargs
    recipe = load_recipe(recipe_path)
    if recipe["task"] != "ifbench":
        raise SystemExit(
            f"Recipe task is '{recipe['task']}', expected 'ifbench'. "
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
    }, compound_name(recipe)


def main() -> int:
    # Two-pass argparse so --recipe can seed the other defaults.
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
                    help="Cap on # of prompts (0 = all 300).")
    ap.add_argument("--split", default=None,
                    help="HF split name. Default: auto-detect (first of "
                         f"{PREFERRED_SPLITS} that exists).")
    ap.add_argument("--max_tokens", type=int,
                    default=dflt.get("max_tokens", MAX_TOKENS),
                    help="Output budget. 8192 fits Qwen3 thinking + "
                         "answer for IFBench prompts.")
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
                    help="Qwen3 docs recommend 1.5 to break repetition loops. "
                         "Binary: 'has token X appeared at all' — insufficient "
                         "once a loop is already running.")
    ap.add_argument("--frequency_penalty", type=float,
                    default=dflt.get("frequency_penalty", FREQUENCY_PENALTY),
                    help="Scales-with-count repetition penalty. 0.5 catches "
                         "Qwen3 thinking-mode paragraph-level rewrite loops "
                         "that presence_penalty doesn't touch.")
    ap.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction,
                    default=dflt.get("enable_thinking", ENABLE_THINKING),
                    help="Toggle Qwen3 thinking mode. Pass --no-enable_thinking "
                         "to disable. In non-thinking mode, lower --max_tokens "
                         "(e.g. 2048). Scoring strips <think>...</think> either way.")
    args = ap.parse_args()

    out_jsonl = THIS_DIR / f"{out_stem}.jsonl"
    out_meta = THIS_DIR / f"{out_stem}.meta.jsonl"

    from datasets import get_dataset_split_names, load_dataset
    split = args.split or _resolve_split(HF_REPO, None, PREFERRED_SPLITS,
                                         get_dataset_split_names)
    print(f"Using split: {split}")
    ds = load_dataset(HF_REPO, split=split)

    n = len(ds) if args.limit <= 0 else min(len(ds), args.limit)

    with open(out_jsonl, "w") as fp, open(out_meta, "w") as fm:
        for i in range(n):
            doc = ds[i]
            prompt = doc["prompt"]

            prompt_record = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "presence_penalty": args.presence_penalty,
                "frequency_penalty": args.frequency_penalty,
                "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
            }
            meta_record = {
                "row_index": i,
                "key": doc.get("key"),
                "prompt": prompt,
                "instruction_id_list": list(doc.get("instruction_id_list") or []),
                "kwargs": list(doc.get("kwargs") or []),
            }
            fp.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
            fm.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

    print(f"Wrote {n} prompts → {out_jsonl}")
    print(f"Wrote {n} metadata rows → {out_meta}")
    print(
        f"  max_tokens={args.max_tokens} temperature={args.temperature} "
        f"top_p={args.top_p} top_k={args.top_k} "
        f"pres_pen={args.presence_penalty} freq_pen={args.frequency_penalty} "
        f"thinking={args.enable_thinking}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
