"""Prepare LiveCodeBench v6 prompts for the openai-dataset sweep path.

Reads livecodebench/code_generation_lite at release_v6 (1,055 problems
from LeetCode, Codeforces, AtCoder; Apr 2025) and emits THREE files:

    prompt/livecodebench_v6.jsonl                ← openai-chat prompts, sweep input (~2 MB)
    prompt/livecodebench_v6.meta.jsonl           ← light metadata, join key question_id (~500 KB):
                                                   question_id, platform, difficulty, contest_date,
                                                   starter_code, fn_name, public_test_cases
    prompt/livecodebench_v6.private_tests.pkl    ← pickle: {question_id: b64z_string}
                                                   (~4 GB; one problem alone is 92 MB compressed)

The split is required because the private-test-case blobs are base64+zlib+pickle-
encoded and average 4 MB per problem; inlining them would make meta.jsonl
unusable (4+ GB, one 92 MB row).  Keeping them in a separate sidecar lets
the meta file stay small and fast-to-iterate, while the scorer loads the
pickle once at startup and looks up by question_id.

Scoring: ../scoring/score_traces_lcb_v6.py reads all three files, extracts
the Python code from each response, and runs the vendored lcb_runner
checker per problem.  See that script's header for vendoring instructions.

Sampling: Qwen3 thinking-mode recipe (T=0.6, top_p=0.95, top_k=20,
presence_penalty=1.5, max_tokens=8192, chat_template_kwargs.enable_thinking=True).
Greedy (T=0) is avoided because it causes repetition/self-correction loops
in Qwen3 thinking mode.  The scorer strips <think>...</think> blocks before
extracting the ```python``` fenced code block.  Override via prep flags.

Contamination filtering: pass --start_date YYYY-MM-DD / --end_date
YYYY-MM-DD to restrict to a contest-date window.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

HF_REPO = "livecodebench/code_generation_lite"
HF_CONFIG = "release_v6"

LCB_CONFIG_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl",
                   "test5.jsonl", "test6.jsonl"],
}

MAX_TOKENS = 8192
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
PRESENCE_PENALTY = 1.5
FREQUENCY_PENALTY = 0.5
ENABLE_THINKING = True

SYSTEM_INSTRUCTION = (
    "You are an expert Python programmer. "
    "You will be given a competitive-programming problem. "
    "Write a single self-contained Python solution. "
    "Return ONLY the final code inside a ```python ... ``` fenced block. "
    "Do not include any prose outside the code block."
)


def _load_lcb_rows(repo: str, config: str):
    """Stream LCB rows for a given config by downloading its jsonl files
    directly — avoids the `datasets` library's block on dataset loader
    scripts (datasets>=3.0), which is what `code_generation_lite.py` is.
    """
    from huggingface_hub import hf_hub_download
    if config not in LCB_CONFIG_FILES:
        raise ValueError(
            f"Unknown LCB config '{config}'. "
            f"Valid: {sorted(LCB_CONFIG_FILES)}"
        )
    for fname in LCB_CONFIG_FILES[config]:
        path = hf_hub_download(repo, fname, repo_type="dataset")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _format_problem(doc: dict) -> str:
    parts = [doc.get("question_content", "").rstrip()]
    starter = doc.get("starter_code")
    if starter:
        parts.append("")
        parts.append("Starter code:")
        parts.append("```python")
        parts.append(starter.rstrip())
        parts.append("```")
    parts.append("")
    parts.append(
        "Write the complete solution as a ```python``` fenced code block."
    )
    return "\n".join(parts)


def _extract_fn_name(doc: dict) -> str | None:
    meta = doc.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    return meta.get("func_name") if isinstance(meta, dict) else None


def _within_date_window(contest_date: str, start: str | None, end: str | None) -> bool:
    if not contest_date:
        return True
    d = contest_date[:10]
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True


def _load_recipe_defaults(recipe_path: Path | None) -> tuple[dict, str]:
    """Return (argparse defaults dict, output stem).

    Without --recipe: output → `prompt/livecodebench_v6.*` (backward compat).
    With --recipe:    output → `prompt/<task>_<variant>.*` and sampling
                      defaults come from recipe.sampling.* / dataset.*.
    """
    if recipe_path is None:
        return {}, "livecodebench_v6"
    from recipe import load_recipe, compound_name, sampling_kwargs
    recipe = load_recipe(recipe_path)
    if recipe["task"] != "livecodebench_v6":
        raise SystemExit(
            f"Recipe task is '{recipe['task']}', expected 'livecodebench_v6'. "
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
        "start_date": ds.get("start_date"),
        "end_date": ds.get("end_date"),
    }, compound_name(recipe)


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--recipe", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    dflt, out_stem = _load_recipe_defaults(pre_args.recipe)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, default=None,
                    help="Path to a recipe YAML. Seeds all sampling defaults "
                         "and switches output files to "
                         "`prompt/<task>_<variant>.{jsonl,meta.jsonl,private_tests.pkl}`.")
    ap.add_argument("--limit", type=int, default=dflt.get("limit", 0),
                    help="Cap on # of prompts (0 = all ~1,055).")
    ap.add_argument("--start_date", default=dflt.get("start_date"),
                    help="Inclusive lower bound on contest_date (YYYY-MM-DD).")
    ap.add_argument("--end_date", default=dflt.get("end_date"),
                    help="Inclusive upper bound on contest_date (YYYY-MM-DD).")
    ap.add_argument("--config", default=HF_CONFIG,
                    help=f"LCB release config (default {HF_CONFIG}). "
                         f"Valid: {sorted(LCB_CONFIG_FILES)}.")
    ap.add_argument("--max_tokens", type=int,
                    default=dflt.get("max_tokens", MAX_TOKENS),
                    help="Output budget. 8192 fits Qwen3 thinking + code.")
    ap.add_argument("--temperature", type=float,
                    default=dflt.get("temperature", TEMPERATURE),
                    help="Qwen3 thinking-mode recommended: 0.6 (NOT 0).")
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
                         "(e.g. 2048). Scoring strips <think>...</think> before "
                         "extracting the final ```python``` block either way.")
    args = ap.parse_args()

    out_jsonl = THIS_DIR / f"{out_stem}.jsonl"
    out_meta = THIS_DIR / f"{out_stem}.meta.jsonl"
    out_private_tests = THIS_DIR / f"{out_stem}.private_tests.pkl"

    print(f"Loading {HF_REPO} config={args.config} "
          f"({len(LCB_CONFIG_FILES[args.config])} jsonl files)")
    ds = _load_lcb_rows(HF_REPO, args.config)

    n_kept = 0
    private_tests: dict[str, str] = {}
    with open(out_jsonl, "w") as fp, open(out_meta, "w") as fm:
        for i, doc in enumerate(ds):
            if args.limit > 0 and n_kept >= args.limit:
                break
            contest_date = str(doc.get("contest_date") or "")
            if not _within_date_window(contest_date, args.start_date, args.end_date):
                continue

            user_msg = _format_problem(doc)
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

            public_tests_raw = doc.get("public_test_cases")
            if isinstance(public_tests_raw, str):
                try:
                    public_tests = json.loads(public_tests_raw)
                except json.JSONDecodeError:
                    public_tests = []
            else:
                public_tests = list(public_tests_raw or [])

            question_id = doc.get("question_id") or f"row_{i}"
            meta_record = {
                "row_index": i,
                "question_id": question_id,
                "contest_id": doc.get("contest_id"),
                "platform": doc.get("platform"),
                "difficulty": doc.get("difficulty"),
                "contest_date": contest_date,
                "starter_code": doc.get("starter_code") or "",
                "fn_name": _extract_fn_name(doc),
                "public_test_cases": public_tests,
            }
            private_b64z = doc.get("private_test_cases")
            if private_b64z:
                private_tests[question_id] = private_b64z

            fp.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
            fm.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
            n_kept += 1

    with open(out_private_tests, "wb") as ftests:
        pickle.dump(private_tests, ftests, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {n_kept} prompts → {out_jsonl}")
    print(f"Wrote {n_kept} metadata rows → {out_meta}")
    print(f"Wrote {len(private_tests)} private-test blobs → {out_private_tests} "
          f"({out_private_tests.stat().st_size / (1024*1024):.1f} MB)")
    print(
        f"  max_tokens={args.max_tokens} temperature={args.temperature} "
        f"top_p={args.top_p} top_k={args.top_k} "
        f"pres_pen={args.presence_penalty} freq_pen={args.frequency_penalty} "
        f"thinking={args.enable_thinking} "
        f"date_window=[{args.start_date or '-inf'},{args.end_date or '+inf'}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
