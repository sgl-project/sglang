"""Shared recipe loader for the end-to-end eval pipeline.

A *recipe* is a YAML file under `recipes/` describing one eval run:
 - which dataset (`task: ifbench`)
 - which variant label (`variant: nothink`) — artifacts get `<task>_<variant>` prefix
 - sampling knobs (thinking on/off, max_tokens, temperature, penalties)
 - calibration knobs (num_prompts, mc, gpu, port)
 - sweep knobs (mc_list, variants, gpus, num_prompts)

Pipeline stages (`prepare_prompts_*.py`, `run_calib.sh`, `gen_all.py`,
`run_sweep.sh`, `score_traces_*.py`, `collect_results.py`) all read the
same recipe. CLI flags still override recipe values.

Python API:
    from recipe import load_recipe, compound_name, sampling_kwargs
    recipe = load_recipe("recipes/ifbench_nothink.yaml")
    name = compound_name(recipe)          # "ifbench_nothink"
    base = recipe["task"]                 # "ifbench"
    kw = sampling_kwargs(recipe)          # {"max_tokens":2048, ...}

Shell API:
    eval "$(python recipe.py recipes/ifbench_nothink.yaml env)"
    echo $RECIPE_NAME $RECIPE_TASK $RECIPE_VARIANT
    python recipe.py recipes/ifbench_nothink.yaml get sampling.max_tokens
    python recipe.py recipes/ifbench_nothink.yaml name
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

_DEFAULTS: dict[str, Any] = {
    "variant": "default",
    "runtime": {
        # Interpreter + env that every stage uses.  Override per recipe to
        # target a different venv / conda env.  Explicit $PYTHON in the shell
        # still wins; after that comes `runtime.python`, then this default.
        "python": "/data/junzhou/.venv-bfcl/bin/python",
        # BF16 baseline checkpoint used by run_calib.sh and run_sweep.sh.
        # Pinning a specific snapshot hash avoids silent model drift when the
        # HF cache re-downloads.
        "model_path": (
            "/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/"
            "snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
        ),
        "host": "127.0.0.1",
    },
    "dataset": {"limit": 0},
    "sampling": {
        "enable_thinking": True,
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "frequency_penalty": 0.5,
    },
    "calibration": {
        "num_prompts": 256,
        "mc": 128,
        "gpu": 4,
        "port": 31304,
        "nccl_port": 41304,
    },
    "sweep": {
        "mc_list": [8, 16, 32, 64, 128, 256],
        "variants": [
            "hot0", "hot20", "hot40", "hot60", "hot80", "hot100",
            "thr32", "thr64", "thr128", "thr256", "thr512",
        ],
        "gpus": [4, 5, 6, 7],
        "num_prompts": 0,
    },
    "scoring": {
        # When false, `run_pipeline.sh --stage score` becomes a no-op — the
        # sweep's .jsonl traces are left without .scores.json sidecars, and
        # `collect` still emits a summary.csv (perf columns only, no accuracy
        # columns).  Useful when: (a) smoke-testing perf only, (b) the scorer
        # needs vendored deps that aren't set up, (c) LCB and you don't want
        # to execute model-generated code on this machine.
        "enabled": True,
        # Extra CLI args forwarded to each `score_traces_<task>.py` invocation
        # (e.g. ["--per-doc"] to include per-row records, or LCB's --timeout).
        "extra_args": [],
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_recipe(path: str | Path) -> dict:
    """Load a YAML recipe, merged over built-in defaults.  Requires `task`."""
    import yaml  # lazy: bench_serving stages don't need yaml at import time
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    if not isinstance(user, dict):
        raise ValueError(f"Recipe {path} must be a YAML mapping.")
    if not user.get("task"):
        raise ValueError(f"Recipe {path} is missing required `task` field.")
    merged = _deep_merge(_DEFAULTS, user)
    return merged


def compound_name(recipe: dict) -> str:
    """Return `<task>_<variant>` — the prefix used for all artifacts."""
    task = recipe["task"]
    variant = recipe.get("variant") or "default"
    return f"{task}_{variant}"


def sampling_kwargs(recipe: dict) -> dict[str, Any]:
    """Per-request sampling params that prep scripts write into prompts.jsonl."""
    return dict(recipe.get("sampling") or {})


def _get_dotted(recipe: dict, key: str) -> Any:
    cur: Any = recipe
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


def _fmt_env_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return str(v)


def _emit_env(recipe: dict) -> str:
    """Emit `export VAR=val` lines a shell can `eval`.

    Flattens nested dicts with `__` (double underscore) separators:
        sampling.max_tokens → RECIPE_SAMPLING__MAX_TOKENS
    Plus convenience vars RECIPE_NAME (task_variant), RECIPE_TASK, RECIPE_VARIANT.
    """
    lines: list[str] = []
    lines.append(f"export RECIPE_NAME={shlex.quote(compound_name(recipe))}")
    lines.append(f"export RECIPE_TASK={shlex.quote(str(recipe['task']))}")
    lines.append(
        f"export RECIPE_VARIANT={shlex.quote(str(recipe.get('variant') or 'default'))}"
    )

    def walk(prefix: str, obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}__{k.upper()}" if prefix else f"RECIPE_{k.upper()}", v)
        else:
            lines.append(f"export {prefix}={shlex.quote(_fmt_env_value(obj))}")

    for k, v in recipe.items():
        if k in {"task", "variant"}:  # already emitted above
            continue
        walk(f"RECIPE_{k.upper()}", v)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recipe", type=Path, help="Path to a recipe YAML.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("name", help="Print the compound name `<task>_<variant>`.")
    sub.add_parser("task", help="Print just the base `task` field.")
    sub.add_parser("env", help="Emit `export VAR=...` lines for shells.")
    sub.add_parser("dump", help="Print the merged-with-defaults recipe as JSON.")

    g = sub.add_parser("get", help="Print one dotted field, e.g. `sampling.max_tokens`.")
    g.add_argument("key")

    args = ap.parse_args()
    recipe = load_recipe(args.recipe)

    if args.cmd == "name":
        print(compound_name(recipe))
    elif args.cmd == "task":
        print(recipe["task"])
    elif args.cmd == "env":
        print(_emit_env(recipe))
    elif args.cmd == "dump":
        print(json.dumps(recipe, indent=2, sort_keys=True))
    elif args.cmd == "get":
        val = _get_dotted(recipe, args.key)
        if isinstance(val, (list, tuple)):
            print(" ".join(str(x) for x in val))
        elif isinstance(val, bool):
            print("1" if val else "0")
        else:
            print(val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
