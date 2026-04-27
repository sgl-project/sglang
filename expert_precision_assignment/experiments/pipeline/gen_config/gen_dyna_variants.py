"""Generate per-mc dynamic-dispatch variants.

For each mc in MC_LIST, reads `<cfg_dir>/mc{mc}/heter_config.json` (the
per-mc base assignment produced by `gen_heter_configs.py`) and emits the
requested variant files into `<cfg_dir>/mc{mc}/variants/`.

Variant naming convention — the prefix encodes the policy:
  - hess<N>  policy=hessian_weighted_routing_weights, hot ratio N%
             (importance × per-expert total routing weight)
             Requires the base heter_config.json to carry
             ``expert_importance_file`` (produced by
             ``gen_heter_configs.py --hessian``, the default).
  - hot<N>   policy=expert_load,                       hot ratio N%
  - thr<N>   policy=expert_batch,                      threshold=N

`VARIANTS` and `MC_LIST` env vars control what gets generated. The
recipe pipeline exports both via run_pipeline.sh, so the YAML is the
single source of truth at runtime.

Use --task=<name> to point at data/configs/<task>/ (per-task amortized
assignments from `gen_heter_configs.py --task`). Default is the flat
data/configs/ layout.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent.parent

# Env override (matches gen_heter_configs.py + run_sweep.sh). For RULER or
# other long-context tasks, use e.g. MC_LIST="1 2 4 8 16 32 64".
_DEFAULT_MC_LIST = [8, 16, 32, 64, 128, 256]
MC_LIST = (
    [int(x) for x in os.environ["MC_LIST"].split()]
    if os.environ.get("MC_LIST") else _DEFAULT_MC_LIST
)

# `VARIANTS` env var lets the recipe pipeline (run_pipeline.sh) drive
# exactly which variant files to emit. Default mirrors the historical
# six-point hess ladder so standalone calls still produce the canonical
# sweep matrix.
_DEFAULT_VARIANTS = ["hess0", "hess20", "hess40", "hess60", "hess80", "hess100"]
VARIANTS = (
    os.environ["VARIANTS"].split()
    if os.environ.get("VARIANTS") else _DEFAULT_VARIANTS
)


def _build_variant(base: dict, name: str) -> dict:
    """Return a deep-copied variant config for the given variant name.

    Variant names parse as a literal prefix + integer parameter:
      hess<N>  → hessian_weighted_routing_weights, hot ratio N%
      hot<N>   → expert_load,                      hot ratio N%
      thr<N>   → expert_batch,                     threshold=N
    """
    cfg = copy.deepcopy(base)

    if name.startswith("hess"):
        param = int(name[len("hess"):])
        if not 0 <= param <= 100:
            raise ValueError(f"hess hot ratio out of [0,100]: {name}")
        cfg["policy"] = "hessian_weighted_routing_weights"
        cfg.pop("policy_params", None)
        hot_r = param / 100.0
        for g in cfg["groups"]:
            if g["name"] == "cold":
                g["size_ratio"] = 1.0 - hot_r
            elif g["name"] == "hot":
                g["size_ratio"] = hot_r
        return cfg

    if name.startswith("hot"):
        param = int(name[len("hot"):])
        if not 0 <= param <= 100:
            raise ValueError(f"hot ratio out of [0,100]: {name}")
        cfg["policy"] = "expert_load"
        cfg.pop("policy_params", None)
        hot_r = param / 100.0
        for g in cfg["groups"]:
            if g["name"] == "cold":
                g["size_ratio"] = 1.0 - hot_r
            elif g["name"] == "hot":
                g["size_ratio"] = hot_r
        return cfg

    if name.startswith("thr"):
        param = int(name[len("thr"):])
        cfg["policy"] = "expert_batch"
        cfg["policy_params"] = {"threshold": param}
        for g in cfg["groups"]:
            g.pop("size_ratio", None)
        return cfg

    raise ValueError(
        f"Unknown variant prefix: {name!r}. Expected hess<N>/hot<N>/thr<N>."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task",
        help="When set, operate on data/configs/<task>/ instead of "
             "data/configs/. Must match the --task value passed to "
             "gen_heter_configs.py.",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="List the variant files that would be written per mc without "
             "creating any files. Useful for verifying which configs a run "
             "would touch before committing.",
    )
    args = ap.parse_args()

    cfg_dir = EXPERIMENTS_DIR / "data" / "configs"
    if args.task:
        cfg_dir = cfg_dir / args.task

    for mc in MC_LIST:
        base_path = cfg_dir / f"mc{mc}" / "heter_config.json"
        if not base_path.exists():
            raise FileNotFoundError(
                f"Missing base config: {base_path}\n"
                f"Run `python gen_heter_configs.py"
                f"{' --task ' + args.task if args.task else ''}` first."
            )
        with open(base_path) as f:
            base = json.load(f)

        variants_dir = cfg_dir / f"mc{mc}" / "variants"
        if not args.dry_run:
            variants_dir.mkdir(parents=True, exist_ok=True)

        planned: list[str] = []
        for variant in VARIANTS:
            cfg = _build_variant(base, variant)
            out = variants_dir / f"{variant}.json"
            planned.append(out.name)
            if not args.dry_run:
                with open(out, "w") as f:
                    json.dump(cfg, f, indent=2)

        verb = "DRY-RUN would write" if args.dry_run else "wrote"
        print(
            f"[mc{mc}] {verb} {len(planned)} variant(s) → {variants_dir}: "
            f"{', '.join(planned)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
