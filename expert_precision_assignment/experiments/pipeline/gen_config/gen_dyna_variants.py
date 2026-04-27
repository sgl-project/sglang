"""Generate per-mc dynamic-dispatch variants.

For each mc in MC_LIST, reads `<cfg_dir>/mc{mc}/heter_config.json` (the
per-mc base assignment produced by `gen_heter_configs.py`) and emits the
variant files into `<cfg_dir>/mc{mc}/variants/`.

Current primary variants:
  - hess0..hess100  (policy=hessian_weighted_routing_weights)
    score(E) = importance(E) × per-expert total routing weight.
    Six hot/cold splits (0/20/40/60/80/100%) drop straight into the
    existing sweep slot that hot0..hot100 used for expert_load.
    Requires the base heter_config.json to carry
    ``expert_importance_file`` (produced by ``gen_heter_configs.py
    --hessian``, the default).

The legacy hot0..hot100 / thr32..thr512 variant loops are preserved
below as ``if False:`` blocks — flip them back on to regenerate the full
sweep matrix.

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
HOT_PCTS = [0, 20, 40, 60, 80, 100]
THRESHOLDS = [32, 64, 128, 256, 512]


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
        planned: list[str] = []

        # --- Primary variants: hessian-weighted routing weights × hot% ---
        # Six hot/cold splits, same ratios as the legacy hot0..hot100 set,
        # with the new hessian-weighted scoring policy replacing expert_load.
        if not args.dry_run:
            variants_dir.mkdir(parents=True, exist_ok=True)
        for hot_pct in HOT_PCTS:
            cfg = copy.deepcopy(base)
            cfg["policy"] = "hessian_weighted_routing_weights"
            cfg.pop("policy_params", None)
            hot_r = hot_pct / 100.0
            cold_r = 1.0 - hot_r
            for g in cfg["groups"]:
                if g["name"] == "cold":
                    g["size_ratio"] = cold_r
                elif g["name"] == "hot":
                    g["size_ratio"] = hot_r
            out = variants_dir / f"hess{hot_pct}.json"
            planned.append(out.name)
            if not args.dry_run:
                with open(out, "w") as f:
                    json.dump(cfg, f, indent=2)

        # --- Legacy variants (disabled). Flip `if False` to `if True` to
        # regenerate the hot0..hot100 + thr32..thr512 matrix alongside
        # hess.json. Kept close to the primary path so re-enabling is a
        # single edit rather than a git archaeology expedition.
        if False:
            for hot_pct in HOT_PCTS:
                cfg = copy.deepcopy(base)
                cfg["policy"] = "expert_load"
                cfg.pop("policy_params", None)
                hot_r = hot_pct / 100.0
                cold_r = 1.0 - hot_r
                for g in cfg["groups"]:
                    if g["name"] == "cold":
                        g["size_ratio"] = cold_r
                    elif g["name"] == "hot":
                        g["size_ratio"] = hot_r
                out = variants_dir / f"hot{hot_pct}.json"
                planned.append(out.name)
                if not args.dry_run:
                    with open(out, "w") as f:
                        json.dump(cfg, f, indent=2)

            for thr in THRESHOLDS:
                cfg = copy.deepcopy(base)
                cfg["policy"] = "expert_batch"
                cfg["policy_params"] = {"threshold": thr}
                for g in cfg["groups"]:
                    g.pop("size_ratio", None)
                out = variants_dir / f"thr{thr}.json"
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
