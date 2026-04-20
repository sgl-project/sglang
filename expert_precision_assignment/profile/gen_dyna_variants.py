"""Generate per-mc dynamic-dispatch variants.

For each mc in MC_LIST, reads `<cfg_dir>/mc{mc}/heter_config.json` (the
per-mc base assignment produced by `gen_heter_configs.py`) and emits 11
dispatch variants into `<cfg_dir>/mc{mc}/variants/`:

  - hot0..hot100  (policy=random, 6 hot_pcts {0,20,40,60,80,100})
  - thr32..thr512 (policy=expert_batch, 5 thresholds {32,64,128,256,512})

All variants share the same int4-only / heter experts from the base; only
`policy`, `policy_params`, and group `size_ratio` differ.

Use --task=<name> to point at configs/<task>/ (per-task amortized
assignments from `gen_heter_configs.py --task`). Default is the flat
configs/ layout.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

MC_LIST = [8, 16, 32, 64, 128, 256]
HOT_PCTS = [0, 20, 40, 60, 80, 100]
THRESHOLDS = [32, 64, 128, 256, 512]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task",
        help="When set, operate on configs/<task>/ instead of flat configs/. "
             "Must match the --task value passed to gen_heter_configs.py.",
    )
    args = ap.parse_args()

    cfg_dir = THIS_DIR / "configs"
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
        variants_dir.mkdir(parents=True, exist_ok=True)

        for hot_pct in HOT_PCTS:
            cfg = copy.deepcopy(base)
            cfg["policy"] = "random"
            cfg.pop("policy_params", None)
            hot_r = hot_pct / 100.0
            cold_r = 1.0 - hot_r
            for g in cfg["groups"]:
                if g["name"] == "cold":
                    g["size_ratio"] = cold_r
                elif g["name"] == "hot":
                    g["size_ratio"] = hot_r
            out = variants_dir / f"hot{hot_pct}.json"
            with open(out, "w") as f:
                json.dump(cfg, f, indent=2)

        for thr in THRESHOLDS:
            cfg = copy.deepcopy(base)
            cfg["policy"] = "expert_batch"
            cfg["policy_params"] = {"threshold": thr}
            for g in cfg["groups"]:
                g.pop("size_ratio", None)
            out = variants_dir / f"thr{thr}.json"
            with open(out, "w") as f:
                json.dump(cfg, f, indent=2)

        print(
            f"[mc{mc}] wrote {len(HOT_PCTS)} hot_pct + "
            f"{len(THRESHOLDS)} thresh variants → {variants_dir}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
