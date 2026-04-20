"""Generate sweep variant heter configs.

Reuses the base int4_only_experts.json (which experts are int4-only vs heter)
and just mutates the dispatch policy + size_ratio / threshold.

Sweep A (hot_pct, policy=random):   hot ∈ {0,20,40,60,80,100}
Sweep B (threshold, policy=expert_batch): threshold ∈ {32,64,128,256,512}
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "configs" / "base"
SWEEP_A_DIR = Path(__file__).resolve().parent / "configs" / "sweep_a"
SWEEP_B_DIR = Path(__file__).resolve().parent / "configs" / "sweep_b"
SWEEP_C_DIR = Path(__file__).resolve().parent / "configs" / "sweep_c"

HOT_PCTS = [0, 20, 40, 60, 80, 100]
HOT_PCTS_GRANULAR = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
THRESHOLDS = [32, 64, 128, 256, 512]


def main() -> None:
    with open(BASE_DIR / "heter_config.json") as f:
        base = json.load(f)

    SWEEP_A_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_B_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_C_DIR.mkdir(parents=True, exist_ok=True)

    for hot_pct in HOT_PCTS:
        cfg = copy.deepcopy(base)
        cfg["policy"] = "random"
        cfg.pop("policy_params", None)
        cold_ratio = 1.0 - hot_pct / 100.0
        hot_ratio = hot_pct / 100.0
        for g in cfg["groups"]:
            if g["name"] == "cold":
                g["size_ratio"] = cold_ratio
            elif g["name"] == "hot":
                g["size_ratio"] = hot_ratio
        out = SWEEP_A_DIR / f"hot{hot_pct}.json"
        with open(out, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  wrote {out}")

    for hot_pct in HOT_PCTS_GRANULAR:
        cfg = copy.deepcopy(base)
        cfg["policy"] = "expert_load"
        cfg.pop("policy_params", None)
        cold_ratio = 1.0 - hot_pct / 100.0
        hot_ratio = hot_pct / 100.0
        for g in cfg["groups"]:
            if g["name"] == "cold":
                g["size_ratio"] = cold_ratio
            elif g["name"] == "hot":
                g["size_ratio"] = hot_ratio
        out = SWEEP_C_DIR / f"hot{hot_pct}.json"
        with open(out, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  wrote {out}")

    for thr in THRESHOLDS:
        cfg = copy.deepcopy(base)
        cfg["policy"] = "expert_batch"
        cfg["policy_params"] = {"threshold": thr}
        for g in cfg["groups"]:
            g.pop("size_ratio", None)
        out = SWEEP_B_DIR / f"thr{thr}.json"
        with open(out, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
