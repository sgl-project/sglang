"""Generate sweep_dynamic_dispatch configs for the accuracy profile.

Two stages:
  1. Build the K=3072 base assignment via assign_experts.py --force_k 3072.
  2. Emit 11 variants on top of the base:
     - Sweep A: policy=random,        hot_pct in {0,20,40,60,80,100}    (6)
     - Sweep B: policy=expert_batch,  threshold in {32,64,128,256,512}  (5)

Output:
  configs/sweep_dynamic_dispatch/base/{int4_only_experts.json, heter_config.json, assignment_report.json}
  configs/sweep_dynamic_dispatch/sweep_a/hot{0,20,40,60,80,100}.json
  configs/sweep_dynamic_dispatch/sweep_b/thr{32,64,128,256,512}.json
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "expert_precision_assignment" / "accuracy_profile"
OUT_BASE = ROOT / "configs" / "sweep_dynamic_dispatch"
BASE_DIR = OUT_BASE / "base"
SWEEP_A_DIR = OUT_BASE / "sweep_a"
SWEEP_B_DIR = OUT_BASE / "sweep_b"

LAYER_SENS = REPO / "expert_precision_assignment" / "sensitivity" / "per_moe_layer" / "results" / "summary.json"
EXPERT_SENS = REPO / "expert_precision_assignment" / "sensitivity" / "per_expert" / "results" / "summary.json"

BF16_MODEL = "/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
INT4_CHECKPOINT = "/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441"

K_BASE = 48 * 64  # 3072
MAX_CONCURRENCY = 256
MAX_PROMPT_LEN = 2048
MAX_OUTPUT_LEN = 2048
GPU_VRAM_BYTES = 85899345920

HOT_PCTS = [0, 20, 40, 60, 80, 100]
THRESHOLDS = [32, 64, 128, 256, 512]


def _build_base() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "expert_precision_assignment.policy.static.assign_experts",
        "--layer_sensitivity", str(LAYER_SENS),
        "--expert_sensitivity", str(EXPERT_SENS),
        "--model_path", BF16_MODEL,
        "--int4_checkpoint", INT4_CHECKPOINT,
        "--max_concurrency", str(MAX_CONCURRENCY),
        "--max_prompt_len", str(MAX_PROMPT_LEN),
        "--max_output_len", str(MAX_OUTPUT_LEN),
        "--gpu_vram_bytes", str(GPU_VRAM_BYTES),
        "--force_k", str(K_BASE),
        "--out_dir", str(BASE_DIR),
    ]
    print(f"  + assign_experts --force_k {K_BASE} -> {BASE_DIR}")
    subprocess.run(cmd, check=True, cwd=REPO)


def _emit_variants() -> None:
    SWEEP_A_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_B_DIR.mkdir(parents=True, exist_ok=True)

    with open(BASE_DIR / "heter_config.json") as f:
        base = json.load(f)

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


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    _build_base()
    _emit_variants()
    print(f"\nDone. base + 6 sweep_a + 5 sweep_b configs under {OUT_BASE}")


if __name__ == "__main__":
    main()
