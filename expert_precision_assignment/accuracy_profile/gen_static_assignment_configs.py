"""Generate sweep_static_assignment configs for the accuracy profile.

For N in {0, 8, 16, 24, 32, 40, 48, 56, 64} (so K = 48*N), invoke
expert_precision_assignment.policy.static.assign_experts with --force_k K
and patch the emitted heter_config.json so the dispatch policy is
expert_batch with threshold=0 (every heter expert always runs BF16).

Run from anywhere; output lands at:
  expert_precision_assignment/accuracy_profile/configs/sweep_static_assignment/K{K}/
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "expert_precision_assignment" / "accuracy_profile"
OUT_BASE = ROOT / "configs" / "sweep_static_assignment"

LAYER_SENS = REPO / "expert_precision_assignment" / "sensitivity" / "per_moe_layer" / "results" / "summary.json"
EXPERT_SENS = REPO / "expert_precision_assignment" / "sensitivity" / "per_expert" / "results" / "summary.json"

BF16_MODEL = "/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
INT4_CHECKPOINT = "/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441"

# 48 experts per group times these multipliers.
N_VALUES = [0, 8, 16, 24, 32, 40, 48, 56, 64]
EXPERTS_PER_STEP = 48

# SLO matches the static-assignment defaults used to derive K=3072.
MAX_CONCURRENCY = 256
MAX_PROMPT_LEN = 2048
MAX_OUTPUT_LEN = 2048

# Force VRAM budget so this can run on a host without a GPU and so K is
# always exactly --force_k (no clamping by the budget). 80 GiB.
GPU_VRAM_BYTES = 85899345920


def _run_assign(k: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
        "--force_k", str(k),
        "--out_dir", str(out_dir),
    ]
    print(f"  + assign_experts --force_k {k} -> {out_dir}")
    subprocess.run(cmd, check=True, cwd=REPO)


def _patch_policy(heter_config_path: Path) -> None:
    """Force expert_batch policy with threshold=0 so all heter experts
    always run BF16 — K becomes the only knob across the sweep."""
    with open(heter_config_path) as f:
        cfg = json.load(f)
    cfg["policy"] = "expert_batch"
    cfg["policy_params"] = {"threshold": 0}
    with open(heter_config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for n in N_VALUES:
        k = EXPERTS_PER_STEP * n
        out_dir = OUT_BASE / f"K{k}"
        _run_assign(k, out_dir)
        _patch_policy(out_dir / "heter_config.json")
        print(f"    patched policy=expert_batch threshold=0 in {out_dir/'heter_config.json'}")
    print(f"\nDone. {len(N_VALUES)} configs under {OUT_BASE}")


if __name__ == "__main__":
    main()
