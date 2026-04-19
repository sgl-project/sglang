# Accuracy Profile for Heter-MoE Configurations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 20-config accuracy + serving-metric sweep over heter-MoE configurations (Qwen3-30B-A3B, gsm8k), mirroring the existing `serving_profile/dynamic` layout.

**Architecture:** Two 1-D sweeps. Sweep 1 varies the static assignment K via `policy/static/assign_experts.py --force_k`; sweep 2 varies the dynamic dispatch policy at fixed K=3072. Each config is benched via `python -m sglang.bench_eval` against gsm8k, producing one JSON per config under `results/gsm8k/<sweep>/`.

**Tech Stack:** Python 3 (subprocess + json), bash (parallel GPU workers), sglang server, lm-evaluation-harness via `sglang.bench_eval`.

**Spec:** `docs/superpowers/specs/2026-04-19-accuracy-profile-design.md`.

---

## Constants used throughout

```
BF16_MODEL=/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
INT4_CHECKPOINT=/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441
LAYER_SENS=/data/huanchen/sglang/expert_precision_assignment/sensitivity/per_moe_layer/results/summary.json
EXPERT_SENS=/data/huanchen/sglang/expert_precision_assignment/sensitivity/per_expert/results/summary.json

REPO=/data/huanchen/sglang
ROOT=$REPO/expert_precision_assignment/accuracy_profile

# Qwen3-30B-A3B has 48 layers × 128 experts = 6144 (layer,expert) pairs.
# Sweep 1 K values: K = 48 * N for N in {0, 8, 16, 24, 32, 40, 48, 56, 64}.
```

Conda env (matches existing scripts): `conda activate sglang` (`/data/huanchen/miniforge3/envs/sglang`).

---

## File structure (everything below this line is created by this plan)

```
expert_precision_assignment/accuracy_profile/
├── gen_static_assignment_configs.py    # Task 3
├── gen_dynamic_dispatch_configs.py     # Task 4
└── run_accuracy_sweep.sh               # Task 5
```

`assign_experts.py` is patched in Task 1 to accept `--force_k 0`. No new test files (small project, JSON outputs are validated by `python -c` spot checks in each task).

---

## Task 1: Allow `assign_experts.py --force_k 0`

`assign_experts.py:291` reads `k = args.force_k if args.force_k > 0 else budget.k_heter_experts` — `--force_k 0` silently falls back to the VRAM budget. We need K=0 for the all-INT4 baseline of sweep 1, so the gate must be `>= 0`. Default sentinel becomes `-1` already (already is), so `-1` still means "use budget" and `0` now means "force K=0".

**Files:**
- Modify: `expert_precision_assignment/policy/static/assign_experts.py:291`

- [ ] **Step 1: Patch the conditional**

Edit `expert_precision_assignment/policy/static/assign_experts.py` line 291:

```python
# before
k = args.force_k if args.force_k > 0 else budget.k_heter_experts
if args.force_k > 0:
    logger.info("Forcing K=%d (VRAM-budget K was %d)", k, budget.k_heter_experts)
```

```python
# after
k = args.force_k if args.force_k >= 0 else budget.k_heter_experts
if args.force_k >= 0:
    logger.info("Forcing K=%d (VRAM-budget K was %d)", k, budget.k_heter_experts)
```

- [ ] **Step 2: Smoke-verify with `--force_k 0`**

Run from repo root:

```bash
mkdir -p /tmp/aprof_smoke_k0
python -m expert_precision_assignment.policy.static.assign_experts \
  --layer_sensitivity  /data/huanchen/sglang/expert_precision_assignment/sensitivity/per_moe_layer/results/summary.json \
  --expert_sensitivity /data/huanchen/sglang/expert_precision_assignment/sensitivity/per_expert/results/summary.json \
  --model_path         /data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --int4_checkpoint    /data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441 \
  --max_concurrency 256 --max_prompt_len 2048 --max_output_len 2048 \
  --gpu_vram_bytes 85899345920 \
  --force_k 0 \
  --out_dir /tmp/aprof_smoke_k0/
```

Expected: log line `Forcing K=0 (VRAM-budget K was N)`. Then verify:

```bash
python -c "
import json
r = json.load(open('/tmp/aprof_smoke_k0/assignment_report.json'))
assert r['K_heter_experts'] == 0, r['K_heter_experts']
assert r['num_int4_only'] == 6144, r['num_int4_only']
print('OK: K=0 enforced, all 6144 experts INT4-only')
"
rm -rf /tmp/aprof_smoke_k0
```

Expected: prints `OK: K=0 enforced, all 6144 experts INT4-only`. (`6144 = 48 layers × 128 experts`.)

- [ ] **Step 3: Commit**

```bash
git add expert_precision_assignment/policy/static/assign_experts.py
git commit -m "$(cat <<'EOF'
heter-moe/static: allow --force_k 0 for all-INT4 baseline

The previous '> 0' check silently dropped --force_k 0 back to the
VRAM-budget K. Use '>= 0' so 0 means "force zero heter experts" and
the sentinel "use budget" stays at the argparse default of -1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create the `accuracy_profile/` directory

**Files:**
- Create: `expert_precision_assignment/accuracy_profile/.gitkeep`

- [ ] **Step 1: Make the directory**

```bash
mkdir -p /data/huanchen/sglang/expert_precision_assignment/accuracy_profile
touch    /data/huanchen/sglang/expert_precision_assignment/accuracy_profile/.gitkeep
```

`configs/` and `results/` are created at runtime by the gen + run scripts; no need to seed them.

- [ ] **Step 2: Commit**

```bash
git add expert_precision_assignment/accuracy_profile/.gitkeep
git commit -m "$(cat <<'EOF'
heter-moe: scaffold accuracy_profile directory

Stub for upcoming accuracy sweep scripts. Configs and results are
generated at runtime under this dir.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `gen_static_assignment_configs.py`

Generate 9 configs at `configs/sweep_static_assignment/K{N}/` by invoking `assign_experts.py --force_k` and patching the emitted `heter_config.json` to use `policy=expert_batch, threshold=0` (so all heter experts always run BF16 — K is the only variable).

**Files:**
- Create: `expert_precision_assignment/accuracy_profile/gen_static_assignment_configs.py`

- [ ] **Step 1: Write the script**

Create `expert_precision_assignment/accuracy_profile/gen_static_assignment_configs.py`:

```python
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
```

- [ ] **Step 2: Run it**

```bash
cd /data/huanchen/sglang
python expert_precision_assignment/accuracy_profile/gen_static_assignment_configs.py
```

Expected: 9 lines like `+ assign_experts --force_k 0 -> .../K0`, each followed by `patched policy=expert_batch threshold=0 ...`. Final line `Done. 9 configs under .../sweep_static_assignment`.

- [ ] **Step 3: Spot-check the configs**

```bash
python -c "
import json, pathlib
base = pathlib.Path('/data/huanchen/sglang/expert_precision_assignment/accuracy_profile/configs/sweep_static_assignment')
expected = [0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072]
for k in expected:
    d = base / f'K{k}'
    cfg = json.load(open(d / 'heter_config.json'))
    rep = json.load(open(d / 'assignment_report.json'))
    assert cfg['policy'] == 'expert_batch', (k, cfg['policy'])
    assert cfg['policy_params'] == {'threshold': 0}, (k, cfg['policy_params'])
    assert rep['K_heter_experts'] == k, (k, rep['K_heter_experts'])
    assert rep['num_int4_only'] == 6144 - k, (k, rep['num_int4_only'])
    print(f'K={k}: OK  ({cfg[\"policy\"]} thr=0, heter={rep[\"K_heter_experts\"]}, int4_only={rep[\"num_int4_only\"]})')
print('all 9 configs valid')
"
```

Expected: 9 `K=...: OK` lines, then `all 9 configs valid`.

- [ ] **Step 4: Commit**

```bash
git add expert_precision_assignment/accuracy_profile/gen_static_assignment_configs.py \
        expert_precision_assignment/accuracy_profile/configs/sweep_static_assignment/
git commit -m "$(cat <<'EOF'
heter-moe/accuracy: gen_static_assignment_configs.py + 9 K-variants

Generates sweep 1 of the accuracy profile: 9 configs at K = 48*N for
N in {0, 8, 16, 24, 32, 40, 48, 56, 64}. Each emitted heter_config.json
is patched to expert_batch policy with threshold=0 so all heter
experts always run BF16 (K is the only sweep variable).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `gen_dynamic_dispatch_configs.py`

Build the K=3072 base config, then emit 11 variants on top of it (6 hot-pct random + 5 expert-batch threshold) — same shape as `serving_profile/dynamic/gen_variant_configs.py` but pointing at our own base.

**Files:**
- Create: `expert_precision_assignment/accuracy_profile/gen_dynamic_dispatch_configs.py`

- [ ] **Step 1: Write the script**

Create `expert_precision_assignment/accuracy_profile/gen_dynamic_dispatch_configs.py`:

```python
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
```

- [ ] **Step 2: Run it**

```bash
cd /data/huanchen/sglang
python expert_precision_assignment/accuracy_profile/gen_dynamic_dispatch_configs.py
```

Expected: 1 `+ assign_experts --force_k 3072` line, 11 `wrote ...` lines, then `Done. base + 6 sweep_a + 5 sweep_b configs under ...`.

- [ ] **Step 3: Spot-check the configs**

```bash
python -c "
import json, pathlib
base = pathlib.Path('/data/huanchen/sglang/expert_precision_assignment/accuracy_profile/configs/sweep_dynamic_dispatch')

rep = json.load(open(base / 'base' / 'assignment_report.json'))
assert rep['K_heter_experts'] == 3072, rep['K_heter_experts']
print('base K=3072: OK')

for hp in [0, 20, 40, 60, 80, 100]:
    cfg = json.load(open(base / 'sweep_a' / f'hot{hp}.json'))
    assert cfg['policy'] == 'random', (hp, cfg['policy'])
    sr = {g['name']: g.get('size_ratio') for g in cfg['groups']}
    assert sr['hot'] == hp/100.0 and sr['cold'] == 1 - hp/100.0, (hp, sr)
    print(f'hot{hp}: random size_ratio cold={sr[\"cold\"]} hot={sr[\"hot\"]}: OK')

for thr in [32, 64, 128, 256, 512]:
    cfg = json.load(open(base / 'sweep_b' / f'thr{thr}.json'))
    assert cfg['policy'] == 'expert_batch', (thr, cfg['policy'])
    assert cfg['policy_params'] == {'threshold': thr}, (thr, cfg['policy_params'])
    print(f'thr{thr}: expert_batch threshold={thr}: OK')

print('all 11 variants valid')
"
```

Expected: `base K=3072: OK`, 6 `hot...: OK` lines, 5 `thr...: OK` lines, `all 11 variants valid`.

- [ ] **Step 4: Commit**

```bash
git add expert_precision_assignment/accuracy_profile/gen_dynamic_dispatch_configs.py \
        expert_precision_assignment/accuracy_profile/configs/sweep_dynamic_dispatch/
git commit -m "$(cat <<'EOF'
heter-moe/accuracy: gen_dynamic_dispatch_configs.py + base+11 variants

Generates sweep 2 of the accuracy profile at fixed K=3072: a base
assignment plus 6 random+size_ratio variants and 5 expert_batch
threshold variants. Mirrors serving_profile/dynamic/gen_variant_configs
but uses an accuracy_profile-owned base.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `run_accuracy_sweep.sh`

Per-GPU worker shell driver: launch sglang server with each `heter_config.json`, run `bench_eval` on gsm8k, save the merged JSON. Mirrors `serving_profile/dynamic/run_dynamic_sweep.sh` (server lifecycle, port allocation, resumability) but inner loop is one `bench_eval` call per config (no request-rate sweep).

**Files:**
- Create: `expert_precision_assignment/accuracy_profile/run_accuracy_sweep.sh`

- [ ] **Step 1: Write the script**

Create `expert_precision_assignment/accuracy_profile/run_accuracy_sweep.sh` and make it executable:

```bash
#!/usr/bin/env bash
# Accuracy sweep on Qwen3-30B-A3B over heter-MoE configurations.
#
# Sweep 1 (sweep_static_assignment): 9 configs, K varies via
#   policy/static/assign_experts.py --force_k {0,384,...,3072}; dispatch
#   patched to expert_batch threshold=0 (all heter always BF16).
#
# Sweep 2 (sweep_dynamic_dispatch): 11 configs at K=3072; dispatch
#   policy varies (6 random hot-pct + 5 expert_batch threshold).
#
# Total: 20 runs. Distributed 5/5/5/5 across GPUs 4-7.
# Per config we run ONE bench_eval on gsm8k at max_concurrency=256.
#
# Usage:
#   bash run_accuracy_sweep.sh               # full sweep
#   TASK=gsm8k bash run_accuracy_sweep.sh    # override task
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_CFG_DIR="$SCRIPT_DIR/configs/sweep_static_assignment"
DYN_BASE_DIR="$SCRIPT_DIR/configs/sweep_dynamic_dispatch"

TASK="${TASK:-gsm8k}"
OUT_STATIC="$SCRIPT_DIR/results/$TASK/sweep_static_assignment"
OUT_DYN="$SCRIPT_DIR/results/$TASK/sweep_dynamic_dispatch"
mkdir -p "$OUT_STATIC" "$OUT_DYN" "$SCRIPT_DIR/results"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HOST="127.0.0.1"

# bench_eval gsm8k settings
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-512}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

# Per-GPU job lists — "label:configfile:outdir", space-separated.
# 20 runs split 5/5/5/5 across GPUs 4-7.
declare -A GPU_JOBS
GPU_JOBS[4]="\
s_K0:$STATIC_CFG_DIR/K0/heter_config.json:$OUT_STATIC \
s_K384:$STATIC_CFG_DIR/K384/heter_config.json:$OUT_STATIC \
s_K768:$STATIC_CFG_DIR/K768/heter_config.json:$OUT_STATIC \
s_K1152:$STATIC_CFG_DIR/K1152/heter_config.json:$OUT_STATIC \
s_K1536:$STATIC_CFG_DIR/K1536/heter_config.json:$OUT_STATIC"

GPU_JOBS[5]="\
s_K1920:$STATIC_CFG_DIR/K1920/heter_config.json:$OUT_STATIC \
s_K2304:$STATIC_CFG_DIR/K2304/heter_config.json:$OUT_STATIC \
s_K2688:$STATIC_CFG_DIR/K2688/heter_config.json:$OUT_STATIC \
s_K3072:$STATIC_CFG_DIR/K3072/heter_config.json:$OUT_STATIC \
d_hot0:$DYN_BASE_DIR/sweep_a/hot0.json:$OUT_DYN"

GPU_JOBS[6]="\
d_hot20:$DYN_BASE_DIR/sweep_a/hot20.json:$OUT_DYN \
d_hot40:$DYN_BASE_DIR/sweep_a/hot40.json:$OUT_DYN \
d_hot60:$DYN_BASE_DIR/sweep_a/hot60.json:$OUT_DYN \
d_hot80:$DYN_BASE_DIR/sweep_a/hot80.json:$OUT_DYN \
d_hot100:$DYN_BASE_DIR/sweep_a/hot100.json:$OUT_DYN"

GPU_JOBS[7]="\
d_thr32:$DYN_BASE_DIR/sweep_b/thr32.json:$OUT_DYN \
d_thr64:$DYN_BASE_DIR/sweep_b/thr64.json:$OUT_DYN \
d_thr128:$DYN_BASE_DIR/sweep_b/thr128.json:$OUT_DYN \
d_thr256:$DYN_BASE_DIR/sweep_b/thr256.json:$OUT_DYN \
d_thr512:$DYN_BASE_DIR/sweep_b/thr512.json:$OUT_DYN"

declare -A GPU_PORT
GPU_PORT[4]=31104
GPU_PORT[5]=31105
GPU_PORT[6]=31106
GPU_PORT[7]=31107

# shellcheck disable=SC1091
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate sglang 2>/dev/null || source activate sglang

run_one_config() {
    local gpu=$1
    local label=$2
    local config=$3
    local out_dir=$4
    local port=$5
    local out="$out_dir/${label}.json"
    local server_log="$out_dir/${label}_server.log"
    local bench_log="$out_dir/${label}_bench.log"

    if [ -f "$out" ]; then
        echo "[gpu$gpu $label] result exists, skip ($out)"
        return 0
    fi
    if [ ! -f "$config" ]; then
        echo "[gpu$gpu $label] MISSING config: $config" >&2
        return 1
    fi

    echo "[gpu$gpu $label] launching server on port $port"
    CUDA_VISIBLE_DEVICES="$gpu" python3 -m sglang.launch_server \
        --model-path "$BF16_MODEL" \
        --host "$HOST" --port "$port" \
        --trust-remote-code \
        --heter-precision-config "$config" > "$server_log" 2>&1 &
    local server_pid=$!

    local elapsed=0
    while ! curl -s "http://${HOST}:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "[gpu$gpu $label] server died during startup (see $server_log)" >&2
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge 900 ]; then
            echo "[gpu$gpu $label] server didn't start within 900s" >&2
            kill -KILL "$server_pid" 2>/dev/null || true
            return 1
        fi
    done
    echo "[gpu$gpu $label] server ready after ${elapsed}s"

    curl -s -X POST "http://${HOST}:${port}/flush_cache" > /dev/null 2>&1 || true
    sleep 1

    echo "[gpu$gpu $label] bench_eval $TASK -> $out"
    if ! python3 -m sglang.bench_eval \
            --task "$TASK" \
            --base-url "http://${HOST}:${port}" \
            --backend sglang \
            --model "$BF16_MODEL" \
            --tokenizer "$BF16_MODEL" \
            --num-fewshot "$NUM_FEWSHOT" \
            --max-gen-toks "$MAX_GEN_TOKS" \
            --request-rate "$REQUEST_RATE" \
            --max-concurrency "$MAX_CONCURRENCY" \
            --apply-chat-template \
            --output-file "$out" > "$bench_log" 2>&1; then
        echo "[gpu$gpu $label] bench_eval FAILED (see $bench_log)" >&2
    fi

    pkill -TERM -P "$server_pid" 2>/dev/null || true
    kill -TERM "$server_pid" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "$server_pid" 2>/dev/null || true
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "[gpu$gpu $label] server stopped"
    sleep 3
}

run_gpu_worker() {
    local gpu=$1
    local port=${GPU_PORT[$gpu]}
    local jobs=${GPU_JOBS[$gpu]}
    for entry in $jobs; do
        local label="${entry%%:*}"
        local rest="${entry#*:}"
        local config="${rest%%:*}"
        local out_dir="${rest##*:}"
        run_one_config "$gpu" "$label" "$config" "$out_dir" "$port" || true
    done
    echo "[gpu$gpu] DONE"
}

echo "============================================================"
echo "  Accuracy profile sweep"
echo "  task:             $TASK"
echo "  GPUs:             4,5,6,7"
echo "  num_fewshot:      $NUM_FEWSHOT"
echo "  max_gen_toks:     $MAX_GEN_TOKS"
echo "  max_concurrency:  $MAX_CONCURRENCY"
echo "  request_rate:     $REQUEST_RATE"
echo "  out_static:       $OUT_STATIC"
echo "  out_dynamic:      $OUT_DYN"
for g in 4 5 6 7; do
    echo "  gpu$g (port ${GPU_PORT[$g]}): ${GPU_JOBS[$g]}"
done
echo "============================================================"

START_TS=$(date +%s)

for gpu in 4 5 6 7; do
    run_gpu_worker "$gpu" > "$SCRIPT_DIR/results/gpu${gpu}_worker.log" 2>&1 &
done
wait

END_TS=$(date +%s)
echo ""
echo "============================================================"
echo "  All GPU workers done in $((END_TS - START_TS))s."
echo "  Results: $OUT_STATIC/  and  $OUT_DYN/"
echo "============================================================"
```

Then:

```bash
chmod +x /data/huanchen/sglang/expert_precision_assignment/accuracy_profile/run_accuracy_sweep.sh
```

- [ ] **Step 2: Lint with shellcheck (optional but cheap)**

```bash
shellcheck /data/huanchen/sglang/expert_precision_assignment/accuracy_profile/run_accuracy_sweep.sh || true
```

Don't fail on shellcheck — `serving_profile/dynamic/run_dynamic_sweep.sh` already disables one warning. Just confirm there's no syntax-level issue (line count > 0 in output isn't fatal; a hard-fail with a parse error would be).

- [ ] **Step 3: Confirm the script parses (bash -n)**

```bash
bash -n /data/huanchen/sglang/expert_precision_assignment/accuracy_profile/run_accuracy_sweep.sh && echo OK
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add expert_precision_assignment/accuracy_profile/run_accuracy_sweep.sh
git commit -m "$(cat <<'EOF'
heter-moe/accuracy: run_accuracy_sweep.sh driver

Per-GPU shell worker that launches sglang with each heter config and
runs sglang.bench_eval (gsm8k by default). 20 runs distributed 5/5/5/5
across GPUs 4-7; outputs grouped by benchmark under results/<task>/.
Resumable: skips configs whose result JSON already exists.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: One-config smoke test

End-to-end validation against a single config (cheapest: K=3072 from sweep 1) before committing to a 20-run sweep. Caps gsm8k at `--limit 32` so the run completes in ~minutes rather than ~hour.

**Files:** none created; this is a runtime check.

- [ ] **Step 1: Free GPU 4**

```bash
nvidia-smi --id=4 --query-compute-apps=pid --format=csv,noheader
```

Expected: empty (or only your own processes you can kill). If something else is running, abort and ask the user.

- [ ] **Step 2: Launch server with the K=3072 config**

```bash
conda activate sglang
CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server \
  --model-path /data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --host 127.0.0.1 --port 31104 \
  --trust-remote-code \
  --heter-precision-config /data/huanchen/sglang/expert_precision_assignment/accuracy_profile/configs/sweep_static_assignment/K3072/heter_config.json \
  > /tmp/aprof_smoke_server.log 2>&1 &
SERVER_PID=$!
```

Wait for readiness:

```bash
until curl -s http://127.0.0.1:31104/health > /dev/null; do
  if ! kill -0 $SERVER_PID; then echo "server died"; tail -50 /tmp/aprof_smoke_server.log; break; fi
  sleep 5
done && echo READY
```

Expected: `READY` within ~5 minutes.

- [ ] **Step 3: Run a tiny bench_eval (limit=32)**

```bash
python3 -m sglang.bench_eval \
  --task gsm8k \
  --base-url http://127.0.0.1:31104 \
  --backend sglang \
  --model /data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --tokenizer /data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --num-fewshot 5 \
  --max-gen-toks 512 \
  --max-concurrency 32 \
  --apply-chat-template \
  --limit 32 \
  --output-file /tmp/aprof_smoke_K3072.json
```

Expected: prints `Task: gsm8k`, `N samples: 32`, an `Accuracy:` block (e.g. `exact_match,strict-match: 0.X`), and a `Performance:` block (TTFT/ITL/throughput).

- [ ] **Step 4: Validate the merged JSON**

```bash
python -c "
import json
r = json.load(open('/tmp/aprof_smoke_K3072.json'))
assert r['task'] == 'gsm8k', r['task']
assert r['n_samples'] == 32, r['n_samples']
assert 'accuracy' in r and r['accuracy'], r.get('accuracy')
assert 'perf' in r and r['perf'], r.get('perf')
print('smoke OK:', {k: r['accuracy'][k] for k in r['accuracy']})
"
```

Expected: `smoke OK: {...}` with at least one accuracy metric.

- [ ] **Step 5: Tear down + cleanup**

```bash
kill -TERM $SERVER_PID 2>/dev/null || true
sleep 2
kill -KILL $SERVER_PID 2>/dev/null || true
rm -f /tmp/aprof_smoke_K3072.json /tmp/aprof_smoke_server.log
echo "smoke test complete"
```

- [ ] **Step 6: (No commit — runtime-only.)**

If smoke passed, the implementation is complete. Hand back to the user to launch the full sweep:

```bash
cd /data/huanchen/sglang
nohup bash expert_precision_assignment/accuracy_profile/run_accuracy_sweep.sh \
  > expert_precision_assignment/accuracy_profile/results/driver.log 2>&1 &
```

If smoke failed, debug before launching the 20-run sweep — most likely culprits: (a) sglang import errors in the bench_eval entry point, (b) `--heter-precision-config` rejected by the server (check `/tmp/aprof_smoke_server.log`), (c) lm-eval-harness gsm8k task not registered.

---

## Self-review notes (for the implementer)

- All paths in tasks are absolute; no `cd` required to read the plan.
- `gen_*` scripts both use `--gpu_vram_bytes 85899345920` (80 GiB) so they can run on a CPU-only box; the actual VRAM constraint is not relevant for sweep 1 (all configs fit because we control K) or sweep 2 (we explicitly target K=3072).
- `assign_experts.py` writes `int4_only_experts.json` with absolute paths in `int4_only_experts_file`. Variants in `gen_dynamic_dispatch_configs.py` inherit this pointer untouched, so all 11 variants reference the same K=3072 base — what we want.
- `run_accuracy_sweep.sh` ports use 31104–31107 (different from `run_dynamic_sweep.sh`'s 31004–31007) to allow concurrent runs without port collisions.
- Skipping logic in `run_accuracy_sweep.sh` keys on `${label}.json`. To rerun a config, delete its result JSON.
