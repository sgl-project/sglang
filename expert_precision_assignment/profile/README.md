# profile/ — heter-precision config generation + sweep

End-to-end pipeline for generating task-specific heter-MoE configs
(BF16/INT4 expert assignment) and running the efficiency/accuracy
sweep on Qwen3-30B-A3B.

All commands run from `profile/`. The conda `sglang` env must be
activated (the shell scripts activate it; the Python CLIs don't).

---

## One-time: worst-case configs

Generate the flat `configs/mc{mc}/...` tree sized against the
worst-case envelope (`max_prompt=2048, max_output=2048`). These are
the starting point — used both as a conservative fallback and as the
server config that `run_calib.sh` runs against.

```bash
python gen_all.py
```

(which runs `gen_heter_configs.py` then `gen_dyna_variants.py`, the
latter emitting the 6 runtime-dispatch variants `hess0..hess100` —
hessian-weighted total-routing-weight scoring at cold/hot splits
{0,20,40,60,80,100}% — per mc level. The legacy `hot0..hot100` and
`thr32..thr512` loops live under `if False:` blocks in
`gen_dyna_variants.py`; flip them on to regenerate the full matrix.)

---

## Per-task pipeline (recommended)

For each workload (sharegpt, gsm8k, mmlu, ...):

### 1. Calibrate KV from one short live run

```bash
bash run_calib.sh sharegpt
```

What it does:

1. Launches one full-BF16 sglang server (GPU 4, mc=128; no `--heter-precision-config`).
2. Runs `bench_serving` (sharegpt) or `bench_eval` (gsm8k, mmlu, …) and captures per-request `(input_len, output_len)` pairs.
3. Feeds that JSONL into `../policy/heter_assign/calib_kv.py` → `kv_calib/<task>/calib.json` with `mean_total_len` + `std_total_len`.
4. Shuts the server down.

Outputs live under `kv_calib/<task>/` (`calib.json`, `details_*.jsonl`,
`server.log`). The whole `kv_calib/` tree is gitignored — regenerate
with `run_calib.sh`, don't commit.

Env overrides: `CALIB_GPU=4 CALIB_MC=128 NUM_PROMPTS=256 CALIB_LIMIT=128`.

### 2. Regenerate configs with amortized KV

```bash
python gen_all.py --task sharegpt --calib_json kv_calib/sharegpt/calib.json
```

Writes `configs/sharegpt/mc{mc}/{heter_config,int4_only_experts,expert_importance,variants/...}.json`.
KV is sized as `mc × (μ + k·σ)` instead of `mc × 0.5 × (max_in + max_out)`,
typically freeing several GB of VRAM that become additional BF16
experts (larger `K_heter`). With `--hessian` (the default), this also
emits `expert_importance.json` — per-layer per-expert importance (hessian
score clamped to 0 below the first-order noise floor, or all-ones
fallback for layers with no signal). `heter_config.expert_importance_file`
points at it; `HeterFusedMoE` loads its layer's row at module init and
hands it to the `hessian_weighted_routing_weights` policy.

Equivalent to running `gen_heter_configs.py` and `gen_dyna_variants.py`
in sequence with the same flags.

### 3. Run the sweep

```bash
NUM_PROMPTS=2048 bash run_sweep.sh sharegpt
```

`run_sweep.sh` auto-picks `configs/sharegpt/mc{mc}/` when that
directory exists (falls back to the flat `configs/mc{mc}/` otherwise).
The 6 × 6 grid (mc ∈ {8,16,32,64,128,256} × `hess0..hess100`) runs
across GPUs 4–7 with round-robin scheduling.

Env overrides: `NUM_PROMPTS=2048 SHAREGPT_CONTEXT_LEN=4096 VARIANTS="hess0 hess60" MC_LIST="32 64 128"`.

### 4. Collect results

`run_sweep.sh` already runs `collect_results.py` at the end; if you
need to recollect:

```bash
python collect_results.py --results_dir results/sharegpt --out_csv results/sharegpt/summary.csv
```

---

## Supported tasks

| task | bench backend | calib path | notes |
|------|--------------|------------|-------|
| `sharegpt` | `bench_serving` | live GPU run | primary efficiency target |
| `gsm8k` | `bench_eval --task=gsm8k` | live GPU run | `max_gen_toks=512`, chat template on, 5-shot multiturn |
| `mmlu_flan_cot_zeroshot` | `bench_eval` | live GPU run | same defaults as gsm8k |
| `niah_single_*` / RULER subtasks | `bench_eval` | CPU tokenize-only | long-context; output_len is fixed by the task YAML |

For bench_eval tasks, output length is bounded by `--max-gen-toks`
(default 512 in `run_sweep.sh`), so the worst-case envelope is close
to tight even without calibration.

### RULER (long-context) — special case

RULER subtasks (`niah_single_1..3`, `niah_multikey_*`, `ruler_qa_*`,
…) differ from the other tasks in three ways that `run_calib.sh`
handles automatically when `RULER_MAX_SEQ` is set:

1. **CPU-only calibration.** RULER locks `max_gen_toks=128` and
   `until=[]` in the task YAML, so every request decodes a constant
   128 tokens regardless of input length. That makes a live GPU run
   uninformative for output statistics — `calib_kv.py` runs in
   tokenize-only mode on CPU and synthesizes the `bench_details`
   block (`output_len` is a δ at 128, `total_len = prompt_len + 128`).
   You can calibrate 64k / 128k / 256k contexts without burning
   GPU KV budget.
2. **No chat template.** Qwen3-30B emits empty responses under the
   instruct chat wrap on RULER (0/32 at 8k) — `APPLY_CHAT_TEMPLATE`
   defaults to `0` when `RULER_MAX_SEQ` is set. Override to `1` only
   if you specifically want the wrapped form.
3. **Zero-shot.** `NUM_FEWSHOT` defaults to `0` and
   `FEWSHOT_AS_MULTITURN` to `0` — RULER is a raw-completion NIAH
   task, few-shot scaffolding doesn't apply.

All RULER calibration artifacts are grouped under `kv_calib/ruler/`
regardless of which subtask produced them, so 8k / 64k / 128k sit
side by side (`calib_<seq>_<task>.json`).

```bash
# 8k context
RULER_MAX_SEQ=8192 bash run_calib.sh niah_single_2
# → kv_calib/ruler/calib_8192_niah_single_2.json

# 128k context (still CPU-only)
RULER_MAX_SEQ=131072 bash run_calib.sh niah_single_2

# Feed into the planner. KV budget at long context is tight, so cap
# the mc ladder at the low end:
MC_LIST="1 2 4 8 16 32 64" \
  python gen_all.py --task niah_single_2 \
    --calib_json kv_calib/ruler/calib_8192_niah_single_2.json

# Sweep (mirror MC_LIST so run_sweep.sh picks the same mc values):
MC_LIST="1 2 4 8 16 32 64" \
  NUM_PROMPTS=128 RULER_MAX_SEQ=8192 \
  bash run_sweep.sh niah_single_2
```

Both `gen_heter_configs.py` and `gen_dyna_variants.py` honor
`MC_LIST` as an env override; `run_sweep.sh` does the same. Set it on
every step of the RULER cycle so the three stay in sync.

---

## Directory layout

```
profile/
  gen_heter_configs.py      # planning (VRAM budget + expert assignment + importance)
  gen_dyna_variants.py      # emit hess0..hess100 variants per mc (legacy hot/thr disabled)
  run_calib.sh              # 1-shot KV calibration (sharegpt | bench_eval | RULER-CPU)
  run_sweep.sh              # mc × variant sweep dispatcher
  collect_results.py        # summary CSV
  configs/                  # gitignored
    mc{mc}/...              # flat: worst-case SLO (fallback)
    <task>/mc{mc}/          # per-task: amortized SLO
      heter_config.json     # policy + group ratios + pointers
      int4_only_experts.json
      expert_importance.json  # per-layer per-expert importance (hessian ranking)
      variants/hess{pct}.json
  kv_calib/                 # gitignored — reproducible via run_calib.sh
    <task>/calib.json       # consumed by gen_heter_configs --calib_json
    ruler/calib_<seq>_<sub>.json   # RULER runs grouped together
  results/                  # gitignored
    <task>/mc{mc}_{variant}*.{jsonl,json}
    <task>/summary.csv
```

Dependencies live in sibling dirs:

- `../policy/heter_assign/` — `vram_estimator.py`, `assign_experts.py`, `calib_kv.py`, `test_configs.py`
- `../sensitivity/` — per-layer PPL and per-expert L2 sensitivity summaries

---

## Quick reference — full sharegpt cycle

```bash
# Activate env (the shell scripts do this themselves)
conda activate sglang

# Step 0: flat worst-case configs (run once per model checkpoint)
python gen_all.py

# Step 1: calibrate
bash run_calib.sh sharegpt

# Step 2: per-task amortized configs + variants
python gen_all.py --task sharegpt --calib_json kv_calib/sharegpt/calib.json

# Step 3: full sweep (uses configs/sharegpt/ automatically)
NUM_PROMPTS=2048 bash run_sweep.sh sharegpt
```
