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
latter emitting the 11 runtime-dispatch variants — `hot0..hot100`,
`thr32..thr512` — per mc level).

---

## Per-task pipeline (recommended)

For each workload (sharegpt, gsm8k, mmlu, ...):

### 1. Calibrate KV from one short live run

```bash
bash run_calib.sh sharegpt
```

What it does:

1. Launches one sglang server (GPU 4, mc=128, `configs/mc128/variants/thr128.json`).
2. Runs `bench_serving --num-prompts 256 --output-details` → per-request `(input_len, output_len)` pairs.
3. Feeds that JSONL into `../policy/heter_assign/calib_kv.py` → `kv_calib/sharegpt.json` with `mean_total_len` + `std_total_len`.
4. Shuts the server down.

Env overrides: `CALIB_GPU=4 CALIB_MC=128 CALIB_VARIANT=thr128 NUM_PROMPTS=256`.

### 2. Regenerate configs with amortized KV

```bash
python gen_all.py --task sharegpt --calib_json kv_calib/sharegpt.json
```

Writes `configs/sharegpt/mc{mc}/{heter_config,int4_only_experts,variants/...}.json`.
KV is sized as `mc × (μ + k·σ)` instead of `mc × 0.5 × (max_in + max_out)`,
typically freeing several GB of VRAM that become additional BF16
experts (larger `K_heter`). Equivalent to running `gen_heter_configs.py`
and `gen_dyna_variants.py` in sequence with the same flags.

### 3. Run the sweep

```bash
NUM_PROMPTS=2048 bash run_sweep.sh sharegpt
```

`run_sweep.sh` auto-picks `configs/sharegpt/mc{mc}/` when that
directory exists (falls back to the flat `configs/mc{mc}/` otherwise).
The 6 × 11 grid (mc ∈ {8,16,32,64,128,256} × 11 variants) runs across
GPUs 4–7 with round-robin scheduling.

Env overrides: `NUM_PROMPTS=2048 SHAREGPT_CONTEXT_LEN=4096 VARIANTS="thr128 hot0"`.

### 4. Collect results

`run_sweep.sh` already runs `collect_results.py` at the end; if you
need to recollect:

```bash
python collect_results.py --results_dir results/sharegpt --out_csv results/sharegpt/summary.csv
```

---

## Supported tasks

| task | bench backend | calib supported | notes |
|------|--------------|------------------|-------|
| `sharegpt` | `bench_serving` | yes (via `run_calib.sh`) | primary efficiency target |
| `gsm8k` | `bench_eval --task=gsm8k` | prompt-only (calib_kv `--task`) | no `--output-details` → no σ for outputs |
| `mmlu_flan_cot_zeroshot` | `bench_eval` | prompt-only | same as gsm8k |

For bench_eval tasks, output length is bounded by `--max-gen-toks`
(default 512 in `run_sweep.sh`), so the worst-case envelope is close
to tight even without calibration.

---

## Directory layout

```
profile/
  gen_heter_configs.py      # planning (VRAM budget + expert assignment)
  run_calib.sh              # 1-shot KV calibration (sharegpt)
  run_sweep.sh              # 6×11 sweep dispatcher (sharegpt or bench_eval)
  collect_results.py        # summary CSV
  configs/
    mc{mc}/...              # flat: worst-case SLO (fallback)
    <task>/mc{mc}/...       # per-task: amortized SLO
  kv_calib/
    <task>.json             # calib_kv output — consumed by gen_heter_configs
  results/
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
python gen_all.py --task sharegpt --calib_json kv_calib/sharegpt.json

# Step 3: full sweep (uses configs/sharegpt/ automatically)
NUM_PROMPTS=2048 bash run_sweep.sh sharegpt
```
