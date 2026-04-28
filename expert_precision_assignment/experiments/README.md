# experiments/ — heter-precision config generation + sweep

End-to-end pipeline for generating task-specific heter-MoE configs
(BF16/INT4 expert assignment) and running the efficiency/accuracy
sweep on Qwen3-30B-A3B.

All commands run from `experiments/`. The conda `sglang` env must be
activated (the shell scripts activate it themselves; the Python CLIs
expect it to already be active).

---

## TL;DR — regeneration

`data/configs/`, `data/kv_calib/`, and every per-run JSON/JSONL under
`data/results/` are gitignored and fully reproducible. Only aggregated
artifacts (summary CSVs + plots under `data/results/<task>/`) are
committed. To rebuild everything from a clean checkout:

```bash
conda activate sglang
cd experiments/

# 1. Flat worst-case configs (fallback for tasks without a calib).
python pipeline/gen_config/gen_all.py

# 2. Per-task pipelines (see sections below for details).
#    Each re-creates data/configs/<task>/ and data/kv_calib/<task>-or-ruler/.
bash pipeline/kv_calib/run_calib.sh sharegpt                                    # generic
python pipeline/gen_config/gen_all.py --task sharegpt \
    --calib_json data/kv_calib/sharegpt/calib.json
NUM_PROMPTS=2048 bash pipeline/run_sweep.sh sharegpt

RULER_MAX_SEQ=65536 NIAH_CACHE_DIR=$(pwd)/data/kv_calib/ruler/niah_cache \
    bash pipeline/kv_calib/run_calib.sh niah_single_2                           # RULER NIAH
# → python pipeline/gen_config/gen_all.py --task ruler_niah_64k \
#       --calib_json data/kv_calib/ruler/calib_65536_niah_single_2.json
# → bash pipeline/run_sweep.sh ruler_niah_64k   (with the env knobs below)

MAX_GEN_TOKS=32 RULER_MAX_SEQ=65536 \
    NIAH_CACHE_DIR=$(pwd)/data/kv_calib/ruler/niah_cache \
    bash pipeline/kv_calib/run_calib.sh ruler_qa_squad                          # RULER QA
# → python pipeline/gen_config/gen_all.py --task ruler_qa1_64k \
#       --calib_json data/kv_calib/ruler/calib_65536_ruler_qa_squad.json
# → bash pipeline/run_sweep.sh ruler_qa1_64k    (with the env knobs below)
```

Full details below.

---

## Env var reference

| Var | Used by | Default | Purpose |
|---|---|---|---|
| `MC_LIST` | gen_heter_configs / gen_dyna_variants / run_sweep.sh | `8 16 32 64 128 256` | mc ladder. For long-context RULER, override to e.g. `"1 2 4 8 16 32 64"` — KV dominates at 64k+ and heter budget collapses above mc=8. |
| `VARIANTS` | run_sweep.sh | `hess0 hess20 hess40 hess60 hess80 hess100` | Which variant files per mc to sweep. |
| `GPUS` | run_sweep.sh | `0..7` | GPU round-robin pool. |
| `RULER_MAX_SEQ` | run_calib.sh / run_sweep.sh | unset | Enables RULER mode. Sets `--metadata='{"max_seq_lengths":[$seq]}'`, `num_fewshot=0`, no chat template, no multiturn. Sweep also sets `--context-length` + YaRN scaling when > 40960. |
| `BENCH_TASK` | run_sweep.sh | `$TASK` | Decouples the config dir name (e.g. `ruler_qa1_64k`) from the lm_eval task (e.g. `ruler_qa_squad`). Essential for any virtual task name. |
| `LIMIT` | run_sweep.sh | unset (full task) | Passed as `--limit` to bench_eval. For RULER, `n=32` smoke / `n=128` comparative / full for paper-grade. |
| `MAX_GEN_TOKS` | run_calib.sh / run_sweep.sh | 128 (RULER) / 512 (other) | Per-task decode budget. **Must match task YAML** (niah_*: 128, ruler_qa_*: 32, ruler_vt: 30, ruler_cwe: 120, ruler_fwe: 50). Wrong value → over- or under-decoding. |
| `CALIB_LIMIT` | run_calib.sh | 128 | How many eval docs to tokenize for KV calibration. |
| `NIAH_CACHE_DIR` | bench_eval / calib_kv.py (both via `install_niah_disk_cache`) | unset (no cache) | Opt-in disk cache for RULER dataset builders. First build writes; subsequent HITs load in seconds. |
| `APPLY_CHAT_TEMPLATE` / `FEWSHOT_AS_MULTITURN` | run_calib.sh / run_sweep.sh | `1` (generic) / `0` (RULER) | Chat-template wrap & multiturn fewshot expansion. Both off for RULER (Qwen3 emits empty responses under the wrap). |
| `NUM_PROMPTS` / `SHAREGPT_CONTEXT_LEN` | run_calib.sh / run_sweep.sh (sharegpt only) | `1024` / `4096` | sharegpt knobs. |
| `CALIB_GPU` / `CALIB_MC` | run_calib.sh | `4` / `128` | GPU / max-concurrency for the live calibration server. Ignored in RULER CPU-only mode. |

---

## One-time: worst-case configs

Generate the flat `data/configs/mc{mc}/...` tree sized against the
worst-case envelope (`max_prompt=2048, max_output=2048`). Used both as
a conservative fallback and as the server config that
`pipeline/kv_calib/run_calib.sh` runs against for generic tasks.

```bash
python pipeline/gen_config/gen_all.py
```

(runs `gen_heter_configs.py` then `gen_dyna_variants.py` — the
latter emitting 6 runtime-dispatch variants `hess0..hess100` per mc:
hessian-weighted total-routing-weight scoring at cold/hot splits
{0,20,40,60,80,100}%. Legacy `hot0..hot100` / `thr32..thr512` loops
live under `if False:` blocks in `gen_dyna_variants.py`; flip on to
regenerate the full matrix.)

---

## Per-task pipeline (recommended)

### 1. Calibrate KV

```bash
bash pipeline/kv_calib/run_calib.sh <task>          # sharegpt, gsm8k, mmlu_flan_cot_zeroshot, …
```

Launches one full-BF16 server on `CALIB_GPU` (default GPU 4), drives
it with `bench_serving` (sharegpt) or `bench_eval` (bench_eval tasks),
captures per-request `(input_len, output_len)`, feeds into
`calib_kv.py --bench_details_jsonl` → `data/kv_calib/<task>/calib.json`
with `mean_total_len` + `std_total_len`. Shuts server down on exit.

### 2. Regenerate configs with amortized KV

```bash
python pipeline/gen_config/gen_all.py --task <task> \
    --calib_json data/kv_calib/<task>/calib.json
```

Writes `data/configs/<task>/mc{mc}/{heter_config,int4_only_experts,expert_importance,variants/*}.json`.
KV is sized as `mc × (μ + k·σ)` instead of worst-case
`mc × 0.5 × (max_in + max_out)`, typically freeing several GB that
become additional BF16 experts. `expert_importance.json` carries
per-layer per-expert hessian scores (clamped to 0 below the
first-order noise floor; all-ones fallback for layers with no signal);
`heter_config.expert_importance_file` points at it, and
`HeterFusedMoE` loads the relevant row at module init.

Equivalent to running `gen_heter_configs.py` then
`gen_dyna_variants.py` in sequence with the same flags. Both honor
`MC_LIST` so a per-task mc ladder stays consistent across steps.

### 3. Run the sweep

```bash
NUM_PROMPTS=2048 bash pipeline/run_sweep.sh <task>
```

Auto-picks `data/configs/<task>/mc{mc}/` when that directory exists
(falls back to the flat `data/configs/mc{mc}/`). mc × variant grid
fans out across `GPUS` with round-robin scheduling; each pair launches
an sglang server on its assigned GPU, runs the bench, and tears down.

### 4. Collect results

`pipeline/run_sweep.sh` runs `collect_results.py` at the end. To
recollect:

```bash
python pipeline/collect_result/collect_results.py \
    --results_dir data/results/<task> \
    --out_csv data/results/<task>/summary.csv
```

---

## RULER (long-context) — special case

RULER subtasks (`niah_single_1..3`, `niah_multikey_*`, `niah_multiquery`,
`niah_multivalue`, `ruler_qa_squad`, `ruler_qa_hotpot`, `ruler_vt`,
`ruler_cwe`, `ruler_fwe`) differ from generic tasks in several ways.
Setting `RULER_MAX_SEQ=<bytes>` on both `pipeline/kv_calib/run_calib.sh`
and `pipeline/run_sweep.sh` flips the appropriate defaults
automatically.

1. **Fixed output length.** Every RULER task YAML sets `until: []`
   plus a task-specific `max_gen_toks`. Every request decodes exactly
   that many tokens regardless of input length (scoring is substring
   match over the decode window). A live GPU run is therefore
   uninformative for output statistics — `run_calib.sh` in RULER mode
   skips the server launch entirely and calls `calib_kv.py`
   tokenize-only on CPU, synthesizing the `bench_details` block with
   a constant `output_len = MAX_GEN_TOKS`, `total_len = prompt_len + MAX_GEN_TOKS`.
   Works at 128k / 256k without burning any GPU KV.

   **Per-task `MAX_GEN_TOKS` — must match the YAML or the calibration
   is wrong:**

   | task(s) | MAX_GEN_TOKS |
   |---|---|
   | `niah_*` (8 tasks) | 128 (default) |
   | `ruler_qa_squad`, `ruler_qa_hotpot` | 32 |
   | `ruler_vt` | 30 |
   | `ruler_cwe` | 120 |
   | `ruler_fwe` | 50 |

2. **No chat template, no multiturn.** Qwen3-30B emits empty
   responses under the instruct chat wrap (0/32 on NIAH 8k). RULER
   mode sets `APPLY_CHAT_TEMPLATE=0`, `FEWSHOT_AS_MULTITURN=0`.

3. **Zero-shot.** `NUM_FEWSHOT=0` — RULER is raw completion.

4. **Server context window + YaRN.** `pipeline/run_sweep.sh` in RULER
   mode launches the server with `--context-length $((RULER_MAX_SEQ+512))`
   and, when that exceeds Qwen3's native 40960, injects YaRN via
   `--json-model-override-args` (rope_parameters: yarn, factor =
   ctx/40960, original_max_position_embeddings = 40960) plus
   `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.

5. **VRAM budget collapses with mc × seq.** On an 80 GB GPU, heter
   budget goes negative around mc=8 at 64k and mc=2 at 256k. For
   RULER use `MC_LIST="1 2 4 8 16 32 64"` (small mc where heter has
   runway) instead of the default `"8 16 32 64 128 256"`.

### Disk cache for RULER dataset construction

`lm_eval/tasks/ruler/prepare_niah.py:generate_samples` is a serial
Python tokenization loop over 500 deterministic haystacks per
`(task, max_seq_length)`. At long contexts this takes minutes per
invocation, and every bench_eval worker rebuilds independently — so
across a 6-worker sweep that's 6×. The `sglang.benchmark.eval_harness.niah_cache`
module intercepts lm_eval's YAML `!function` resolver and wraps the
13 RULER builder functions (`niah_single_*`, `niah_multikey_*`,
`niah_multiquery`, `niah_multivalue`, `get_squad`, `get_hotpotqa`,
`get_vt_dataset`, `get_cw_dataset`, `fwe_download`) with
`Dataset.save_to_disk` / `load_from_disk`, keyed by
`md5(fn_name + max_seq_lengths + tokenizer)[:12]`. Both `bench_eval.py`
and `calib_kv.py` call `install_niah_disk_cache()` lazily; no-op
unless `NIAH_CACHE_DIR` is set.

First-build cost (single worker on CPU, 500 samples):

| seq | build time | cache size |
|---|---|---|
| 8k   | ~8s   | 17 MB  |
| 16k  | ~18s  | 36 MB  |
| 32k  | ~33s  | 72 MB  |
| 64k  | ~72s  | 144 MB |
| 128k | ~177s | 288 MB |

Parallel build across 5 seq lengths → critical path = 128k ≈ 3 min.
Warm reload is ~5s regardless of seq length, so subsequent sweeps at
the same seq_len instantly skip the rebuild in every worker.

Cache layout:

```
data/kv_calib/ruler/niah_cache/
  <fn_name>/<hash12>/           # HF Dataset (arrow + metadata)
  niah_single_2/4b1ff37453e8/   # e.g. seq=8192, tokenizer=Qwen3
  niah_single_2/180b0208a237/   # seq=16384
  get_squad/58169f4fc29f/       # ruler_qa_squad seq=8192
  …
```

### Task-naming convention

When the bench_eval task doesn't uniquely identify the config (you
want separate configs for the same task at different seq lengths),
use `ruler_<subtask>_<seqtag>` for the config dir and pass
`BENCH_TASK=<actual lm_eval task>` to the sweep. This keeps
`data/configs/ruler_niah_8k/`, `data/configs/ruler_niah_64k/`,
`data/configs/ruler_qa1_64k/`, etc. independent while all routing to
the right lm_eval task internally.

### Worked example — NIAH at 64k (single-needle retrieval)

```bash
export NIAH_CACHE_DIR=$(pwd)/data/kv_calib/ruler/niah_cache

# 1. Calibrate all seq lengths in parallel (CPU-only, ~3 min).
for seq in 8192 16384 32768 65536 131072; do
  (RULER_MAX_SEQ=$seq bash pipeline/kv_calib/run_calib.sh niah_single_2) &
done
wait
# → data/kv_calib/ruler/calib_{8192,…,131072}_niah_single_2.json

# 2. Amortized configs for the 64k sweep (mc ladder capped at 64).
MC_LIST="1 2 4 8 16 32 64" \
  python pipeline/gen_config/gen_all.py --task ruler_niah_64k \
    --calib_json data/kv_calib/ruler/calib_65536_niah_single_2.json

# 3. Sweep — mc=4 × 6 variants × n=128 on 8 GPUs, ~20 min on A100-80GB.
MC_LIST="4" \
  VARIANTS="hess0 hess20 hess40 hess60 hess80 hess100" \
  LIMIT=128 \
  RULER_MAX_SEQ=65536 \
  BENCH_TASK=niah_single_2 \
  bash pipeline/run_sweep.sh ruler_niah_64k
# → data/results/ruler_niah_64k/{mc4_hess*,summary.csv}
```

### Worked example — QA at 64k (RULER qa_1)

```bash
export NIAH_CACHE_DIR=$(pwd)/data/kv_calib/ruler/niah_cache

# 1. Calibrate — note MAX_GEN_TOKS=32 for qa_squad (not the 128 default).
for seq in 8192 16384 32768 65536 131072; do
  (MAX_GEN_TOKS=32 RULER_MAX_SEQ=$seq bash pipeline/kv_calib/run_calib.sh ruler_qa_squad) &
done
wait

# 2. Configs (same MC_LIST as NIAH since KV budget scales identically).
MC_LIST="1 2 4 8 16 32 64" \
  python pipeline/gen_config/gen_all.py --task ruler_qa1_64k \
    --calib_json data/kv_calib/ruler/calib_65536_ruler_qa_squad.json

# 3. Sweep — BENCH_TASK routes to the lm_eval task; MAX_GEN_TOKS must
#    match the qa_squad YAML or bench_eval over-decodes.
MC_LIST="4" \
  VARIANTS="hess0 hess20 hess40 hess60 hess80 hess100" \
  LIMIT=128 \
  RULER_MAX_SEQ=65536 \
  BENCH_TASK=ruler_qa_squad \
  MAX_GEN_TOKS=32 \
  bash pipeline/run_sweep.sh ruler_qa1_64k
```

---

## Supported tasks

| task | bench backend | calib path | max_gen_toks | notes |
|---|---|---|---|---|
| `sharegpt` | `bench_serving` | live GPU run | — | primary efficiency target |
| `gsm8k` | `bench_eval --task=gsm8k` | live GPU run | 512 | chat template on, 5-shot multiturn |
| `mmlu_flan_cot_zeroshot` | `bench_eval` | live GPU run | 512 | same defaults as gsm8k |
| `niah_single_*`, `niah_multi*` (8 tasks) | `bench_eval` | CPU tokenize-only | 128 | long-context; pass `RULER_MAX_SEQ` |
| `ruler_qa_squad` (qa_1), `ruler_qa_hotpot` (qa_2) | `bench_eval` | CPU tokenize-only | 32 | needs `MAX_GEN_TOKS=32` |
| `ruler_vt` | `bench_eval` | CPU tokenize-only | 30 | variable tracking |
| `ruler_cwe` | `bench_eval` | CPU tokenize-only | 120 | common-words |
| `ruler_fwe` | `bench_eval` | CPU tokenize-only | 50 | frequent-words |

See the 13-task RULER paper recipe in the NVIDIA README; the `ruler`
group in lm_eval aggregates these with equal weight and
`weight_by_size: False`.

---

## Directory layout

```
experiments/
  README.md                    # this file
  PIPELINE.md                  # one-recipe-per-yaml pipeline guide
  run_pipeline.sh              # top-level driver (recipe → 6 stages)

  pipeline/                    # all stage code lives here
    recipe.py                  # shared YAML-recipe loader (imported by stages)
    run_sweep.sh               # mc × variant sweep dispatcher
    run_bench_eval_all.sh      # post-sharegpt sweep across canonical tasks
    run_calib_think_compare.sh # IFBench think-on/off A/B
    prompt/                    # prep stage
      prepare_prompts_*.py     # one per task (ifbench, supergpqa, lcb_v6, …)
    kv_calib/                  # calib stage
      run_calib.sh             # KV calibration; RULER_MAX_SEQ → CPU tokenize-only
    gen_config/                # gen stage
      gen_all.py               # gen_heter_configs + gen_dyna_variants oneshot
      gen_heter_configs.py     # VRAM budget → K experts → BF16/INT4 split
      gen_dyna_variants.py     # emit hess0..hess100 variants per mc
    scoring/                   # score stage
      score_traces_*.py        # one per task (offline accuracy)
      vendored/                # third-party scorer code (ifbench, lcb_runner)
    collect_result/            # collect stage
      collect_results.py       # summary CSV aggregation

  recipe/
    yamls/                     # eval recipes (ifbench_*, supergpqa_*, lcb_v6_*)

  plots/
    plot_sharegpt_grid.py      # sharegpt heatmap
    plot_gsm8k_grid.py         # gsm8k heatmap

  data/                        # all runtime outputs (mostly gitignored)
    configs/                   # gitignored — regenerate via gen_all.py
      mc{mc}/...               # flat worst-case (fallback)
      <task>/mc{mc}/           # per-task amortized
        heter_config.json      # policy + group ratios + pointers
        int4_only_experts.json
        expert_importance.json # hessian score grid (loaded by HeterFusedMoE)
        assignment_report.json # K_heter, fo_cap, SLO breakdown
        variants/hess{pct}.json
    kv_calib/                  # gitignored — regenerate via run_calib.sh
      <task>/calib.json        # generic tasks (gsm8k, sharegpt, …)
      ruler/
        calib_<seq>_<subtask>.json    # RULER calibs (one per seq × subtask)
        niah_cache/<fn>/<hash>/       # HF Dataset on disk (opt-in via NIAH_CACHE_DIR)
    results/                   # PARTIALLY gitignored — keep only summary.csv + plots
      <task>/
        mc{mc}_{variant}.{json,jsonl}   # gitignored per-run artifacts
        server.log / bench.log          # gitignored
        summary.csv                     # committed
        *.png                           # committed (from plots/plot_*.py)

  monitor/
    monitor.sh                 # one-shot sweep progress snapshot
    show_calib.py              # inspect calib/sweep trace JSONL
```

Dependencies in sibling dirs:

- `../policy/heter_assign/` — `vram_estimator.py`, `assign_experts.py`,
  `calib_kv.py`, `test_configs.py`
- `../legacy/sensitivity/` — per-layer PPL and per-expert L2 sensitivity summaries
- `../hessian/` — per-expert hessian scores (`hessian_scores.json`)

Cache module:

- `../../python/sglang/benchmark/eval_harness/niah_cache.py`
  (`install_niah_disk_cache()` — called at start of bench_eval and
  calib_kv when `NIAH_CACHE_DIR` is set)

---

## What's committed vs regenerable

| path | status | regenerate with |
|---|---|---|
| `data/configs/` | gitignored | `python pipeline/gen_config/gen_all.py [--task X --calib_json …]` |
| `data/kv_calib/` | gitignored | `bash pipeline/kv_calib/run_calib.sh <task>` |
| `data/results/<task>/*.json` `*.jsonl` `*.log` | gitignored | `bash pipeline/run_sweep.sh <task>` |
| `data/results/<task>/summary.csv` | **committed** | `python pipeline/collect_result/collect_results.py --results_dir …` |
| `data/results/<task>/*.png` | **committed** | `python plots/plot_*_grid.py` |

Deleting `data/configs/ data/kv_calib/ data/results/` and following the
TL;DR at the top of this README will rebuild everything except the
committed CSVs and plots — those come back from `git checkout`. The
1+ GB `niah_cache/` subdir is also safe to delete; it's a pure speedup
cache, first call after deletion takes ~3 min (parallel across 5 seq
lengths) to rebuild.
