# `run_pipeline.sh` — one-entrypoint eval pipeline

Think of the pipeline as one function:

```
run_pipeline(recipe.yaml) → results/<name>/summary.csv
```

Where `recipe.yaml` is a single config file that describes the whole run
(dataset, sampling, calibration, sweep) and the function drives **six stages
in order**: `prep → calib → gen → sweep → score → collect`.

---

## 1. INPUT — what you prepare

### 1a. Pick (or write) a recipe

Recipes live in `profile/recipes/`. Four starters ship with the repo:

| recipe | task | thinking | use when |
|---|---|---|---|
| `ifbench_nothink.yaml` | IFBench (300 prompts) | off | **recommended** for IFBench — higher scores, 4× faster |
| `ifbench_think.yaml` | IFBench | on | comparison baseline / ablation |
| `supergpqa_think.yaml` | SuperGPQA (26,529 MCQs) | on | grad-level reasoning — thinking helps |
| `lcb_v6_think.yaml` | LiveCodeBench v6 (1,055 problems) | on | code gen — thinking helps plan algorithms |

To make a new recipe, copy one and edit. All fields are optional except `task`.

### 1b. Recipe schema

```yaml
task: ifbench                # REQUIRED — chooses prep + scorer module
variant: nothink             # artifact suffix — all outputs named `<task>_<variant>`

dataset:
  limit: 0                   # 0 = all prompts; N = first N only (for smoke tests)
  seed: 1234                 # supergpqa shuffle seed (optional)
  start_date: null           # lcb_v6 contest date window lower bound (optional)
  end_date:   null           # lcb_v6 contest date window upper bound (optional)

sampling:                    # written into every prompt_record in prompts/<name>.jsonl
  enable_thinking: true      # Qwen3 thinking toggle
  max_tokens: 8192           # output budget per request
  temperature: 0.6           # Qwen3 thinking recipe: 0.6 (NOT 0)
  top_p: 0.95
  top_k: 20
  presence_penalty: 1.5      # has-appeared penalty
  frequency_penalty: 0.5     # scales-with-count penalty (breaks repetition loops)

calibration:                 # one short live run on full BF16 to measure KV usage
  num_prompts: 256           # how many prompts to run
  mc: 128                    # max_running_requests (must match sweep if using it)
  gpu: 4                     # which GPU (single)
  port: 31304
  nccl_port: 41304

sweep:                       # 6 × 11 grid, round-robin across GPUs
  mc_list:  [8, 16, 32, 64, 128, 256]
  variants: [hot0, hot20, hot40, hot60, hot80, hot100,
             thr32, thr64, thr128, thr256, thr512]
  gpus:     [4, 5, 6, 7]
  num_prompts: 0             # 0 = all prompts

scoring:                     # optional — controls the `score` stage
  enabled: true              # false → stage becomes a no-op; collect still runs
  extra_args: []             # forwarded to score_traces_<task>.py per cell
                             # e.g. ["--per-doc"] or LCB's ["--timeout", "6"]
```

### 1c. Parameter glossary

| field | meaning | when to change it |
|---|---|---|
| `task` | which `prepare_prompts_<task>.py` and `score_traces_<task>.py` to call | add a new supported dataset |
| `variant` | artifact suffix; `<task>_<variant>` prevents collisions across runs | any time you want side-by-side comparison |
| `dataset.limit` | 0 = all prompts, N = first N | smoke tests (`limit: 16`) before the full run |
| `sampling.enable_thinking` | Qwen3 `<think>` block on/off | off is usually better for IFBench, on for math/code/reasoning |
| `sampling.max_tokens` | per-request output cap | thinking=true → 8192; thinking=false → 2048 |
| `sampling.frequency_penalty` | penalty per repeat token | raise if you see repetition loops in responses |
| `calibration.mc` | max_running_requests during calib | must match one of `sweep.mc_list` — calib KV μ/σ is valid at that concurrency |
| `calibration.num_prompts` | how many prompts to calibrate on | 128–256 is enough; higher wastes time |
| `sweep.mc_list` | concurrency levels to sweep | subset for smoke tests (e.g. `[128]`) |
| `sweep.variants` | heter-precision variants per mc | subset to test one policy at a time |
| `sweep.gpus` | round-robin GPU pool | adjust to your hardware |
| `scoring.enabled` | run the offline scorer in the `score` stage | set `false` for perf-only runs, for LCB-on-untrusted-models (safety), or when the vendored scorer deps aren't installed |
| `scoring.extra_args` | CLI args forwarded to `score_traces_<task>.py` | e.g. `["--per-doc"]` to emit per-row records, or `["--timeout", "6"]` for LCB |

---

## 2. INVOCATION — how you call it

### 2a. Full pipeline (all 6 stages)

```bash
cd expert_precision_assignment/profile
bash run_pipeline.sh recipes/ifbench_nothink.yaml
```

### 2b. One stage at a time

```bash
bash run_pipeline.sh recipes/ifbench_nothink.yaml --stage prep
bash run_pipeline.sh recipes/ifbench_nothink.yaml --stage calib
bash run_pipeline.sh recipes/ifbench_nothink.yaml --stage gen
bash run_pipeline.sh recipes/ifbench_nothink.yaml --stage sweep
bash run_pipeline.sh recipes/ifbench_nothink.yaml --stage score
bash run_pipeline.sh recipes/ifbench_nothink.yaml --stage collect
```

### 2c. Resume from a stage / subset

```bash
bash run_pipeline.sh recipes/ifbench_nothink.yaml --from sweep         # sweep → score → collect
bash run_pipeline.sh recipes/ifbench_nothink.yaml --stages gen,sweep   # exactly these two
```

### 2d. Env overrides (one-off tweaks without editing the recipe)

```bash
# Quick 16-prompt calib smoke test
NUM_PROMPTS=16 bash run_pipeline.sh recipes/ifbench_nothink.yaml --stage calib

# Sweep only two variants on GPU 0
VARIANTS="thr128 hot0" GPUS="0" \
  bash run_pipeline.sh recipes/ifbench_nothink.yaml --stage sweep
```

### 2e. Two variants side-by-side

```bash
bash run_pipeline.sh recipes/ifbench_think.yaml    &
bash run_pipeline.sh recipes/ifbench_nothink.yaml  &
wait
```

They won't collide — every artifact uses the compound name.

---

## 3. OUTPUT — what you get back

Each stage writes to a predictable path. Let `<name>` = `<task>_<variant>`
(e.g. `ifbench_nothink`).

| stage | produces | contains |
|---|---|---|
| `prep`    | `prompts/<name>.jsonl`                       | openai-chat prompts — one JSON line per request, with sampling knobs baked in |
|           | `prompts/<name>.meta.jsonl`                  | ground-truth per prompt (instruction IDs, kwargs, answer letter, …) |
|           | `prompts/<name>.private_tests.pkl`           | LCB only — compressed private test cases, 1.5–4 GB |
| `calib`   | `kv_calib/calib_<name>_n<N>.jsonl`           | bench_serving trace — per-request `input_lens`, `output_lens`, `generated_texts`, timings |
|           | `kv_calib/<name>.json`                       | summary μ/σ of `total_len` — input to `gen_heter_configs.py` for amortized KV sizing |
|           | `kv_calib/server_<name>_bf16.log`            | raw sglang server log for the calib run |
| `gen`     | `configs/<name>/mc<mc>/heter_config.json`    | per-mc base heter assignment (which experts → INT4 vs BF16) |
|           | `configs/<name>/mc<mc>/int4_only_experts.json` | fallback int4 list |
|           | `configs/<name>/mc<mc>/variants/<variant>.json` | 11 runtime-dispatch variants (hot0..hot100, thr32..thr512) |
| `sweep`   | `results/<name>/mc<mc>_<variant>.jsonl`      | full bench_serving trace for one (mc, variant) cell — 66 files by default |
|           | `results/<name>/mc<mc>_<variant>_server.log` | per-cell sglang server log |
|           | `results/<name>/mc<mc>_<variant>_bench.log`  | per-cell bench_serving log |
|           | `results/<name>/gpu<g>_worker.log`           | per-GPU orchestration log |
| `score`   | `results/<name>/mc<mc>_<variant>.scores.json` | sidecar with task-specific accuracy fields (e.g. IFBench's 4 metrics) |
| `collect` | `results/<name>/summary.csv`                 | one row per cell — mc, variant, throughput, latency, accuracy — joins sweep traces with scores sidecars |

---

## 4. UNDERSTANDING THE OUTPUT

### 4a. `kv_calib/<name>.json` (calibration summary)

```json
{
  "bench_details": {
    "total_len": {"n": 256, "mean": 2160.2, "std": 31.5, "p99": 2200, ...}
  },
  "recommended_slo": {
    "mean_total_len": 2160.2, "std_total_len": 31.5, "kv_headroom_sigmas": 2.0
  }
}
```

**What to check**: `mean` and `std` of `total_len` should be non-degenerate
(std > 0). If every request hit the cap, std=0 and the KV envelope is just
`max_tokens` — that's fine for sizing but means your model saturated the
output budget (see §5 pathologies).

### 4b. `results/<name>/summary.csv` (the final answer)

Columns you'll actually read:

| column | meaning |
|---|---|
| `mc` / `variant` | one row per grid cell |
| `output_throughput` | tok/s — the primary efficiency number |
| `median_ttft_ms`, `p99_ttft_ms` | time-to-first-token |
| `median_itl_ms`, `p99_itl_ms` | inter-token latency |
| `score_prompt_level_strict_acc` (ifbench) | strict pass rate per prompt |
| `score_accuracy` (supergpqa) | MCQ accuracy |
| `score_pass@1` (lcb_v6) | fraction of problems whose generated code passes all tests |

Plot `output_throughput` vs. `score_*` across variants to see the
efficiency-accuracy frontier per mc. Higher `hotN` → more experts in BF16;
higher `thrN` → more experts above the |f·o| threshold.

### 4c. `results/<name>/mc<mc>_<variant>.jsonl` (raw trace)

Use `show_calib.py` to inspect:

```bash
python show_calib.py results/ifbench_nothink/mc128_thr128.jsonl          # summary + row 0
python show_calib.py results/ifbench_nothink/mc128_thr128.jsonl -n 5     # first 5 answers
python show_calib.py results/ifbench_nothink/mc128_thr128.jsonl --row 3  # row 3 in full
python show_calib.py results/ifbench_nothink/mc128_thr128.jsonl --raw    # raw bytes, no framing
```

It auto-joins with the matching `prompts/<name>.jsonl` + `prompts/<name>.meta.jsonl`
so you see the prompt, constraint, and answer side-by-side.

### 4d. `kv_calib/calib_<name>_n<N>.jsonl` (calibration trace)

Same format as sweep traces. Inspect the same way:

```bash
python show_calib.py kv_calib/calib_ifbench_nothink_n16.jsonl            # summary
python show_calib.py kv_calib/calib_ifbench_nothink_n16.jsonl --errors   # only failed requests
```

---

## 5. QUICK SANITY CHECKS

Before trusting a full sweep's numbers, spot-check the calib trace:

1. **Input lengths reasonable**: `input_lens` should look like real prompt
   sizes (IFBench ~100-200, SuperGPQA ~200-400, LCB ~500-2000). If every
   row shows `input_lens=[2, 2, …]`, you hit the old bench_serving
   tokenization bug — make sure `transformers<5.0` or the fix is in place.
2. **Output lengths spread**: `output_lens` should have non-zero std. If
   every request saturates `max_tokens`, the model is either looping or
   the cap is too low — raise it or disable thinking.
3. **No repetition loops**: Use `show_calib.py <calib_trace> --row 0`. If
   you see the same paragraph repeated 10+ times, raise `frequency_penalty`
   (e.g. 0.7) or set `enable_thinking: false`.
4. **`</think>` appears exactly once per row** (when thinking=on): multiple
   closures mean the model is emitting spurious delimiters inside its
   answer — another repetition pathology.

---

## 6. ADDING A NEW TASK

1. Write `prompts/prepare_prompts_<newtask>.py` (use `prepare_prompts_ifbench.py`
   as a template — it has the `--recipe` two-pass argparse + compound output path).
2. Write `scoring/score_traces_<newtask>.py` (task-specific accuracy metric).
3. Copy an existing recipe and set `task: <newtask>` + whatever variant name.
4. `bash run_pipeline.sh recipes/<newtask>_<variant>.yaml` — the rest Just Works.

---

## 7. BACKWARDS COMPATIBILITY

The old per-stage invocations still work for one-off debugging — none of
them require `--recipe`:

```bash
python prompts/prepare_prompts_ifbench.py            # writes prompts/ifbench.jsonl (no suffix)
bash run_calib.sh sharegpt                           # still works
bash run_sweep.sh ifbench                            # still works (no variant suffix)
```

`--recipe` is additive: pass it and artifact paths gain the `<task>_<variant>`
prefix; omit it and everything is exactly where it used to be.
