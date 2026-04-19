# Accuracy Profile for Heter-MoE Configurations

**Date:** 2026-04-19
**Status:** Draft
**Scope:** Offline accuracy + serving-metric sweep over heter-MoE configurations on Qwen3-30B-A3B. Mirrors `expert_precision_assignment/serving_profile/dynamic/` but answers an accuracy question instead of an efficiency question. Single-GPU only (TP=1). gsm8k for the demo; additional benchmarks slot in later as sibling result directories.

## Problem

`serving_profile/dynamic/` already produces an efficiency table (TTFT / ITL / throughput) across heter-MoE configurations and request rates. It does not answer the upstream question:

> Does having more BF16 experts actually lead to better accuracy?

We need a parallel sweep that varies the same kinds of knobs but reports accuracy. Two independent 1-D axes (kept separate to keep total runs tractable; benchmark is the implicit third axis, handled by re-running with a different `--task`):

1. **Sweep static assignment** — vary `K` (number of heter experts produced by `policy/static/assign_experts.py`). All heter experts always run BF16. Tests "more BF16 weights loaded → better accuracy."
2. **Sweep dynamic dispatch** — fix `K = 48 × 64 = 3072` and reuse the 11 `serving_profile/dynamic` variants (6 hot-pct + 5 threshold). Tests "at fixed K, does runtime dispatch policy affect accuracy?"

## Inputs

| Input | Source | Notes |
|-------|--------|-------|
| Per-layer PPL sensitivity | `expert_precision_assignment/sensitivity/per_moe_layer/results/summary.json` | reused by `assign_experts.py` |
| Per-expert L2 sensitivity | `expert_precision_assignment/sensitivity/per_expert/results/summary.json` | reused by `assign_experts.py` |
| BF16 model | `/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/.../` | 48 layers × 128 experts |
| INT4 checkpoint | `/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/.../` | for the cold group in `heter_config.json` |
| Existing variant generator | `expert_precision_assignment/serving_profile/dynamic/gen_variant_configs.py` | template for `gen_dynamic_dispatch_configs.py` |
| Bench driver | `python/sglang/bench_eval.py` | runs lm-eval-harness through `bench_serving`; emits accuracy + serving metrics |

## Layout (under `expert_precision_assignment/accuracy_profile/`)

```
accuracy_profile/
├── configs/
│   ├── sweep_static_assignment/
│   │   ├── K0/
│   │   │   ├── int4_only_experts.json
│   │   │   ├── heter_config.json
│   │   │   └── assignment_report.json
│   │   ├── K384/  K768/  K1152/  K1536/  K1920/  K2304/  K2688/  K3072/
│   │   └── (same structure each)
│   └── sweep_dynamic_dispatch/
│       ├── base/
│       │   ├── int4_only_experts.json     # produced by assign_experts.py --force_k 3072
│       │   ├── heter_config.json
│       │   └── assignment_report.json
│       ├── sweep_a/
│       │   └── hot{0,20,40,60,80,100}.json
│       └── sweep_b/
│           └── thr{32,64,128,256,512}.json
├── results/
│   └── gsm8k/
│       ├── sweep_static_assignment/        # K0.json, K384.json, ..., K3072.json
│       └── sweep_dynamic_dispatch/         # hot0.json, ..., hot100.json, thr32.json, ..., thr512.json
├── gen_static_assignment_configs.py
├── gen_dynamic_dispatch_configs.py
└── run_accuracy_sweep.sh
```

Future benchmarks (mmlu, hellaswag, …) slot in as `results/<task>/sweep_*/`. No code change required for new tasks beyond passing a different `--task` to the driver.

## Sweep 1 — `sweep_static_assignment` (9 configs)

For `N ∈ {0, 8, 16, 24, 32, 40, 48, 56, 64}` (i.e. `K ∈ {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072}`):

```
python -m expert_precision_assignment.policy.static.assign_experts \
  --layer_sensitivity   .../sensitivity/per_moe_layer/results/summary.json \
  --expert_sensitivity  .../sensitivity/per_expert/results/summary.json \
  --model_path          $BF16_MODEL \
  --int4_checkpoint     $INT4_CHECKPOINT \
  --max_concurrency 256 --max_prompt_len 2048 --max_output_len 2048 \
  --force_k $((48 * N)) \
  --out_dir configs/sweep_static_assignment/K$((48 * N))/
```

Then **patch each emitted `heter_config.json`** so the dispatch policy is `expert_batch` with `threshold=0` — every heter expert always runs BF16. With `K=0` no experts are heter, so the patch is a no-op (everything is INT4). With `K=3072` all heter experts always BF16. K is the only variable across the 9 configs.

`gen_static_assignment_configs.py` is the script that does both steps (driver loop over N + post-write patch).

## Sweep 2 — `sweep_dynamic_dispatch` (11 configs)

1. Generate base at K=3072:
   ```
   python -m expert_precision_assignment.policy.static.assign_experts \
     ... --force_k 3072 --out_dir configs/sweep_dynamic_dispatch/base/
   ```
2. `gen_dynamic_dispatch_configs.py` is a near-copy of `serving_profile/dynamic/gen_variant_configs.py`:
   - Reads `configs/sweep_dynamic_dispatch/base/heter_config.json`.
   - Sweep A: 6 variants with `policy=random`, hot/cold `size_ratio` set per `hot_pct ∈ {0,20,40,60,80,100}`.
   - Sweep B: 5 variants with `policy=expert_batch`, `policy_params.threshold ∈ {32,64,128,256,512}`.
   - Writes to `configs/sweep_dynamic_dispatch/sweep_a/hotN.json` and `.../sweep_b/thrN.json`.
   - Each variant inherits the same `int4_only_experts_file` pointer (the K=3072 base).

## Run driver — `run_accuracy_sweep.sh`

Same per-GPU worker pattern as `serving_profile/dynamic/run_dynamic_sweep.sh` (GPUs 4–7, ports 31004–31007, server-launch + readiness wait + cleanup). Differences:

- Per config it runs **`bench_eval`** instead of `bench_serving`:
  ```
  python -m sglang.bench_eval \
    --base-url http://127.0.0.1:$PORT --backend sglang \
    --model $BF16_MODEL --tokenizer $BF16_MODEL \
    --task gsm8k --num-fewshot 5 --max-gen-toks 512 \
    --apply-chat-template \
    --max-concurrency 256 \
    --output-file results/gsm8k/<sweep>/<label>.json
  ```
- One run per config (no inner `request_rate` loop — accuracy-side traffic config is fixed at `max_concurrency=256` to match the SLO that produced K=3072).
- Total runs: **9 (static) + 11 (dynamic) = 20**, distributed roughly evenly across GPUs 4–7.
- Resumability: skip a run if its output JSON already exists (same convention as `run_dynamic_sweep.sh`).

## Output

Each `results/gsm8k/<sweep>/<label>.json` is the merged `bench_eval` report — accuracy block (e.g. `gsm8k.exact_match,strict-match`) plus serving block (TTFT/ITL/throughput percentiles). Two final tables (rendered post hoc from these files; not part of this spec):

**Sweep 1 (K varies, dispatch=always-bf16):**

| K | bf16_frac | gsm8k em | tput (tok/s) | TTFT p50 | ITL p50 |
|---|-----------|----------|--------------|----------|---------|
| 0 | 0% | … | … | … | … |
| 384 | 6.25% | … | … | … | … |
| … | … | … | … | … | … |
| 3072 | 50% | … | … | … | … |

**Sweep 2 (K=3072 fixed):**

| variant | policy | hot_pct or threshold | gsm8k em | tput | TTFT p50 | ITL p50 |
|---------|--------|----------------------|----------|------|----------|---------|
| hot0 | random | 0% | … | … | … | … |
| … | … | … | … | … | … | … |
| thr512 | expert_batch | 512 | … | … | … | … |

## Out of scope

- Multi-benchmark sweep (mmlu, hellaswag, …). Slots in trivially via new `results/<task>/...` dirs and re-runs of the driver with a different `--task`. Not implemented in this spec.
- Multi-GPU / TP>1.
- Sweeping `request_rate` for accuracy. Single fixed traffic config per run.
- Plotting / table rendering. The spec stops at JSON outputs.
- Confidence intervals / repeated runs. Single seed, single pass per config.

## Risks & open notes

- **Sweep 1 K=0 edge case** — `assign_experts.py` currently logs an error when `bf16_budget_bytes == 0` but still emits outputs; here we *intend* K=0, so the error log is expected and ignorable. The patched `heter_config.json` (threshold=0, no heter experts) is the all-INT4 baseline.
- **Sweep 2 base must be regenerated** if `serving_profile/dynamic/configs/base/` was produced with a different K than 3072 — we own our own base under `accuracy_profile/configs/sweep_dynamic_dispatch/base/`, so no shared state.
- **`bench_eval` output schema** — driver assumes one `gsm8k` task per invocation; if `simple_evaluate` returns no requests (e.g. `--limit` filtered everything), it raises. We pass no `--limit`, so the full ~1.3k items run.
