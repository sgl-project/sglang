# Static Expert Precision Assignment

**Date:** 2026-04-19
**Status:** Draft (supersedes `expert-assignment-design.md` for the SLO-driven input layer)
**Scope:** Offline planner that picks which experts are INT4-only vs heter (dual BF16+INT4) for a target SLO. Produces a `--heter-precision-config` JSON ready for `sglang.launch_server`. Single-GPU only (TP=1). Dynamic / runtime policy is out of scope.

## Problem

`HeterFusedMoE` supports three expert precision categories per layer:

- **INT4-only** — only INT4 (Marlin) weights, always runs INT4 kernel.
- **Heter (dual)** — BF16 + INT4 weights, per-batch policy chooses which to run.
- **BF16-only** — only BF16; used by sensitivity sweeps, not in serving.

For online serving, we must decide, across all MoE layers, which experts get dual-precision vs INT4-only. More heter experts = better accuracy but more VRAM for BF16 weight copies. The assignment must fit within the VRAM left over after KV cache, activations, INT4 weights, and non-MoE weights, at the operator's SLO.

## Inputs

| Input | Source | Notes |
|-------|--------|-------|
| Per-layer PPL sensitivity | `per_moe_layer_sensitivity/results/summary.json` | `per_layer[L].ppl_increase` — WikiText-2 ΔPPL when layer L is all INT4. |
| Per-expert L2 sensitivity | `per_expert_sensitivity/results/summary.json` | `per_layer[L].experts[E].sensitivity` — `‖mixed − bf16‖₂` when expert E alone is INT4. |
| Per-expert token count | same file | `per_layer[L].experts[E].token_count` — experts with zero routed tokens force score 0. |
| GPU VRAM | `torch.cuda.get_device_properties(gpu).total_memory` | auto-detected |
| Model config | `AutoConfig.from_pretrained(model_path)` | `num_experts`, `hidden_size`, `moe_intermediate_size`, `num_hidden_layers`, `num_key_value_heads`, `head_dim` |
| SLO: `max_concurrency` | CLI | max concurrent requests in flight |
| SLO: `max_prompt_len` | CLI | max prompt tokens per request |
| SLO: `max_output_len` | CLI | max generated tokens per request |
| Budget knob: `kv_reserve_frac` | CLI, default 0.5 | fraction of worst-case KV we actually reserve (see § KV policy) |
| Budget knob: `headroom_gb` | CLI, default 2.0 | absolute floor for CUDA context / fragmentation |
| Budget knob: `headroom_frac` | CLI, default 0.05 | fraction-of-VRAM headroom; actual headroom = `max(gb, frac × total)` |
| INT4 checkpoint | CLI | path to GPTQ-INT4 checkpoint (for the generated config; not loaded) |

## Output (in `--out_dir`)

- `int4_only_experts.json` — per-layer lists of expert IDs that are INT4-only (experts not listed are heter):
  ```json
  {"0": [3, 7, 12, ...], "1": [0, 5, 8, ...], ...}
  ```
- `heter_config.json` — complete config for `--heter-precision-config`:
  ```json
  {
    "groups": [
      {"name": "cold", "num_bits": 4, "group_size": 128, "checkpoint": "<int4_ckpt>"},
      {"name": "hot",  "num_bits": 16}
    ],
    "policy": "expert_batch",
    "policy_params": {"threshold": 128},
    "int4_only_experts_file": "<abs path to int4_only_experts.json>"
  }
  ```
- `assignment_report.json` — SLO echo, full VRAM breakdown, `K`, per-layer heter/int4 counts, top scores, layers forced to INT4-only by non-positive PPL.

## CLI

```bash
python assign_experts.py \
    --layer_sensitivity   ../per_moe_layer_sensitivity/results/summary.json \
    --expert_sensitivity  ../per_expert_sensitivity/results/summary.json \
    --model_path      /data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e77... \
    --int4_checkpoint /data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4... \
    --max_concurrency 128 \
    --max_prompt_len  4096 \
    --max_output_len  1024 \
    --kv_reserve_frac 0.5 \
    --headroom_gb     2.0 \
    --headroom_frac   0.05 \
    --gpu     0 \
    --out_dir ./output/
```

`--group_size` defaults to 128 (Qwen3 GPTQ). The script fails loudly if the sensitivity JSONs don't cover all MoE layers and all experts.

## VRAM budget

```
# --- Fixed terms (workload-independent) -----------------------------------
GPU_VRAM       = device total memory
NON_MOE_VRAM   = 2 × non_moe_param_count                   # embeds, norms, attn, router gates
INT4_VRAM      = num_experts × num_moe_layers × int4_expert_size(H, I, group_size)
HEADROOM       = max(headroom_gb, headroom_frac × GPU_VRAM)

# --- Workload terms -------------------------------------------------------
peak_tokens    = max_concurrency × (max_prompt_len + max_output_len)
kv_per_token   = 2 × num_kv_heads × head_dim × num_layers × 2       # K+V, bf16
KV_VRAM        = kv_reserve_frac × peak_tokens × kv_per_token
ACT_VRAM       = prefill_budget_tokens × hidden_size × 2 × safety   # small, ~1–2 GB

# --- BF16 expert budget ---------------------------------------------------
BF16_BUDGET    = GPU_VRAM - NON_MOE_VRAM - INT4_VRAM - HEADROOM - KV_VRAM - ACT_VRAM
bf16_expert_sz = 2 × 3 × H × I                              # fused w13 [2I,H] + w2 [H,I]
K              = clamp(floor(BF16_BUDGET / bf16_expert_sz), 0, num_experts × num_moe_layers)
```

Defaults:

- `prefill_budget_tokens = 16384` (sglang `--max-prefill-tokens` default; not an SLO knob here).
- `safety` factor on activations ≈ 1.5.

The `assignment_report.json` prints every term of the breakdown so the squeezing term is visible.

### Sanity check (Qwen3-30B-A3B on H100 80 GB, SLO from CLI example)

| Term | Value |
|------|-------|
| `NON_MOE` | ≈ 4 GB |
| `INT4` (48 × 128 experts) | ≈ 14 GB |
| `HEADROOM` = max(2, 0.05 × 80) | 4 GB |
| `KV` @ 128 × (4096+1024) × 0.5 × 96 KB | ≈ 32 GB |
| `ACT` | ≈ 1 GB |
| **`BF16_BUDGET`** | **≈ 25 GB** |
| `bf16_expert_sz` = 2×3×2048×768 | 9.4 MB |
| **`K` heter experts** | **≈ 2600 / 6144** |

At `max_concurrency=256` with the same per-request bounds, worst-case KV alone (~64 GB at frac=0.5) already exceeds the post-fixed-term budget — the script would report `K=0` and the operator must relax either concurrency or `kv_reserve_frac`. This is the expected surface for the VRAM term squeezing everything else out.

## KV reservation policy

sglang does not OOM when KV is tight (`schedule_batch.py:2001 retract_decode`, `scheduler.py:2629`). Its actual behavior:

1. Admission control — new requests wait in the queue until enough free KV.
2. Retraction — under pressure, the scheduler kicks the longest-output running requests, releases their KV, and re-queues them to be re-prefilled later (pure recompute; no CPU KV offload in the normal path).
3. Hard abort — only if a single request alone cannot fit.

**Implication:** reserving KV for the worst case is a perf knob, not a correctness guarantee. Undersizing KV costs throughput (fewer reqs in flight) and, under peak, long-output latency (retract → re-prefill → retract). Oversizing KV costs accuracy (fewer BF16 slots).

We default `kv_reserve_frac = 0.5` — workloads rarely hit simultaneous worst-case on every slot, and the accuracy downside of over-reserving is measurable in PPL while the retract downside is invisible until peak burst. Operators who want strict no-retract behavior can pass `--kv_reserve_frac 1.0`.

## Assignment algorithm

```
# 1. Composite score per (layer, expert):
for each MoE layer L:
    S_L = Σ_E expert_sensitivity[L][E]                 # per-layer L2 total
    for each expert E in layer L:
        score[L,E] = max(0, ppl_increase[L]) × (expert_sensitivity[L][E] / S_L)

# 2. Edge cases → force INT4-only (score 0):
#    - ppl_increase[L] ≤ 0       (quantization neutral/helpful for that layer)
#    - token_count[L,E] == 0     (expert unused on calibration data)

# 3. Greedy top-K:
rank all (L, E) by (score DESC, layer_id ASC, expert_id ASC)
top K → heter (dual BF16 + INT4)
rest  → INT4-only
```

Ties broken by `(layer_id, expert_id)` for determinism.

### Additivity assumption

The composite score assumes per-expert PPL contributions are approximately additive. Justification:

- Each token routes to only top-k experts (top-8 of 128 for Qwen3-30B-A3B).
- Quantizing expert E only perturbs tokens routed to E.
- Different experts affect mostly disjoint token sets.
- For disjoint perturbations the combined error is `sqrt(Σ l2²)`, but the greedy ranking is still optimal for uniform-cost allocation.

## Script location

```
expert_precision_assignment/
  per_moe_layer_sensitivity/      # existing
  per_expert_sensitivity/         # existing
  dynamic/                        # existing — see note below
    assign/
      assign_experts.py           # NEW — main entry
      vram_estimator.py           # NEW — budget computation + model-config helpers
    hypothesis/                   # existing invariance sweep
```

**Note on folder name:** `dynamic/` is a misnomer; this work is *static* assignment. It is kept as-is because another experiment is currently running under that path. Rename to `static/` after that experiment completes.

## Validation (post-hoc, not part of this script)

1. **Accuracy** — launch sglang with `heter_config.json`, run `ppl_client.py` on WikiText-2, compare PPL against the BF16 baseline in `per_moe_layer_sensitivity/results/bf16_baseline.json`.
2. **Throughput** — reuse the `dynamic/hypothesis/run_invariance.sh` pattern with the generated config at the target request rate; check TTFT/ITL vs SLO.
3. **VRAM** — `nvidia-smi` during serving confirms actual usage fits within budget.

## Non-goals (deferred)

- Multi-GPU / TP / EP budgeting.
- Dynamic policy tuning (importance weights, dynamic threshold) — covered in a separate doc.
- Activation-memory modeling beyond the simple `prefill_budget_tokens × H` formula.
- Any per-layer threshold or mixed-policy assignment (all heter layers share one `expert_batch` threshold).
