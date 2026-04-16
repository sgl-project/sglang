# Static Expert Precision Assignment

**Date:** 2026-04-16
**Status:** Draft
**Scope:** Static assignment only (which experts are INT4-only vs heter). Dynamic policy brainstorm is separate.

## Problem

HeterFusedMoE supports three expert precision categories per layer:
- **INT4-only**: only INT4 (Marlin) weights, always runs INT4 kernel
- **Heter (dual)**: both BF16 + INT4 weights, dynamically assigned per-batch
- **BF16-only**: only BF16 weights (sensitivity sweep VRAM optimization, not used in serving)

For online serving, we need to decide which experts across all layers get dual-precision (heter) vs INT4-only. More heter experts = better accuracy but more VRAM for BF16 weight copies. The assignment must fit within the GPU VRAM budget after accounting for KV cache, activations, and INT4 weights.

## Inputs

All inputs are pre-computed offline or provided by the user:

| Input | Source | Format |
|-------|--------|--------|
| Per-layer PPL sensitivity | `per_moe_layer_sensitivity/results/summary.json` | `per_layer[L].ppl_increase` — PPL increase when all experts in layer L are INT4 |
| Per-expert L2 sensitivity | `per_expert_sensitivity/results/summary.json` | `per_layer[L].experts[E].sensitivity` — `\|\|mixed - bf16_baseline\|\|_2` when expert E alone is INT4 |
| Per-expert baseline norm | `per_expert_sensitivity/results/summary.json` | `per_layer[L].baseline_norm` — L2 norm of all-BF16 output for the layer |
| GPU VRAM | Auto-detected | `torch.cuda.get_device_properties().total_memory` |
| Max batch size | User-provided (from SLO) | Integer — peak concurrent sequences the server must handle |
| Model config | Auto-detected from model | `num_experts`, `hidden_size`, `moe_intermediate_size`, `num_hidden_layers`, `num_key_value_heads`, `head_dim`, etc. |
| INT4 checkpoint path | User-provided | Path to GPTQ-INT4 checkpoint |

## Output

A heter precision config JSON (for `--heter-precision-config`) containing:

```json
{
  "groups": [
    {"name": "cold", "num_bits": 4, "group_size": 128, "checkpoint": "<path>"},
    {"name": "hot", "num_bits": 16}
  ],
  "policy": "expert_batch",
  "policy_params": {"threshold": 128},
  "int4_only_experts_file": "<path to generated JSON>"
}
```

And the `int4_only_experts_file` JSON:

```json
{
  "0": [3, 7, 12, ...],
  "1": [0, 5, 8, ...],
  ...
}
```

Per-layer lists of expert IDs that are INT4-only. Experts NOT in this file are heter (dual-precision).

## Algorithm

### Phase 1: Compute VRAM weight budget

```
GPU_VRAM = detected total GPU memory
KV_VRAM = kv_cache_size(model_config, max_batch_size, max_seq_len)
ACT_VRAM = activation_memory(model_config, max_batch_size)
INT4_VRAM = int4_weight_size(model_config)  # all experts need INT4 regardless
NON_MOE_VRAM = non_moe_weight_size(model_config)  # embeddings, norms, attention, etc.
OVERHEAD = runtime_overhead()  # CUDA context, fragmentation, etc.

BF16_BUDGET = GPU_VRAM - KV_VRAM - ACT_VRAM - INT4_VRAM - NON_MOE_VRAM - OVERHEAD
```

Each BF16 expert copy costs:
```
bf16_expert_size = 2 bytes × (2 × hidden_size × intermediate_size + hidden_size × intermediate_size)
                 = 2 bytes × 3 × hidden_size × intermediate_size
                 # w13 (gate+up fused) + w2 (down)
```

Number of BF16 expert slots available:
```
K = floor(BF16_BUDGET / bf16_expert_size)
K = clamp(K, 0, total_experts)  # can't exceed 48 × 128 = 6144
```

### Phase 2: Global greedy assignment

**Step 1 — Compute composite score for every expert:**

For each expert E in layer L:
```
sum_l2[L] = sum over all experts E' in layer L of l2_sensitivity[L][E']
score(L, E) = ppl_increase[L] × (l2_sensitivity[L][E] / sum_l2[L])
```

This gives each expert a score in units of "estimated PPL contribution." The per-layer PPL provides cross-layer calibration so L2 norms are comparable across layers.

**Step 2 — Handle edge cases:**

- Layers with `ppl_increase <= 0` (quantization helps or is neutral): all experts get score 0 → all go INT4-only. These layers never consume BF16 budget.
- Experts with `token_count == 0` in the calibration data: sensitivity is 0, go INT4-only.

**Step 3 — Rank and assign:**

1. Sort all experts by composite score descending
2. Top-K experts → **heter** (dual BF16 + INT4)
3. Remaining experts → **INT4-only**
4. Group by layer → write `int4_only_experts_file` JSON

### Additivity assumption

The composite score assumes per-expert PPL contributions are approximately additive. This is reasonable for MoE because:
- Each token routes to only top-k experts (top-8 out of 128 for Qwen3-30B-A3B)
- Quantizing expert E only perturbs tokens routed to E
- Different experts affect mostly disjoint token sets
- For disjoint perturbations, total error is `sqrt(sum(l2^2))`, but the greedy ranking is still optimal for uniform-cost allocation

## VRAM Estimation Details

### KV cache

```
kv_per_token = 2 × num_kv_heads × head_dim × num_layers × 2 bytes  # K + V, bf16
KV_VRAM = kv_per_token × max_batch_size × max_seq_len
```

For Qwen3-30B-A3B (4 KV heads, head_dim=128, 48 layers):
```
kv_per_token = 2 × 4 × 128 × 48 × 2 = 98,304 bytes ≈ 96 KB/token
```

### INT4 weight size (all MoE layers)

```
# Per expert: qweight (packed INT4) + scales + metadata
int4_per_expert ≈ H×I×4/8 + H×I×4/8 + scales + qzeros  # roughly H×I bytes
INT4_VRAM = int4_per_expert × num_experts × num_moe_layers
```

### BF16 expert size

For Qwen3-30B-A3B (H=2048, I=768):
```
bf16_expert_size = 2 × (2×2048×768 + 2048×768) = 2 × 3 × 2048 × 768 = 9.4 MB
```

So K=1000 heter experts costs ~9.2 GB of BF16 weight VRAM.
Total for all 6,144 experts: ~56.6 GB (full BF16 MoE weights).

## Script location

```
expert_precision_assignment/
  per_moe_layer_sensitivity/    # existing
  per_expert_sensitivity/        # existing
  assign/                        # NEW
    assign_experts.py            # main assignment script
    vram_estimator.py            # VRAM budget computation
```

## CLI interface

```bash
python assign_experts.py \
    --layer_sensitivity  ../per_moe_layer_sensitivity/results/summary.json \
    --expert_sensitivity ../per_expert_sensitivity/results/summary.json \
    --model_path /path/to/Qwen3-30B-A3B \
    --int4_checkpoint /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
    --max_batch_size 256 \
    --max_seq_len 4096 \
    --gpu 0 \
    --out_dir ./output/
```

Outputs:
- `output/int4_only_experts.json` — the per-layer INT4-only expert lists
- `output/heter_config.json` — complete config for `--heter-precision-config`
- `output/assignment_report.json` — summary: per-layer counts, VRAM breakdown, top-K scores

## Validation

After generating the config, validate with existing bench infra:

1. **Accuracy:** Launch server with the config, run `ppl_client.py` on WikiText-2, compare PPL against BF16 baseline
2. **Throughput:** Run `bench_serving.sh` with the trace at target request rate, verify TTFT/ITL meet SLO
3. **VRAM:** Check `nvidia-smi` during serving to confirm actual usage fits within budget

## Dynamic policy TODOs (pending discussion with professor)

### Remove global batch short-circuit from `ExpertBatchGatedHeterDispatch`

`should_skip_group()` (heter_policy.py:452) skips the BF16 kernel when
`num_tokens < threshold`. This branch is baked into CUDA graphs at capture
time and cannot be changed dynamically. More importantly, it encodes the
**wrong assumption** for the SLO-aware direction: small batch should mean
*more* BF16 (throughput headroom → spend on accuracy), not less.

**Plan:** Remove the `should_skip_group` override from `ExpertBatchGatedHeterDispatch`
(the base class returns `False`, so both kernels always run). The BF16 kernel
with all-zero weights is a no-op — one kernel launch overhead per layer, likely
negligible vs the INT4 kernel.

**Keep:** The layer-level skip in `heter_moe.py:700` (`if self._num_bf16_experts == 0: continue`)
is safe — it's a static, load-time fact (no BF16 weights exist at all for this layer).

### Dynamic policy design: importance-weighted threshold

Extend the per-expert gating from raw token count to an importance-weighted score:

```
effective_count[e] = token_count[e] × importance_weight[e]
is_hot = effective_count >= threshold
```

- `importance_weight[e]` is a per-expert, per-layer static weight (precomputed offline).
  Highly sensitive experts get large weights → always clear threshold → always BF16.
  Insensitive experts get weight ~1.0 → pure load-based gating.
- `threshold` is the single dynamic knob, tunable per-batch based on SLO.

**Key properties:**
- No architectural change to dispatch — still a per-expert score vs threshold comparison
- Sensitivity baked into importance weights (static); load-awareness preserved via token counts
- Among equally-important experts, hot ones (more tokens) get BF16 first
- `_assign()` uses GPU tensor ops → threshold is dynamically adjustable without CUDA graph issues
  (once the `should_skip_group` short-circuit is removed)

**Threshold tuning via groupGEMM profiling:**
- Profile BF16 and INT4 groupGEMM latency at various expert counts / token distributions
- Given a batch size + candidate threshold → predict how many experts qualify for BF16
  → predict BF16 groupGEMM time + INT4 groupGEMM time → predict per-layer latency
- Set threshold = max value where predicted latency meets TTFT/ITL SLO
- Profiling infra already exists: `test/test_heter_moe/e2e/bench_serving/`

**Importance weight derivation (open exploration):**
- Could be a function of the static assignment composite score
  (`ppl_increase[L] × l2[L][E] / sum_l2[L]`)
- Could be learned from profiling data (e.g., grid search over weight schemes,
  optimize for best accuracy at each SLO point)
- This is an exploration step — try simple mappings first, then consider learned approaches

### Other future directions

- Per-layer thresholds (different layers may have different latency/accuracy trade-offs)
- Whether to dynamically upgrade INT4-only experts to BF16 when load is low
  (requires they have BF16 weights → changes the static assignment boundary)
- Runtime-adaptive threshold tuning based on observed TTFT/ITL (feedback loop)
