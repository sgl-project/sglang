# RelayKV Positioning Update: VRAM-Constrained KV Working Set Manager

## Date

2026-04-29

## Summary

RelayKV is repositioned from a KV cache reduction prototype to a **VRAM-constrained KV working set manager**.

The core target is not simply extending context length, but controlling the decode-time KV working set under a fixed GPU-resident KV budget after loading a large quantized dense model.

## Practical Target

Primary practical environment:

```text
GPU:
  RTX 3060 12GB single GPU

Model class:
  dense quantized model

Near-term target:
  Qwen3.5-9B Q4 class

Future target:
  Qwen3.6 9B–12B dense Q4 class

Long-term target:
  Qwen3.6 27B dense with strong weight/KV quantization
```

## Core Assumption

Local LLM users usually load the largest quantized model that fits in VRAM, then use the remaining VRAM for context / KV cache.

Therefore RelayKV should be designed as a system that maximizes usefulness under the small residual KV VRAM budget left after model loading.

## New Definition

```text
RelayKV
= decode-time KV working set manager
  under a fixed GPU-resident KV budget
```

RelayKV may preserve the full KV logically or in cold storage, but it does not assume the full KV is always resident on GPU or always included in the attention working set.

## Working KV Structure

The default working set should be:

```text
working KV
= recent full window
+ always-on anchor blocks
+ retrieved cold blocks within remaining budget
```

### Layer 1: Recent full window

Purpose:

```text
Keep local coherence and short-range generation stable.
```

Properties:

```text
always resident
highest priority
contiguous tail of context
```

### Layer 2: Anchor blocks

Purpose:

```text
Keep stable global instruction / format / identity / task anchors.
```

Properties:

```text
always resident
small budget
usually early context or selected stable blocks
```

### Layer 3: Retrieved cold blocks

Purpose:

```text
Bring back task-relevant old context only when useful.
```

Properties:

```text
selected from cold KV store
limited by remaining budget
retrieval_top_k / scoring controlled
```

## Key Design Goal

The decode-time attention working KV should fit within a fixed budget:

```text
working_kv_budget_mib
or
working_kv_budget_tokens
```

This budget is derived from the residual VRAM after model loading and runtime workspace.

## Evaluation Reframing

Previous RelayKV evaluation focused mainly on:

```text
seq_len
coverage_ratio
working_ratio
mean_abs_diff
```

These remain useful, but are not enough.

New evaluation should explicitly model residual KV VRAM constraints.

## New Evaluation Question

The key question is:

```text
After loading a large quantized dense model and leaving only
512 MiB / 1 GiB / 2 GiB for KV,
which working KV configuration preserves task quality?
```

This is more important than simply asking how long the logical context can become.

## Evaluation Metrics

Core budget metrics:

```text
available_kv_budget_mib
kv_working_budget_tokens
recent_window
anchor_blocks
retrieval_top_k
working_ratio
```

Quality / behavior metrics:

```text
mean_abs_diff
top5_overlap
first_divergence_step
task_accuracy
same_first_code
```

Operational metrics:

```text
planned_resident_kv_mib
planned_cold_kv_mib
host_backup_candidate_kv_mib
working_kv_mib
budget_overflow
```

## Implementation Priority

The next implementation direction should be changed.

### Previous direction

```text
host backup dry-copy
KV layout observation
token pool mapping
eventual CPU copy
```

This was useful for understanding SGLang internals, but it is not the next most important step.

### New priority

```text
1. Add / clarify KV budget mode
2. Keep recent full window stable
3. Treat anchor blocks as always-on memory
4. Select retrieved cold blocks within remaining budget
5. Simulate RTX 3060 12GB residual KV constraints
6. Improve scoring only after budget behavior is explicit
```

## Budget Mode

RelayKV should expose a mode where the user can specify a KV working budget directly.

Possible args:

```text
--relaykv-kv-budget-mib
--relaykv-working-budget-tokens
--relaykv-recent-window
--relaykv-anchor-blocks
--relaykv-retrieval-top-k
```

The budget planner should compute:

```text
total_budget_tokens
recent_tokens
anchor_tokens
remaining_retrieval_tokens
retrieval_top_k_effective
budget_overflow
```

## Budget Allocation

Initial policy:

```text
B_total = kv_working_budget_tokens

B_recent = min(recent_window, B_total)
B_anchor = min(anchor_blocks * block_size, B_total - B_recent)
B_retrieval = max(0, B_total - B_recent - B_anchor)
```

Then:

```text
working_kv = recent + anchor + retrieved_cold_blocks(B_retrieval)
```

If anchor/recent exceed budget:

```text
recent has highest priority
anchor may be clipped or disabled by policy
retrieval becomes 0
```

## RTX 3060 12GB Residual KV Simulation

The evaluation should include explicit residual KV budgets.

Suggested budget points:

```text
512 MiB
1024 MiB
2048 MiB
```

For the observed Qwen2.5-1.5B GQA profile:

```text
kv_bytes_per_token = 28672
```

Budget-to-token conversion:

```text
512 MiB  -> about 18724 KV tokens
1024 MiB -> about 37449 KV tokens
2048 MiB -> about 74898 KV tokens
```

For larger models, this conversion will be much smaller due to more layers / heads / dimensions.

Therefore the planner must not assume that token budgets generalize across models.

## TurboQuant / PolarQuant Relationship

TurboQuant / PolarQuant are not competitors to RelayKV.

They are complementary.

```text
TurboQuant / PolarQuant:
  reduce representation size of model weights and/or KV cache

RelayKV:
  controls which KV blocks are resident and active during decode
```

RelayKV should not hard-code cold KV storage format.

Short-term interface:

```python
ColdKVStore.get_block(block_id) -> fp16/bf16 K/V block
```

Future implementation can replace the backend:

```text
plain KV
or
TurboQuant / PolarQuant compressed KV
  -> dequantize selected block
  -> insert into working KV
```

## Storage Abstraction

RelayKV should introduce or keep a storage abstraction:

```text
ColdKVStore
  get_block(block_id)
  get_blocks(block_ids)
  estimate_block_bytes(block_id)
```

Backends:

```text
PlainHostKVStore
CompressedKVStore
TurboQuantKVStore
PolarQuantKVStore
```

MVP should still use plain fp16/bf16 metadata or no-copy shadow planning.

## Immediate Work Rebuild

The work should now be reorganized into a budget-first track.

### Stop / pause

Pause deeper host backup dry-copy implementation.

Do not proceed to actual CPU tensor copy yet.

### Continue using

Keep the already implemented pieces:

```text
server args
shadow planner
memory estimate
host backup candidate metadata
range metadata
KV layout observation
token pool mapping observation
prefill final guard
```

These are useful diagnostics.

### New next task

Add budget mode / budget planner metadata.

## Proposed Next Phase

### Phase B0: Budget planner design note

Add:

```text
notes/relaykv_budget_mode_design_ja.md
```

### Phase B1: Budget planner metadata

Add planner fields:

```text
available_kv_budget_mib
kv_working_budget_tokens
recent_window_tokens
anchor_budget_tokens
retrieval_budget_tokens
retrieval_top_k_requested
retrieval_top_k_effective
budget_overflow
budget_policy_reason
```

### Phase B2: Shadow log integration

Extend `relaykv_shadow_plan_prefill` to show budget allocation.

### Phase B3: Evaluation script

Add a lightweight script to sweep:

```text
available_kv_budget_mib: 512 / 1024 / 2048
recent_window: 512 / 1024 / 2048 / 4096
anchor_blocks: 0 / 4 / 8
retrieval_top_k: 0 / 2 / 4 / 8
```

Output table fields:

```text
available_kv_budget_mib
kv_working_budget_tokens
recent_window
anchor_blocks
retrieval_top_k_effective
working_kv_mib
working_ratio
budget_overflow
```

### Phase B4: Quality integration

Only after budget behavior is explicit, connect to quality metrics:

```text
mean_abs_diff
top5_overlap
first_divergence_step
task_accuracy
same_first_code
```
