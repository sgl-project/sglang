# Loop 7 — AC-1 closure: oracle-off zero-hot-path + stride reference

The two remaining AC-1 evidence gaps (Codex R7 gap #1) are now closed with durable
artifacts + a CI-backed GPU test.

## 1. Oracle-off byte-identical + zero-alloc under CUDA-graph replay (task4)
`oracle_off_graph_replay_alloc.json` (produced by `oracle_off_replay_alloc.py`;
8×H200 CUDA): with `recall_oracle=false` (the default), the **production
graph-safe DS selector** is captured under CUDA graph and replayed 120 steps:
- `replay_indices_byte_identical_to_eager: true`, `replay_lengths_..._identical: true`
  (eager and replay share `selected_indices` sha `87426fc4269cd235`).
- `replay_allocation_delta_bytes: 0`, `replay_zero_new_allocations: true`
  (`assert_no_alloc_in_region`).
- **verdict PASS** — the "zero hot-path cost" claim is **demonstrated, not
  asserted**: oracle-off selection is byte-for-byte the eager result and replay
  allocates nothing.
- CI backing: `test_double_sparsity_unit.py::test_oracle_off_replay_byte_identical_and_zero_alloc`
  (asserts `sel.config.recall_oracle is False` + byte-identical replay + zero
  alloc), alongside the pre-existing `test_cuda_graph_100_step_replay_matches_eager`
  and `test_cuda_graph_replay_zero_allocations`.

## 2. Oracle sampling-stride reference — default == stride=1 (dense) (task6 remnant)
`oracle_stride_reference.json` (from `oracle_stride_reference.py`):
- The oracle's emitted `stride` field is **1 for all 14,640 R4 success records**
  (`emitted_stride_value_counts: {"1": 14640}`); the hook hardcodes `stride=1`
  (`selection_kernel.py` → `oracle_payload_for_row(stride=1)`).
- **`default_equals_stride1: true`** — the oracle samples score-only recall over
  EVERY needle token (dense, no subsampling), so the "default stride" IS stride=1
  and a separate sparse-stride run is N/A; proven from the emitted records.
- **Dense-DS within-budget reference**: at 1024w (≤2048 tokens, context fits the
  budget ⇒ DS selects densely = dense-DS) DS-default and DS-hybrid both recall
  **100%**; recorded next to the default-stride beyond-budget served recall (4K
  80%, 16K default 6% / hybrid 38%) and the per-length score-only recall@K.

## AC-1 verdict: MET
- Oracle records the required per-trial fields on the live all-reduced score
  tensor; fail-closed; dedicated sink (R1–R4).
- Oracle-off byte-identical + zero-alloc under graph replay — demonstrated (this round).
- Separated baseline (served vs admission) at mem 0.7 with the dense/stride=1
  reference (R0 baseline + this round's stride artifact).
- AC-1.1 dense-within-window post-topK replacement (R1/task5).
All AC-1 sub-criteria now have committed evidence.

## Artifacts
`oracle_off_graph_replay_alloc.json`, `oracle_off_replay_alloc.py`,
`oracle_stride_reference.json`, `oracle_stride_reference.py`,
`test_double_sparsity_unit.py::test_oracle_off_replay_byte_identical_and_zero_alloc`.
