# Round 0 Contract

## Mainline Objective

Complete the AC-0 architecture rotation: rename the entire Double Sparsity package from page-level to token-level storage, rewrite the adapter to <150 LOC, update all call sites, and migrate the 150 unit tests. This establishes the token-level foundation that every subsequent AC depends on.

## Target ACs

- **AC-0** (Architecture rotation): All rename + shape + adapter + config + validator + deepseek_v2.py + cuda_graph changes land in a single atomic set of commits.
- **AC-13** (Regression gate): 150 unit tests pass (same count) after `task-ac0-tests` completes.

## Blocking Side Issues in Scope

None known.

## Queued Side Issues Out of Scope

- task-m1-hook (AC-1): kv_b_proj hook at dsa_backend.py — depends on AC-0 but is Phase A2, not Phase A1.
- task-m2-rangemask (AC-3): per-request token range ownership — also Phase A2.
- All hardware runs (AC-1 hwtest, AC-4 hwrun, AC-6 hwrun, AC-8, AC-9, etc.).

## Round Success Criteria

1. `from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write` succeeds.
2. `from sglang.srt.layers.attention.double_sparsity import PageSignatureTable` raises `ImportError`.
3. `pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py` passes with 150 tests.
4. `page_table_adapter.py` is < 150 LOC and implements logical→physical via `req_to_token` gather.
5. `validator.py` references `dsa_prefill_backend` / `dsa_decode_backend`, not the dead `nsa_*` variants.
6. `_bind_double_sparsity_runtime_data` in `deepseek_v2.py` derives `max_tokens = req_to_token_pool.size`.

## Notes

- Cancel signal per plan: if AC-0 is not closed by end of Round 1, the estimate is wrong — stop and re-scope.
- task-ac0-cuda-graph depends on task-m2-rangemask for the ownership mask parameter; the CUDA graph module can be updated for the new token-level shapes and ABI while leaving the ownership mask parameter as a placeholder (None/optional) until M2 lands in a later round.
- Do NOT fix `page_signature_write.py` in-place; instead create `token_label_write.py` with the new token-level interface and delete the old file.
