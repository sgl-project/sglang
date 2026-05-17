---
id: 2026-05-17-ftka-evaluation-scoped-to-selector-only
type: decision
title: FTKA evaluation — selector-only scope; measured-out and REJECTED for DS production
status: active
created: 2026-05-17
updated: 2026-05-17
tags: [double-sparsity, ftka, sparse-attention, selector, benchmark, measured]
---

# FTKA evaluation scoped to selector-only; score substitution rejected on layout grounds

## One-line Conclusion
> Evaluate `tsinghua-ideal/flash-topk-attention` (FTKA) as an
> opt-in **selector-only** substitute (`ftka_raft_topk` backend); do
> **not** wire FTKA's `batched_sparse_gemv` as a score-kernel
> replacement, because the layout transform from our flat K-label cache
> to FTKA's paged-cache view is per-step and dominates the kernel-level
> speedup.

## Context Links
- Based on: [[knowledge/ds-flashinfer-top-k-page-table-boundaries/content]]
- Based on: [[knowledge/ds-native-sparse-decode-pareto/content]]
- Leads to: [[knowledge/ds-external-sparse-kernel-layout-mismatch/content]]
- Related: [[maxims/cuda-graph-capture-rejects-host-syncs-in-eligibility-gates]]

## Context

DS v2 ships with two selector backends — `torch` (production default)
and `flashinfer_topk_page_table` (wired, microbench-fast at bs≥16, but
capture-unsafe inside SGLang's nested-stream graph context). FTKA
(commit `d8803b29...`) provides two kernels relevant to our score+select
pipeline: a RAFT radix `raft_topk` and a paged-cache `batched_sparse_gemv`.

We needed to decide which FTKA components to consider for integration
and at what scope, before doing any engineering work, so the
microbench harness wouldn't accidentally drift into algorithm change.

## Problem

- DS's K-label side cache is `[T_pool, H_kv, S]` indexed by physical
  row id from `req_to_token[bs, max_ctx]`. There is no "page table"
  beyond the per-request `req_to_token` row.
- FTKA's GEMV expects `kv_indices` (flat physical-id list),
  `kv_indptr` (per-request page-count cumsum), and
  `kv_last_page_len`. With `page_size=1` this is buildable in O(bs ×
  max_ctx) per step.
- That transform is *per decode step* — `req_to_token` changes whenever
  active sequences advance — so the layout cost cannot be amortized
  across steps. At bs=32 / max_ctx=128K it materializes ~16 MB of
  int32 cells inside what should be a tight decode hot path.
- Meanwhile, `raft_topk` substitutes only the top-k step. The same
  per-step cost concern does not apply because RAFT consumes the
  pre-existing score buffer directly.

## Alternatives Considered

- **Option A — wire FTKA at both score AND top-k stages.** Rejected:
  the layout transform turns a kernel-level speedup into a wash (or
  worse) at production shapes, and adds an allocation-per-step path
  that breaks CUDA-graph capture.
- **Option B — rebuild our K-label cache as a paged structure to
  match FTKA.** Rejected for this pass: it would require coordinated
  changes across `K_label` allocation (`DoubleSparsityAlgorithm.
  _allocate_native_scratch`), the score kernel, the
  K-label set/append path, and the calibration tooling. Cost-out-of-
  proportion to the speculative win.
- **Option C — wire FTKA at top-k only, gated as an opt-in selector
  backend.** Chosen. Reuses the existing `_BaseSelector` contract and
  the `_build_selected_physical` Triton path, so the only new code is
  one selector class + microbench wiring.
- **Option D — port Twilight's top-p pruner / adaptive budget /
  weight estimator.** Rejected: those change token-selection
  semantics. Out of scope for this evaluation pass.

## Decision

1. Add `ftka_raft_topk` to `SUPPORTED_SELECTOR_BACKENDS`. Constructor
   raises `RuntimeError` with installation guidance when `ftka` is not
   importable. Default remains `torch`.
2. Add `microbench_ftka_backends.py` that measures all four paths
   (torch / flashinfer / ftka_raft_topk / ftka_gemv+ftka_topk) at the
   PLAN-spec shape grid. Path P4 is included specifically to *measure*
   the layout-transform cost, not because we expect to integrate it.
3. Production default is unchanged. No CLI behavior change for users
   who don't pass `--double-sparsity-selector-backend`.

## Consequence

- Future sessions wanting to evaluate a different external sparse-attn
  kernel can follow the same template: opt-in selector backend +
  microbench row + parity test. The selector contract is the integration
  surface, not the kernel.
- A future change that introduces a paged K-label cache (for any
  reason, e.g. prefix-cache integration) re-opens FTKA `batched_sparse_
  gemv` as a candidate. At that point the gate is "score parity within
  bf16 tol + ≥1.10× at 128K bs≥16 + graph-safe."

## Exploration Reduction

- **What to ask less next time:** "Should we wire FTKA's score kernel?"
  → No, not against the current flat K-label layout. The PLAN gate
  was structurally adverse before measurement even started.
- **What to look up less next time:** the four-path matrix
  (torch / flashinfer / ftka_topk / ftka_gemv+ftka_topk) is in
  `benchmark/double_sparsity/repro_session/microbench_ftka_backends.py`
  and the per-component recommendation table is in
  `benchmark/double_sparsity/repro_session/ftka_comparison/REPORT.md`.
- **Invalidation condition:** if (a) DS adopts a paged K-label cache,
  or (b) FTKA upstream changes `batched_sparse_gemv` to accept our
  flat-indexed layout, or (c) Twilight's algorithmic ideas are
  separately approved for production (different decision, different
  scope), re-open this and re-evaluate.

## Measured Outcome (2026-05-17, post-install)

After resolving FTKA build glue against `flashinfer 0.6.11.post1` /
`torch 2.11.0+cu130` / CUDA 13 (carved-out `ftka_topk_only.cu` TU; see
REPORT §"FTKA build-glue port"), the full 76-shape microbench ran with
P3 ftka_raft_topk actually measured. Parity 76/76, graph capture 76/76,
top_k up to 8192 supported. But the speedup vs `torch.topk` is
**strongly context-dependent**:

| `max_ctx` | n | min ratio | median | max | regime |
|---:|---:|---:|---:|---:|---|
| 32K  | 25 | 1.97× | **2.36×** | 3.06× | torch under-utilized; RAFT wins |
| 64K  | 26 | 0.85× | **1.00×** | 1.19× | wash |
| 128K | 25 | 0.47× | **0.63×** | 0.83× | RAFT loses everywhere |

**DS production target is `ctx ≥ 64K`, typically 128K.** Of the 20
shapes in the production envelope (`bs ≥ 16, ctx ≥ 64K`), only **2**
pass the ≥1.15× speedup gate and **14** fail outright (<0.95×). At
the README headline operating points (bs ∈ {16, 32}, ctx = 128K,
top_k ∈ {1024, 2048}), RAFT runs at **0.63×–0.79×** of torch — i.e.,
21–37% slower.

Root cause is the **one-block-per-batch radix** in RAFT's
`decode_select_k` (see `[[knowledge/external-topk-one-block-radix-
scaling-regime/content]]`). Documented in the report; no production
code changed.

**Final decision: REJECT P3 (`ftka_raft_topk`) for production.** PLAN
gate "≥1.15× at bs≥16 / top_k ∈ {1024, 2048}" fails at every
ctx ≥ 64K production shape. e2e (README headline) was not run —
PLAN-gated on P3 passing the microbench gate, which it does not.

P4 stays structurally rejected per the original decision body
(`BatchedSparseGEMV<128, ...>` compile-time vs DS S=32) — confirmed by
source at `csrc/src/ftka_ops.cu:70`.

Default `--double-sparsity-selector-backend=torch` is unchanged.
