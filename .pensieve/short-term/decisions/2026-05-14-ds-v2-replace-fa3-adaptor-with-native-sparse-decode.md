---
id: 2026-05-14-ds-v2-replace-fa3-adaptor-with-native-sparse-decode
type: decision
title: DS v2 replaces FA3-page-table sparse path with a native sparse-decode kernel pipeline
status: active
created: 2026-05-14
updated: 2026-05-14
tags: [double-sparsity, attention, sparse-decode, fa3, triton]
---

# DS v2 replaces FA3-page-table sparse path with a native sparse-decode kernel pipeline

## One-line Conclusion
> For decode, `DoubleSparsityAlgorithm.try_native_sparse_decode` runs a self-contained Triton pipeline (score → torch.topk → fused build_selected_physical → split-K sparse attention) instead of writing into FA3's page table, eliminating the per-layer metadata-rewrite cost that dominated legacy DS-on GPU time.

## Context Links
- Based on: [[knowledge/ds-native-sparse-decode-pareto/content]]
- Leads to: [[maxims/cuda-graph-capture-rejects-host-syncs-in-eligibility-gates]]
- Related: [[maxims/preserve-user-visible-behavior-as-a-hard-rule]]

## Context

The v2 branch initially threaded sparse decode through `DSFlashAttentionAdaptor`,
which rewrote FA3's page-table / `cu_seqlens_k` / `cache_seqlens_int32` in-place
each layer so FA3 would attend only to selected tokens. nsys at 70B/TP=8/32K
showed this path was ~92% of DS-on GPU time, split between:

  * `_ds_select_stage2_merge_kernel` (51% — single Triton program with constexpr
    `num_blocks` that wouldn't compile at 128K)
  * `ds_union_per_batch` torch-on-CUDA ~20-op pipeline (61% of selection cost,
    dominated by Python kernel-launch overhead)
  * FA3 page-table rewrite ops (`prepare_varlen_num_blocks` 10× ratio,
    `index_put` 356× ratio in the nsys diff)

DS-on TBT at 32K/bs=1 was 100.14 ms vs DS-off 8.52 ms — 11.76× slower than
dense, a regression so large the legacy v2 path never produced a 128K JSON.

## Problem

Two distinct failure modes:

  * `_ds_select_stage2_merge_kernel` takes `NUM_CANDIDATES_PADDED` and
    `EFFECTIVE_BUDGET_PADDED` as `tl.constexpr` and runs
    `for k in tl.static_range(EFFECTIVE_BUDGET): ...` with `tl.where` over a
    `[NUM_CANDIDATES_PADDED]` register tensor. At 128K context with
    `block_t=2048`, `num_blocks=64`, `k_block=64`, `NUM_CANDIDATES=4096` →
    LLVM O3 does not converge in 7+ minutes.
  * Even where the stage-2 kernel compiles, the FA3 page-table rewrite plus
    the ~20-op union path costs >9 ms / decode step at bs=1, far more than
    the dense forward they're replacing.

The 1106-line v1 sparse-decode kernel from PR #22992 — split into
`_sparse_fwd_kernel_flash_decode_stage1/2/3` — already exists upstream and
solves the same problem with its own attention kernel (not FA3) that loads
K/V only at provided indices.

## Alternatives Considered

  * **Optimize stage-2 merge (refactor `num_blocks` from constexpr to runtime)** —
    Solves only the compile-time symptom, leaves the union pipeline + FA3
    page-table rewrite intact. Wouldn't close the 9 ms gap.
  * **Adopt v1's full backend wholesale** — Drops v2's K-label *side cache*
    (which doesn't pollute the KV pool's allocation accounting), v2's
    capture-safe scratch pattern, v2's sink/recent window semantics, and the
    `H_kv_local==1` GQA-shared-set specialization. Too much regression for
    one feature delta.
  * **FlashInfer's generic sparse attention** — Doesn't eliminate the
    selection-side overhead at all, and adds a third backend to maintain.

## Decision

Port the v1 sparse-decode Triton kernels into v2's tree at
`python/sglang/srt/layers/attention/triton_ops/double_sparsity_native_decode.py`,
adapted to read K_label from v2's side cache (`DoubleSparsityAlgorithm.k_label`,
not v1's `DoubleSparseTokenToKVPool.label_buffer`). Layer dispatch in
`RadixAttention.forward` tries `try_native_sparse_decode` first; only on
return-None does it fall through to the legacy FA3 + adaptor path.
Specialize selection so one selected set per local kv head is shared by the
local GQA query heads (v2 design departure from v1's per-query-head topk).
Build_selected_physical is a single Triton kernel, not torch ops.

`_handle_piecewise_cuda_graph` already disables piecewise compile when
`enable_double_sparsity` is set; the main `cuda_graph_runner` stays on. The
native dispatch path must be capture-safe (see related maxim).

## Consequence

  * 70B/TP=8/128K/conc=32: DS-on TBT 31.19 ms vs DS-off 34.68 ms (ratio
    **0.8995 ≤ 0.90**, visible-win PASS); NIAH 0.90 vs dense 0.80 (delta
    +0.10, quality-guard PASS). Both gates pass at the same operating point
    at `token_budget=8192`.
  * Synthetic per-phase profile drops legacy DS-on's 91 ms selection overhead
    to ~12 ms (8× faster); end-to-end TBT is bounded by `total_selected`,
    not `seq_len` (sparse-attn microbench: 37 µs at selected=512 across
    32K/64K/128K, ≤2% jitter).
  * Legacy stage-2 capacity guard demoted to a warning in `model_runner.py`
    — native path doesn't touch the constrained kernels.
  * Legacy path remains as fallback for `bs > scratch_max_bs` (sized from
    `server_args.max_running_requests`).
  * The perf/quality Pareto across `token_budget` is now calibration-bound,
    not kernel-bound — wikitext calibration limits NIAH at low budgets even
    though kernels are correct (see [[knowledge/ds-native-sparse-decode-pareto/content]]).

## Exploration Reduction

  * **What to ask less next time:** "Why is DS-on so slow at long context?"
    The legacy v2 path was selection-bound by torch-op launch overhead and
    FA3 metadata-rewrite cost, not by kernel compute. The native path runs
    one Triton selection + one Triton sparse-attention; its TBT is flat
    across `seq_len`.
  * **What to look up less next time:** v1 PR #22992 already has the
    sparse-decode kernels (`_sparse_fwd_kernel_flash_decode_stage1/2/3`);
    fetch via `git fetch https://github.com/Jiminator/sglang.git
    dev/double-sparsity-reintro:refs/pr/22992`.
  * **Invalidation condition:** If `cuda_graph_runner` adds capture for
    long-running selection state, or if FA3 grows a fused
    sparse-attention-with-index-list kernel, revisit. Either would let the
    legacy path achieve the native path's bound.
