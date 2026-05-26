# Attention Unit Test Mutation Journal

Date: 2026-05-26
Worktree: `/sgl-workspace/sglang-mut` (detached from `cheng/test/attention-runner-combination`)
GPU: H200 #1 (`CUDA_VISIBLE_DEVICES=1`)
PYTHONPATH: `/sgl-workspace/sglang-mut/python:$PYTHONPATH`

## Goal

Validate whether the unit tests under `test/manual/attention/unittest/` actually detect
bugs in the production attention backend code. Method: inject ~28 small mutations into
backend production code (overwhelmingly in `init_forward_metadata*` variants and spec
metadata builders, no `sm_scale` changes), run the relevant test files, record
CAUGHT/MISSED, then revert before the next mutation.

The main repo at `/sgl-workspace/sglang` was never touched. All edits happened in the
isolated `sglang-mut` worktree on a detached HEAD; after each mutation `git checkout --
<file>` reverted it.

## Headline Result

- **28 mutations total**
- **18 CAUGHT** (64%) → tests detected the bug
- **10 MISSED** (36%) → tests passed despite the bug

Coverage is strong for Triton dense, FlashInfer dense (eager + capture + replay), MLA
Triton/FlashInfer eager and graph paths, torch_native extend, and the EAGLE
`generate_attn_arg_prefill` spec-metadata builder. Coverage is weak for SWA above-window
metadata, FlashMLA target_verify metadata, and the `HybridLinearAttnBackend` /
`MambaAttnBackendBase` dispatch layer.

## Mutation Matrix

| #  | File:line                                 | Function                                           | Mutation                                                    | Test file                            | Result   | Detail                                                       |
|----|-------------------------------------------|----------------------------------------------------|-------------------------------------------------------------|--------------------------------------|----------|--------------------------------------------------------------|
| 1  | triton_backend.py:300                     | init_forward_metadata (eager decode)               | `cumsum(seq_lens - 1)` off-by-one                           | dense/test_triton.py                 | CAUGHT   | 6/8 failed, max abs diff 0.14                                |
| 2  | triton_backend.py:475                     | init_forward_metadata (eager extend)               | swap `extend_seq_lens` → `extend_prefix_lens` in qo_indptr  | dense/test_triton.py                 | CAUGHT   | 9 failures, NaN outputs                                      |
| 3  | triton_backend.py:764                     | init_forward_metadata_replay_cuda_graph (decode)   | skip kv_indptr cumsum update (stale capture-time)           | dense/test_triton.py                 | CAUGHT   | 3 failures in cuda_graph_decode tests, max abs diff 1.51     |
| 4  | triton_backend.py:849                     | init_forward_metadata_replay_cuda_graph (verify)   | skip mask_indptr cumsum                                     | dense/test_triton.py                 | CAUGHT   | 3 failures in spec_verify_cuda_graph_cases                   |
| 5  | triton_backend.py:786                     | init_forward_metadata_replay_cuda_graph (SWA dec)  | `sliding_window_size + 1`                                   | swa/test_triton.py                   | MISSED   | SWA decode test uses within-window lengths only              |
| 6  | triton_backend.py:832                     | init_forward_metadata_replay_cuda_graph (SWA tgt)  | `sliding_window_size + 1`                                   | swa/test_triton.py                   | MISSED   | window+1 fits comfortably above test seq_lens                |
| 7  | triton_backend.py:412                     | init_forward_metadata (eager target_verify)        | `cumsum(seq_mask_len[:bs]) + 1`                             | dense/test_triton.py                 | CAUGHT   | 8 failures                                                   |
| 8  | triton_backend.py:862                     | init_forward_metadata_replay_cuda_graph (draft_ex) | skip kv_indptr cumsum in draft_extend_v2                    | dense/test_triton.py                 | CAUGHT   | 2 failures in draft_extend_v2 graph tests                    |
| 9  | flashinfer_backend.py:479                 | init_forward_metadata (eager extend)               | `prefix_lens = forward_batch.seq_lens` (full instead of pre)| dense/test_flashinfer.py             | CAUGHT   | 13 errors (crash)                                            |
| 10 | flashinfer_backend.py:727                 | init_forward_metadata_replay_cuda_graph (decode)   | `seq_lens[:bs] - 1`                                         | dense/test_flashinfer.py             | CAUGHT   | 4 failures                                                   |
| 11 | flashinfer_backend.py:746                 | init_forward_metadata_replay_cuda_graph (verify)   | `spec_info=None` (drop spec_info)                           | dense/test_flashinfer.py             | CAUGHT   | 3 errors                                                     |
| 12 | flashinfer_mla_backend.py:318             | init_forward_metadata (eager target_verify)        | `prefix_lens = forward_batch.seq_lens` (full)               | mla/test_flashinfer.py               | MISSED   | target_verify path apparently robust to wrong prefix_lens    |
| 13 | flashinfer_mla_backend.py:471             | init_forward_metadata_replay_cuda_graph (decode)   | skip cuda_graph_kv_indptr_cpu cumsum                        | mla/test_flashinfer.py               | CAUGHT   | 2 failures                                                   |
| 14 | flashmla_backend.py:120-121               | init_forward_metadata (eager target_verify)        | `seq_lens + num_draft_tokens + 1` (extra token)             | mla/test_flashmla.py                 | MISSED   | adding 1 has no effect on existing test seq_lens             |
| 15 | flashmla_backend.py:120-121               | init_forward_metadata (eager target_verify)        | drop `+ num_draft_tokens` entirely                          | mla/test_flashmla.py                 | MISSED   | test still passed — eager target_verify metadata not asserted? |
| 16 | flashmla_backend.py:347-348               | init_forward_metadata_replay_cuda_graph (verify)   | drop `+ num_draft_tokens`                                   | mla/test_flashmla.py                 | MISSED   | replay target_verify metadata robust to draft-count error   |
| 17 | triton_backend.py:812                     | init_forward_metadata_replay_cuda_graph (verify)   | skip kv_indptr cumsum                                       | dense + mla/test_triton.py           | CAUGHT   | dense 3 fail, mla 1 fail                                     |
| 18 | torch_native_backend.py:98                | _run_sdpa_forward_extend                           | `prefill_seq_len_q = extend_prefix_lens[seq_idx] - 1`       | dense/test_torch_native.py           | CAUGHT   | 10 errors                                                    |
| 19 | hybrid_linear_attn_backend.py:850         | init_forward_metadata_replay_cuda_graph (Hybrid)   | only init first sub-backend (`attn_backend_list[:1]`)       | gdn/test_triton.py                   | MISSED   | GDN test fixture doesn't go through HybridLinearAttnBackend? |
| 20 | hybrid_linear_attn_backend.py:786         | init_forward_metadata (Hybrid eager)               | skip first sub-backend (`attn_backend_list[1:]`)            | gdn/test_triton.py                   | MISSED   | same — dispatch layer not exercised                          |
| 21 | hybrid_linear_attn_backend.py:417         | init_forward_metadata_replay_cuda_graph (Mamba)    | `seq_lens_cpu - 1`                                          | gdn/test_triton.py                   | MISSED   | replay path not exercised by GDN test or robust to shift     |
| 22 | speculative/eagle_info.py:197             | generate_attn_arg_prefill                          | `paged_kernel_lens + draft_token_num + 1` (extra token)    | dense/test_flashinfer.py             | CAUGHT   | 5 failures (FlashInfer DRAFT_EXTEND covered)                 |
| 23 | triton_backend.py:1395 (MLA Class 2)      | init_forward_metadata_replay_cuda_graph (MLA-MS)   | skip `get_num_kv_splits` computation                        | mla + dense/test_triton.py           | MISSED   | MLA multi-step backend not exercised by unit tests           |
| 24 | triton_backend.py:652 (capture verify)    | init_forward_metadata_capture_cuda_graph (verify)  | `cumsum(seq_lens - 1)` in capture                           | dense/test_triton.py                 | CAUGHT   | 3 failures                                                   |
| 25 | flashinfer_backend.py:438                 | init_forward_metadata (eager decode)               | `forward_batch.seq_lens - 1`                                | dense/test_flashinfer.py             | CAUGHT   | 7 failures                                                   |
| 26 | flashinfer_backend.py:590                 | init_forward_metadata_capture_cuda_graph (decode)  | `seq_lens - 1` in capture                                   | dense/test_flashinfer.py             | CAUGHT   | 3 failures                                                   |
| 27 | speculative/eagle_info.py:188             | generate_attn_arg_prefill (qo_indptr)              | drop `+1` factor → qo_indptr too short by one entry         | dense/test_flashinfer.py             | CAUGHT   | 5 errors (crash on missing entry)                            |
| 28 | flashmla_backend.py:197                   | init_forward_metadata_capture_cuda_graph (decode)  | `seq_lens - 1` in capture                                   | mla/test_flashmla.py                 | CAUGHT   | 1 failure in cuda_graph_decode case                          |

## Summary of MISSES (coverage gaps to file)

10 mutations escaped detection. Three thematic gaps:

### 1. SWA above-window metadata is not exercised in graph paths (M5, M6)

The SWA `cuda_graph_decode_cases` use `prefix_lens=(1,2,3)` with `window=4`, so all
decode lengths stay within the window. Mutating `sliding_window_size + 1` therefore has
no observable effect — both `M5` (decode replay) and `M6` (target_verify replay) MISSED.

The PLAN.md already notes "Phase 3 SWA graph cases currently use decode lengths within
the configured window; an above-window Triton SWA decode case exposes a backend/reference
semantic mismatch before graph replay". This mutation campaign confirms that gap is real:
**no current SWA graph test would catch a window-size off-by-one bug in
`init_forward_metadata_replay_cuda_graph` or `init_forward_metadata_capture_cuda_graph`.**

Recommendation: re-investigate the above-window Triton SWA reference mismatch and
unblock at least one above-window SWA decode/verify graph case.

### 2. FlashMLA target_verify metadata is under-asserted (M14, M15, M16)

Three independent mutations to FlashMLA target_verify metadata all MISSED:
- M14: adding `+1` to seq_lens
- M15: dropping `+ num_draft_tokens` in eager
- M16: dropping `+ num_draft_tokens` in replay

`test_runner_mode_eagle_verify_cases` and `test_runner_mode_eagle_verify_cuda_graph_cases`
ran green even though the per-request KV span passed into `get_mla_metadata` /
`create_flashmla_kv_indices_triton` was off by `num_draft_tokens`.

Recommendation: add an MLA test case where the verify output is sensitive to the
draft-token region of the KV cache (e.g., draft tokens that diverge from a chain of
all-equal tokens), or assert the contents of the constructed `block_kv_indices` directly.

### 3. HybridLinearAttnBackend dispatch + Mamba metadata isn't exercised (M19, M20, M21)

Mutations to `HybridLinearAttnBackend.init_forward_metadata` and
`MambaAttnBackendBase.init_forward_metadata_replay_cuda_graph` ran green on
`gdn/test_triton.py`. The GDN test fixture likely constructs the linear-attention path
directly (or substitutes a simpler shim) instead of going through the production
hybrid-linear dispatch wrapper.

Recommendation: confirm via inspection whether the GDN unit fixture exercises the
production `HybridLinearAttnBackend.init_forward_metadata*` path. If not, either
plumb the dispatch wrapper into the fixture, or add a focused dispatch-layer test that
verifies sub-backend `init_forward_metadata*` calls actually fire.

### 4. Other isolated misses

- **M12** (`flashinfer_mla_backend.py:318`): `prefix_lens = forward_batch.seq_lens`
  in eager target_verify went undetected. The MLA target_verify path may compute
  effective extend lengths from `draft_token_num` rather than `(seq_lens − prefix_lens)`,
  making `prefix_lens` effectively a no-op in this branch. Worth a follow-up.
- **M23** (`triton_backend.py:1395`): `MultiStepDraftBackend.init_forward_metadata_replay_cuda_graph`
  is not exercised by the unit test matrix. The multi-step draft backend is only
  reachable through full-server production EAGLE draft flow; the Phase 4 production
  draft-runner unit tests do not currently route through it. Worth confirming whether
  this gap is intentional (`ml/test_triton.py` graph tests are draft-extend v2, not
  multi-step v1).

## Caught categories (for completeness)

The other 18 mutations were caught with diff signatures ranging from `max abs diff ~0.05`
to NaN-producing crashes:

- **Eager `init_forward_metadata`** off-by-ones: M1, M2, M7, M9, M18, M25 — all caught.
- **`init_forward_metadata_capture_cuda_graph`** mutations: M24, M26, M28 — all caught.
- **`init_forward_metadata_replay_cuda_graph`** mutations: M3, M4, M8, M10, M11, M13, M17 —
  all caught. Notably this is the highest-value category (stale-state bugs) and the
  tests demonstrably detect them.
- **`generate_attn_arg_prefill` (spec metadata builder)**: M22, M27 — both caught,
  including a subtle off-by-one that only fires through FlashInfer DRAFT_EXTEND.

## Methodology

For each mutation:
1. `Edit` the production file in `/sgl-workspace/sglang-mut`.
2. Run only the relevant test file(s) with
   `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/sgl-workspace/sglang-mut/python:$PYTHONPATH python <test_file>`.
3. Record pass/fail + first failing test name + max abs diff if reported.
4. Revert with `git checkout -- <file>` before the next mutation.

Each mutation modified exactly one production source file. All mutations targeted code
under `python/sglang/srt/layers/attention/` or `python/sglang/srt/speculative/`. None
modified the test sources. No `sm_scale` changes (excluded by user request).
