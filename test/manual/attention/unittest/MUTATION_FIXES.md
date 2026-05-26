# Attention Unit Test Mutation Coverage Fixes

Date: 2026-05-26

This worktree closes 7 of the 10 MISSED mutations from the
`MUTATION_JOURNAL.md` on `origin/cheng/test/attention-mutation-testing-journal`.
The remaining 3 are documented as structurally undetectable through forward
output and are recorded here and in `PLAN.md` rather than papered over with
implementation-detail assertions.

All validation runs use
`PYTHONPATH=$WORKTREE/python:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python <test_file>`
so that the worktree's `python/sglang/` reflects the mutated production code
while the foreground repo stays untouched.

## MISSES addressed

### M5 — `triton_backend.py:786`, `init_forward_metadata_replay_cuda_graph` decode SWA, `sliding_window_size + 1`

- **Closure**: Added an above-window decode case
  (`prefix_lens=(7, 8, 9)`, `sliding_window_size=4`) to
  `swa/test_triton.py::CUDA_GRAPH_CASES`. The existing case stayed within the
  window, so `min(seq_lens, sliding_window_size) == min(seq_lens,
  sliding_window_size + 1)` was an invariant and the mutation was invisible.
- **Reference fix**: `dense_attention_reference` previously used
  `key_start = max(0, query_pos - sliding_window_size)` (extend-kernel rule:
  `kv_id >= q_id - window`, `window + 1` keys). The SWA-aware decode backends
  (`triton`, `flashinfer`) instead clip to `min(seq_lens, window)` (exactly
  `window` keys). Reference now picks the matching rule based on
  `case.forward_mode.is_decode()` and `case.backend in
  _SWA_AWARE_DECODE_BACKENDS`. `torch_native` keeps the old extend-style rule
  since it does not implement SWA at all.
- **Validation**: Re-applied `self.sliding_window_size + 1` at
  `triton_backend.py:786` locally, ran
  `swa/test_triton.py::test_runner_mode_cuda_graph_decode_cases`, the
  `runner_cuda_graph_swa_decode_above_window` sub-case failed with
  "Mismatched elements: 130 / 256 (50.8%), greatest abs diff 0.325" before
  revert.

### M6 — `triton_backend.py:832`, `init_forward_metadata_replay_cuda_graph` verify SWA, `sliding_window_size + 1`

- **Outcome**: STRUCTURALLY UNDETECTABLE through forward output.
- **Why**: With the mutation,
  `update_sliding_window_buffer_cuda_graph` writes `window_kv_lens = min(seq_lens,
  window + 1)` and `window_kv_offsets = seq_lens - window_kv_lens` (each +1
  longer / -1 earlier than the unmutated values). Two downstream invariants
  then re-mask out the difference:
  1. The extend kernel applies its own per-token mask
     `kv_id >= q_id - layer.sliding_window_size`, where
     `layer.sliding_window_size` is unmutated. Any extra prefix token the
     mutation adds at the front of `window_kv_indices` is at an absolute
     position below the kernel's threshold and is rejected.
  2. The custom-mask row stride is computed as `cur_seq_len +
     window_kv_offset = window_kv_lens + draft + (seq_lens -
     window_kv_lens) = seq_lens + draft`, which is invariant under a +1
     shift between `window_kv_lens` and `window_kv_offset`.
- **Action taken**: Added an above-window verify case
  (`prefix_lens=(6, 8)`, `sliding_window_size=4`) to
  `swa/test_triton.py::SPEC_VERIFY_CUDA_GRAPH_CASES` for general above-window
  coverage. It does NOT catch M6 on its own (validated by re-applying the
  mutation locally and observing the test still passes), but it does
  exercise an above-window verify path that the suite previously lacked.
  Documented in `PLAN.md` under "Intentionally Untested Mutation-Surfaces".

### M12 — `flashinfer_mla_backend.py:318`, eager target_verify `prefix_lens = forward_batch.seq_lens`

- **Outcome**: STRUCTURALLY UNDETECTABLE (no-op mutation).
- **Why**: `prefix_lens` flows to
  `FlashInferMLAIndicesUpdaterPrefill.call_begin_forward`, which only
  consumes the argument inside the `spec_info is None` branch. The eager
  target_verify path always supplies a non-None `spec_info`, so the value
  passed for `prefix_lens` is dead in this branch.
- **Action taken**: Documented in
  `mla/test_flashinfer.py::test_eager_target_verify_prefix_lens_is_noop`
  with `@unittest.skip` and a full call-graph explanation, plus a matching
  entry in `PLAN.md`.

### M14 — `flashmla_backend.py:120-121`, eager target_verify `seq_lens + num_draft_tokens + 1`

- **Closure**: Added
  `mla/test_flashmla.py::test_eager_target_verify_block_kv_indices_metadata`.
  Uses page-boundary-crossing `prefix_lens=(61, 63)` with `draft=3` so:
  - correct: `seq_lens + draft = (64, 66)`, `cdiv = (1, 2)`, `max_seqlen_pad = 2`;
  - M14: `seq_lens + draft + 1 = (65, 67)`, `cdiv = (2, 2)` — per-row valid
    page-count differs in row 0.
  The test asserts `block_kv_indices` shape and the per-row count of
  `>= 0` entries (eager path initialises to `-1`).
- **Why output-diff missed**: `forward_extend` recomputes
  `cache_seqlens = forward_batch.seq_lens + self.num_draft_tokens` from the
  unmutated batch fields and only reads
  `cdiv(cache_seqlens, PAGE_SIZE)` block entries, so the extra entry the
  mutation writes is never read.
- **Validation**: Re-applied
  `seq_lens_cpu/seq_lens + self.num_draft_tokens + 1` at
  `flashmla_backend.py:120-121`, ran
  `test_eager_target_verify_block_kv_indices_metadata`, failed with
  `assertEqual([2, 2], [1, 2])` on the per-row valid-page-count
  assertion before revert.

### M15 — `flashmla_backend.py:120-121`, eager target_verify drop `+ num_draft_tokens`

- **Closure**: Same `test_eager_target_verify_block_kv_indices_metadata`.
  With M15, `seq_lens = prefix = (61, 63)`, `cdiv = (1, 1)`,
  `max_seqlen_pad = 1`. Shape changes from `(2, 2)` to `(2, 1)`.
- **Validation**: Re-applied
  `seq_lens_cpu = forward_batch.seq_lens_cpu` (and same for seq_lens) at
  `flashmla_backend.py:120-121`, the same test failed with
  "First differing element 1: 1 vs 2" on the shape assertion before
  revert.

### M16 — `flashmla_backend.py:347-348`, replay target_verify drop `+ num_draft_tokens`

- **Closure**: Added
  `mla/test_flashmla.py::test_replay_target_verify_block_kv_indices_metadata`.
  Builds a capture+replay sequence with the same
  page-boundary-crossing prefix lens, asserts the *sliced*
  `block_kv_indices[:bs, :max_seqlen_pad]` shape. (Replay's underlying
  buffer is initialised to `1`, not `-1`, so per-row valid counts cannot
  be tested by sentinel comparison; shape alone is enough.)
- **Validation**: Re-applied
  `seq_lens = seq_lens[:bs]; seq_lens_cpu = seq_lens_cpu[:bs]` (dropping
  `+ self.num_draft_tokens`) at `flashmla_backend.py:347-348`, the replay
  metadata test failed on the shape assertion before revert.

### M19 — `hybrid_linear_attn_backend.py:879-900`, `HybridLinearAttnBackend.init_forward_metadata_replay_cuda_graph`, `attn_backend_list[:1]`

- **Closure**: Added
  `gdn/test_triton.py::test_hybrid_dispatch_replay_init_forward_metadata_fan_out`.
  Constructs `HybridLinearAttnBackend` with two `MagicMock`
  sub-backends (one typed as `MambaAttnBackendBase`), calls
  `init_forward_metadata_replay_cuda_graph` once with sentinel args, and
  asserts BOTH sub-backends saw exactly one matching call. Slicing the
  list to `[:1]` reduces the linear sub-backend's call-count to 0 and
  the assertion fires immediately.
- **Why output-diff missed**: For the GDN cuda-graph decode case both
  capture and replay used the same `req_pool_indices` arange, so the
  linear sub-backend's `state_indices_list[bs - 1]` was already at the
  replay-correct values from the capture pass even when the replay
  dispatch was skipped.
- **Validation**: Re-applied `[:1]` slice in
  `HybridLinearAttnBackend.init_forward_metadata_replay_cuda_graph`, ran
  the new test, it failed with
  "AssertionError: Expected 'init_forward_metadata_replay_cuda_graph' to
  be called once. Called 0 times." (on the linear sub-backend) before
  revert.

### M20 — `hybrid_linear_attn_backend.py:825-827`, `HybridLinearAttnBackend.init_forward_metadata`, `attn_backend_list[1:]`

- **Closure**: Added
  `gdn/test_triton.py::test_hybrid_dispatch_eager_init_forward_metadata_fan_out`.
  Same spy approach for the eager path: `assert_called_once_with` on
  both sub-backends.
- **Why output-diff missed**: GDN fixture sets `full_attn_layers=[]`,
  so the full sub-backend's `init_forward_metadata` does no work that
  the actual GDN forward depends on.
- **Validation**: Re-applied `[1:]` slice in
  `HybridLinearAttnBackend.init_forward_metadata`, ran the new test, it
  failed with "AssertionError: Expected 'init_forward_metadata' to be
  called once. Called 0 times." (on the full sub-backend) before revert.
- **Bonus**: Also added
  `test_hybrid_dispatch_capture_init_forward_metadata_fan_out` for the
  symmetric capture path (no journal entry, but the same dispatch shape
  would silently miss an analogous mutation).

### M21 — `hybrid_linear_attn_backend.py:417` (`MambaAttnBackendBase._replay_metadata` via `Mamba2AttnBackend.init_forward_metadata_replay_cuda_graph`), `seq_lens_cpu - 1`

- **Closure**: Added
  `mamba/test_mamba2.py::test_mamba2_replay_metadata_padding_indices`.
  Builds the Mamba2 fixture, allocates cuda-graph state, then calls
  `Mamba2AttnBackend.init_forward_metadata_replay_cuda_graph` with a
  hand-rolled batch where `seq_lens_cpu = [5, 1, 1]` (two rows at the
  cuda-graph fill value of 1). Asserts the trailing two entries of
  `state_indices_list[bs - 1]` are `-1`. Under M21,
  `count_nonzero(seq_lens_cpu - 1 == 1) = count_nonzero(seq_lens_cpu ==
  2) = 0`, so the padding accounting collapses and the trailing entries
  retain the real mamba indices instead of being zeroed to `-1`.
- **Why output-diff missed**: The GDN cuda-graph runner uses
  `allow_padding=False`, so the existing GDN replay batch never
  contains the fill value. The existing Mamba2 fixture only had an
  eager EXTEND case and no decode-replay coverage at all.
- **Why we use a hand-rolled batch instead of a full forward**: The
  existing Mamba2 fixture's forward path trips a pre-existing
  `enable_symm_mem` baseline bug in the foreground repo (the mock
  `server_args` `SimpleNamespace` lacks `enable_symm_mem`, which the
  production `is_symmetric_memory_enabled()` reads via
  `get_global_server_args()` inside `MambaMixer2.in_proj` /
  `out_proj`). Going metadata-only sidesteps that pre-existing bug
  cleanly and is sufficient to detect M21.
- **Validation**: Re-applied
  `(seq_lens_cpu - 1) == self.get_cuda_graph_seq_len_fill_value()`
  in `MambaAttnBackendBase._replay_metadata`, ran the new test, it
  failed with `assertEqual([7, 0, 0], [7, -1, -1])` on the
  state-indices assertion before revert.

### M23 — `triton_backend.py:1395`, `TritonMultiStepDraftBackend.init_forward_metadata_replay_cuda_graph`, skip `get_num_kv_splits`

- **Outcome**: STRUCTURALLY UNDETECTABLE through forward output.
- **Why**: `cuda_graph_num_kv_splits` is initialised in
  `init_cuda_graph_state` to `max_kv_splits`. The decode kernel reads
  `kv_splits = tl.load(num_kv_splits + cur_batch)` and splits the work
  across `kv_splits` parts; the combine_kv kernel handles any
  split-count >= 1 correctly. So leaving the buffer at its init value
  preserves correctness and only loses scheduling efficiency.
- **Action taken**: Documented in `PLAN.md` under
  "Intentionally Untested Mutation-Surfaces". The
  `MultiStepDraftBackend` lifecycle/shape is already covered
  end-to-end by
  `dense/test_triton.py::test_runner_mode_eagle_draft_cuda_graph_runner_cases`;
  a worker-integration test (Phase 4b deferred follow-up) is the
  appropriate place to catch any regression that makes
  `cuda_graph_num_kv_splits` load-bearing.

## Summary

| MISS | Status | Test |
|------|--------|------|
| M5 | CLOSED | `swa/test_triton.py::test_runner_mode_cuda_graph_decode_cases` (`runner_cuda_graph_swa_decode_above_window`) |
| M6 | DOCUMENTED (kernel re-masking invariant) | `swa/test_triton.py::test_runner_mode_spec_verify_cuda_graph_cases` (above-window case as exploratory coverage) |
| M12 | DOCUMENTED (no-op argument in spec_info branch) | `mla/test_flashinfer.py::test_eager_target_verify_prefix_lens_is_noop` (skip with reason) |
| M14 | CLOSED | `mla/test_flashmla.py::test_eager_target_verify_block_kv_indices_metadata` |
| M15 | CLOSED | `mla/test_flashmla.py::test_eager_target_verify_block_kv_indices_metadata` |
| M16 | CLOSED | `mla/test_flashmla.py::test_replay_target_verify_block_kv_indices_metadata` |
| M19 | CLOSED | `gdn/test_triton.py::test_hybrid_dispatch_replay_init_forward_metadata_fan_out` |
| M20 | CLOSED | `gdn/test_triton.py::test_hybrid_dispatch_eager_init_forward_metadata_fan_out` |
| M21 | CLOSED | `mamba/test_mamba2.py::test_mamba2_replay_metadata_padding_indices` |
| M23 | DOCUMENTED (kv-split-count is non-load-bearing for correctness) | `PLAN.md`, deferred to worker-integration follow-up |

7 of 10 MISSES converted into actually-failing assertions; 3 documented as
structurally undetectable in `PLAN.md`.

## Notes on baseline

- The pre-existing `mamba/test_mamba2.py::test_projected_mamba2_attention_cases`
  failure (`'SimpleNamespace' object has no attribute 'enable_symm_mem'`) was
  observed in the baseline before any changes and remains. Per task
  instructions, that is being fixed in the foreground and is ignored here.
- The final `python -m unittest discover -s test/manual/attention/unittest -p
  'test_*.py'` sweep reports `Ran 93 tests in ~30s, FAILED (errors=1,
  skipped=4)`. The single error is the baseline `enable_symm_mem` bug; the
  four skips are the existing three hardware-gated MLA tests
  (`cutlass_mla`, `trtllm_mla`, `tokenspeed_mla`) plus the new documented
  M12 skip.
