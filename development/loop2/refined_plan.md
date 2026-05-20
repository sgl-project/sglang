# Loop 2 — Standalone Double Sparsity, M1-C Closeout on DeepSeek-V3.2 (FP8)

## Goal Description

Land the page-table adapter that completes the standalone Double Sparsity (DS) attention path on DeepSeek-V3.2 (FP8), fix the per-layer skip-topk gate so the DS branch is actually reached, wire the missing per-request observability hook (with production-grade error signalling), add the M3-B page-stability CI hook, ship the operator runbook with a hardware preflight, and remove every `SGLANG_DS_ALLOW_*` startup gate.

In code terms:

- `DeepseekV2AttentionMLA._select_topk_indices` stops raising `NotImplementedError`. Both DS and NSA branches return the **same Tensor type** — `page_table_1: int32[bs, max_seqlen_pad]` — to the downstream consumer. There is no typed-union return, no `isinstance` dispatch, and only one `_forward_flashmla_kv` call site downstream.
- The DS branch produces its `page_table_1` either by (a) generalising `transform_index_page_table_decode` to accept a sparse page selection, or (b) expanding DS-selected pages to a token-index buffer and reusing the existing transform. The choice is an implementation detail; the externally observable contract is one tensor type with one downstream consumer.
- The DS selector's per-request stats (`sparsity_rate`, `selected_pages`, `dense_fallback`, plus a new error counter) reach `meta_info["double_sparsity"]` as a **per-request summary**, not as a per-output-token list. This requires a dedicated request-summary bridge through the scheduler, because the existing `customized_info` accumulator is per-output-token and would otherwise expose a list of dicts instead of one dict.
- `forward_absorb_prepare` (both alt-stream and normal branches) gates `skip_topk` on `not self.use_double_sparsity` so the DS selector runs on every DS-routed step. The reuse short-circuit is not a DS-branch concern; it is a prepare-path concern.
- `validate_double_sparsity` and the `DoubleSparsitySelector` placeholder path stop reading `SGLANG_DS_ALLOW_NO_ADAPTER` and `SGLANG_DS_ALLOW_PLACEHOLDER`. `DoubleSparsitySelector.bind_runtime_data(...)` gets a real production caller at the per-layer DS attention init hook, and has an idempotence contract (second call either raises a typed `DoubleSparsityRebindError` or is an exact same-object no-op).
- The page-table adapter consumes preallocated output buffers owned by the FlashMLA forward-metadata (or an equivalent NSA-metadata-owned tensor) so that capture-phase CUDA graph allocations stay zero on the hot path.
- DS contract assertions on `selected_indices` / `valid_lengths` / adapter outputs run **either** ahead of CUDA graph capture (in `init_forward_metadata`) **or** entirely inside Triton kernels where the host does not synchronise. There is no host-side assertion that disables itself during capture.
- `m3b_page_stability_fixture` (standalone in `double_sparsity/page_signature_write.py`) gains a CI driver in `test_double_sparsity_unit.py` that runs it on a synthetic V3.2-shape input and does not touch `_double_sparsity_radix_fixture_passed` (the loop-1 DEC-2 default flip remains operator-phase work).
- The operator runbook lives at `development/loop2/RUNBOOK.md` and begins with a preflight that fails before serving if the actual node's backend, FP8 dtype, CUDA arch, page size, top-k, or TP world size do not match the assumptions baked into `serve_double_sparsity.sh`.
- `serve_double_sparsity.sh` no longer exports any `SGLANG_DS_ALLOW_*` variable.

Hardware-bound work (real calibration run, real benchmark runs, real M3-B hardware run, loop-1 DEC-2 default flip, the loop-1 SLO and NIAH/MMLU numbers) remains explicitly out of scope.

## Acceptance Criteria

Each criterion includes positive and negative tests for deterministic verification. All CI tests use synthetic shapes; no real DeepSeek-V3.2 weight loads.

- AC-1: `--enable-double-sparsity` boots without any `SGLANG_DS_ALLOW_*` override; production code paths contain no references to the two removed env gates; `bind_runtime_data` has a typed idempotence contract; channel-mask rejection covers structural AND value-domain corruption.
  - Positive Tests:
    - Server boot with `--enable-double-sparsity --double-sparsity-config <path>.json` and no environment overrides succeeds; first synthetic decode returns without raising.
    - After boot, every DS-enabled attention layer's selector reports `IS_PLACEHOLDER == False` and has been bound exactly once.
    - A second `bind_runtime_data` call with the SAME `page_signature_table`, `channel_mask`, and `process_group` objects is a no-op (does not raise, does not swap tensors).
  - Negative Tests:
    - `rg "SGLANG_DS_ALLOW_NO_ADAPTER|SGLANG_DS_ALLOW_PLACEHOLDER" python/sglang/srt development/serve_double_sparsity.sh` returns zero hits (matches in `test/` are allowed; `SGLANG_DS_RADIX_OVERRIDE` is unaffected).
    - Server boot with a missing channel-mask file raises `DoubleSparsityChannelMaskMissing` (typed).
    - Server boot with a channel mask whose `channel_weights` contain NaN, +/-Inf, or whose projections are all-zero per row raises `DoubleSparsityChannelMaskCorrupt` (typed) at startup, not at first decode.
    - A second `bind_runtime_data` call with a DIFFERENT `channel_mask` (or different `process_group` / `page_signature_table`) raises `DoubleSparsityRebindError` (typed) instead of silently swapping internal pointers.

- AC-2: A DS-routed decode request reaches FlashMLA via the same `_forward_flashmla_kv` call site as the NSA path; `_select_topk_indices` returns the same `page_table_1: int32[bs, max_seqlen_pad]` Tensor type from both branches; the downstream consumer makes one call, has no `isinstance` branch, and the adapter rejects each named contract violation with a named exception.
  - Positive Tests:
    - End-to-end synthetic decode through the DS branch returns the same shape and dtype as an NSA-branch decode; the call counter on `_forward_flashmla_kv` increments exactly once per decode step.
    - The downstream consumer (in `forward_absorb_prepare` / `forward_absorb_core`) contains no `isinstance` dispatch on the return of `_select_topk_indices`. (Static-grep negative check: zero matches for `isinstance(.*DSSelectionResult|isinstance(.*DoubleSparsitySelectionResult` under `python/sglang/srt/`.)
    - DS-produced `page_table_1` content matches what the NSA path would have produced if given a logical-pages-only `topk_indices` derived from DS's selection (bit-for-bit equivalence on a synthetic fixture).
  - Negative Tests (each raises a distinct named exception class with a distinct message; one named test per failure):
    - `DSAdapterPageOutOfRange`: a page ID outside the range covered by synthetic `req_to_token` raises this exception in the validation layer, not in the kernel.
    - `DSAdapterValidLengthOverflow`: `valid_lengths > max_top_k` raises this exception in the validation layer.
    - `DSAdapterNonAscending`: a row of `selected_indices` that is not ascending raises this exception (Triton-side device assertion or pre-capture host check).
    - `DSAdapterPaddingViolation`: missing `-1` padding past `valid_lengths` raises this exception.
    - `DSAdapterDtypeMismatch` / `DSAdapterDeviceMismatch` / `DSAdapterBatchMismatch`: dtype, device, or batch-size disagreement between `selected_indices`, `valid_lengths`, `req_to_token`, and `req_pool_indices` raises the corresponding typed exception.

- AC-3: A successful DS request's `meta_info["double_sparsity"]` is a **per-request summary dict** (not a list of per-token dicts), with `sparsity_rate: float`, `selected_pages: int`, `dense_fallback: int`. Production failures emit Prometheus counters with labels and structured log lines.
  - Positive Tests:
    - End-to-end synthetic decode test verifies `meta_info["double_sparsity"]` is a single dict, not a list; field types are `float`, `int`, `int`; field values match the per-request stats produced by the selector during that decode.
    - When a request generates N > 1 tokens, `meta_info["double_sparsity"]` is still a single summary dict whose semantics are defined (use the last-step values, or the per-request aggregate; the choice is documented in `customized_info_for_request` and exercised by the test).
    - Prometheus registry contains a `sglang_double_sparsity_errors_total{cls=<bad_mask|bad_adapter_input|selector_runtime_error|rank_mismatch>}` counter that increments under the corresponding failure path.
    - Structured log lines for each error class include the request ID, layer ID, and selector ID at WARNING or ERROR level.
  - Negative Tests:
    - When DS is disabled, `meta_info["double_sparsity"]` is absent (key missing, not present-with-empty).
    - When the request-summary bridge is bypassed (simulated by patching the scheduler glue), the transport test fails at the tokenizer side, not at the helper level.
    - When a DS request's `meta_info["double_sparsity"]` becomes a list of per-token dicts instead of one dict (regression), the test detects and fails — preventing accidental reliance on the per-output-token accumulator.

- AC-4: The standalone `m3b_page_stability_fixture` function has a synthetic-input CI hook in `test_double_sparsity_unit.py` that does not mutate `_double_sparsity_radix_fixture_passed`.
  - Positive Tests:
    - CI test calls `m3b_page_stability_fixture(...)` with synthetic V3.2-shape input; cold and warm signatures match.
    - Before and after the test runs, `server_args._double_sparsity_radix_fixture_passed` remains its pre-test value.
  - Negative Tests:
    - Perturbing the synthetic warm-run input causes the fixture to surface a mismatch and fail loudly.
    - Any attempt by the CI hook to set `_double_sparsity_radix_fixture_passed = True` is caught by a side-effect probe.

- AC-5: `development/loop2/RUNBOOK.md` exists with five numbered operator phases and a numbered Phase 0 preflight. The preflight fails before serving when the running node's backend, dtype, page size, top-k, CUDA arch, or TP world size do not match the assumptions baked into `serve_double_sparsity.sh`.
  - Positive Tests:
    - File present; contains Phase 0 (preflight) plus Phases 1–5 (calibrate / boot / benchmark / compare / M3-B hardware).
    - The preflight script (or equivalent embedded section) checks: backend == `flashmla_kv`, dtype == `fp8_e4m3`, page_size == 64, top_k == 2048, TP world size == 8, CUDA arch matches H200 (compute capability 9.x).
    - The runbook explicitly states that the calibration / benchmark / hardware-M3-B runs themselves are operator work, not part of this loop.
  - Negative Tests:
    - Removing or omitting Phase 0 or any of Phases 1–5 fails the regression-sweep structural lint.
    - A preflight invocation against a node that fails ANY of the checked invariants exits non-zero before launching the server (verified by a fixture that fakes the relevant environment readouts).

- AC-6: Existing 87 unit tests stay green. The test suite gains coverage of every distinct correctness property introduced by this loop (the seven plan-named tests below are the **expected** anchors; implementation may add more, must not skip these, and must not gate count growth on a numeric floor alone).
  - Adapter mapping (`test_ds_page_table_adapter_basic_mapping`): synthetic `(selected_indices, valid_lengths)` produces the same `page_table_1` content as the NSA path would for the equivalent logical pages.
  - Adapter contract negatives (one named test per exception class listed in AC-2): each test asserts both the exception class AND the validation layer that catches it (host pre-capture vs Triton kernel).
  - Rebind idempotence (`test_ds_rebind_idempotence`): same-object rebind is a no-op; different-object rebind raises `DoubleSparsityRebindError`.
  - Skip-topk gate fix (`test_ds_skip_topk_gate_alt_stream_and_normal`): with `use_double_sparsity = True`, both alt-stream and normal branches of `forward_absorb_prepare` invoke `_select_topk_indices` on every decode step regardless of `prev_topk_indices` state.
  - Per-request summary semantics (`test_ds_meta_info_request_summary`): `meta_info["double_sparsity"]` is a single dict, not a list, for any N > 1 generated tokens.
  - Mask corruption boot rejection (`test_ds_channel_mask_value_corruption`): NaN / Inf / all-zero `channel_weights` are rejected at startup with a typed exception.
  - M3-B synthetic CI hook (`test_ds_m3b_synthetic_ci_hook`): per AC-4.
  - Positive Tests:
    - All listed tests exist and pass; the regression sweep verifies their presence by name and confirms none are `@skip`-decorated.
  - Negative Tests:
    - Deleting or `@skip`-marking any listed test trips the regression sweep.
    - Mutating an adapter contract negative test so it asserts only "raises something" (rather than the named exception class) trips a meta-check that scans for `assertRaises(Exception)` patterns in the new tests.

- AC-7 (new): On TP world size > 1, every TP rank selects the same logical pages for every DS-routed decode step; missing or broken TP all-reduce is detected at startup, not via silent divergence.
  - Positive Tests:
    - Two-rank synthetic test (mocked `torch.distributed` or two real processes) confirms `selected_indices` is identical across ranks for the same input.
    - Selector logs `process_group` membership at INFO level on bind for TP > 1.
  - Negative Tests:
    - With `process_group=None` AND `world_size > 1`, server boot raises `DoubleSparsityTPMisconfigured` at startup. A unit test forces this configuration and asserts the named exception.
    - Disabling the page-score all-reduce (patched to a no-op) on a synthetic two-rank fixture is detected: rank-local scores produce a divergent `selected_indices` and the test fails the agreement assertion.

- AC-8 (new): The DS attention forward path makes no per-step allocations on the CUDA-graph captured path. Output buffers (`page_table_1`, intermediate adapter buffers) are allocator-owned and reused across steps.
  - Positive Tests:
    - A CUDA-graph capture test (synthetic backend, mocked Triton) captures one DS decode step and replays it; per-step memory delta is zero on the captured path (verified via `torch.cuda.memory_allocated` deltas or a buffer-allocation probe).
    - The adapter signature accepts pre-allocated `page_table_1_out` and `adapter_scratch` tensors and writes into them in-place.
  - Negative Tests:
    - A regression test that re-introduces a per-step `torch.empty(...)` in the adapter (simulated by monkey-patching) trips the zero-allocation invariant on replay.

- AC-9 (new): A selector or adapter failure mid-decode raises a typed exception that is contained to the offending request; the batch and worker continue serving other requests; partially accumulated DS stats for the failed request are discarded; failure is observable via Prometheus and structured logs.
  - Positive Tests:
    - A synthetic decode test injects a `DSAdapterPageOutOfRange` on the 2nd of 3 in-flight requests; only that request is aborted with a non-2xx status (or equivalent error path); the other two requests complete normally; the worker process does not exit.
    - The scheduler clears the failed request's partial `customized_info` accumulator (no leaked per-token entries into the next batch).
    - The Prometheus error counter increments by 1 with `cls=bad_adapter_input`.
  - Negative Tests:
    - Removing the per-request error containment causes the synthetic mid-batch failure to abort all three requests; the regression test detects this and fails.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
- Unified `page_table_1` Tensor return from `_select_topk_indices` on both DS and NSA branches; one downstream consumer call site; no `isinstance` dispatch in `forward_absorb_*`.
- DS-side `page_table_1` produced inside `_select_topk_indices` either by generalising `transform_index_page_table_decode` to accept a sparse page selection or by expanding DS-selected pages to a token-index buffer and reusing the existing transform. Implementation choice is recorded in `## Implementation Notes`; externally observable contract is unified.
- Fix of `skip_topk` gating in `forward_absorb_prepare` (both alt-stream and normal branches) so the DS selector runs on every DS-routed step. The fix lives in the prepare path, not inside `_select_topk_indices`.
- `bind_runtime_data` rewritten with a typed idempotence contract (same-object no-op vs different-object `DoubleSparsityRebindError`).
- Channel-mask validation broadened to reject value-domain corruption (NaN, Inf, all-zero per-row projections) at startup with a typed exception.
- Production wiring of `bind_runtime_data` at the per-layer DS attention init hook with channel mask sliced for the local TP rank, page signature table allocator-owned, TP process group, and device.
- Full removal of `SGLANG_DS_ALLOW_NO_ADAPTER` from `validate_double_sparsity` and full removal of `SGLANG_DS_ALLOW_PLACEHOLDER` from production code in `DoubleSparsitySelector`. Both exports stripped from `development/serve_double_sparsity.sh`. `SGLANG_DS_RADIX_OVERRIDE` is unaffected.
- Per-request DS stats published into a new **per-request summary** field on the scheduler output (NOT through the per-output-token `customized_info` accumulator), and unpacked by the existing tokenizer hook into `meta_info["double_sparsity"]` as a single dict.
- DS error taxonomy implemented as a `sglang_double_sparsity_errors_total` Prometheus counter with `cls` label, plus structured log lines at WARNING/ERROR. Error classes: `bad_mask`, `bad_adapter_input`, `selector_runtime_error`, `rank_mismatch`.
- DS adapter consumes pre-allocated output buffers owned by FlashMLA forward-metadata (or an equivalent NSA-metadata-owned tensor). No per-step `torch.empty(...)` on the CUDA-graph captured path.
- Contract assertions live either pre-capture (in `init_forward_metadata` or equivalent) for shape/dtype/device, or inside Triton kernels (device-side `tl.device_assert`) for value-domain checks. No host-side assertion that silently no-ops during capture.
- Mid-decode error containment policy: typed exception aborts the one offending request, batch and worker continue, partial stats discarded, observability counters increment.
- TP-rank invariance: startup fails fast for TP > 1 with no process group; per-decode test verifies identical `selected_indices` across ranks.
- CI hook for `m3b_page_stability_fixture` driven by synthetic V3.2-shape input, gated only on CUDA availability where absolutely necessary.
- `development/loop2/RUNBOOK.md` with Phase 0 preflight + Phases 1–5 (calibrate / boot / benchmark / compare / M3-B hardware), including concrete commands referencing the existing scripts and exact preflight invariant checks.
- The 87 prior unit tests stay green; the new tests listed in AC-6 plus AC-7 / AC-8 / AC-9 coverage land.

### Lower Bound (Minimum Acceptable Scope)
- All upper-bound scope items are required. The plan does not enumerate a minimum below this; the design changes triggered by the comment ledger are non-optional for correctness.
- Discretion is limited to: prose density in the runbook, named exception class spelling (one or several classes per category, both acceptable), and whether the DS-side `page_table_1` builder is implemented as a generalised transform or as an expand-then-transform helper — provided the externally observable unified-shape contract holds.

### Allowed Choices
- Can use:
  - Existing Triton kernels under `python/sglang/srt/layers/attention/double_sparsity/`.
  - The existing `transform_index_page_table_decode` (generalised in-place to accept a sparse selection, or wrapped by a thin expansion helper).
  - The existing `customized_info_for_request` helper, with the per-request summary bridge added on top.
  - The existing `m3b_page_stability_fixture` function (standalone in `double_sparsity/page_signature_write.py`).
  - Mocking/patching of `req_to_token`, `req_pool_indices`, and `torch.distributed` in unit tests to drive synthetic shapes and TP topologies.
  - Test-only references to `SGLANG_DS_ALLOW_PLACEHOLDER` are allowed (e.g., to construct a placeholder selector via direct attribute injection or `object.__new__`); production code paths must not read the env var.
- Cannot use:
  - A typed-union return from `_select_topk_indices` that encodes the selector's identity in its return type (the previously-considered `DSSelectionResult`). The return type must be the same `page_table_1` Tensor for both branches.
  - `isinstance` dispatch on the return of `_select_topk_indices` in `forward_absorb_prepare` / `forward_absorb_core` / the backend.
  - The top-level FlashMLA dense `block_table` path (`python/sglang/srt/layers/attention/flashmla_backend.py` per-batch metadata mutation) as the DS handoff. DS targets NSA's `_forward_flashmla_kv` sparse path.
  - Imports from `python/sglang/srt/mem_cache/sparsity/` (legacy HiSparse).
  - Real DeepSeek-V3.2 weight loads in CI.
  - Any code path that flips `_double_sparsity_radix_fixture_passed` from the CI hook (loop-1 DEC-2 default flip is out of scope).
  - Compatibility shims that keep `SGLANG_DS_ALLOW_NO_ADAPTER` / `SGLANG_DS_ALLOW_PLACEHOLDER` callable from production paths after this loop.
  - Host-side contract assertions that silently no-op inside CUDA graph capture. Cheap shape/device checks must run pre-capture; value-domain checks must live in device-side Triton assertions.
  - Per-step `torch.empty(...)` / `torch.zeros(...)` allocations inside the DS adapter on the captured path.
  - Per-output-token `customized_info` accumulator as the DS-stats transport (the contract is per-request summary; a dedicated bridge is required).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

`_select_topk_indices` returns one Tensor type from both branches:

```
# DeepseekV2AttentionMLA._select_topk_indices
def _select_topk_indices(self, x, q_lora, positions, forward_batch, layer_id,
                        return_indices=True) -> torch.Tensor:
    if self.use_double_sparsity:
        # DS-side build of physical page_table_1:
        selected_indices, valid_lengths, stats = \
            self.double_sparsity_selector.retrieve_topk(...)
        # contract checks live pre-capture (in init_forward_metadata) or in Triton.
        page_table_1 = build_ds_page_table_1(
            selected_indices, valid_lengths,
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            page_size=PAGE_SIZE,
            out=forward_batch.attn_metadata.ds_page_table_1_buf,  # preallocated
            scratch=forward_batch.attn_metadata.ds_adapter_scratch,
        )
        self._publish_ds_stats(forward_batch, stats)   # side-channel
        return page_table_1
    # NSA path — unchanged shape:
    topk_indices = self.indexer(...)
    page_table_1 = transform_index_page_table_decode(page_table, topk_indices)
    return page_table_1
```

Downstream consumer (single path, no `isinstance`):

```
# In forward_absorb_prepare / forward_absorb_core
page_table_1 = self._select_topk_indices(...)
output = nsa_backend._forward_flashmla_kv(..., page_table_1=page_table_1, ...)
```

`forward_absorb_prepare` skip-topk fix (both alt-stream and normal branches):

```
# was: if self.skip_topk and prev_topk_indices is not None: reuse
# now:
if self.skip_topk and (not self.use_double_sparsity) and prev_topk_indices is not None:
    reuse
```

`bind_runtime_data` idempotence (sketch):

```
def bind_runtime_data(self, page_signature_table, channel_mask, *, process_group=None):
    if not self.IS_PLACEHOLDER:
        same = (self._page_signature_table is page_signature_table and
                self._channel_mask is channel_mask and
                self._process_group is process_group)
        if same:
            return
        raise DoubleSparsityRebindError(
            "bind_runtime_data called twice with different objects; "
            "this would silently invalidate captured graphs and TP state."
        )
    # ... first-time bind ...
```

Per-request DS stats transport (sketch):

```
# Scheduler output gains a per-request summary field that bypasses the
# per-token customized_info accumulator:
batch_out.per_request_summary = {
    "double_sparsity": [customized_info_for_request(stats_req_i) for i in range(bs)],
}

# Tokenizer hook reads per_request_summary alongside customized_info:
if getattr(recv_obj, "per_request_summary", None):
    for k, v in recv_obj.per_request_summary.items():
        meta_info[k] = v[i]   # one dict per request — not a list per request
```

DS error taxonomy (sketch):

```
# metrics.py — register once at server start:
SGLANG_DS_ERRORS = Counter(
    "sglang_double_sparsity_errors_total",
    ["cls"],   # bad_mask | bad_adapter_input | selector_runtime_error | rank_mismatch
)
```

### Relevant References

- `python/sglang/srt/models/deepseek_v2.py` — `DeepseekV2AttentionMLA._select_topk_indices` (the seam; the `NotImplementedError` raise lives here today).
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` — `forward_absorb_prepare`, where the `skip_topk` short-circuit currently runs BEFORE `_select_topk_indices` and must be gated on `use_double_sparsity` in both alt-stream and normal branches.
- `python/sglang/srt/layers/attention/double_sparsity/selector.py` — `DoubleSparsitySelector.retrieve_topk`, `bind_runtime_data` (production caller needed; idempotence contract added).
- `python/sglang/srt/layers/attention/double_sparsity/validator.py` — `validate_double_sparsity`; the `SGLANG_DS_ALLOW_NO_ADAPTER` gate to remove.
- `python/sglang/srt/layers/attention/double_sparsity/channel_mask.py` — channel-mask loader; broaden value-domain rejection.
- `python/sglang/srt/layers/attention/double_sparsity/metrics.py` — `customized_info_for_request` helper; register the new `sglang_double_sparsity_errors_total` counter here.
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `compute_page_scores`, `all_reduce_page_scores`, `retrieve_topk` kernel entry points.
- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py` — DS CUDA-graph capture helper; reference for buffer ownership.
- `python/sglang/srt/layers/attention/double_sparsity/page_signature_write.py` — `m3b_page_stability_fixture` (standalone).
- `python/sglang/srt/layers/attention/nsa_backend.py` — `_forward_flashmla_kv`, `transform_index_page_table_decode` call sites. DS uses the same `_forward_flashmla_kv` consumer.
- `python/sglang/srt/layers/attention/nsa/transform_index.py` — `transform_index_page_table_decode`; candidate for sparse-selection generalisation OR wrapping by an expand-then-transform helper.
- `python/sglang/srt/managers/tokenizer_manager.py` — `customized_info` unpack loop and the new `per_request_summary` unpack path.
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py` — `BatchTokenIDOutput.customized_info` (per-token); new `per_request_summary` field set here.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — destination for the new named tests.
- `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh`, `development/benchmark.sh`, `development/benchmark_baseline.sh`, `development/benchmark_compare.py` — operator-runbook target scripts.

## Dependencies and Sequence

### Milestones

1. **Milestone A — Unified shape and adapter**: Decide DS-side builder strategy (generalise transform vs expand-then-transform), implement, return same Tensor type from `_select_topk_indices` on both branches; downstream consumer loses its `isinstance` dispatch.
   - Phase A1: Decide and document the DS-side builder strategy.
   - Phase A2: Implement DS-side `build_ds_page_table_1` (with preallocated output and scratch buffers).
   - Phase A3: Replace `NotImplementedError` in the DS branch with the new builder; remove the typed-union sketch; collapse the downstream consumer to one call site.

2. **Milestone B — Prepare-path and runtime data**: Fix `skip_topk` gate in `forward_absorb_prepare` (alt-stream AND normal); add `bind_runtime_data` idempotence; wire production caller for `bind_runtime_data`.

3. **Milestone C — Observability and stats transport**: Add per-request summary field on `BatchTokenIDOutput`; route DS stats through it; register `sglang_double_sparsity_errors_total` Prometheus counter; add structured log lines for each error class.

4. **Milestone D — Safety hardening**: CUDA-graph allocation invariant (preallocated buffers); contract assertions move to pre-capture / Triton; TP-rank fail-fast at startup for missing process group; mid-decode error containment policy.

5. **Milestone E — Mask validation breadth and env-gate removal**: Broaden channel-mask validation (NaN / Inf / all-zero rejection at startup); remove `SGLANG_DS_ALLOW_NO_ADAPTER` and `SGLANG_DS_ALLOW_PLACEHOLDER` from production code and launcher.

6. **Milestone F — Tests and CI hook**: Add the AC-6 anchor tests plus AC-7 / AC-8 / AC-9 coverage; add M3-B synthetic CI hook test.

7. **Milestone G — Runbook and regression**: Write `development/loop2/RUNBOOK.md` with Phase 0 preflight + Phases 1–5; final regression sweep.

Dependency notes:
- Milestone A blocks B, C, D, F (everything else assumes unified-shape return).
- Milestone B's `bind_runtime_data` wiring blocks Milestone E's removal of `SGLANG_DS_ALLOW_PLACEHOLDER`.
- Milestone D's containment policy blocks Milestone F's AC-9 tests.
- Milestone G's regression sweep depends on every prior milestone.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Decide DS-side `page_table_1` builder strategy (generalise transform vs expand-then-transform) and document the unified-shape return contract; lock the absence of `isinstance` dispatch downstream. | AC-2 | analyze | - |
| task2 | Implement DS-side builder; replace `NotImplementedError` in `_select_topk_indices` so the DS branch returns the same `page_table_1` Tensor type as the NSA branch; remove the typed-union sketch; collapse the downstream consumer to one call site. | AC-2 | coding | task1 |
| task3 | Fix `skip_topk` gating in `forward_absorb_prepare` (alt-stream AND normal branches) so the DS selector runs on every DS-routed step. | AC-6 (skip-topk gate fix) | coding | task2 |
| task4 | Add `bind_runtime_data` idempotence (same-object no-op; different-object `DoubleSparsityRebindError`); wire production caller at per-layer DS attention init hook. | AC-1 | coding | task2 |
| task5 | Add a per-request summary field on `BatchTokenIDOutput`; route DS stats through it; verify tokenizer hook produces one dict per request (not a list). | AC-3 | coding | task2 |
| task6 | Register `sglang_double_sparsity_errors_total{cls}` Prometheus counter; add structured log lines for `bad_mask`, `bad_adapter_input`, `selector_runtime_error`, `rank_mismatch`. | AC-3 | coding | task5 |
| task7 | Add preallocated output buffers (`page_table_1_out`, `adapter_scratch`) to FlashMLA / NSA forward-metadata; rewire adapter to write in-place. Move host-side shape/device contract checks pre-capture; move value-domain checks into Triton device assertions. | AC-8 | coding | task2 |
| task8 | Add TP-rank fail-fast startup invariant for TP > 1 with missing `process_group`; add structured log line on bind for TP > 1. | AC-7 | coding | task4 |
| task9 | Implement mid-decode error containment: typed exceptions abort the one offending request only, partial customized_info / per-request-summary state is cleared, worker keeps serving. | AC-9 | coding | task5, task6 |
| task10 | Broaden channel-mask validation to reject NaN / Inf / all-zero projections at startup with `DoubleSparsityChannelMaskCorrupt`. | AC-1 | coding | - |
| task11 | Remove `SGLANG_DS_ALLOW_NO_ADAPTER` from `validate_double_sparsity`; remove `SGLANG_DS_ALLOW_PLACEHOLDER` from production code in `DoubleSparsitySelector`; strip both env exports from `development/serve_double_sparsity.sh`; update tests that depended on the placeholder env to construct a placeholder selector directly. | AC-1 | coding | task4, task5 |
| task12 | Add AC-6 anchor tests + AC-7 / AC-8 / AC-9 coverage tests in `test_double_sparsity_unit.py` (and additional tests as needed to cover the design changes). Each adapter-contract negative test asserts a distinct named exception class. | AC-2, AC-3, AC-6, AC-7, AC-8, AC-9 | coding | task2, task3, task4, task5, task6, task7, task8, task9, task10 |
| task13 | Add M3-B synthetic CI hook test driving `m3b_page_stability_fixture` with synthetic V3.2-shape inputs; assert cold/warm match; assert no mutation of `_double_sparsity_radix_fixture_passed`. | AC-4 | coding | - |
| task14 | Write `development/loop2/RUNBOOK.md` with Phase 0 preflight (backend / dtype / page size / top-k / TP world size / CUDA arch checks) and Phases 1–5 (calibrate / boot / benchmark / compare / M3-B hardware). | AC-5 | coding | - |
| task15 | Regression sweep — verify all existing tests pass; verify every AC-6 anchor test exists and is not `@skip`-marked; verify `rg "SGLANG_DS_ALLOW_NO_ADAPTER\|SGLANG_DS_ALLOW_PLACEHOLDER" python/sglang/srt development/serve_double_sparsity.sh` returns zero hits; verify no `isinstance(.*DSSelectionResult)` matches anywhere; verify zero per-step allocations on the CUDA-graph captured path. | AC-1, AC-2, AC-6, AC-8 | analyze | task11, task12, task13 |

## Claude-Codex Deliberation

### Agreements
- DS branch's `_select_topk_indices` is currently an NSA token-topk seam; a dense top-level FlashMLA `block_table` is not a viable DS handoff.
- Two FlashMLA paths exist (top-level dense vs NSA `_forward_flashmla_kv` sparse); DS targets the NSA sparse path.
- `m3b_page_stability_fixture` is already a standalone function; the CI hook is a new test method that invokes it on a synthetic shape.
- `_double_sparsity_radix_fixture_passed` must not be flipped by a synthetic CI hook.
- `SGLANG_DS_RADIX_OVERRIDE` is a separate concern and stays in place.
- The DS branch must not honour `prev_topk_indices` reuse (selector runs every decode step); the fix lives in `forward_absorb_prepare`, not inside `_select_topk_indices`.
- Test design: behaviorally framed expectations with named anchor tests; numeric count floors alone are gameable and were dropped.

### Resolved Disagreements

- Topic: Adapter return type and dispatch shape.
  - Earlier candidates: Design A (token-level NSA `topk_indices` produced by adapter) — rejected as a contract violation; Design B (typed `DSSelectionResult` + downstream `isinstance` dispatch + bypass of `transform_index_page_table_decode`) — rejected after review (information leakage; permanent dual-arm runtime path; symptomatic tests).
  - Resolution: Design C. `_select_topk_indices` returns the same `page_table_1` Tensor type from both DS and NSA branches; downstream consumer makes one `_forward_flashmla_kv` call with no `isinstance` branch. DS-side `page_table_1` is produced inside `_select_topk_indices` via a generalised transform or an expand-then-transform helper (implementation choice noted under Path Boundaries; externally observable contract is unified).

- Topic: `meta_info` shape (top-level fields vs nested) AND per-request vs per-token semantics.
  - Earlier resolution: Nested under `meta_info["double_sparsity"]`.
  - Revised resolution: Nested AND per-request summary semantics. The existing `customized_info` accumulator is per-output-token; surfacing DS stats through it would produce a list of dicts. A new `per_request_summary` channel on the scheduler output is added; the tokenizer hook unpacks it into a single dict per request.

- Topic: `SGLANG_DS_ALLOW_PLACEHOLDER` disposition and `bind_runtime_data` wiring.
  - Resolution: Both env gates removed; `bind_runtime_data` wired at the per-layer DS attention init hook AND given a typed idempotence contract (same-object no-op; different-object `DoubleSparsityRebindError`).

- Topic: AC-6 framing (numeric count vs named tests vs behaviorally framed list).
  - Resolution: Behaviorally framed list with named anchor tests; no isolated numeric floor.

- Topic: `skip_topk` / `prev_topk_indices` reuse on the DS path.
  - Resolution: The fix lives in `forward_absorb_prepare`, in both alt-stream and normal branches, by gating the reuse short-circuit on `not self.use_double_sparsity`. The test covers both branches.

- Topic: Contract-assertion strategy under CUDA-graph capture.
  - Earlier wording: "skip host-sync paths inside CUDA graph capture".
  - Resolution: No host-side assertion that disables itself during capture. Cheap shape/dtype/device checks run pre-capture; value-domain checks live inside Triton kernels (`tl.device_assert` or equivalent). The "skip during capture" phrasing is removed.

- Topic: Hot-path allocations under CUDA-graph capture.
  - Resolution: New AC-8. Adapter writes into preallocated, allocator-owned buffers. Per-step `torch.empty(...)` on the captured path is prohibited.

- Topic: TP-rank agreement.
  - Resolution: New AC-7. Startup fails fast for TP > 1 with `process_group is None`. A two-rank synthetic test verifies identical `selected_indices` and detects a no-op all-reduce.

- Topic: Mid-decode failure containment.
  - Resolution: New AC-9. Typed exception aborts only the offending request; batch and worker continue; partial summary state for the failed request is cleared; Prometheus counter increments.

- Topic: Hardware assumptions vs "out of scope" framing.
  - Resolution: AC-5 expanded with a numbered Phase 0 preflight that fails before serving when the running node does not match the assumptions baked into `serve_double_sparsity.sh` (backend, dtype, page size, top-k, TP world size, CUDA arch).

- Topic: Channel-mask validation breadth.
  - Resolution: AC-1 broadened to reject NaN / Inf / all-zero projections at startup with `DoubleSparsityChannelMaskCorrupt`.

- Topic: Adapter contract negatives.
  - Resolution: AC-2 enumerates one named exception class per failure mode; AC-6 anchors one test per class.

### Convergence Status
- Final Status: `converged`

## Pending User Decisions

None. The unified-shape redesign and the new ACs (AC-7 / AC-8 / AC-9) are baked into the plan; the implementation choice between "generalise the transform" and "expand-then-transform" is an implementation note, not a user-facing decision.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must not contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- Use descriptive, domain-appropriate naming. Examples: name the new exception classes after the failure mode (`DoubleSparsityRebindError`, `DSAdapterPageOutOfRange`), not after their AC labels.

### DS-Side `page_table_1` Builder
Two implementation strategies are acceptable; pick one and document it in the PR description, do not ship both:
1. Generalise `transform_index_page_table_decode` to accept a sparse `(selected_indices, valid_lengths)` page-level input alongside its existing token-level `topk_indices` input. The NSA path keeps its current call; the DS path uses the new entry.
2. Add a thin `expand_ds_selection_to_topk_indices(selected_indices, valid_lengths, page_size)` helper that fills a token-index buffer from the DS page selection, then call the existing `transform_index_page_table_decode` unchanged. Less code reuse efficiency but smaller blast radius on shared NSA code.

### Branch and PR Scope
- Branch: `dev/double-sparsity-standalone` (continues from loop 1).
- All adapter / wiring / observability / safety / runbook work lands as a single coherent series; no compatibility shims for the removed env gates.

--- Original Design Draft Start ---

Complete M1-C of the standalone Double Sparsity path on DeepSeek-V3.2 (FP8): write the page-table adapter that takes the DS selector's (selected_indices, valid_lengths) page-level tuple, maps logical page IDs to physical via `req_to_token` / `req_pool_indices`, and emits the FlashMLA `block_table` in sequence order. The adapter REPLACES the current NotImplementedError raise in DeepseekV2AttentionMLA._select_topk_indices's DS branch; it must also bypass the existing NSA topk_indices consumer on the DS branch (per AC-2, DS does not stack on NSA — it is an alternative selection path that drives FlashMLA directly).

Remove both the per-step NotImplementedError and the SGLANG_DS_ALLOW_NO_ADAPTER startup gate in validate_double_sparsity once the adapter is in place. The dev override env vars exported from serve_double_sparsity.sh become unused and are deleted from the launcher.

Wire the scheduler-side `customized_info_for_request` glue at the existing `customized_info` hook in tokenizer_manager.py (search for the symbol; line numbers drift) so the sglang_double_sparsity_* metrics actually surface in per-request meta_info.

Land the M3-B page-stability fixture in CI against a synthetic V3.2-shape fixture (the function already exists; the CI hook is the new piece). Provide a numbered operator runbook at development/loop2/RUNBOOK.md covering: (1) calibrate against /cluster-storage/models/deepseek-ai/DeepSeek-V3.2/ to produce the channel mask safetensors; (2) boot serve_double_sparsity.sh + serve_native_nsa.sh; (3) run benchmark.sh twice (DS + baseline) at the agreed concurrencies; (4) run benchmark_compare.py to produce the side-by-side SLO + quality report; (5) run M3-B on real hardware and decide DEC-2's default.

Out of scope (operator phase 2 / phase 3):
- The actual calibration run, benchmark run, and M3-B hardware run.
- Flipping the DEC-2 default based on the M3-B result.
- Capturing AC-8 SLO numbers and AC-9 NIAH/MMLU results.

Branch: dev/double-sparsity-standalone (continues from the same branch). Existing 87-test suite must stay green; add adapter unit tests (synthetic FlashMLA shape fixture verifies block_table emission), integration test that runs end-to-end through the adapter without raising, and a CI test that drives the M3-B fixture against a small synthetic prompt.

Acceptance criteria for the loop close:
- `--enable-double-sparsity` boots successfully without any SGLANG_DS_ALLOW_* override.
- A request through the DS branch reaches FlashMLA and returns a result without raising.
- meta_info on a DS request contains the sparsity_rate / selected_pages / dense_fallback fields.
- M3-B fixture has a CI hook (synthetic shape).
- development/loop2/RUNBOOK.md exists and is reviewable.
- 87+ unit tests pass; the test count grows by at least 5.

--- Original Design Draft End ---
