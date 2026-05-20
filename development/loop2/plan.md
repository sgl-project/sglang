# Loop 2 — Standalone Double Sparsity, M1-C Closeout on DeepSeek-V3.2 (FP8)

<comment>
LINUS — OVERALL REACTION:

The actual code change is small. Write an adapter, wire one init hook,
populate one dict field, add a CI test method, write a runbook, delete two
env vars. A one-day patch series, two if you're being thorough.

This plan is 308 lines of process around that change:
  - 6 acceptance criteria + 7 sub-criteria with positive/negative tests
  - 9 numbered tasks, 4 milestones, 12 phases
  - 6 "Resolved Disagreements" between Claude and Codex
  - A "Convergence Status: converged" line
  - A "Speculative-Naming Rule Cross-Check" section that says the rule
    doesn't apply
  - A "Pending User Decisions: None" section

Worse, the central architecture choice — typed-union return with
`isinstance` dispatch downstream vs unified tensor shape — was decided by a
two-agent "deliberation" that never put the obvious unified-shape option on
the table. The plan converged on the wrong design through process.

Show me the code. Then we'll argue about the code.
</comment>

## Goal Description

Land the page-table adapter that completes the standalone Double Sparsity (DS) attention path on DeepSeek-V3.2 (FP8), wire the missing observability hook, add the M3-B page-stability CI hook, ship the operator runbook for the calibrate / benchmark / hardware-M3-B phase, and remove every `SGLANG_DS_ALLOW_*` startup gate so a deployment can boot DS without dev-only overrides.

In code terms:
- `DeepseekV2AttentionMLA._select_topk_indices` DS branch stops raising `NotImplementedError` and begins emitting a typed DS selection that the downstream consumer routes directly to `nsa_backend._forward_flashmla_kv` (bypassing the NSA `topk_indices → transform_index_page_table_decode → page_table_1` pipeline).
<comment>
LINUS: This bullet contains the whole problem. You're introducing a typed
union return so downstream code can `isinstance()`-dispatch DS vs NSA.
That's not engineering — that's two parallel universes glued together with
a tag.

The data-structure question you didn't ask: why can't `_select_topk_indices`
return the SAME `page_table_1` tensor on BOTH paths — DS branch having
pre-computed it, NSA branch having run the transform? Then downstream has
zero branches. The selection logic stays inside the method, where it
belongs. That's good taste. What you've written is a special case.

[pensieve maxims/eliminate-special-cases-by-redesigning-data-flow:
 "Sometimes you can see a problem in a different way and rewrite it so that
  the special case goes away and becomes the normal case."]
</comment>
- `validate_double_sparsity` and the `DoubleSparsitySelector` placeholder path stop reading `SGLANG_DS_ALLOW_NO_ADAPTER` and `SGLANG_DS_ALLOW_PLACEHOLDER`; `DoubleSparsitySelector.bind_runtime_data(...)` gets a real production caller inside the per-layer DS attention init hook (where channel mask, page signature table, TP process group, and device all exist).
- Per-request DS stats (`sparsity_rate`, `selected_pages`, `dense_fallback`) reach `meta_info["double_sparsity"]` via the existing `customized_info` transport (selector → scheduler `BatchTokenIDOutput.customized_info` → tokenizer hook).
- `m3b_page_stability_fixture` (already standalone in `double_sparsity/page_signature_write.py`) gains a CI driver in `test_double_sparsity_unit.py` that exercises it on a synthetic V3.2-shape input — without touching `_double_sparsity_radix_fixture_passed` (DEC-2 default flip remains operator-phase work).
- The operator-phase runbook lives at `development/loop2/RUNBOOK.md` and the launcher (`serve_double_sparsity.sh`) no longer exports any `SGLANG_DS_ALLOW_*` variable.

Hardware-bound work (real calibration run, benchmark runs, real M3-B hardware fixture, DEC-2 default flip, AC-8 SLO numbers, AC-9 NIAH/MMLU numbers) is explicitly out of scope and deferred to operator phases 2/3.
<comment>
CODEX: The plan says hardware is out of scope while baking in one target

serve_double_sparsity.sh assumes a single H200 node, 8-way TP, FP8, page 64,
top_k 2048, and the validator only allows fp8_e4m3 with flashmla_kv. Synthetic
CI cannot validate that backend, CUDA arch, FP8 dtype, or TP group behavior.
The runbook needs a preflight that fails before serving if the actual node and
backends do not match those assumptions.

[evidence: development/serve_double_sparsity.sh:5-7,23-31,48-56; python/sglang/srt/layers/attention/double_sparsity/validator.py:42-45,130-151]
</comment>

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. All CI tests use synthetic shapes; no real DeepSeek-V3.2 weight loads.

- AC-1: `--enable-double-sparsity` boots without any `SGLANG_DS_ALLOW_*` override, and no production code path references the two removed env gates.
  - Positive Tests (expected to PASS):
    - Server boot with `--enable-double-sparsity --double-sparsity-config <path>.json` and no environment overrides succeeds; first synthetic decode completes without raising.
    - After boot, every DS-enabled attention layer's selector reports `IS_PLACEHOLDER == False` and has been bound via `bind_runtime_data(...)` exactly once before serving begins.
  - Negative Tests (expected to FAIL):
    - `rg "SGLANG_DS_ALLOW_NO_ADAPTER|SGLANG_DS_ALLOW_PLACEHOLDER" python/sglang/srt development/serve_double_sparsity.sh` returns zero hits (matches in `test/` are allowed; `SGLANG_DS_RADIX_OVERRIDE` is unaffected).
    - With `--enable-double-sparsity` and a missing/malformed channel mask, server boot raises a typed, actionable error (not `NotImplementedError`, not a placeholder fallback).
<comment>
CODEX: `bind_runtime_data` needs an idempotence contract

AC-1 says every selector is bound exactly once, but the current method does
not track that. A second call silently swaps page_signature_table,
channel_mask, and process_group after IS_PLACEHOLDER is already false. If
graph capture or per-rank setup has seen the first tensors, this is a stale
pointer or rank-divergence bug, not a harmless re-init. Make rebind either a
typed error or an exact same-object no-op, and test both.

[evidence: python/sglang/srt/layers/attention/double_sparsity/selector.py:76-129]
</comment>
<comment>
CODEX: "Malformed channel mask" is too narrow

The loader rejects missing tensors, schema drift, hash mismatch, dtype/page
size mismatch, and out-of-range channel_selection. It does not reject NaN or
Inf channel_weights, nor all-zero projections. Those can produce NaN or flat
scores in compute_page_scores and degrade into arbitrary topk without a
startup error. AC-1's negative test should include weights-level corruption,
not just missing files and bad JSON.

[evidence: python/sglang/srt/layers/attention/double_sparsity/channel_mask.py:147-211; python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:222-286]
</comment>

- AC-2: A DS-routed decode request reaches FlashMLA via NSA's `_forward_flashmla_kv` sparse path, with the adapter producing the physical `page_table_1` tensor directly. The DS branch bypasses the `transform_index_page_table_decode` pipeline.
<comment>
LINUS: "Bypasses the transform pipeline" is engineerese for "we couldn't be
bothered to fit into the existing path, so we built a sidecar." The
transform pipeline exists for a reason — it turns logical page IDs into
physical ones. The DS adapter ALSO turns logical page IDs into physical
ones. You have two implementations of the same operation running in
parallel.

Pick one. Either the DS adapter calls into the existing transform (passing
its pre-selected page list), or you generalise the transform to accept a
sparse selection. Don't bypass code that does what you need — teach it to
take what you have.

[pensieve pipelines/run-when-refactoring Task 2: "If downstream code lacks
 fields or works by guessing, add the fields to the upstream authority
 path. Delete downstream patch branches created for missing data."]
</comment>
  - Positive Tests (expected to PASS):
    - Synthetic end-to-end decode test invokes the DS branch; a probe / call-counter asserts `_forward_flashmla_kv` is invoked with the adapter's physical `page_table_1` (not with token-level `topk_indices` flowing through the transform).
    - Adapter integration test verifies the adapter maps logical page IDs to physical page IDs through synthetic `req_to_token` / `req_pool_indices` correctly and preserves sequence-order ascending output with `-1` padding.
  - Negative Tests (expected to FAIL):
    - Adapter rejects out-of-range page IDs (page ID beyond what `req_to_token` covers).
    - Adapter rejects `valid_lengths > max_top_k`.
    - Adapter rejects non-ascending `selected_indices`.
    - Adapter rejects missing `-1` padding past `valid_lengths`.
    - Adapter rejects dtype / device / shape / batch-size mismatch between `selected_indices`, `valid_lengths`, and `req_to_token`.

- AC-3: A successful DS request's `meta_info` contains nested `meta_info["double_sparsity"] = {sparsity_rate, selected_pages, dense_fallback}` per the existing `customized_info_for_request` helper, and the transport works end-to-end.
  - Positive Tests (expected to PASS):
    - End-to-end transport test (synthetic backend) verifies the response payload contains the nested dict with `float`, `int`, and `int` field types respectively.
    - Same test verifies that field values match the per-request stats produced by the selector during that decode.
  - Negative Tests (expected to FAIL):
    - When DS is disabled, the `"double_sparsity"` key is absent from `meta_info`.
    - When the scheduler accumulator on `BatchTokenIDOutput.customized_info` is bypassed (simulated by patching the scheduler glue), the transport test detects the missing field at the tokenizer side rather than at the helper level.
<comment>
CODEX: `customized_info` is not a per-request summary channel today

The existing scheduler path accumulates customized_info per generated token,
then slices req.customized_info by output-token offset before
BatchTokenIDOutput. TokenizerManager blindly assigns meta_info[k] = v[i]. If DS
stats are appended through the existing accumulator, the user sees a list of
per-token dicts, not the single dict AC-3 demands. Either bypass that
accumulator with a request-summary field or define last/aggregate semantics.

[evidence: python/sglang/srt/managers/scheduler_output_processor_mixin.py:163-179; python/sglang/srt/managers/scheduler_output_processor_mixin.py:1246-1252; python/sglang/srt/managers/tokenizer_manager.py:1739-1741]
</comment>
<comment>
CODEX: AC-3 ignores production observability for failures

meta_info on successful requests is not enough. The risky failures are startup
mask rejection, adapter contract rejection, selector exceptions during decode,
and capture-disabled metric emission. metrics.py has healthy-selection metrics
and a dense-fallback counter, but no error counter or error-class labels.
Operators need logs and Prometheus signals that separate bad mask, bad adapter
input, selector runtime error, and rank mismatch.

[evidence: python/sglang/srt/layers/attention/double_sparsity/metrics.py:68-87; python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:469-477; python/sglang/srt/layers/attention/double_sparsity/validator.py:178-240]
</comment>

- AC-4: The standalone `m3b_page_stability_fixture` function has a synthetic-input CI hook in `test_double_sparsity_unit.py` that does not mutate `_double_sparsity_radix_fixture_passed`.
  - Positive Tests (expected to PASS):
    - The new CI test method drives `m3b_page_stability_fixture(...)` with a synthetic V3.2-shape input; cold and warm-run page signatures match.
    - Before and after the test runs, `server_args._double_sparsity_radix_fixture_passed` remains `False` (or `None`).
  - Negative Tests (expected to FAIL):
    - Perturbing the synthetic warm-run input causes the fixture to surface a mismatch and fail the test loudly.
    - Any attempt by the CI hook to set `_double_sparsity_radix_fixture_passed = True` is caught by a side-effect probe.

- AC-5: `development/loop2/RUNBOOK.md` exists and is reviewable, with five numbered operator phases covering calibrate → boot → benchmark → compare → M3-B hardware.
  - Positive Tests (expected to PASS):
    - File present at `development/loop2/RUNBOOK.md`.
    - Contains five numbered sections matching: (1) calibrate channel mask from `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2/`, (2) boot `serve_double_sparsity.sh` + `serve_native_nsa.sh`, (3) run `benchmark.sh` twice (DS + baseline) at agreed concurrencies, (4) run `benchmark_compare.py` for the side-by-side SLO + quality report, (5) run M3-B fixture on real hardware and decide DEC-2 default.
  - Negative Tests (expected to FAIL):
    - Removing or omitting any of the five sections fails reviewable-runbook lint (a simple structural check in the regression sweep).

- AC-6: The existing 87 unit tests stay green, and the following seven named tests are added (each must exist and pass).
<comment>
LINUS: What's the right test count? The answer is "however many it takes
to be confident the change is correct." Not "seven." Not "five or more."
Not "named ones with assigned AC sub-IDs in a plan document."

You enumerated test names BEFORE you wrote the implementation. That's
backwards. You'll either (a) write tests that exactly match the names and
miss real bugs, or (b) discover during implementation that a different
test set is more useful and have to update the plan to match. The plan
becomes a hostage to its own specificity.

Also: AC-6.3 wants a test that "the startup hook invokes `bind_runtime_data`
for every DS-enabled attention layer." That tests the WIRING you wrote.
It's "I called this function in this loop", not "the system works."
Pointless mockery.

[pensieve maxims/prefer-pragmatic-solutions; knowledge/taste-review —
 Google review order puts Tests AFTER Design/Functionality/Complexity for
 a reason]
</comment>
  - AC-6.1: `test_ds_page_table_adapter_basic_mapping` — synthetic `(selected_indices, valid_lengths)` → physical `page_table_1` through synthetic `req_to_token` / `req_pool_indices`.
  - AC-6.2: `test_ds_page_table_adapter_bounds_negative` — single parametrised test covering out-of-range page IDs, `valid_lengths > max_top_k`, non-ascending `selected_indices`, missing `-1` padding, and dtype/device/shape mismatches.
<comment>
CODEX: One parametrized negative test will hide the adapter contract

Out-of-range page IDs, padding holes, non-ascending rows, length overflow,
dtype mismatch, and device mismatch are different bugs. A single parametrized
test usually asserts "raises something", which lets one broad early guard
satisfy every case. Split them or at least assert exact exception classes,
messages, and the validation layer that must catch each case.

[evidence: python/sglang/srt/layers/attention/double_sparsity/selector.py:90-112; python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:456-462]
</comment>
  - AC-6.3: `test_ds_runtime_data_binding_production` — startup hook invokes `bind_runtime_data` for every DS-enabled attention layer; after init, all selectors report `IS_PLACEHOLDER == False`.
  - AC-6.4: `test_ds_meta_info_transport_end_to_end` — synthetic decode produces `meta_info["double_sparsity"]` with three correctly typed fields, transported via `BatchTokenIDOutput.customized_info`.
  - AC-6.5: `test_ds_m3b_synthetic_ci_hook` — drives the standalone `m3b_page_stability_fixture` with synthetic V3.2 shape; never touches `_double_sparsity_radix_fixture_passed`.
  - AC-6.6: `test_ds_bypasses_skip_topk_cache` — verifies the DS branch does not honour `prev_topk_indices` reuse: the selector is invoked on every DS layer/step regardless of `skip_topk`.
<comment>
CODEX: `skip_topk` is applied before the DS selector can defend itself

The plan says DS must ignore prev_topk_indices and run each step, but current
forward_absorb_prepare decides reuse before calling _select_topk_indices. If
self.skip_topk is true and prev_topk_indices exists, the DS branch is never
reached. AC-6.6 cannot be fixed inside _select_topk_indices; the prepare path
must gate skip_topk on not self.use_double_sparsity, and both alt_stream and
normal branches need coverage.

[evidence: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:227-270; python/sglang/srt/models/deepseek_v2.py:1428-1447]
</comment>
  - AC-6.7: `test_ds_decode_reaches_flashmla_kv_sparse_path` — probe asserts `_forward_flashmla_kv` is called with the adapter's physical `page_table_1` (not the NSA `topk_indices` → transform pipeline) when the DS branch is active.
<comment>
LINUS: AC-6.6 and AC-6.7 are not testing the feature. They're testing that
your `isinstance` dispatch landed on the right arm. They exist because the
DESIGN forces a routing branch.

In a properly designed version where DS and NSA return the same shape:
  - AC-6.7 is vacuous (there's no other codepath to reach)
  - AC-6.6 collapses to a one-line invariant ("selector ran every step"),
    not a probe-instrumented call-counter test

Two of your seven mandatory tests are scar tissue from the bad design.
That's not test coverage. That's a maintenance burden you wrote yourself a
ticket for.

[pensieve knowledge/taste-review — Google review order: Design FIRST, then
 Functionality, Complexity, Tests. Tests don't fix a bad design; they
 enshrine it.]
</comment>


  - Positive Tests (expected to PASS):
    - `pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` reports ≥ 94 passed (87 prior + 7 new).
  - Negative Tests (expected to FAIL):
    - Deleting or skipping any of the seven new tests trips the regression-sweep guard (`task9` enumerates the expected list and fails if any are missing or skipped).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
- A complete page-table adapter on the DS branch of `_select_topk_indices` that performs logical-to-physical page mapping via `req_to_token` / `req_pool_indices` and emits the physical `page_table_1` tensor directly consumed by `nsa_backend._forward_flashmla_kv` (bypassing `transform_index_page_table_decode`).
- A typed DS selection result returned from `_select_topk_indices` that the downstream consumer in `forward_absorb_*` recognises and routes to `_forward_flashmla_kv`; the NSA branch's existing `topk_indices` Tensor return is left unchanged.
<comment>
LINUS: STOP. The return type of `_select_topk_indices` should not encode
which selector ran. That's information leakage in its purest form: an
internal routing decision (DS vs NSA) leaking into the public return type,
forcing every caller to know about it and dispatch on it.

The caller doesn't care whether DS or NSA produced the result. The caller
cares about USING the result. Return the same tensor type from both
branches. Hide the difference behind the method, not in the method's
signature.

Also note "NSA branch's existing Tensor return is left unchanged" — you
just told me both arms of this `isinstance` are permanent. That's not a
refactor, that's coexistence forever.

[pensieve knowledge/taste-review — Information leakage CRITICAL: internal
 decisions exposed externally]
</comment>
- Replacement of the per-step `NotImplementedError` with light contract assertions that skip host-sync paths inside CUDA graph capture (dtype, device, shape, bounds, ascending order, padding consistency).
<comment>
LINUS: A check that "only runs sometimes" is, structurally, a fallback for
the case when it cannot run. You removed the `SGLANG_DS_ALLOW_*` env-gate
fallbacks (good), and now you're inventing a new one — assertions that
silently disable themselves during CUDA graph capture.

Either the contract holds always (push the cheap shape/dtype/device checks
to ahead-of-capture; push the value-dependent ones to Triton kernels where
the host doesn't have to sync), or it holds never. "Sometimes" is the
worst answer: bugs that only repro outside CUDA graph capture, escape into
production with capture on, and nobody can reproduce them.

Pick one. Don't paper over an architectural mismatch with a runtime gate.

[pensieve knowledge/taste-review — Fallback masks upstream issues]
</comment>
<comment>
CODEX: The hot path still allocates under the planned graph boundary

The plan talks about skipping host-sync checks during capture, but capture
safety is also about allocation. The current selector path creates q_proj,
scores, topk outputs, and selected buffers, and the graph helper captures
retrieve_topk then copies into preallocated state. A DS page-table adapter that
returns a fresh page_table_1 per step repeats the same problem. The ABI should
accept graph-owned output buffers or reuse NSA metadata buffers.

[evidence: python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:155,282-286,377-419; python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py:157-182]
</comment>
- Full removal of `SGLANG_DS_ALLOW_NO_ADAPTER` from `validate_double_sparsity` and full removal of `SGLANG_DS_ALLOW_PLACEHOLDER` from production code in `DoubleSparsitySelector`. Both exports stripped from `development/serve_double_sparsity.sh`. `SGLANG_DS_RADIX_OVERRIDE` is unaffected.
- Production wiring of `DoubleSparsitySelector.bind_runtime_data(...)` at the per-layer DS attention init hook (where channel mask, page signature table, TP process group, and device all exist).
<comment>
CODEX: TP-rank agreement is assumed, not tested

Passing a process_group into bind_runtime_data is not the same as proving all
ranks pick the same logical pages. all_reduce_page_scores silently becomes a
no-op when process_group is None or torch.distributed is not initialized, which
is fine for single rank and dangerous for an accidentally unbound TP deployment.
Add a fail-fast startup invariant for TP>1 and a test that rank-local scores
would diverge without the all-reduce.

[evidence: python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:289-310; python/sglang/srt/layers/attention/double_sparsity/selector.py:126-129; python/sglang/srt/layers/attention/double_sparsity/channel_mask.py:304-351]
</comment>
- Selector stats published into `BatchTokenIDOutput.customized_info` so the existing tokenizer hook at `tokenizer_manager.py` unpacks them into `meta_info["double_sparsity"]`.
- CI hook for `m3b_page_stability_fixture` driven by a synthetic V3.2-shape input, deterministic and CPU-runnable where possible (gated on CUDA availability only if absolutely necessary).
- `development/loop2/RUNBOOK.md` with all five numbered operator phases, including concrete commands referencing the existing scripts.
- The 87 prior unit tests stay green; the seven named tests above are added.

### Lower Bound (Minimum Acceptable Scope)
- All upper-bound scope items are themselves required (the resolved design decisions under `## Claude-Codex Deliberation` leave no degrees of freedom): the adapter, the bind wiring, the env-gate removals, the metrics transport, the M3-B CI hook, the runbook, and the seven named tests are mandatory.
- Discretion is limited to prose density (runbook may be terse), refactor scope (no unrelated cleanups), and assertion verbosity (assertions must exist but may be combined where the failure modes share a check).

### Allowed Choices
- Can use:
  - Existing Triton kernels under `python/sglang/srt/layers/attention/double_sparsity/`.
  - Existing `create_flashmla_kv_indices_triton` as a reference implementation pattern for the adapter, but the adapter itself is DS-specific and selects from `selected_indices` rather than from a dense range.
  - Existing `m3b_page_stability_fixture` function (standalone in `double_sparsity/page_signature_write.py`).
  - Existing `customized_info_for_request` helper in `double_sparsity/metrics.py` (output shape unchanged).
  - Mocking/patching of `req_to_token` and `req_pool_indices` in unit tests to drive synthetic shapes.
  - Test-only references to `SGLANG_DS_ALLOW_PLACEHOLDER` are allowed (e.g., to construct a placeholder selector via direct attribute injection or `object.__new__`), but production code paths must not read the env var.
- Cannot use:
  - The top-level FlashMLA dense `block_table` path (`python/sglang/srt/layers/attention/flashmla_backend.py` per-batch metadata mutation) as the DS handoff. The DS path is via NSA's `_forward_flashmla_kv` sparse-indices route (DEC-1).
  - Imports from `python/sglang/srt/mem_cache/sparsity/` (legacy HiSparse). DS is standalone.
  - Real DeepSeek-V3.2 weight loads in CI.
  - Any code path that flips `_double_sparsity_radix_fixture_passed` from the CI hook. DEC-2 default flipping is out of scope.
  - Compatibility shims that keep `SGLANG_DS_ALLOW_NO_ADAPTER` / `SGLANG_DS_ALLOW_PLACEHOLDER` callable from production paths after this loop.
  - Reuse of `prev_topk_indices` on the DS branch (DS selector runs every decode layer/step; `skip_topk` short-circuit is disabled on DS path).

> **Note on Deterministic Designs**: The draft fixes the design space tightly (specific adapter responsibilities, specific env gates to remove, specific runbook structure, specific test count floor). Path boundaries reflect that narrow specification; upper and lower bound effectively converge.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Adapter sketch (pseudocode):

```
DSSelectionResult:
    page_table_1: int32[bs, max_seqlen_pad]   # physical page IDs in sequence order, -1 padded
    valid_lengths: int32[bs]
    stats: DoubleSparsityRequestStats          # per-request: sparsity_rate, selected_pages, dense_fallback

DeepseekV2AttentionMLA._select_topk_indices DS branch:
    selected_indices, valid_lengths, stats = self.double_sparsity_selector.retrieve_topk(...)
    # Contract assertions (skip host-sync in CUDA capture):
    assert_shape_dtype(selected_indices, valid_lengths)
    assert_ascending_with_minus_one_padding(selected_indices, valid_lengths)  # Triton-side
    page_table_1 = ds_page_table_adapter(
        selected_indices=selected_indices,
        valid_lengths=valid_lengths,
        req_to_token=forward_batch.req_to_token_pool.req_to_token,
        req_pool_indices=forward_batch.req_pool_indices,
        page_size=PAGE_SIZE,
    )
    return DSSelectionResult(page_table_1=page_table_1, valid_lengths=valid_lengths, stats=stats)

# Downstream consumer (forward_absorb_prepare/core or backend dispatch):
if isinstance(result, DSSelectionResult):
    output = nsa_backend._forward_flashmla_kv(..., page_table_1=result.page_table_1, ...)
    bind_request_stats_to_scheduler_output(result.stats)   # see below
else:
    # existing NSA topk_indices Tensor path
    page_table_1 = transform_index_page_table_decode(page_table, topk_indices=result)
    output = nsa_backend._forward_flashmla_kv(..., page_table_1=page_table_1, ...)
```
<comment>
LINUS: Read your own pseudocode out loud.

  if isinstance(result, DSSelectionResult):
      ...DS path, calls _forward_flashmla_kv...
  else:
      ...NSA path, calls _forward_flashmla_kv...

Both arms call the same backend function. Both arms pass it a
`page_table_1`. The ONLY difference is where `page_table_1` came from. If
I saw this in a patch I would NAK it in five seconds and tell you to
delete the `if`.

Move the transform call inside the DS branch's adapter (so the DS branch
produces `page_table_1` exactly the way the else-arm does), and then the
ENTIRE consumer becomes:

  output = nsa_backend._forward_flashmla_kv(..., page_table_1=result, ...)
  bind_request_stats_if_any(result_extras)

One path. Zero `isinstance`. The stats accumulator handles the DS-vs-NSA
difference (DS produces stats, NSA produces None), and you do not need a
named dataclass to carry it.

Your refactor pipeline literally says "Two runtime paths must be kept:
redesign the boundary until the old path can be deleted." You do not have
a deletion plan for the else-arm. You have a coexistence plan.

[pensieve pipelines/run-when-refactoring Hard Rule 3 + Failure Fallback]
</comment>

`bind_runtime_data` production wiring (sketch):

```
# In the per-layer DS attention init hook (called once per layer at model load,
# after channel mask + page signature table are allocated and TP group is ready)
selector.bind_runtime_data(
    page_signature_table=layer_owned_page_signature_table,
    channel_mask=channel_mask_sliced_for_this_tp_rank,
    process_group=tp_group,
)
assert selector.IS_PLACEHOLDER is False
```

Customized info transport (sketch):

```
# Selector emits per-request stats during retrieve_topk
# Stats accumulate per-request inside the forward batch loop
# Scheduler glue collects them into BatchTokenIDOutput.customized_info:
batch_out.customized_info = {
    "double_sparsity": [customized_info_for_request(stats_req_i) for i in range(bs)],
}
# Tokenizer hook at tokenizer_manager.py L1739-1741 already unpacks:
#   for k, v in recv_obj.customized_info.items():
#       meta_info[k] = v[i]
# → meta_info["double_sparsity"] = {sparsity_rate, selected_pages, dense_fallback}
```

M3-B CI hook (sketch):

```python
# In test_double_sparsity_unit.py
def test_ds_m3b_synthetic_ci_hook(self):
    fixture_inputs = build_synthetic_v32_shape_inputs()
    cold_signature = m3b_page_stability_fixture(fixture_inputs, mode="cold")
    warm_signature = m3b_page_stability_fixture(fixture_inputs, mode="warm")
    self.assertEqual(cold_signature, warm_signature)
    # Side-effect probe: confirm radix fixture flag untouched
    self.assertFalse(getattr(self.server_args, "_double_sparsity_radix_fixture_passed", False))
```

### Relevant References

- `python/sglang/srt/models/deepseek_v2.py` — `DeepseekV2AttentionMLA._select_topk_indices` (DS branch seam; the `NotImplementedError` raise lives here).
- `python/sglang/srt/layers/attention/double_sparsity/selector.py` — `DoubleSparsitySelector.retrieve_topk` (page-level output) and `bind_runtime_data` (needs production caller).
- `python/sglang/srt/layers/attention/double_sparsity/validator.py` — `validate_double_sparsity`, the `SGLANG_DS_ALLOW_NO_ADAPTER` startup gate.
- `python/sglang/srt/layers/attention/double_sparsity/metrics.py` — `customized_info_for_request` helper.
- `python/sglang/srt/layers/attention/double_sparsity/page_signature_write.py` — `m3b_page_stability_fixture` (already standalone).
- `python/sglang/srt/layers/attention/nsa_backend.py` — `_forward_flashmla_kv` and the existing `transform_index_page_table_decode → page_table_1 → _forward_flashmla_kv` pipeline that the DS branch bypasses.
- `python/sglang/srt/layers/attention/nsa/transform_index.py` — the transform the DS path skips.
- `python/sglang/srt/layers/attention/flashmla_backend.py` — `create_flashmla_kv_indices_triton` reference pattern; **not** the integration target for DS.
- `python/sglang/srt/managers/tokenizer_manager.py` — existing `customized_info` unpack loop.
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py` — where `BatchTokenIDOutput.customized_info` is populated.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — destination for all seven new named tests.
- `development/serve_double_sparsity.sh`, `development/serve_native_nsa.sh`, `development/benchmark.sh`, `development/benchmark_baseline.sh`, `development/benchmark_compare.py` — operator-runbook target scripts.

## Dependencies and Sequence

### Milestones

1. **Milestone A — Contract & Adapter**: Lock the DS handoff ABI and land the page-table adapter.
   - Phase A1: Define `DSSelectionResult` type or equivalent typed return that the downstream consumer dispatches on.
   - Phase A2: Implement `ds_page_table_adapter` (logical → physical via `req_to_token` / `req_pool_indices`) as a Triton kernel or Python function with Triton kernel for the hot path.
   - Phase A3: Replace `NotImplementedError` in `_select_topk_indices` DS branch with contract assertions + adapter call; route downstream consumer to `_forward_flashmla_kv` with physical `page_table_1`.

2. **Milestone B — Production Wiring**: Bridge DS data into runtime and metrics into the response.
   - Phase B1: Wire `bind_runtime_data` from the per-layer DS attention init hook (model load path).
   - Phase B2: Wire selector stats → `BatchTokenIDOutput.customized_info["double_sparsity"]` in the scheduler output processor; tokenizer hook is unchanged.

3. **Milestone C — Tests & Gate Removal**: Add seven named tests, then remove env gates.
   - Phase C1: Add the seven named tests (six adapter/transport tests + M3-B CI hook).
   - Phase C2: Remove `SGLANG_DS_ALLOW_NO_ADAPTER` from validator; remove `SGLANG_DS_ALLOW_PLACEHOLDER` from production code in selector; strip both env exports from `serve_double_sparsity.sh`.

4. **Milestone D — Runbook & Regression**: Operator-phase documentation and final sweep.
   - Phase D1: Write `development/loop2/RUNBOOK.md` (five numbered operator phases).
   - Phase D2: Regression sweep — verify 94+ unit tests pass; verify `rg` negative grep against `SGLANG_DS_ALLOW_NO_ADAPTER` / `SGLANG_DS_ALLOW_PLACEHOLDER` in `python/sglang/srt/` + `development/serve_double_sparsity.sh`.

Dependency notes:
- Milestone A blocks Milestones B and C (adapter must exist before binding / tests / gate removal are meaningful).
- Phase C2 (gate removal) must follow Phase B1 (`bind_runtime_data` wired) — otherwise removing `SGLANG_DS_ALLOW_PLACEHOLDER` breaks boot.
- Phase D1 (runbook) is independent and can land in parallel with A/B/C.
- Phase D2 (regression sweep) is the final gate.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Lock the DS handoff ABI: define `DSSelectionResult` (or equivalent typed return), document the downstream dispatch contract for `forward_absorb_*` and `_forward_flashmla_kv`. | AC-2 | analyze | - |
| task2 | Implement the page-table adapter (logical → physical via `req_to_token` / `req_pool_indices`, emits physical `page_table_1` in sequence order, `-1` padded). Replace the per-step `NotImplementedError` in `_select_topk_indices` DS branch with contract assertions + adapter call. Skip host-sync inside CUDA graph capture. | AC-2 | coding | task1 |
| task3 | Wire `DoubleSparsitySelector.bind_runtime_data(...)` at the per-layer DS attention init hook so every DS-enabled layer's selector is bound (channel mask sliced for TP rank, page signature table allocator-owned, TP process group, device) before serving. | AC-1 | coding | task2 |
| task4 | Wire selector stats → `BatchTokenIDOutput.customized_info["double_sparsity"]` in the scheduler output processor mixin; verify existing tokenizer hook unpacks correctly to `meta_info["double_sparsity"]`. | AC-3 | coding | task2 |
| task5 | Add named tests AC-6.1–6.4, 6.6, 6.7 to `test_double_sparsity_unit.py`: adapter mapping, adapter bounds negative, runtime data binding production, meta_info transport end-to-end, `skip_topk` bypass, `_forward_flashmla_kv` sparse-path probe. | AC-2, AC-3, AC-6 | coding | task2, task3, task4 |
| task6 | Remove `SGLANG_DS_ALLOW_NO_ADAPTER` from `validate_double_sparsity`; remove `SGLANG_DS_ALLOW_PLACEHOLDER` from production code in `DoubleSparsitySelector`; strip both env exports from `development/serve_double_sparsity.sh`. Update any unit tests that depended on the placeholder env to construct a placeholder selector directly. | AC-1, AC-6 | coding | task5 |
| task7 | Add `test_ds_m3b_synthetic_ci_hook` (AC-6.5) driving `m3b_page_stability_fixture` with synthetic V3.2-shape inputs; assert cold/warm signatures match; assert `_double_sparsity_radix_fixture_passed` is not mutated. | AC-4, AC-6 | coding | task1 |
| task8 | Write `development/loop2/RUNBOOK.md` with five numbered operator phases (calibrate / boot / benchmark / compare / M3-B hardware), referencing concrete commands and the existing scripts. | AC-5 | coding | - |
| task9 | Regression sweep — verify ≥ 94 unit tests pass; verify `rg "SGLANG_DS_ALLOW_NO_ADAPTER|SGLANG_DS_ALLOW_PLACEHOLDER" python/sglang/srt development/serve_double_sparsity.sh` returns zero hits; verify all seven named tests in AC-6 exist and are not skipped. | AC-1, AC-6 | analyze | task5, task6, task7 |

<comment>
CODEX: Mid-decode selector failures have no containment policy

Once the env-gate fallback is gone, retrieve_topk or adapter failures propagate
from the attention prepare path in the middle of a batch. The plan does not say
whether that aborts one request, the whole batch, or the worker, and it does not
define how partially accumulated customized_info is cleared. Fail-loud is fine,
but the error boundary must be explicit or operators will get an opaque worker
crash for a single corrupt request state.

[evidence: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:241-270; python/sglang/srt/managers/scheduler_output_processor_mixin.py:163-179]
</comment>

## Claude-Codex Deliberation

### Agreements
- DS branch's `_select_topk_indices` is currently an NSA token-topk seam; returning a dense FlashMLA `block_table` is not contract-compatible with the existing call chain.
- Two FlashMLA paths exist (top-level dense `block_table` vs NSA `_forward_flashmla_kv` sparse); DS targets the NSA sparse path.
- `m3b_page_stability_fixture` is already a standalone function in `double_sparsity/page_signature_write.py`; the CI hook is a new test method that invokes it on a synthetic shape.
- `_double_sparsity_radix_fixture_passed` must not be flipped by a synthetic CI hook (DEC-2 default flip is operator-phase work).
- `SGLANG_DS_RADIX_OVERRIDE` is a separate concern and stays in place; this loop only removes the two ALLOW gates.
- AC-6 uses named tests (not just a count) to prevent gaming the test-growth floor; the named list also makes the regression sweep deterministic.

### Resolved Disagreements

- Topic: Adapter target path.
  - Claude initial position: Token-level NSA-shape `topk_indices` produced by adapter (Design A), reusing existing pipeline.
  - Codex initial position: Same shape claim is incompatible with claiming the adapter does logical→physical mapping; pick one.
  - Resolution: Adapter does logical→physical via `req_to_token` / `req_pool_indices` and emits the physical `page_table_1` directly. The DS branch bypasses `transform_index_page_table_decode` by returning a typed `DSSelectionResult` that the downstream consumer dispatches to `_forward_flashmla_kv` without the transform. The NSA branch's existing `topk_indices` Tensor return is unchanged. This matches the draft's "drives FlashMLA directly" wording and avoids contract conflation.
<comment>
LINUS: This is a deliberation between Design A (token-level, reuse
pipeline) and Design B (typed result, bypass pipeline). Design C — "DS
branch produces its `page_table_1` by calling the existing transform with
its own selected logical pages, returning the SAME tensor type as the NSA
path" — was never on the table.

You "converged" by picking between two bad options without considering the
obvious third one. That's not deliberation, that's a coin flip with extra
steps. Ousterhout's rule: design it twice. You designed it once and held a
vote between two variants of the same design.

[pensieve knowledge/taste-review — Ousterhout: "Design it twice. You'll
 end up with a much better result."]
</comment>

- Topic: `meta_info` shape (top-level fields vs nested).
  - Claude initial position: Nested under `meta_info["double_sparsity"]` (matches existing helper, minimum-diff).
  - Codex initial position: Agreed nested; warned that AC-3 must verify end-to-end transport, not just helper shape.
  - Resolution: Nested under `meta_info["double_sparsity"]`. AC-3 includes both positive end-to-end transport check and a negative test that breaks the scheduler accumulator to confirm the failure is detected at the tokenizer side.

- Topic: `SGLANG_DS_ALLOW_PLACEHOLDER` disposition.
  - Claude initial position: Remove both env gates and wire `bind_runtime_data` in production.
  - Codex initial position: Agreed, but flagged that `bind_runtime_data` is not pure startup data — it needs the per-layer selector, allocator-owned page signature table, TP-sliced channel mask, device, and TP process group all in scope.
  - Resolution: Both env gates removed; `bind_runtime_data` wired at the per-layer DS attention init hook (where all required objects exist). Tests that previously relied on the placeholder env construct a placeholder selector directly (or use the existing `IS_PLACEHOLDER` toggle path in tests).

- Topic: AC-6 framing (numeric count vs named tests).
  - Claude initial position: "≥ 5 new tests" per draft.
  - Codex initial position: Numeric count is gameable; list named tests and required failure modes.
  - Resolution: Seven named tests are enumerated as AC-6.1–6.7; the count floor (≥ 5) is preserved as a side effect.

- Topic: `skip_topk` / `prev_topk_indices` reuse on DS branch.
  - Claude initial position: Implementation note ("DS does not consume `prev_topk_indices`").
  - Codex initial position: Should be an AC-backed test, not a constraint, because existing control flow honours reuse before `_select_topk_indices`.
  - Resolution: Added AC-6.6 (`test_ds_bypasses_skip_topk_cache`) and the corresponding constraint in path boundaries.

- Topic: `_forward_flashmla_kv` codepath verification.
  - Claude initial position: Single adapter mapping test (AC-6.1).
  - Codex initial position: Single mapping test does not prove the runtime control flow reaches the sparse path.
  - Resolution: Added AC-6.7 (`test_ds_decode_reaches_flashmla_kv_sparse_path`) with a probe / call-counter that confirms `_forward_flashmla_kv` is invoked with the adapter's physical `page_table_1` (not via the transform).

### Convergence Status
- Final Status: `converged`

## Pending User Decisions

None at the time of plan generation. All Claude/Codex disagreements have been resolved through the convergence loop; the resolutions are recorded above and baked into the acceptance criteria, path boundaries, and task breakdown. The draft's quantitative metric ("test count grows by at least 5") is treated as a hard floor and is satisfied with seven named tests; this is not an optimisation-direction metric, so no further user confirmation is required.

If the user disagrees with any of the resolved decisions (DEC-1 adapter target, DEC-2 meta_info shape, DEC-3 placeholder-gate disposition), they should call those out before the RLCR loop begins so the affected ACs and tasks can be re-scoped.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead. Examples: name the typed return `DoubleSparsitySelectionResult` (or similar), not `DSResultForAC2`; name the test methods after the behaviour they verify, not after their AC label.

### Speculative-Naming Rule Cross-Check
This loop does not introduce new speculative-decoding identifiers; the DS naming follows the established double-sparsity conventions in `python/sglang/srt/layers/attention/double_sparsity/` (which are unrelated to the speculative-decoding `accept_*` / `correct_*` / `bonus_*` rule set in `.claude/rules/speculative-naming.md`). No exceptions are required.
<comment>
LINUS: You wrote a paragraph in a 335-line plan to formally notarise that
an unrelated naming convention does not apply to this change. Why is this
here? Who is the audience?

If the answer is "a future reviewer" — they don't need this notarisation;
they can look at the diff. If the answer is "the process required it" —
the process is broken. The original draft was 22 lines. You expanded it to
308 lines of structured plan. The ratio of ceremony to substance is bad
enough without adding "rules that do not apply" sections.

Plans should fit in your head. This one doesn't.

[pensieve maxims/prefer-pragmatic-solutions-over-theoretical-completeness:
 "Avoid speculative abstractions without clear near-term value."]
</comment>

### Branch and PR Scope
- Branch: `dev/double-sparsity-standalone` (continues from loop 1).
- All adapter / wiring / test / runbook work lands as a single coherent series; no compatibility shims for the removed env gates.

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
