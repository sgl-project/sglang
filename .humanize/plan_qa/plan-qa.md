# Refine Plan QA

## Summary

Twenty `<comment>` blocks were extracted from `development/loop2/plan.md`: ten written in a Linus Torvalds voice (LINUS) targeting design taste, and ten from a Codex pass (CODEX) targeting correctness, observability, and operational gaps Linus missed. All twenty were classified as `change_request` (none were questions or research requests). The dominant change is architectural: the previously-resolved Design B (typed-union `DSSelectionResult` + downstream `isinstance` dispatch that bypasses the NSA `transform_index_page_table_decode` pipeline) is replaced by Design C — a unified `page_table_1` Tensor return from both DS and NSA branches of `_select_topk_indices`, with the downstream consumer making one call and no `isinstance` branch. Around that central change, Codex's comments lifted nine separate correctness and operability properties to first-class plan content: `bind_runtime_data` idempotence, broadened channel-mask validation, per-request (not per-token) `meta_info` semantics, production observability with a Prometheus error counter and structured logs, the `forward_absorb_prepare` skip-topk gate fix (which is not a DS-branch concern), CUDA-graph allocation safety, TP-rank invariance at startup and at runtime, mid-decode error containment, and a hardware preflight in the runbook. Three new top-level ACs (AC-7 TP-rank invariance, AC-8 CUDA-graph allocation safety, AC-9 mid-decode error containment) were added; AC-1, AC-2, AC-3, AC-5, AC-6 were materially rewritten; AC-4 is unchanged. The Speculative-Naming Cross-Check section, identified by LINUS as pure process noise, was removed. The refined plan converges; no Pending User Decisions remain because the design space is now constrained tightly enough that the remaining choice (generalise the existing transform vs expand-then-transform helper) is an implementation note, not a user-facing decision.

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1 | change_request | Preamble (top of file) | "LINUS — OVERALL REACTION: … The actual code change is small. … the central architecture choice … was decided by a two-agent 'deliberation' that never put the obvious unified-shape option on the table." | applied |
| CMT-2 | change_request | Goal Description | "LINUS: This bullet contains the whole problem. You're introducing a typed union return so downstream code can `isinstance()`-dispatch DS vs NSA. … why can't `_select_topk_indices` return the SAME `page_table_1` tensor on BOTH paths" | applied |
| CMT-3 | change_request | Goal Description (after "out of scope") | "CODEX: The plan says hardware is out of scope while baking in one target … The runbook needs a preflight that fails before serving if the actual node and backends do not match those assumptions." | applied |
| CMT-4 | change_request | AC-2 | "LINUS: 'Bypasses the transform pipeline' is engineerese for 'we couldn't be bothered to fit into the existing path' … Pick one. Either the DS adapter calls into the existing transform … or you generalise the transform to accept a sparse selection." | applied |
| CMT-5 | change_request | AC-1 (after positive/negative blocks) | "CODEX: `bind_runtime_data` needs an idempotence contract … A second call silently swaps page_signature_table, channel_mask, and process_group … Make rebind either a typed error or an exact same-object no-op, and test both." | applied |
| CMT-6 | change_request | AC-1 (after CMT-5) | "CODEX: 'Malformed channel mask' is too narrow … It does not reject NaN or Inf channel_weights, nor all-zero projections … AC-1's negative test should include weights-level corruption." | applied |
| CMT-7 | change_request | AC-3 | "CODEX: `customized_info` is not a per-request summary channel today … If DS stats are appended through the existing accumulator, the user sees a list of per-token dicts, not the single dict AC-3 demands." | applied |
| CMT-8 | change_request | AC-3 (after CMT-7) | "CODEX: AC-3 ignores production observability for failures … metrics.py has healthy-selection metrics and a dense-fallback counter, but no error counter or error-class labels. Operators need logs and Prometheus signals that separate bad mask, bad adapter input, selector runtime error, and rank mismatch." | applied |
| CMT-9 | change_request | AC-6 (preamble) | "LINUS: What's the right test count? The answer is 'however many it takes to be confident the change is correct.' Not 'seven.' … You enumerated test names BEFORE you wrote the implementation. That's backwards." | applied |
| CMT-10 | change_request | AC-6.2 | "CODEX: One parametrized negative test will hide the adapter contract … A single parametrized test usually asserts 'raises something', which lets one broad early guard satisfy every case. Split them or at least assert exact exception classes." | applied |
| CMT-11 | change_request | AC-6.6 | "CODEX: `skip_topk` is applied before the DS selector can defend itself … AC-6.6 cannot be fixed inside `_select_topk_indices`; the prepare path must gate skip_topk on `not self.use_double_sparsity`, and both alt_stream and normal branches need coverage." | applied |
| CMT-12 | change_request | AC-6.7 | "LINUS: AC-6.6 and AC-6.7 are not testing the feature. They're testing that your `isinstance` dispatch landed on the right arm. … In a properly designed version where DS and NSA return the same shape: AC-6.7 is vacuous." | applied |
| CMT-13 | change_request | Path Boundaries / Upper Bound (typed return bullet) | "LINUS: STOP. The return type of `_select_topk_indices` should not encode which selector ran. … Return the same tensor type from both branches. Hide the difference behind the method, not in the method's signature." | applied |
| CMT-14 | change_request | Path Boundaries / Upper Bound (contract assertions bullet) | "LINUS: A check that 'only runs sometimes' is, structurally, a fallback for the case when it cannot run. … Either the contract holds always … or it holds never." | applied |
| CMT-15 | change_request | Path Boundaries / Upper Bound (after CMT-14) | "CODEX: The hot path still allocates under the planned graph boundary … A DS page-table adapter that returns a fresh page_table_1 per step repeats the same problem. The ABI should accept graph-owned output buffers or reuse NSA metadata buffers." | applied |
| CMT-16 | change_request | Path Boundaries / Upper Bound (after `bind_runtime_data` bullet) | "CODEX: TP-rank agreement is assumed, not tested … all_reduce_page_scores silently becomes a no-op when process_group is None … Add a fail-fast startup invariant for TP>1 and a test that rank-local scores would diverge without the all-reduce." | applied |
| CMT-17 | change_request | Feasibility Hints / pseudocode `isinstance` block | "LINUS: Read your own pseudocode out loud. … One path. Zero `isinstance`. The stats accumulator handles the DS-vs-NSA difference (DS produces stats, NSA produces None), and you do not need a named dataclass to carry it." | applied |
| CMT-18 | change_request | Task Breakdown (after task9 in original) | "CODEX: Mid-decode selector failures have no containment policy … the error boundary must be explicit or operators will get an opaque worker crash for a single corrupt request state." | applied |
| CMT-19 | change_request | Resolved Disagreements / Adapter target path | "LINUS: This is a deliberation between Design A (token-level, reuse pipeline) and Design B (typed result, bypass pipeline). Design C — 'DS branch produces its `page_table_1` by calling the existing transform with its own selected logical pages, returning the SAME tensor type as the NSA path' — was never on the table." | applied |
| CMT-20 | change_request | Implementation Notes / Speculative-Naming Rule Cross-Check | "LINUS: You wrote a paragraph in a 335-line plan to formally notarise that an unrelated naming convention does not apply to this change. … Plans should fit in your head. This one doesn't." | applied |

## Answers

No `question`-classified comments were extracted. The Linus comments are stylistically rhetorical ("What's the right test count?"), but each one closes with an actionable directive that pushes the comment into `change_request` per the classification heuristics.

## Research Findings

No `research_request`-classified comments were extracted. Codex's evidence-bearing comments already supply concrete file paths and line ranges; no additional repository research was required to apply the corresponding plan changes. The CMT-3 hardware-preflight content was sourced from the `[evidence: development/serve_double_sparsity.sh:5-7,23-31,48-56; python/sglang/srt/layers/attention/double_sparsity/validator.py:42-45,130-151]` references inside the comment itself.

## Plan Changes Applied

### CMT-1: Plan ceremony and missing-design-option meta-critique

**Original Comment:**
```
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
```

**Changes Made:**
Cumulative effect of CMT-2, CMT-13, CMT-17, CMT-19 implementing Design C; CMT-20 removing the speculative-naming subsection; trimming the Pending User Decisions block. AC-2 now verifies the **observable** unified-shape contract (one `_forward_flashmla_kv` call site, zero `isinstance` matches in `forward_absorb_*`).

**Affected Sections:** Goal Description; Path Boundaries (Upper Bound + Cannot use); Claude-Codex Deliberation; Implementation Notes; Pending User Decisions.

**Cross-Reference Updates:** "DEC-1/2/3" labels from the previous deliberation are dropped from the refined plan to avoid clashing with the loop-1 DEC-2 (radix cache) reference that still appears in the Goal Description and Original Design Draft appendix.

---

### CMT-2: Typed-union return is a special case

**Original Comment:**
```
LINUS: This bullet contains the whole problem. You're introducing a typed
union return so downstream code can `isinstance()`-dispatch DS vs NSA.
That's not engineering — that's two parallel universes glued together with
a tag.

The data-structure question you didn't ask: why can't `_select_topk_indices`
return the SAME `page_table_1` tensor on BOTH paths — DS branch having
pre-computed it, NSA branch having run the transform? Then downstream has
zero branches. The selection logic stays inside the method, where it
belongs. That's good taste. What you've written is a special case.

[pensieve maxims/eliminate-special-cases-by-redesigning-data-flow]
```

**Changes Made:**
Goal Description's first code-terms bullet now reads: "both DS and NSA branches return the same Tensor type — `page_table_1: int32[bs, max_seqlen_pad]` — to the downstream consumer. There is no typed-union return, no `isinstance` dispatch, and only one `_forward_flashmla_kv` call site downstream." AC-2 is rewritten to assert this observable contract.

**Affected Sections:** Goal Description; AC-2; Path Boundaries / Cannot use; Feasibility Hints (Conceptual Approach pseudocode); Claude-Codex Deliberation (see CMT-19).

**Cross-Reference Updates:** AC-2 positive test changes from "the adapter's physical `page_table_1` (not topk_indices via the transform)" to "DS-produced `page_table_1` matches what the NSA path would produce for the equivalent logical-pages-only input, bit-for-bit on a synthetic fixture."

---

### CMT-3: Hardware-out-of-scope vs. baked-in target

**Original Comment:**
```
CODEX: The plan says hardware is out of scope while baking in one target

serve_double_sparsity.sh assumes a single H200 node, 8-way TP, FP8, page 64,
top_k 2048, and the validator only allows fp8_e4m3 with flashmla_kv. Synthetic
CI cannot validate that backend, CUDA arch, FP8 dtype, or TP group behavior.
The runbook needs a preflight that fails before serving if the actual node and
backends do not match those assumptions.

[evidence: development/serve_double_sparsity.sh:5-7,23-31,48-56; python/sglang/srt/layers/attention/double_sparsity/validator.py:42-45,130-151]
```

**Changes Made:**
AC-5 mandates a "Phase 0 preflight" in `RUNBOOK.md` that checks backend (`flashmla_kv`), dtype (`fp8_e4m3`), page size (64), top-k (2048), TP world size (8), and CUDA arch (H200 / cc 9.x). The preflight exits non-zero before launching the server if any check fails. AC-5 negatives cover "missing Phase 0" and "preflight against a non-matching fake node exits non-zero".

**Affected Sections:** AC-5; Task Breakdown / task14 (writing RUNBOOK with Phase 0 preflight); Goal Description.

**Cross-Reference Updates:** Runbook structure expands from 5 to 6 phases (Phase 0 + Phases 1–5).

---

### CMT-4: "Bypasses the transform pipeline" is a sidecar

**Original Comment:**
```
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

[pensieve pipelines/run-when-refactoring Task 2]
```

**Changes Made:**
AC-2 no longer says "bypasses `transform_index_page_table_decode`". Implementation Notes lists the two acceptable strategies — generalise the transform or wrap it via `expand_ds_selection_to_topk_indices` helper — and forbids shipping both.

**Affected Sections:** AC-2; Path Boundaries / Allowed Choices; Implementation Notes / "DS-Side `page_table_1` Builder"; Task Breakdown / task1 and task2.

**Cross-Reference Updates:** None.

---

### CMT-5: `bind_runtime_data` lacks an idempotence contract

**Original Comment:**
```
CODEX: `bind_runtime_data` needs an idempotence contract

AC-1 says every selector is bound exactly once, but the current method does
not track that. A second call silently swaps page_signature_table,
channel_mask, and process_group after IS_PLACEHOLDER is already false. If
graph capture or per-rank setup has seen the first tensors, this is a stale
pointer or rank-divergence bug, not a harmless re-init. Make rebind either a
typed error or an exact same-object no-op, and test both.

[evidence: python/sglang/srt/layers/attention/double_sparsity/selector.py:76-129]
```

**Changes Made:**
AC-1 positive adds "second `bind_runtime_data` with SAME objects is a no-op". AC-1 negative adds "second `bind_runtime_data` with DIFFERENT `channel_mask`/`process_group`/`page_signature_table` raises `DoubleSparsityRebindError`". AC-6 anchor `test_ds_rebind_idempotence` covers both. task4 implements idempotence.

**Affected Sections:** AC-1; AC-6 (rebind idempotence anchor); Task Breakdown / task4; Implementation Notes (`bind_runtime_data` idempotence sketch); Claude-Codex Deliberation.

**Cross-Reference Updates:** None.

---

### CMT-6: Channel mask corruption broader than "malformed"

**Original Comment:**
```
CODEX: "Malformed channel mask" is too narrow

The loader rejects missing tensors, schema drift, hash mismatch, dtype/page
size mismatch, and out-of-range channel_selection. It does not reject NaN or
Inf channel_weights, nor all-zero projections. Those can produce NaN or flat
scores in compute_page_scores and degrade into arbitrary topk without a
startup error. AC-1's negative test should include weights-level corruption,
not just missing files and bad JSON.

[evidence: python/sglang/srt/layers/attention/double_sparsity/channel_mask.py:147-211; python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:222-286]
```

**Changes Made:**
AC-1 negatives distinguish "missing channel-mask file" (`DoubleSparsityChannelMaskMissing`) from "channel_weights NaN/Inf/all-zero projection" (`DoubleSparsityChannelMaskCorrupt`). AC-6 anchor `test_ds_channel_mask_value_corruption` exercises the corruption path. task10 implements validation breadth.

**Affected Sections:** AC-1; AC-6; Task Breakdown / task10.

**Cross-Reference Updates:** None.

---

### CMT-7: `customized_info` is per-output-token, not per-request

**Original Comment:**
```
CODEX: `customized_info` is not a per-request summary channel today

The existing scheduler path accumulates customized_info per generated token,
then slices req.customized_info by output-token offset before
BatchTokenIDOutput. TokenizerManager blindly assigns meta_info[k] = v[i]. If DS
stats are appended through the existing accumulator, the user sees a list of
per-token dicts, not the single dict AC-3 demands. Either bypass that
accumulator with a request-summary field or define last/aggregate semantics.

[evidence: python/sglang/srt/managers/scheduler_output_processor_mixin.py:163-179; python/sglang/srt/managers/scheduler_output_processor_mixin.py:1246-1252; python/sglang/srt/managers/tokenizer_manager.py:1739-1741]
```

**Changes Made:**
AC-3 distinguishes per-request summary from per-token list semantics, mandates a new `per_request_summary` channel on `BatchTokenIDOutput`. AC-3 positive verifies single-dict shape for N > 1 tokens. AC-3 negative includes "regression if meta_info becomes a list of per-token dicts". AC-6 anchor `test_ds_meta_info_request_summary`. task5 implements.

**Affected Sections:** AC-3; AC-6; Path Boundaries / Cannot use; Task Breakdown / task5; Feasibility Hints (transport sketch); Claude-Codex Deliberation.

**Cross-Reference Updates:** None.

---

### CMT-8: AC-3 ignores production observability for failures

**Original Comment:**
```
CODEX: AC-3 ignores production observability for failures

meta_info on successful requests is not enough. The risky failures are startup
mask rejection, adapter contract rejection, selector exceptions during decode,
and capture-disabled metric emission. metrics.py has healthy-selection metrics
and a dense-fallback counter, but no error counter or error-class labels.
Operators need logs and Prometheus signals that separate bad mask, bad adapter
input, selector runtime error, and rank mismatch.

[evidence: python/sglang/srt/layers/attention/double_sparsity/metrics.py:68-87; python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:469-477; python/sglang/srt/layers/attention/double_sparsity/validator.py:178-240]
```

**Changes Made:**
AC-3 requires a Prometheus counter `sglang_double_sparsity_errors_total{cls=...}` with four labels (`bad_mask`, `bad_adapter_input`, `selector_runtime_error`, `rank_mismatch`) and structured WARNING/ERROR logs. task6 implements.

**Affected Sections:** AC-3; Task Breakdown / task6; Path Boundaries / Upper Bound; Feasibility Hints (error taxonomy sketch).

**Cross-Reference Updates:** None.

---

### CMT-9: AC-6 numeric-count fixation; AC-6.3 is wiring-only mockery

**Original Comment:**
```
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
```

**Changes Made:**
AC-6 preamble rewritten as behaviorally framed expectations; numeric count floor (≥ 94 passed) is dropped. The wiring-only AC-6.3 test is replaced by `test_ds_rebind_idempotence`, which verifies bind correctness AND rebind containment in one assertion suite.

**Affected Sections:** AC-6; Task Breakdown / task12.

**Cross-Reference Updates:** None.

---

### CMT-10: AC-6.2 parametrised test hides per-failure-mode coverage

**Original Comment:**
```
CODEX: One parametrized negative test will hide the adapter contract

Out-of-range page IDs, padding holes, non-ascending rows, length overflow,
dtype mismatch, and device mismatch are different bugs. A single parametrized
test usually asserts "raises something", which lets one broad early guard
satisfy every case. Split them or at least assert exact exception classes,
messages, and the validation layer that must catch each case.

[evidence: python/sglang/srt/layers/attention/double_sparsity/selector.py:90-112; python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:456-462]
```

**Changes Made:**
AC-2 negatives enumerate distinct named exception classes (`DSAdapterPageOutOfRange`, `DSAdapterValidLengthOverflow`, `DSAdapterNonAscending`, `DSAdapterPaddingViolation`, `DSAdapterDtypeMismatch` / `DSAdapterDeviceMismatch` / `DSAdapterBatchMismatch`). Each anchor test asserts exception class AND the validation layer (host pre-capture vs Triton kernel) that catches it. AC-6 negatives include a meta-check that scans for `assertRaises(Exception)` regressions.

**Affected Sections:** AC-2; AC-6; Task Breakdown / task12.

**Cross-Reference Updates:** None.

---

### CMT-11: `skip_topk` applied before DS selector can defend itself

**Original Comment:**
```
CODEX: `skip_topk` is applied before the DS selector can defend itself

The plan says DS must ignore prev_topk_indices and run each step, but current
forward_absorb_prepare decides reuse before calling _select_topk_indices. If
self.skip_topk is true and prev_topk_indices exists, the DS branch is never
reached. AC-6.6 cannot be fixed inside _select_topk_indices; the prepare path
must gate skip_topk on not self.use_double_sparsity, and both alt_stream and
normal branches need coverage.

[evidence: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:227-270; python/sglang/srt/models/deepseek_v2.py:1428-1447]
```

**Changes Made:**
A new task3 fixes the gate in `forward_absorb_prepare` (both alt-stream and normal branches). AC-6 anchor `test_ds_skip_topk_gate_alt_stream_and_normal` covers both branches.

**Affected Sections:** Goal Description (one new code-terms bullet); AC-6; Task Breakdown / task3 and task12; Feasibility Hints (gate-fix sketch).

**Cross-Reference Updates:** Prior `test_ds_bypasses_skip_topk_cache` framing (which implied the test belonged inside `_select_topk_indices`) is replaced.

---

### CMT-12: AC-6.6 / AC-6.7 are scar tissue from a bad design

**Original Comment:**
```
LINUS: AC-6.6 and AC-6.7 are not testing the feature. They're testing that
your `isinstance` dispatch landed on the right arm. They exist because the
DESIGN forces a routing branch.

In a properly designed version where DS and NSA return the same shape:
  - AC-6.7 is vacuous (there's no other codepath to reach)
  - AC-6.6 collapses to a one-line invariant ("selector ran every step"),
    not a probe-instrumented call-counter test
```

**Changes Made:**
AC-6.7 (isinstance probe) is dropped — the unified-shape redesign makes it vacuous. AC-6.6 spirit is preserved as the skip-topk gate anchor (location moved to `forward_absorb_prepare` per CMT-11). AC-2 keeps a positive test that verifies "no `isinstance` matches in `forward_absorb_*`" via static grep.

**Affected Sections:** AC-2 (static grep added); AC-6 (AC-6.7 removed; AC-6.6 spirit retained).

**Cross-Reference Updates:** Original task5's "AC-6.1–6.4, 6.6, 6.7" list is replaced by task12's behaviorally framed anchors; references to AC-6.7 are removed entirely.

---

### CMT-13: Information leakage in return type

**Original Comment:**
```
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
```

**Changes Made:**
Upper Bound bullet for "typed DS selection result" is replaced by "Unified `page_table_1` Tensor return … no `isinstance` dispatch in `forward_absorb_*`". Cannot use adds "A typed-union return … that encodes the selector's identity."

**Affected Sections:** Path Boundaries / Upper Bound and Cannot use; AC-2; Goal Description; Feasibility Hints.

**Cross-Reference Updates:** None.

---

### CMT-14: Skip-during-capture is a new fallback

**Original Comment:**
```
LINUS: A check that "only runs sometimes" is, structurally, a fallback for
the case when it cannot run. You removed the `SGLANG_DS_ALLOW_*` env-gate
fallbacks (good), and now you're inventing a new one — assertions that
silently disable themselves during CUDA graph capture.

Either the contract holds always … or it holds never.
```

**Changes Made:**
The Upper Bound bullet for contract assertions is rewritten: pre-capture host checks for shape/dtype/device; Triton device-side `tl.device_assert` for value-domain checks. No host-side assertion that silently no-ops during capture. Cannot use adds the explicit prohibition. task7 implements.

**Affected Sections:** Path Boundaries / Upper Bound and Cannot use; Task Breakdown / task7; Claude-Codex Deliberation (new contract-assertion topic).

**Cross-Reference Updates:** None.

---

### CMT-15: Hot path still allocates under graph boundary

**Original Comment:**
```
CODEX: The hot path still allocates under the planned graph boundary

The plan talks about skipping host-sync checks during capture, but capture
safety is also about allocation. … A DS page-table adapter that returns a
fresh page_table_1 per step repeats the same problem. The ABI should accept
graph-owned output buffers or reuse NSA metadata buffers.

[evidence: python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:155,282-286,377-419; python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py:157-182]
```

**Changes Made:**
New top-level AC-8: zero per-step allocations on the captured path; adapter writes into pre-allocated `page_table_1_out` and `adapter_scratch` buffers owned by FlashMLA / NSA forward-metadata. AC-8 negative includes a regression probe that detects re-introduction of `torch.empty(...)`. task7 implements buffer ownership.

**Affected Sections:** New AC-8; Path Boundaries / Upper Bound and Cannot use; Task Breakdown / task7 and task12; Claude-Codex Deliberation (new hot-path-allocations topic).

**Cross-Reference Updates:** None.

---

### CMT-16: TP-rank agreement assumed, not tested

**Original Comment:**
```
CODEX: TP-rank agreement is assumed, not tested

Passing a process_group into bind_runtime_data is not the same as proving all
ranks pick the same logical pages. all_reduce_page_scores silently becomes a
no-op when process_group is None or torch.distributed is not initialized, which
is fine for single rank and dangerous for an accidentally unbound TP deployment.
Add a fail-fast startup invariant for TP>1 and a test that rank-local scores
would diverge without the all-reduce.

[evidence: python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:289-310; python/sglang/srt/layers/attention/double_sparsity/selector.py:126-129; python/sglang/srt/layers/attention/double_sparsity/channel_mask.py:304-351]
```

**Changes Made:**
New top-level AC-7: TP-rank invariance with fail-fast startup for TP > 1 + `process_group is None`. AC-7 positive includes a two-rank synthetic test of identical `selected_indices`; AC-7 negative includes both the `DoubleSparsityTPMisconfigured` startup raise and the "all-reduce disabled" divergence detection. task8 implements.

**Affected Sections:** New AC-7; Task Breakdown / task8 and task12; Path Boundaries / Upper Bound; Claude-Codex Deliberation (new TP-rank-agreement topic).

**Cross-Reference Updates:** None.

---

### CMT-17: Pseudocode shows `isinstance` over both arms calling the same backend

**Original Comment:**
```
LINUS: Read your own pseudocode out loud.

  if isinstance(result, DSSelectionResult):
      ...DS path, calls _forward_flashmla_kv...
  else:
      ...NSA path, calls _forward_flashmla_kv...

Both arms call the same backend function. … One path. Zero `isinstance`.
```

**Changes Made:**
Feasibility Hints pseudocode rewritten: `_select_topk_indices` returns a Tensor on both branches; downstream consumer makes one `_forward_flashmla_kv` call with no `isinstance`. Stats side-channel is shown as a `self._publish_ds_stats(forward_batch, stats)` call inside the DS branch, not as a return-value field.

**Affected Sections:** Feasibility Hints / Conceptual Approach.

**Cross-Reference Updates:** None.

---

### CMT-18: Mid-decode selector failures have no containment policy

**Original Comment:**
```
CODEX: Mid-decode selector failures have no containment policy

Once the env-gate fallback is gone, retrieve_topk or adapter failures propagate
from the attention prepare path in the middle of a batch. The plan does not say
whether that aborts one request, the whole batch, or the worker, and it does not
define how partially accumulated customized_info is cleared. Fail-loud is fine,
but the error boundary must be explicit or operators will get an opaque worker
crash for a single corrupt request state.

[evidence: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:241-270; python/sglang/srt/managers/scheduler_output_processor_mixin.py:163-179]
```

**Changes Made:**
New top-level AC-9: typed exception aborts one request only; batch and worker continue; partial summary state for the failed request is cleared; Prometheus counter increments. AC-9 positive uses a 3-request synthetic decode with an injected mid-batch failure. task9 implements.

**Affected Sections:** New AC-9; Task Breakdown / task9 and task12; Path Boundaries / Upper Bound; Claude-Codex Deliberation (new mid-decode-failure-containment topic).

**Cross-Reference Updates:** None.

---

### CMT-19: Deliberation never put unified-shape Design C on the table

**Original Comment:**
```
LINUS: This is a deliberation between Design A (token-level, reuse
pipeline) and Design B (typed result, bypass pipeline). Design C — "DS
branch produces its `page_table_1` by calling the existing transform with
its own selected logical pages, returning the SAME tensor type as the NSA
path" — was never on the table.

You "converged" by picking between two bad options without considering the
obvious third one.
```

**Changes Made:**
Resolved Disagreements "Adapter target path" topic is rewritten. Designs A and B are listed as **rejected**; Resolution names Design C explicitly: unified `page_table_1` Tensor return, one `_forward_flashmla_kv` call, no `isinstance`. Implementation Notes documents the two acceptable internal strategies (generalise transform vs expand-then-transform) under the constraint that only one ships.

**Affected Sections:** Claude-Codex Deliberation; Implementation Notes / "DS-Side `page_table_1` Builder"; Goal Description; AC-2.

**Cross-Reference Updates:** None.

---

### CMT-20: Speculative-Naming Cross-Check is process noise

**Original Comment:**
```
LINUS: You wrote a paragraph in a 335-line plan to formally notarise that
an unrelated naming convention does not apply to this change. Why is this
here? Who is the audience?
…
Plans should fit in your head. This one doesn't.
```

**Changes Made:**
The `### Speculative-Naming Rule Cross-Check` subsection is removed from Implementation Notes. Branch and PR Scope subsection preserved.

**Affected Sections:** Implementation Notes (subsection removed).

**Cross-Reference Updates:** None.

---

## Remaining Decisions

None. All 20 change requests were applied. The one remaining implementation choice — whether to generalise the existing `transform_index_page_table_decode` to accept a sparse selection, or to add an `expand_ds_selection_to_topk_indices` helper that calls the existing transform unchanged — is captured under Implementation Notes / "DS-Side `page_table_1` Builder" as a non-user-facing choice constrained by Path Boundaries (only one strategy ships).

## Refinement Metadata

- **Input Plan:** `/sgl-workspace/sglang/development/loop2/plan.md`
- **Output Plan:** `/sgl-workspace/sglang/development/loop2/refined_plan.md`
- **QA Document:** `/sgl-workspace/sglang/.humanize/plan_qa/plan-qa.md`
- **Total Comments Processed:** 20
  - Questions: 0
  - Change Requests: 20
  - Research Requests: 0
- **Plan Sections Modified:**
  - Goal Description (rewritten)
  - Acceptance Criteria (AC-1 broadened; AC-2 rewritten; AC-3 rewritten; AC-5 preflight added; AC-6 reframed; AC-7 added; AC-8 added; AC-9 added)
  - Path Boundaries (Upper Bound + Cannot use rewritten)
  - Feasibility Hints and Suggestions (Conceptual Approach pseudocode rewritten; new sketches added)
  - Dependencies and Sequence (Milestones expanded from 4 to 7)
  - Task Breakdown (rewritten; 15 tasks)
  - Claude-Codex Deliberation (Adapter target path topic rewritten; new resolved topics for contract-assertion strategy, hot-path allocations, TP-rank agreement, mid-decode containment, hardware preflight, channel-mask validation breadth, adapter contract negatives, rebind idempotence)
  - Pending User Decisions (trimmed)
  - Implementation Notes (Speculative-Naming Cross-Check removed; new "DS-Side `page_table_1` Builder" subsection added)
- **Convergence Status:** `converged`
- **Refinement Date:** 2026-05-20
- **Mode:** `discussion`
