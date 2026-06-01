# Loop 7 — DeepSeek-V3.2 Double-Sparsity Long-Context Recall R&D (Tier-2 / AC-10)

## Goal Description

Close — or rigorously characterize — the DeepSeek-V3.2 (DSv3.2) FP8 **double-sparsity (DS) long-context recall gap**. Today DS recalls **4K 75% / 16K 5% / 64K 0%** (NIAH) against DSA's **100% at every length using the same 2048 selection budget and the same decode kernel**. Loop 6 landed the Tier-1 engineering spine (footprint → mem-lift → admission → TTFT) and deliberately did **not** touch selection quality; this loop is the deferred, high-priority Tier-2 recall carryover, gated-open by the strategic decision `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`.

The work is **diagnosis-led**. Before committing to any heavy kernel, an instrumented oracle diagnostic measures **why** DS misses the needle — is the gap *budget-limited* (the 2048 cap is the wall) or *scorer-limited* (DS picks the wrong 2048)? The available evidence already leans scorer-limited: at 4K, DS selects ~50% of all tokens yet still misses 25% of needles. The plan therefore **measures first, then leads with the cheap selector improvement (Tier-2.B), and only pursues the heavy adjustable-budget decode kernel (Tier-2.A) if the diagnostic proves a wider budget recovers recall.** The DSA production default and the Loop-6 Tier-1 operating point are never regressed; all new code is opt-in and reversible.

<comment>CRITIQUE — [DON'T RE-OPEN DECIDED SCOPE / major] "the internally-contradictory strategic gate is corrected" (echoed in DEC-1's Tradeoff and task15) mis-frames the move. `runs/20260530_dsv32_loop6/ds_on_v32_decision.md` did not contradict itself — it explicitly SELECTED Tier-2.A as primary with a stated rationale (the kernel is the only lever that removes the hard cap; the selector is secondary). What M0 actually does is REVERSE that decided ordering on new evidence. That's legitimate science, but name it. "Fixing a contradiction" lets a future reader treat a scope reversal as a janitorial tweak. task15 should read: "M0 oracle evidence supersedes the Tier-2.A-primary ordering; the prior rationale was sound when written." Cite what changed; don't rewrite the record.</comment>

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Recall thresholds are directional/stretch per **DEC-2**; the binding floor is a recorded, characterized, non-regressing result.

- AC-1: **Measure-first oracle diagnostic + separated baseline.** A flag-gated debug-oracle mode quantifies why the needle is missed, and the baseline cleanly separates *served-but-missed* from *admission/HTTP failure*.
  - Positive Tests (expected to PASS):
    - With the oracle enabled, each NIAH trial records: the harness-provided needle token span; the needle's rank in the all-reduced DS token scores (sampled per layer/decode-step at a configurable stride); `selected_contains_needle`; and **score-only** recall@K for K ∈ {512, 1024, 2048, 4096, 8192}.
    - With the oracle disabled (default), decode yields byte-identical `selected_indices` and no extra device allocation versus the current path (zero hot-path cost).
    - The baseline report records 4K served recall, 16K served recall, and 64K admission/servability status at the lifted `mem_fraction_static=0.7` op-point, as distinct fields.
  - Negative Tests (expected to FAIL / be rejected):
    - Enabling the oracle with no harness-provided needle span raises a clear error (no silent guessing of needle position).
    - Reporting recall@K for K > 2048 as a *decode* result is rejected — K > 2048 is score-only/oracle until the Tier-2.A opt-in ABI exists.
    - Reporting 64K as "0% recall" when the request was actually rejected at admission is rejected (must be labeled an admission failure, not a served miss).
  - AC-1.1: **Oracle dense-within-window diagnostic.** Forcing the needle into the selected 2048 set and re-measuring answer recall separates a selector miss from downstream attention/model behavior.
    - Positive: when the needle is force-included within 2048, answer recall recovers (confirming selector-miss attribution).
    - Negative: a recall failure that persists even with the needle force-included is not attributed to the selector (must be flagged as a downstream/model effect).

<comment>CRITIQUE — [EVIDENCE / major] The oracle samples scores "per layer/decode-step at a configurable stride" (AC-1 positive test), but nothing pins which aggregation is authoritative for the recall@K pass/fail, and a coarse stride can make per-layer scores look rosier than the score that actually drives decode-time selection — a misleading upper bound on what the selector sees. Per maxim `prefer-pragmatic-solutions-over-theoretical-completeness` ("expose invalid inputs early instead of hiding them"): the artifact must emit the stride it used, and task4's baseline must record a stride=1 (dense) reference next to the default so any stride-induced optimism is visible, not buried in a default knob.</comment>

<comment>CRITIQUE (Codex) — [EVIDENCE / major] ANCHOR: "zero hot-path cost" (AC-1 positive test / task1). That's a test result, not a property, and no task proves it. An oracle flag can still add a branch, allocation, or host-sync even when "off"; the graph-safe selector relies on caller-owned scratch and skips metrics during capture, so "off" must be demonstrated, not asserted. FIX: add an oracle-off equivalence task that diffs baseline vs oracle-disabled `selected_indices`/`valid_lengths` byte-for-byte and runs the existing CUDA allocation detector under graph replay.</comment>

<comment>CRITIQUE (Codex) — [DURABLE EVIDENCE / major] ANCHOR: "the needle's rank in the all-reduced DS token scores" (AC-1). Singular "rank" is sloppy for a multi-token needle; the worst-rank / all-tokens-in-top-K rule lives only in the Resolved-Disagreements prose, not in AC-1 or task2. The authoritative tensor is the one AFTER `all_reduce_token_scores` and BEFORE `select_topk_sequence_order`. FIX: make the artifact schema require `needle_worst_rank`, `needle_all_tokens_in_topK`, and the invariant `recall@2048 == selected_contains_needle`, all computed from that exact live all-reduced score tensor.</comment>

<comment>CRITIQUE (Codex) — [MINIMUM LEVER / major] ANCHOR: "Forcing the needle into the selected 2048 set" (AC-1.1). Force it HOW? Boosting scores or rewriting labels corrupts the very scorer you're diagnosing; appending the needle blows the 2048 budget. FIX: specify post-topK logical-index replacement ONLY — evict the lowest-ranked non-needle selected positions, insert every needle token, preserve exactly 2048 entries, then go through the existing logical→physical path unchanged.</comment>

- AC-2: **Recall uplift measured (DS-vs-DSA, same node); floor = recorded+characterized, strict target = stretch.**
  - Positive Tests (expected to PASS):
    - The loop produces a recall delta versus the AC-1-measured served baseline for 4K/16K/64K, DS-vs-DSA on the same node, with artifacts.
    - A legitimate negative result (no material uplift, attributed by the oracle to scorer- or budget-limit) still closes the loop, provided it is recorded, characterized, and non-regressing.
  - Negative Tests (expected to FAIL / be rejected):
    - A reported uplift lacking the DSA-same-node comparison artifact is rejected.
    - A recall change that silently regresses MMLU or within-budget (≤ 2048) recall parity is rejected.
    - Claiming the strict stretch target (e.g. 16K materially > 5%, 64K > 0%) as met without the measured artifact crossing that threshold is rejected (but missing the stretch does NOT fail the loop).

<comment>CRITIQUE (Codex) — [DURABLE EVIDENCE / major] ANCHOR: "16K materially > 5%, 64K > 0%". The ac12 baseline is 20 trials/length — at 16K, one hit IS 5 percentage points. Calling a one- or two-needle move "material" is numerology, not evidence. FIX: state the Loop-7 NIAH trial count and a confidence rule up front (paired fixed prompts with exact binomial intervals, or N large enough that a claimed uplift isn't just one lucky needle).</comment>

- AC-3: **Tier-2.B selector uplift — non-learned, flag-gated, non-regression.** (Per DEC-5, learned/distilled scoring is excluded here and deferred behind explicit approval.)
  - Positive Tests (expected to PASS):
    - One or more non-learned scorer variants — channel weighting/normalization, head-aggregation changes in `compute_token_scores`, or deterministic anchor-budget (recency/global/strided) ablations — are flag-gated; the offline channel-mask scorer remains the default and is byte-identical when the flag is off.
    - With a variant enabled: NIAH non-regression at ≤ 2048 (within-budget recall parity preserved), MMLU within ≤ 1.0 pp of DSA (matching the ac12 gate), and dense-DS (≤ 2048) recall stays 100%.
  - Negative Tests (expected to FAIL / be rejected):
    - A variant that alters default (flag-off) selection output is rejected.
    - A variant requiring learned/distilled artifacts is rejected from AC-3 (belongs to the DEC-5 follow-on).
    - A variant that improves 16K NIAH but regresses MMLU or within-budget parity beyond tolerance is rejected.

<comment>CRITIQUE — [EVIDENCE / minor] The "≤ 1.0 pp of DSA (matching the ac12 gate)" anchor was measured in `runs/20260528_dsv32_mvp/ac12_analysis.md` at mem 0.6 on TWO nodes; Loop 7 runs single-node at mem 0.7. MMLU is short-context and probably mem-insensitive — but don't compare against a stale anchor on faith. task4 should re-anchor MMLU at the Loop-7 op-point (a one-shot spot-check), not carry the ac12 number forward unexamined.</comment>

<comment>CRITIQUE (Codex) — [SEQUENCING / major] ANCHOR: "With a variant enabled" (AC-3). Every scorer/anchor variant changes the values entering the TP=8 `all_reduce_token_scores`, and the lifted-budget path changes the selected-index shape — but no AC/task pins cross-rank selected-index EQUALITY for the new modes (existing TP tests prove only the current scorer). Divergent ranks across ranks = silent wrong output. FIX: add parameterized TP multiprocess tests for each scorer/anchor flag and lifted-budget mode, keeping `DoubleSparsityTPMisconfigured` / `DoubleSparsityRebindError` fail-fast coverage in that matrix.</comment>

- AC-4: **Tier-2.A opt-in adjustable-budget decode — pursued only if AC-1 oracle uplift justifies it.**
  - Positive Tests (expected to PASS):
    - Tier-2.A is pursued only when AC-1 score-only recall@4096/8192 shows **material recoverable uplift over recall@2048** (the oracle-uplift gate); the gate evidence is recorded.
    - A new opt-in config/validator/backend flag selects the lifted-budget decode path; the default DSA `flashmla_kv` path and its `indices.shape[-1] == dsa_index_topk` assert are untouched; fp16/DSA defaults are unchanged.
    - `top_k > 2048` without the opt-in path raises a fail-fast config/validator error (NOT via `SGLANG_DS_ALLOW_TOPK_MISMATCH`).
    - The opt-in path remaps **physical selected slots → `page_table_1_flattened` → compact dequantized-KV indices** for `flash_mla_sparse_fwd`; pad/`-1` entries are masked or safe-replaced *before* dequant/index; the budget is a **fixed configured `max_top_k` with padding** (no dynamic tensor shapes); the R23 deterministic tie-break is preserved.
    - Kernel correctness: output matches a reference sparse-attention implementation on small deterministic fp8-KV / dequant cases within a defined tolerance.
    - A prefix-sharing remap test (the same physical slot appearing multiple times in `page_table_1_flattened`) proves request-local compact mapping.
    - Per **DEC-4 + DEC-6**: a slower eager research path is acceptable to first prove recall, but a **landed** path is production-ready — CUDA-graph-safe via an `out=`/scratch `dequantize_k_cache_paged` variant + q-padding scratch, zero-alloc under replay, perf-validated — before Loop 7 closes; otherwise the recall evidence is recorded and production-hardening is explicitly carried to a follow-on with the DSA default untouched.
  - Negative Tests (expected to FAIL / be rejected):
    - Weakening the default DSA path's `dsa_index_topk` assert (instead of adding a separate opt-in path) is rejected.
    - Passing physical indices into the compact dequantized KV domain is rejected.
    - Claiming CUDA-graph / zero-alloc safety while `dequantize_k_cache_paged` still allocates `torch.empty` internally is rejected.
    - Feeding `-1` / pad indices into dequant is rejected.
    - Pursuing Tier-2.A when the AC-1 oracle uplift is ~0 is rejected (no evidence → do not build the kernel).

<comment>CRITIQUE — [COST HONESTY / major] task12 (alloc-free dequant `out=` variant + CUDA-graph capture + perf validation) is the single heaviest, most irreversible item in this loop — `ds_on_v32_decision.md` itself calls the kernel work "heavy" — yet the task table weights it like task8 (a benchmark run) and the plan never states its true magnitude. (The no-time-estimate rule blocks a literal day count, so state it in risk terms.) Mark task12 explicitly as the high-cost / high-risk item, hard-gated behind the M0 oracle-uplift result, so nobody starts kernel hardening before the data earns it. Pairs with the SEQUENCING blocker on the task graph below.</comment>

<comment>CRITIQUE (Codex) — [DON'T RE-OPEN DECIDED SCOPE / major] ANCHOR: "fixed configured `max_top_k` with padding" (AC-4). `config.py` explicitly REJECTS `max_top_k` as a reserved Twilight field, and the only existing escape is `SGLANG_DS_ALLOW_TOPK_MISMATCH` — both of which this plan says not to touch. The wording invites exactly that collision. FIX: name the new ABI now — e.g. `enable_lifted_budget_decode` + `lifted_budget_top_k` — with explicit validators that reject `top_k > index_topk` unless the opt-in backend path is selected; do not reuse `max_top_k`.</comment>

- AC-5: **Tier-2.C servability — separated from recall, 64K-first.** (Per DEC-3, 128k is its own loop.)
  - Positive Tests (expected to PASS):
    - 64K servability is re-confirmed unambiguous at the lifted `mem_fraction_static=0.7` op-point, with served vs admission-status reported as distinct fields.
    - 128k is scoped to its own loop; any 128k probe run here is reported strictly as a servability (not recall) result, with the new ceiling documented.
  - Negative Tests (expected to FAIL / be rejected):
    - Conflating a 128k number with the recall ACs is rejected (servability ≠ recall).
    - Treating full 128k servability engineering as a Loop-7 blocking gate is rejected.

- AC-6: **No Tier-1 regression + perf guardrails.**
  - Positive Tests (expected to PASS):
    - The Loop-6 admission/TTFT spine and the directional AC-5 conc-16 result (P99 TTFT 13.13 s < 22 s at the full-context Option-B point) still hold at the chosen op-point.
    - DSA and fp16 non-DS defaults are behavior-unchanged.
    - Perf guardrails (TTFT, decode TPS/req, GPU memory, graph-replay success, admission) at conc-1/16 are recorded; any landed path stays within the agreed budget.
  - Negative Tests (expected to FAIL / be rejected):
    - Any change that regresses the DS int8 / `mem_fraction_static=0.7` / radix-on / TP=8 operating point is rejected.
    - A recall win that violates the perf/memory guardrail budget for a **landed** path is rejected (a slower **research** path is allowed per DEC-6, but is not the landed deliverable per DEC-4).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation lands: the AC-1 flag-gated oracle diagnostic plus a cleanly separated baseline; at least one production-ready, flag-gated **Tier-2.B** non-learned scorer/anchor improvement with measured recall uplift and full non-regression (MMLU, within-budget parity, dense-DS); and — only if the oracle-uplift gate is met — a **production-ready Tier-2.A** opt-in adjustable-budget decode path (CUDA-graph-safe via an `out=`/scratch dequant variant, explicit physical→compact index remap, fail-fast opt-in ABI, padding safety, R23 tie-break, kernel-correctness tests) with the DSA default untouched. 64K servability is re-confirmed at mem 0.7, and the internally-contradictory strategic gate is corrected by a decision record. Learned/distilled scoring is documented as an approved-but-deferred follow-on.

### Lower Bound (Minimum Acceptable Scope)
The implementation lands the AC-1 oracle diagnostic and a separated, characterized baseline that **empirically attributes** the recall gap (scorer-limited vs budget-limited), DS-vs-DSA on the same node; at least one non-learned Tier-2.B ablation measured with non-regression; and the A-vs-B decision recorded with evidence and the strategic gate corrected. A legitimate negative result (no material uplift) is an acceptable close, provided it is recorded, characterized, and non-regressing. Tier-2.A is **not required** when the oracle-uplift gate is not met; 128k servability is out (its own loop).

### Allowed Choices
- Can use: flag-gated, opt-in, reversible code paths; the existing NIAH pytest harness (`test/manual/test_double_sparsity_v32.py`) plus `development/serve_double_sparsity.sh` / `development/benchmark.sh` (no new serve/bench scaffolding); existing telemetry (`metrics.py::record_selection`) and the unit probe (`channel_mask.py::startup_sanity_probe`); `flash_mla_sparse_fwd` + `dequantize_k_cache_paged` as the Tier-2.A building block; an eager research decode path as an intermediate falsification step (DEC-6); a new explicit opt-in ABI for a lifted decode budget.
- Cannot use: weakening the default DSA `flashmla_kv` `dsa_index_topk` assert; `SGLANG_DS_ALLOW_TOPK_MISMATCH` as the lifted-budget mechanism; the Twilight config fields (`selection_mode` / `top_p` / `min_top_k` / `max_top_k`) without an explicit new ABI; learned/distilled artifacts without DEC-5 follow-on approval; dynamic `top_k` tensor shapes (must be fixed `max_top_k` + padding); new serve/bench scaffolding; any change that regresses the Tier-1 operating point.

> **Note on Determinism**: This draft fixes several non-negotiables (DSA default untouched, opt-in ABI rather than assert-weakening, fixed `max_top_k`+padding rather than dynamic shapes, no new serve/bench scaffolding). Where the draft is deterministic the boundaries above are narrow by design; the genuine choice space is the *order* and *depth* of the A-vs-B investigation, resolved by AC-1 evidence and the recorded decisions.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

```
# Stage 0 — MEASURE FIRST (the pivot; decides everything downstream)
oracle = flag_gated_debug_mode()                     # off by default, zero hot-path cost
for trial in NIAH(lengths=[4K,16K,64K]):
    needle_span = harness.needle_token_positions(trial)   # harness provides; never guessed
    scores      = all_reduced DS token scores (sampled per layer/decode-step)
    record(needle_rank = worst rank over needle_span,      # pass/fail = ALL needle tokens in top-K
           selected_contains_needle,
           recall_at_K = {K: needle_span ⊆ topK(scores)    # SCORE-ONLY for K in 512..8192
                          for K in [512,1024,2048,4096,8192]},
           summary = {needle_score, kth_threshold, margin, percentiles, valid_token_count})
baseline = {4K: served_recall, 16K: served_recall, 64K: admission_status_or_served}

# Decision gate (DEC-1)
if recall_at_4096/8192 materially > recall_at_2048:   budget-limited  -> Tier-2.A is justified
else:                                                  scorer-limited  -> Tier-2.B is the lever

# Stage 1 — Tier-2.B (lead): non-learned scorer/anchor ablations within the locked 2048 ABI
#   touch compute_token_scores / select_topk_sequence_order only; default = offline channel-mask

# Stage 2 — Tier-2.A (only if budget-limited): opt-in lifted-budget decode
#   physical_slots -> page_table_1_flattened -> dequantize_k_cache_paged(out=scratch)
#                  -> COMPACT KV  -> compact per-request indices -> flash_mla_sparse_fwd
#   fixed max_top_k(4096/8192)+padding; mask -1 before dequant; DSA flashmla_kv path untouched
```

### Relevant References
- `python/sglang/srt/layers/attention/dsa_backend.py` — `_forward_flashmla_kv` (the `indices.shape[-1] == dsa_index_topk` cap) and `_forward_flashmla_sparse` (`flash_mla_sparse_fwd`, no cap, the Tier-2.A building block).
- `python/sglang/srt/layers/attention/dsa/dequant_k_cache.py` — `dequantize_k_cache_paged` (returns a **compact** `[num_tokens,1,dim]` tensor keyed by `page_table_1_flattened`; allocates `torch.empty` internally → needs an `out=`/scratch variant for graph safety).
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `retrieve_topk_via_labels`, `compute_token_scores`, `select_topk_sequence_order`, `_topk_by_score_then_pos` (R23 deterministic tie-break) — the Tier-2.B surface.
- `python/sglang/srt/layers/attention/double_sparsity/selector.py` / `token_label_write.py` / `channel_mask.py` — selector entrypoint, label write (channel-mask projection), and `startup_sanity_probe` (NIAH-min unit probe).
- `python/sglang/srt/layers/attention/double_sparsity/config.py` — narrow config surface; explicitly rejects the deferred Twilight fields (any lifted-budget ABI must be added explicitly).
- `python/sglang/srt/layers/attention/double_sparsity/metrics.py` — `record_selection` (telemetry home for the oracle).

<comment>CRITIQUE (Codex) — [CLEAN SEPARATION / major] ANCHOR: "`metrics.py::record_selection` (telemetry home for the oracle)". `record_selection` is Prometheus counter plumbing for selected/valid token counts — it has no request/trial/needle-span/layer/step schema. Routing oracle ranks and percentiles through it either pollutes production metrics or plants a capture-time host-sync trap. FIX: give the oracle its OWN flag-gated artifact sink keyed by request/trial/layer/step, and leave `metrics.py` with at most a cheap "oracle enabled" counter.</comment>
- `test/manual/test_double_sparsity_v32.py` — server NIAH harness (emits `ac12_niah_<len>_*.json`); `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — unit tests.
- `development/serve_double_sparsity.sh`, `development/benchmark.sh` — reuse as-is (no new scaffolding).
- Handoffs: `runs/20260530_dsv32_loop6/ds_on_v32_decision.md` (the open gate), `development/past_implementations/study/08-current-system-architecture.md` (as-built), `runs/20260528_dsv32_mvp/ac12_analysis.md` (recall characterization: 4K~50%→75%, 16K~12.5%→5%, 64K unservable@mem0.6), `.humanize/rlcr/2026-05-30_06-27-19/` (Loop-6 process record).

## Dependencies and Sequence

### Milestones
1. **M0 — Measure-first oracle diagnostic & separated baseline** (the pivot):
   - Phase A: add the flag-gated oracle (needle-rank, `selected_contains_needle`, score-only recall@K, compact score summaries) on the selector score path, off by default.
   - Phase B: re-measure DS-vs-DSA NIAH at 4K/16K/64K at mem 0.7, separating served-recall from admission-status; add the dense-within-window oracle (AC-1.1); emit the budget-vs-scorer attribution.
2. **M1 — Tier-2.B non-learned selector uplift** (lead direction per DEC-1):
   - Step 1: implement flag-gated non-learned scorer variants (channel weighting/normalization, head-aggregation) and deterministic anchor ablations; default stays the offline channel-mask.
   - Step 2: measure recall delta + MMLU + within-budget parity + dense-DS non-regression.
3. **M2 — Tier-2.A opt-in adjustable-budget decode** (conditional on M0 oracle-uplift gate):
   - Step 1: add the explicit opt-in lifted-budget ABI (config/validator/backend flag; fail-fast on `top_k>2048` without opt-in).
   - Step 2: build the opt-in decode path (`flash_mla_sparse_fwd` + dequant) with the physical→compact index remap and padding safety; eager research path first (DEC-6).
   - Step 3: if the recall win justifies it, harden to production-ready (alloc-free dequant `out=` variant, CUDA-graph capture, perf validation) before close (DEC-4), else carry hardening to a follow-on with evidence recorded.
4. **M3 — Tier-2.C 64K servability re-confirmation** (separable; 128k deferred to its own loop per DEC-3):
   - Step 1: re-confirm 64K `/generate` servability at mem 0.7, reporting served vs admission distinctly.
5. **M4 — Consolidation**:
   - Step 1: DS-vs-DSA recall report (same node); Tier-1 spine + directional AC-5 non-regression; perf guardrails recorded.
   - Step 2: decision record updating the contradictory strategic gate to the recorded A-vs-B resolution.

Dependencies: M0 gates everything (its evidence sets the A-vs-B order and the Tier-2.A oracle-uplift gate). M1 depends only on M0. M2 depends on M0's gate being met (and is independent of M1, though informed by it). M3 is independent of M1/M2 (servability ≠ recall). M4 depends on M0 + whichever of M1/M2 ran. Learned scoring (DEC-5) is not in this loop's dependency graph.

## Task Breakdown

Each task includes exactly one routing tag (`coding` = implemented by Claude; `analyze` = executed via Codex / `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Add flag-gated oracle diagnostic (needle-rank, `selected_contains_needle`, compact score summary) on the selector score path; off by default, zero hot-path cost | AC-1 | coding | - |
| task2 | Add score-only recall@K (K∈{512,1024,2048,4096,8192}) computation + harness needle-span hook in the NIAH harness; enforce "K>2048 is score-only, not decode" | AC-1 | coding | task1 |
| task3 | Add the dense-within-window oracle (force needle into selected 2048, re-measure answer recall) | AC-1.1 | coding | task1 |
| task4 | Re-measure DS-vs-DSA NIAH 4K/16K/64K at mem 0.7; produce separated baseline (served-recall vs admission-status) | AC-1, AC-2 | coding | task2 |
| task5 | Analyze M0 evidence: budget-limited vs scorer-limited attribution; decide A-vs-B and whether the Tier-2.A oracle-uplift gate is met | AC-2, AC-4 | analyze | task4 |
| task6 | Implement flag-gated non-learned Tier-2.B scorer variants (channel weighting/normalization, head-aggregation) in `compute_token_scores`; default = offline channel-mask | AC-3 | coding | task5 |
| task7 | Implement deterministic anchor-budget ablation (recency/global/strided) within 2048, flag-gated | AC-3 | coding | task5 |
| task8 | Measure Tier-2.B recall delta + MMLU + within-budget parity + dense-DS non-regression | AC-2, AC-3 | coding | task6 |
| task9 | Design the opt-in lifted-budget ABI (config/validator/backend flag; fail-fast on `top_k>2048` without opt-in); review compact-index remap + padding-safety design | AC-4 | analyze | task5 |
| task10 | Implement the opt-in adjustable-budget decode path (`flash_mla_sparse_fwd` + dequant), physical→compact remap, `-1`/pad masking, fixed `max_top_k`+padding, R23 tie-break; eager research path | AC-4 | coding | task9 |
| task11 | Kernel-correctness + safety tests: reference-attention tolerance (fp8/dequant), prefix-sharing remap, invalid padding, duplicate indices, `valid_lengths`, graph-replay allocation at 4096/8192 | AC-4 | coding | task10 |
| task12 | If recall win justifies: alloc-free dequant `out=` variant + q-padding scratch + CUDA-graph capture + perf validation (production-ready landing) | AC-4, AC-6 | coding | task11 |
| task13 | Re-confirm 64K `/generate` servability at mem 0.7 (served vs admission distinct); keep 128k out of scope | AC-5 | coding | task4 |
| task14 | Consolidation: DS-vs-DSA recall report, Tier-1 spine + directional AC-5 non-regression, perf guardrails at conc-1/16 | AC-2, AC-6 | coding | task8 |
| task15 | Write the decision record correcting the strategic gate to the recorded A-vs-B resolution | AC-2 | coding | task14 |

<comment>CRITIQUE — [SEQUENCING / blocker] task14 (consolidation → "Loop 7 closes") depends on task8, NOT on task12 (Tier-2.A production hardening). With DEC-6 allowing a slow research path first, this graph lets the loop formally close while the most expensive, irreversible piece is still dangling — which directly violates DEC-4 ("production-ready bar for landed code"). AC-4's "before Loop 7 closes" is prose the task graph does not enforce. Fix: make task14 depend on an explicit task12-DISPOSITION record. Its output MAY be "Tier-2.A hardening deferred to a follow-on, evidence recorded" — but the close gates on the disposition EXISTING, not on hope.</comment>

<comment>CRITIQUE — [MINIMUM LEVER / major] task6 bundles three distinct levers — channel weighting, channel normalization, head-aggregation — into one coding task. They interact; bundled, a recall move (up OR down) is unattributable, which guts the entire premise of a measure-first loop: knowing WHICH knob moved the number. Per maxim `reduce-complexity-before-adding-branches`. Split into task6a (channel weighting/normalization) and task6b (head-aggregation), each independently flag-gated — the same discipline task7 (anchor-budget) already follows.</comment>

<comment>CRITIQUE — [CLEAN SEPARATION / minor] task8 (measure Tier-2.B) depends only on task6, not task7 (anchor-budget ablation), so whether task8 covers the anchor variant is undefined — an implicit, unstated merge point in the graph. State the policy: either add task7 as a task8 dependency, or add a task8b for the anchor measurement. The ambiguity is the bug, not which way you resolve it.</comment>

## Claude-Codex Deliberation

### Agreements
- The measure-first pivot is correct: needle-rank + score-only recall@K diagnostics are the right way to quantify whether a wider budget has any recoverable upside before any kernel work.
- Leading with Tier-2.B over Tier-2.A is technically justified — selecting ~50% of tokens at 4K yet missing 25% of needles is hard evidence that a larger budget alone is not the first lever.
- Keeping Tier-2.C servability separate from recall is correct ("can serve 128K" and "recalls the needle" are different failure classes).
- Preserving the default `flashmla_kv` DSA path untouched and making the lifted-budget decode strictly opt-in is mandatory.
- A fixed configured `max_top_k` with padding (not dynamic shapes) is the right CUDA-graph direction for Tier-2.A.

### Resolved Disagreements
- **M0 framing**: Codex argued "selector telemetry" is the wrong frame because the selector does not know the needle. Resolved → M0 is a harness/debug-oracle mode: the needle span comes from the NIAH harness; multi-token needle pass/fail uses the **worst rank / all-needle-tokens-in-top-K** rule (min rank is only a best-token summary); recall@K for K>2048 is score-only/offline, never a decode run.
- **"Reproduce 75/5/0"**: Codex flagged the 64K "0%" as an *admission* failure at mem 0.6, not a served miss. Resolved → the baseline separates served-recall from admission-status and re-measures at mem 0.7.
- **A-vs-B gate**: Codex argued needle rank merely in (2048, 8192] is insufficient justification for Tier-2.A. Resolved → the gate is an **oracle-uplift** gate (score-only recall@4096/8192 must show material recoverable uplift over recall@2048).
- **M1 vagueness / "learned" weight**: Codex argued "learned/query-aware scorer" is too heavy and underspecified for one loop. Resolved → M1 splits into concrete non-learned candidates first; learned/distilled scoring is deferred behind DEC-5.
- **M2 index domain**: Codex flagged that `dequantize_k_cache_paged` returns a **compact** tensor, so `flash_mla_sparse_fwd` needs compact-domain indices, not physical slots (verified in code). Resolved → an explicit physical→`page_table_1_flattened`→compact remap step is required, with a prefix-sharing remap test.
- **M2 allocation / graph safety**: Codex flagged the internal `torch.empty` in `dequantize_k_cache_paged` (verified). Resolved → zero-alloc claims require an `out=`/scratch dequant variant + q-padding scratch; otherwise M2 stays an eager research path and that limit is documented.
- **Opt-in ABI**: Codex argued against reusing `SGLANG_DS_ALLOW_TOPK_MISMATCH`; `config.py` already rejects Twilight fields (verified). Resolved → a new explicit config/validator/backend opt-in ABI with fail-fast on `top_k>2048`.
- **Padding safety**: Resolved → `-1`/pad entries are masked or safe-replaced before any dequant/index.

### Convergence Status
- Final Status: `converged` (2 convergence rounds; round 2 returned no `REQUIRED_CHANGES` and no high-impact `DISAGREE`; the two round-2 `OPTIONAL_IMPROVEMENTS` — worst-rank pass/fail rule and the prefix-sharing remap test — are folded into AC-1 and AC-4/task11).

<comment>CRITIQUE — [ON-RAILS / positive] Pensieve verdict: the plan is genuinely on its rails for the recall objective. Every resolved decision (DEC-1..6) is honored by the task structure; the served-miss vs admission-failure distinction from `ac12_analysis.md` is carried through; the opt-in ABI discipline (never weaken the DSA `dsa_index_topk` assert) is enforced at the "Cannot use" boundary per maxim `preserve-user-visible-behavior-as-a-hard-rule`; and it does not reach into Tier-1, GLM-5.1, multi-node, or learned-artifact scope. The blocker/major items above are sharpening, not redirection.</comment>

<comment>CRITIQUE (Codex) — [ON-RAILS / correction] The note above overstates one clause: "every resolved decision is honored by the task structure" is not quite true — DEC-4 (production-ready bar for landed code) is honored in PROSE but NOT enforced by the task DAG (see the SEQUENCING blocker: task14 can close after task8 with task12 unresolved). Otherwise concur the plan is on-rails.</comment>

## Pending User Decisions

All decisions below were resolved in gen-plan discussion mode; none remain `PENDING`.

- DEC-1: **Which direction leads — Tier-2.A vs Tier-2.B?**
  - Claude Position: Measure-first → B → A-if-evidence (evidence and theory-over-pragmatism favor the cheap selector lever first).
  - Codex Position: B-first is better supported by the evidence; the human must decide whether to override the strategic gate that names A as the selected direction.
  - Tradeoff Summary: A has the higher ceiling but the current data suggests it may not move recall; B is cheaper and evidence-aligned; the gate is internally contradictory.
  - Decision Status: **RESOLVED — Measure-first → B → A-if-evidence.** M0 decides; the strategic gate will be corrected (task15).
- DEC-2: **Recall gate hardness.**
  - Claude Position: Floor = recorded+characterized non-regressing result; strict numeric target is a stretch.
  - Codex Position: Needs owner choice; a strict gate risks failing a legitimate negative R&D result.
  - Tradeoff Summary: A strict gate gives a crisp success bar but can mark a valid scorer-limited finding as failure.
  - Decision Status: **RESOLVED — Floor = recorded+characterized; strict target = stretch.**
- DEC-3: **128k servability (Tier-2.C) — this loop or its own?**
  - Claude Position: Separate; re-confirm 64K unambiguously first, defer 128k to its own loop.
  - Codex Position: Keep 128K in a separate mini-loop unless required; first make 64K unambiguous.
  - Tradeoff Summary: Including 128k adds engineering scope that can mask recall attribution.
  - Decision Status: **RESOLVED — Separate; 64K re-confirmed at mem 0.7, 128k deferred to its own loop.**
- DEC-4: **Deliverable bar — production-ready vs measured evidence?**
  - Claude Position: For an R&D loop, opt-in/reversible code with measured evidence is sufficient.
  - Codex Position: Owner must decide whether landed code must be production-grade.
  - Tradeoff Summary: Production-ready is a higher, slower bar but yields landable code; evidence-only is faster but defers hardening.
  - Decision Status: **RESOLVED — Production-ready is the bar for landed code (before close).** Composes with DEC-6: the slower research path is an intermediate falsification step, not the landed deliverable.
- DEC-5: **Learned/distilled selector artifacts in scope?**
  - Claude Position: Non-learned candidates first; learned behind explicit approval (honors theory-over-pragmatism without front-loading training infra).
  - Codex Position: Learned artifacts need explicit approval — not a small scorer swap.
  - Tradeoff Summary: Learned scoring is theoretically strongest but adds calibration data, artifact versioning, and config/schema cost.
  - Decision Status: **RESOLVED — Non-learned first; learned/distilled deferred behind this explicit approval (follow-on).**
- DEC-6: **Slower opt-in research decode acceptable, or must it meet Loop-6 throughput from the start?**
  - Claude Position: A slower eager/dequant research path is acceptable to first prove recall.
  - Codex Position: Owner must decide whether Loop-6 throughput constraints apply even to Tier-2.A experiments.
  - Tradeoff Summary: Allowing a slow research path preserves the cheap falsification step; requiring full perf up front raises Tier-2.A cost substantially.
  - Decision Status: **RESOLVED — Slower research path OK to prove recall first** (with DEC-4: a winning path is hardened to production-ready before it lands/closes).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Phase", "Step", "M0/M1/M2", "DEC-", "Tier-2.A/B/C", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate names instead (e.g. `selection_recall_oracle`, `lifted_budget_decode`, `adjustable_topk`, `channel_weight_scorer`, `anchor_budget`, `compact_index_remap`).

--- Original Design Draft Start ---

# Loop 7 Draft — DS Long-Context Recall R&D (Tier-2 / AC-10), the deferred high-priority work

> Written 2026-05-31, after **Loop 6 closed at its Minimum Acceptable Scope** (`.humanize/rlcr/2026-05-30_06-27-19`).
> Loop 6 landed the full **Tier-1 engineering spine** (TokenLabelTable footprint → mem-lift → admission →
> TTFT) and closed AC-5 **directional** (conc-16 P99 TTFT 13.13 s < 22 at the full-context Option-B point).
> Tier-2 (DS long-context **recall** R&D) was **explicitly deferred to its own loop** per the plan's Lower
> Bound and the owner's Round-24 close. **This is that loop, and it is the high-priority carryover.**
> Feed this through `gen-plan` once the scope below is confirmed.

---

## Objective

Close the **DS long-context recall gap** on DeepSeek-V3.2 FP8. Today DS recalls **4K 75% / 16K 5% / 64K 0%**
(NIAH) vs DSA **100% at every length using the same 2048 budget and the same decode kernel**. The whole point
of this loop is to make DS *competitive on recall*, which Loop 6 deliberately did not touch (it fixed
admission/TTFT, not selection quality).

This is the **AC-10 / Tier-2** work carried out of Loop 6. It is **gated-open**: the strategic gate
`runs/20260530_dsv32_loop6/ds_on_v32_decision.md` (Loop-6 AC-1) already resolved to *pursue* Tier-2 recall
R&D **strictly after** the Tier-1 spine landed. The Tier-1 spine has landed → the gate is open.

---

## Why this loop exists (the recall root cause — established in Loop 6)

DS decode is **sound**: dense DS (seq ≤ 2048) recalls **100%**, and DS MMLU == DSA MMLU (89.00%). So the
recall collapse at length is **not** a decode bug. It is the product of **two compounding limits**:

1. **The selection budget is kernel-locked at `index_topk = 2048`.** The shared `flashmla_kv` **decode**
   kernel asserts `indices.shape[-1] == self.dsa_index_topk` (`python/sglang/srt/layers/attention/dsa_backend.py`,
   `_forward_flashmla_kv`) during CUDA-graph capture; `SGLANG_DS_ALLOW_TOPK_MISMATCH=1` does **not** bypass it.
   So **DS cannot spend more than 2048 tokens of budget on a hard/long prompt without a new decode kernel.**
2. **DS's offline channel-mask selector is inferior to V3.2's *trained* DSA indexer at that same 2048 budget.**
   V3.2's trained indexer reliably places the needle inside its 2048; DS's offline channel-importance
   projection does not at 16K/64K. This is a **selection-quality** gap, not a budget-size gap *per se* — but
   the two interact: a better selector helps within 2048, and a wider budget helps a fixed selector.

(Full as-built state and evidence: [`development/past_implementations/study/08-current-system-architecture.md`](../past_implementations/study/08-current-system-architecture.md),
the recall characterization in `runs/20260528_dsv32_mvp/ac12_analysis.md`, and the strategic gate
`runs/20260530_dsv32_loop6/ds_on_v32_decision.md`.)

---

## Scope — IN (the two R&D directions, from the strategic gate doc)

### Tier-2.A — PRIMARY: adjustable-`top_k` sparse decode kernel
A `flashmla_kv`-style **decode** kernel (mirroring the native NSA/DSA sparse-matmul decode) that exposes an
**adjustable `top_k`** by relaxing the `indices.shape[-1] == dsa_index_topk` hard cap — as a **new, opt-in DS
decode path**, NOT by weakening the assert on the default DSA path. This lets DS spend a **larger** selection
budget (e.g. 4096 / 8192) on long prompts and is the only lever that can lift recall when the bottleneck is
the 2048 cap itself. Heavy: it is a CUDA-graph-safe sparse-attention decode kernel with a fixed-shape ABI of
its own.

### Tier-2.B — SECONDARY: learned / query-aware DS selector
A **query-aware or learned** DS selector that places the needle inside the existing 2048 budget better than
the offline channel-mask projection — **no kernel change required** (stays within the locked ABI), so it is
much cheaper to try first as a recall-uplift probe. Candidate: a lightweight learned scorer, or pulling the
**top-p / nucleus selection (Twilight, roadmap Loop 11)** forward — top-p can spend more of the 2048 budget
on hard prompts adaptively.

### Tier-2.C — secondary engineering scope: 128k servability
Extends Loop-6's 64K servability (`runs/20260530_dsv32_loop6/ac8_servability/ac8_64k_servability.md`) to the
**128k** context the deferred client requirement (roadmap §6 Loop 7) needs — KV-budget / admission to *serve*
128k. Servability is separate from recall; both are needed for the 128k deliverable.

---

## Scope — OUT

- **Re-litigating the Tier-1 spine.** The lifted **DS int8 / `mem_fraction_static`=0.7 / radix-on / TP=8**
  operating point and the directional AC-5 result stand as the Loop-7 baseline; do not regress them.
- **The strict all-concurrency client SLO** (`P99 TTFT < 22s` AND `≥ 30 TPS/req` at *every* conc). That is a
  **separate downstream** concern: DS per-request decode TPS is ≤ DSA structurally, and conc-64 ≥ 30 is
  unattainable even for DSA (29.4) — so it is an operating-point / DSA-side question, not recall R&D. Only in
  scope here if the owner explicitly merges the two.
- **GLM-5.1 / nvfp4 / multi-node / knob-compat** — their own roadmap loops (§6/§9/§10).

---

## Acceptance criteria (draft — `gen-plan` will formalize positive/negative tests)

1. **Recall uplift, measured.** NIAH 4K/16K/64K recall **delta vs the Loop-5/Loop-6 DS baseline 75 / 5 / 0**,
   on real hardware, DS-vs-DSA on the same node. `gen-plan` sets the binding uplift gate (e.g. 16K materially
   > 5%); a recorded+characterized result is the floor (DEC-3-style), a strict recall target is the stretch.
2. **(If Tier-2.A) the new decode kernel is CUDA-graph-safe and opt-in**: bit-exact selection contract
   (the R23 deterministic tie-break carries over), zero-alloc under graph replay, the **default DSA path's
   `dsa_index_topk` assert is untouched**, fp16/DSA default unchanged.
3. **(If Tier-2.B) the new selector is flag-gated** with the offline channel-mask as default; selection
   equivalence is **NIAH non-regression**, not bitwise (selector granularity changed).
4. **(If Tier-2.C) 128k `/generate` serves** (no HTTP 400) at the lifted op-point, or the new ceiling is
   documented.
5. **No Tier-1 regression**: the Loop-6 admission/TTFT spine and the directional AC-5 conc-16 result still
   hold at the chosen op-point.

---

## Hardware / operating point

Same as Loop 6: single node, **8×H200 (TP=8)**, V3.2 FP8, page_size 64, fp8 KV, `flashmla_kv` prefill+decode,
overlap-schedule + piecewise-cuda-graph disabled, radix-on via the config-bound fixture, **DS int8 compact
table at `mem_fraction_static`=0.7**. Reuse the Loop-5/Loop-6 serve/bench scripts (`development/serve_double_sparsity.sh`,
`benchmark.sh`) and the NIAH harness — no new serve/bench scaffolding.

---

## Inputs / handoffs to read first

- **Strategic gate (the open gate):** `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`
- **As-built system state:** `development/past_implementations/study/08-current-system-architecture.md`
- **Recall characterization:** `runs/20260528_dsv32_mvp/ac12_analysis.md` (+ `ac12_results/`)
- **The kernel cap to relax (Tier-2.A):** `indices.shape[-1] == dsa_index_topk` in
  `python/sglang/srt/layers/attention/dsa_backend.py`
- **The selector to improve (Tier-2.B):** the DS selection path in
  `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` +
  `token_label_write.py` (channel-mask projection)
- **Loop-6 process record:** `.humanize/rlcr/2026-05-30_06-27-19/` (goal-tracker, round summaries)

---

## Pending decisions (resolve in `gen-plan` discussion mode)

- **Which direction leads?** Tier-2.B (cheap, no-kernel, try first) vs Tier-2.A (the only lever if 2048 itself
  is the wall). Recommend B-as-probe then A-if-needed, but A is the higher-ceiling, higher-cost path.
- **Recall gate hardness:** strict recall target vs DEC-3-style recorded directional uplift.
- **Does 128k servability (Tier-2.C) belong here or its own loop?** It is engineering, not recall R&D.
- **Theory-over-pragmatism (standing owner preference):** prefer the theoretically correct
  adjustable-budget/learned-selector design over a cheap hack even at higher engineering cost.

--- Original Design Draft End ---
