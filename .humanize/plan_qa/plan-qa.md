# Refine Plan QA

## Summary

Processed **16** comment blocks from `development/loop7/plan.md` (8 pensieve/Linus-style review comments + 7 independent Codex critiques + 1 Codex correction), producing the comment-free `development/loop7/refined_plan_v1.md`. **15** were `change_request`s and were **applied** to the plan; **1** (CMT-15) was a positive `ON-RAILS` observation requiring no plan change and is marked **resolved** (its one overstated clause was corrected by CMT-16, which is folded into the CMT-12 disposition-gate fix). No `question` needed an answer beyond the positive note and no `research_request` was present, so the codebase claims the comments rest on (e.g. `config.py` rejecting `max_top_k`, `dequantize_k_cache_paged` returning a compact tensor and allocating internally, `record_selection` being a Prometheus counter) were already verified during the gen-plan and review passes and are reflected directly in the edits. No new pending decisions were introduced; the plan remains **converged**.

The refinement is structural and consistency-preserving: the Task Breakdown grew from 15 to 20 tasks (split the bundled scorer task; added an oracle-sink task, an oracle-off equivalence task, a TP-determinism task, and a Tier-2.A landing-disposition gate), the lifted-budget ABI was renamed off the reserved Twilight field, the oracle artifact schema and force-needle mechanism were pinned, a statistical-materiality rule was added, and the "internally contradictory gate" framing was replaced with an evidence-based supersession.

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1 | change_request | Goal Description | "...'the internally-contradictory strategic gate is corrected'... mis-frames the move..." | applied |
| CMT-2 | change_request | Acceptance Criteria (AC-1) | "...the artifact must emit the stride it used, and task4's baseline must record a stride=1 (dense) reference..." | applied |
| CMT-3 | change_request | Acceptance Criteria (AC-1) | "...'zero hot-path cost'... no task proves it... add an oracle-off equivalence task..." | applied |
| CMT-4 | change_request | Acceptance Criteria (AC-1) | "...Singular 'rank' is sloppy for a multi-token needle... require `needle_worst_rank`, `needle_all_tokens_in_topK`..." | applied |
| CMT-5 | change_request | Acceptance Criteria (AC-1.1) | "...'Forcing the needle into the selected 2048 set'... Force it HOW?... specify post-topK logical-index replacement..." | applied |
| CMT-6 | change_request | Acceptance Criteria (AC-2) | "...the ac12 baseline is 20 trials/length... state the Loop-7 NIAH trial count and a confidence rule..." | applied |
| CMT-7 | change_request | Acceptance Criteria (AC-3) | "...the '≤ 1.0 pp of DSA' anchor was measured at mem 0.6 on TWO nodes... re-anchor MMLU at the Loop-7 op-point..." | applied |
| CMT-8 | change_request | Acceptance Criteria (AC-3) | "...no AC/task pins cross-rank selected-index EQUALITY for the new modes... add parameterized TP multiprocess tests..." | applied |
| CMT-9 | change_request | Acceptance Criteria (AC-4) | "...task12... is the single heaviest, most irreversible item... Mark task12 explicitly as the high-cost / high-risk item..." | applied |
| CMT-10 | change_request | Acceptance Criteria (AC-4) | "...`config.py` explicitly REJECTS `max_top_k`... name the new ABI now — `enable_lifted_budget_decode` + `lifted_budget_top_k`..." | applied |
| CMT-11 | change_request | Feasibility / Relevant References | "...`record_selection` is Prometheus counter plumbing... give the oracle its OWN flag-gated artifact sink..." | applied |
| CMT-12 | change_request | Task Breakdown | "...task14... depends on task8, NOT on task12... make task14 depend on an explicit task12-DISPOSITION record..." | applied |
| CMT-13 | change_request | Task Breakdown | "...task6 bundles three distinct levers... Split into task6a (channel weighting/normalization) and task6b (head-aggregation)..." | applied |
| CMT-14 | change_request | Task Breakdown | "...task8... depends only on task6, not task7... add task7 as a task8 dependency, or add a task8b..." | applied |
| CMT-15 | question | Claude-Codex Deliberation / Convergence | "...Pensieve verdict: the plan is genuinely on its rails for the recall objective..." | resolved |
| CMT-16 | change_request | Claude-Codex Deliberation / Convergence | "...DEC-4... is honored in PROSE but NOT enforced by the task DAG..." | applied |

Disposition values used: "applied", "resolved".

## Answers

### CMT-15: Positive "on-rails" verdict (no action required)

**Original Comment:**
```
CRITIQUE — [ON-RAILS / positive] Pensieve verdict: the plan is genuinely on its rails for the recall objective. Every resolved decision (DEC-1..6) is honored by the task structure; the served-miss vs admission-failure distinction from `ac12_analysis.md` is carried through; the opt-in ABI discipline (never weaken the DSA `dsa_index_topk` assert) is enforced at the "Cannot use" boundary per maxim `preserve-user-visible-behavior-as-a-hard-rule`; and it does not reach into Tier-1, GLM-5.1, multi-node, or learned-artifact scope. The blocker/major items above are sharpening, not redirection.
```

**Answer:**
Acknowledged. This is a positive observation, not a change request, so no plan edit is required from it directly. Its one overstated clause — "every resolved decision is honored by the task structure" — was correctly flagged by CMT-16: DEC-4 (production-ready bar) was honored only in prose, not enforced by the task DAG. That gap is fixed via the CMT-12 disposition gate (`task17` → `task19`), after which the clause becomes true.

**Plan Changes:**
None directly. The substantive correction it triggered is recorded under CMT-16 / CMT-12.

## Research Findings

No `research_request`-type comments were present. The codebase facts the change requests depend on were verified in the preceding gen-plan and review passes and are cited inline in the refined plan's `### Relevant References` (e.g. `config.py` rejecting the Twilight `max_top_k` field; `dequantize_k_cache_paged` returning a compact `[num_tokens,1,dim]` tensor keyed by `page_table_1_flattened` and allocating `torch.empty` internally; `metrics.py::record_selection` being Prometheus counter plumbing; `selector.py`'s `DoubleSparsityTPMisconfigured` / `DoubleSparsityRebindError` fail-fast guards; the all-reduced score tensor produced by `all_reduce_token_scores` before `select_topk_sequence_order`).

## Plan Changes Applied

### CMT-1: Reframe "internally contradictory gate" as an evidence-based supersession

**Original Comment:**
```
CRITIQUE — [DON'T RE-OPEN DECIDED SCOPE / major] "the internally-contradictory strategic gate is corrected" (echoed in DEC-1's Tradeoff and task15) mis-frames the move. `runs/20260530_dsv32_loop6/ds_on_v32_decision.md` did not contradict itself — it explicitly SELECTED Tier-2.A as primary with a stated rationale (the kernel is the only lever that removes the hard cap; the selector is secondary). What M0 actually does is REVERSE that decided ordering on new evidence. That's legitimate science, but name it. "Fixing a contradiction" lets a future reader treat a scope reversal as a janitorial tweak. task15 should read: "M0 oracle evidence supersedes the Tier-2.A-primary ordering; the prior rationale was sound when written." Cite what changed; don't rewrite the record.
```

**Changes Made:**
Replaced all "internally-contradictory gate is corrected" language with supersession framing ("the strategic gate's Tier-2.A-primary ordering is superseded by a decision record citing the M0 evidence; the prior rationale was sound when written").

**Affected Sections:**
- Goal Description: added the supersession sentence.
- Path Boundaries (Upper + Lower Bound): "supersedes the strategic gate's Tier-2.A-primary ordering".
- Dependencies and Sequence (M4 Step 2): "supersedes ... with the M0 evidence (citing what changed)".
- Pending User Decisions (DEC-1): Tradeoff + Decision Status reworded from "internally contradictory" to "selected Tier-2.A as primary; M0 evidence may supersede that ordering".
- Task Breakdown (`task20`): "Write the decision record that supersedes the strategic gate's Tier-2.A-primary ordering".

**Cross-Reference Updates:** `task15` → `task20` (renumbered); DEC-1 now references `task20`.

### CMT-2: Oracle must emit its sampling stride; baseline records a stride=1 reference

**Original Comment:**
```
CRITIQUE — [EVIDENCE / major] The oracle samples scores "per layer/decode-step at a configurable stride" (AC-1 positive test), but nothing pins which aggregation is authoritative for the recall@K pass/fail, and a coarse stride can make per-layer scores look rosier than the score that actually drives decode-time selection — a misleading upper bound on what the selector sees. Per maxim `prefer-pragmatic-solutions-over-theoretical-completeness` ("expose invalid inputs early instead of hiding them"): the artifact must emit the stride it used, and task4's baseline must record a stride=1 (dense) reference next to the default so any stride-induced optimism is visible, not buried in a default knob.
```

**Changes Made:** AC-1 positive test now requires the **sampling stride** as an explicit emitted artifact field and pins the authoritative tensor (the live all-reduced score tensor). The baseline (now `task6`) records a stride=1 (dense) reference next to the default stride.

**Affected Sections:** Acceptance Criteria (AC-1); Dependencies (M0 Phase A/B); Task Breakdown (`task6`); Feasibility (pseudocode `sink.record(stride, ...)`).

### CMT-3: Add an oracle-off equivalence task to prove "zero hot-path cost"

**Original Comment:**
```
CRITIQUE (Codex) — [EVIDENCE / major] ANCHOR: "zero hot-path cost" (AC-1 positive test / task1). That's a test result, not a property, and no task proves it. An oracle flag can still add a branch, allocation, or host-sync even when "off"; the graph-safe selector relies on caller-owned scratch and skips metrics during capture, so "off" must be demonstrated, not asserted. FIX: add an oracle-off equivalence task that diffs baseline vs oracle-disabled `selected_indices`/`valid_lengths` byte-for-byte and runs the existing CUDA allocation detector under graph replay.
```

**Changes Made:** AC-1 positive/negative tests now require the oracle-off claim to be *demonstrated*; added **`task4`** (oracle-off byte-equivalence of `selected_indices`/`valid_lengths` + CUDA allocation detector under graph replay). A "zero hot-path cost" claim without this evidence is an AC-1 negative (rejected).

**Affected Sections:** Acceptance Criteria (AC-1); Task Breakdown (`task4`); Feasibility (pseudocode comment).

### CMT-4: Pin the multi-token needle-rank schema on the live all-reduced tensor

**Original Comment:**
```
CRITIQUE (Codex) — [DURABLE EVIDENCE / major] ANCHOR: "the needle's rank in the all-reduced DS token scores" (AC-1). Singular "rank" is sloppy for a multi-token needle; the worst-rank / all-tokens-in-top-K rule lives only in the Resolved-Disagreements prose, not in AC-1 or task2. The authoritative tensor is the one AFTER `all_reduce_token_scores` and BEFORE `select_topk_sequence_order`. FIX: make the artifact schema require `needle_worst_rank`, `needle_all_tokens_in_topK`, and the invariant `recall@2048 == selected_contains_needle`, all computed from that exact live all-reduced score tensor.
```

**Changes Made:** AC-1 now requires `needle_worst_rank`, `needle_all_tokens_in_topK`, and the invariant `recall@2048 == selected_contains_needle`, all computed on the tensor produced after `all_reduce_token_scores` and before `select_topk_sequence_order`.

**Affected Sections:** Acceptance Criteria (AC-1); Task Breakdown (`task1`, `task3`); Feasibility (References note `all_reduce_token_scores`); Claude-Codex Deliberation (Resolved Disagreements: M0 framing).

### CMT-5: Specify the AC-1.1 force-needle mechanism (post-topK replacement)

**Original Comment:**
```
CRITIQUE (Codex) — [MINIMUM LEVER / major] ANCHOR: "Forcing the needle into the selected 2048 set" (AC-1.1). Force it HOW? Boosting scores or rewriting labels corrupts the very scorer you're diagnosing; appending the needle blows the 2048 budget. FIX: specify post-topK logical-index replacement ONLY — evict the lowest-ranked non-needle selected positions, insert every needle token, preserve exactly 2048 entries, then go through the existing logical→physical path unchanged.
```

**Changes Made:** AC-1.1 now mandates post-topK logical-index replacement only (evict lowest-ranked non-needle, insert every needle token, preserve exactly 2048). Boosting scores / rewriting labels / appending beyond budget is an AC-1.1 negative.

**Affected Sections:** Acceptance Criteria (AC-1.1); Task Breakdown (`task5`); Claude-Codex Deliberation (Resolved Disagreements).

### CMT-6: State trial count + binomial confidence for "material" uplift

**Original Comment:**
```
CRITIQUE (Codex) — [DURABLE EVIDENCE / major] ANCHOR: "16K materially > 5%, 64K > 0%". The ac12 baseline is 20 trials/length — at 16K, one hit IS 5 percentage points. Calling a one- or two-needle move "material" is numerology, not evidence. FIX: state the Loop-7 NIAH trial count and a confidence rule up front (paired fixed prompts with exact binomial intervals, or N large enough that a claimed uplift isn't just one lucky needle).
```

**Changes Made:** AC-2 now requires the NIAH trial count per length and an exact (Clopper–Pearson) binomial confidence rule stated up front; "material" must exceed the baseline's binomial CI. A "material" claim whose delta lies within the baseline CI (e.g. one needle at N=20) is an AC-2 negative.

**Affected Sections:** Acceptance Criteria (AC-2); Task Breakdown (`task6`, `task7` gate judged "beyond the binomial CI"); Feasibility (pseudocode); DEC-2 (materiality note).

### CMT-7: Re-anchor MMLU at the Loop-7 operating point

**Original Comment:**
```
CRITIQUE — [EVIDENCE / minor] The "≤ 1.0 pp of DSA (matching the ac12 gate)" anchor was measured in `runs/20260528_dsv32_mvp/ac12_analysis.md` at mem 0.6 on TWO nodes; Loop 7 runs single-node at mem 0.7. MMLU is short-context and probably mem-insensitive — but don't compare against a stale anchor on faith. task4 should re-anchor MMLU at the Loop-7 op-point (a one-shot spot-check), not carry the ac12 number forward unexamined.
```

**Changes Made:** AC-3 now requires the DSA MMLU anchor to be re-measured at the Loop-7 op-point (single-node, mem 0.7), not carried from ac12. The re-anchor is folded into `task6`.

**Affected Sections:** Acceptance Criteria (AC-3); Task Breakdown (`task6`, `task12`).

### CMT-8: Add TP cross-rank determinism tests for the new modes

**Original Comment:**
```
CRITIQUE (Codex) — [SEQUENCING / major] ANCHOR: "With a variant enabled" (AC-3). Every scorer/anchor variant changes the values entering the TP=8 `all_reduce_token_scores`, and the lifted-budget path changes the selected-index shape — but no AC/task pins cross-rank selected-index EQUALITY for the new modes (existing TP tests prove only the current scorer). Divergent ranks across ranks = silent wrong output. FIX: add parameterized TP multiprocess tests for each scorer/anchor flag and lifted-budget mode, keeping `DoubleSparsityTPMisconfigured` / `DoubleSparsityRebindError` fail-fast coverage in that matrix.
```

**Changes Made:** AC-3 now requires identical `selected_indices` across all TP=8 ranks per variant (and an AC-3 negative for cross-rank divergence). Added **`task11`** (parameterized TP multiprocess determinism tests for each scorer/anchor flag, keeping the fail-fast guard coverage); the lifted-budget mode's TP equality is added to AC-4 and `task15`.

**Affected Sections:** Acceptance Criteria (AC-3, AC-4); Task Breakdown (`task11`, `task15`); Claude-Codex Deliberation (Resolved Disagreements: TP correctness); References (`selector.py` guards).

### CMT-9: Mark the Tier-2.A hardening task high-cost / high-risk and gated

**Original Comment:**
```
CRITIQUE — [COST HONESTY / major] task12 (alloc-free dequant `out=` variant + CUDA-graph capture + perf validation) is the single heaviest, most irreversible item in this loop — `ds_on_v32_decision.md` itself calls the kernel work "heavy" — yet the task table weights it like task8 (a benchmark run) and the plan never states its true magnitude. (The no-time-estimate rule blocks a literal day count, so state it in risk terms.) Mark task12 explicitly as the high-cost / high-risk item, hard-gated behind the M0 oracle-uplift result, so nobody starts kernel hardening before the data earns it. Pairs with the SEQUENCING blocker on the task graph below.
```

**Changes Made:** The hardening task (now **`task16`**) is explicitly labeled "HIGH-COST / HIGH-RISK (hard-gated behind the `task7` oracle-uplift result, `ds_on_v32_decision.md` calls this 'heavy')". M2 Step 3 mirrors the label.

**Affected Sections:** Task Breakdown (`task16`); Dependencies (M2 Step 3); Acceptance Criteria (AC-4 oracle-uplift gate).

### CMT-10: Name a distinct lifted-budget ABI; do not reuse the reserved `max_top_k`

**Original Comment:**
```
CRITIQUE (Codex) — [DON'T RE-OPEN DECIDED SCOPE / major] ANCHOR: "fixed configured `max_top_k` with padding" (AC-4). `config.py` explicitly REJECTS `max_top_k` as a reserved Twilight field, and the only existing escape is `SGLANG_DS_ALLOW_TOPK_MISMATCH` — both of which this plan says not to touch. The wording invites exactly that collision. FIX: name the new ABI now — e.g. `enable_lifted_budget_decode` + `lifted_budget_top_k` — with explicit validators that reject `top_k > index_topk` unless the opt-in backend path is selected; do not reuse `max_top_k`.
```

**Changes Made:** Replaced every "fixed configured `max_top_k`" with the new ABI fields `enable_lifted_budget_decode` (bool) + `lifted_budget_top_k` (int). AC-4 adds positive tests (these fields select the path; validator rejects `top_k > index_topk` unless `enable_lifted_budget_decode`) and a negative test (reusing `max_top_k`/Twilight/`SGLANG_DS_ALLOW_TOPK_MISMATCH` is rejected). Added **`task13`** to design the ABI.

**Affected Sections:** Acceptance Criteria (AC-4); Path Boundaries (Upper/Lower Bound, Allowed Choices "Can/Cannot use", Note on Determinism); Feasibility (pseudocode + `config.py` reference); Task Breakdown (`task13`, `task14`); Implementation Notes (descriptive-names example); Claude-Codex Deliberation.

### CMT-11: Give the oracle a dedicated artifact sink; keep `record_selection` as counters

**Original Comment:**
```
CRITIQUE (Codex) — [CLEAN SEPARATION / major] ANCHOR: "`metrics.py::record_selection` (telemetry home for the oracle)". `record_selection` is Prometheus counter plumbing for selected/valid token counts — it has no request/trial/needle-span/layer/step schema. Routing oracle ranks and percentiles through it either pollutes production metrics or plants a capture-time host-sync trap. FIX: give the oracle its OWN flag-gated artifact sink keyed by request/trial/layer/step, and leave `metrics.py` with at most a cheap "oracle enabled" counter.
```

**Changes Made:** AC-1 now mandates a dedicated flag-gated oracle artifact sink keyed by request/trial/layer/step, with `metrics.py` holding at most a cheap "oracle enabled" counter; routing per-trial oracle data through `record_selection` is an AC-1 negative. Added **`task2`** for the sink. Updated the `metrics.py` reference and Allowed Choices accordingly.

**Affected Sections:** Acceptance Criteria (AC-1); Path Boundaries (Allowed Choices); Feasibility (References + pseudocode `sink`); Task Breakdown (`task2`); Claude-Codex Deliberation.

### CMT-12 + CMT-16: Gate the loop close on a Tier-2.A landing disposition (enforce DEC-4)

**Original Comments:**
```
CRITIQUE — [SEQUENCING / blocker] task14 (consolidation → "Loop 7 closes") depends on task8, NOT on task12 (Tier-2.A production hardening). With DEC-6 allowing a slow research path first, this graph lets the loop formally close while the most expensive, irreversible piece is still dangling — which directly violates DEC-4 ("production-ready bar for landed code"). AC-4's "before Loop 7 closes" is prose the task graph does not enforce. Fix: make task14 depend on an explicit task12-DISPOSITION record. Its output MAY be "Tier-2.A hardening deferred to a follow-on, evidence recorded" — but the close gates on the disposition EXISTING, not on hope.
```
```
CRITIQUE (Codex) — [ON-RAILS / correction] The note above overstates one clause: "every resolved decision is honored by the task structure" is not quite true — DEC-4 (production-ready bar for landed code) is honored in PROSE but NOT enforced by the task DAG (see the SEQUENCING blocker: task14 can close after task8 with task12 unresolved). Otherwise concur the plan is on-rails.
```

**Changes Made:** Added **`task17`** (Tier-2.A landing disposition record: production-ready landing achieved OR hardening explicitly deferred-with-evidence). Consolidation (**`task19`**) now depends on `task12`, `task17`, and `task18`, so the loop close gates on the disposition existing. The Dependencies prose states this enforces DEC-4; DEC-4's Decision Status references the gate.

**Affected Sections:** Task Breakdown (`task17`, `task19`); Dependencies and Sequence (prose + M2 Step 3 / M4); Pending User Decisions (DEC-4); Claude-Codex Deliberation (Resolved Disagreements: DEC-4 enforcement).

**Cross-Reference Updates:** old `task14` (consolidation) → `task19`; old `task15` (decision record) → `task20`.

### CMT-13: Split the bundled Tier-2.B scorer task for attribution

**Original Comment:**
```
CRITIQUE — [MINIMUM LEVER / major] task6 bundles three distinct levers — channel weighting, channel normalization, head-aggregation — into one coding task. They interact; bundled, a recall move (up OR down) is unattributable, which guts the entire premise of a measure-first loop: knowing WHICH knob moved the number. Per maxim `reduce-complexity-before-adding-branches`. Split into task6a (channel weighting/normalization) and task6b (head-aggregation), each independently flag-gated — the same discipline task7 (anchor-budget) already follows.
```

**Changes Made:** Split the old `task6` into **`task8`** (channel weighting/normalization) and **`task9`** (head-aggregation), each independently flag-gated; the anchor-budget ablation is **`task10`**. AC-3 now requires each variant to be independently flag-gated and byte-identical when off, and `task12` measures each variant's recall delta per-variant (attributable).

**Affected Sections:** Task Breakdown (`task8`, `task9`, `task10`, `task12`); Acceptance Criteria (AC-3); Dependencies (M1 Step 1).

### CMT-14: Make the Tier-2.B measurement depend on all variants

**Original Comment:**
```
CRITIQUE — [CLEAN SEPARATION / minor] task8 (measure Tier-2.B) depends only on task6, not task7 (anchor-budget ablation), so whether task8 covers the anchor variant is undefined — an implicit, unstated merge point in the graph. State the policy: either add task7 as a task8 dependency, or add a task8b for the anchor measurement. The ambiguity is the bug, not which way you resolve it.
```

**Changes Made:** Resolved by making the measurement task (**`task12`**) depend on all three variant tasks (`task8`, `task9`, `task10`) and explicitly measure each variant's recall delta per-variant.

**Affected Sections:** Task Breakdown (`task12` Depends On = `task8, task9, task10`).

## Remaining Decisions

None. All six gen-plan decisions (DEC-1..DEC-6) remain `RESOLVED`; this refinement introduced no new pending decision. The 15 change requests had explicit FIX directives and were applied directly; the one positive note (CMT-15) required no decision.

## Refinement Metadata

- **Input Plan:** `development/loop7/plan.md`
- **Output Plan:** `development/loop7/refined_plan_v1.md`
- **QA Document:** `.humanize/plan_qa/plan-qa.md`
- **Total Comments Processed:** 16
  - Questions: 1 (CMT-15, positive note — resolved, no plan change)
  - Change Requests: 15
  - Research Requests: 0
- **Plan Sections Modified:** Goal Description; Acceptance Criteria (AC-1, AC-1.1, AC-2, AC-3, AC-4); Path Boundaries (Upper Bound, Lower Bound, Allowed Choices, Note on Determinism); Feasibility Hints and Suggestions (Conceptual Approach, Relevant References); Dependencies and Sequence (Milestones M0–M4, dependency prose); Task Breakdown (15 → 20 tasks); Claude-Codex Deliberation (Agreements, Resolved Disagreements, Convergence Status); Pending User Decisions (DEC-1, DEC-2, DEC-4); Implementation Notes (Code Style example).
- **Convergence Status:** converged
- **Refinement Date:** 2026-06-01
- **Mode:** discussion
