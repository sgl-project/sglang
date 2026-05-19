# Refine Plan QA

## Summary

Nine comments were extracted from `development/refined_plan_v2.md`. All nine resolved pre-existing `PENDING` user decisions — eight of them as explicit RESOLVED outcomes (DEC-1, DEC-3, DEC-4, DEC-5, DEC-6, DEC-8, DEC-9, DEC-10) and one as a deliberate deferral with a follow-up checkpoint (DEC-2 → revisit after Milestone 1 lands). DEC-7 (Extensions interpretation) received no comment and remains PENDING with Claude's default holding. The refinement applies the resolutions across `## Acceptance Criteria` (AC-1 negative test wording, AC-7 / AC-8 hardware specifics, AC-9 unchanged numerical contract), `## Feasibility Hints and Suggestions` (Hint #5 TP rank synchronization rewritten to "score all-reduce, not signature all-gather"; Future-Work Notes rewritten to drop HiSparse adapter and surface GLM-5.1 as the load-bearing forward-compat target), `## Dependencies and Sequence` (Milestone 0 Phase C and Milestone 5 preamble updated for the new DEC status distribution), and `## Claude-Codex Deliberation` (DEC resolutions appended). Convergence status is `partially_converged` with two PENDING decisions remaining (DEC-2 deferred to M1, DEC-7 low-stakes).

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1 | change_request | Pending User Decisions preamble → DEC-1 | "The hardware is actually 2 nodes of H200 … 16-way cross-node TP or 2x replicas 8-way TP behind SMG …" | applied (DEC-1 RESOLVED) |
| CMT-2 | change_request | DEC-2 | "Let's circle back after Milestone 1 lands." | deferred (DEC-2 stays PENDING with M1 checkpoint) |
| CMT-3 | change_request | DEC-3 | "I think claude's position is good enough … Stay with Claude's position right now." | applied (DEC-3 RESOLVED) |
| CMT-4 | change_request | DEC-4 | "Aggree with claude here, lets provide the calibration script, but the actuall channel mask file will not be tracked by the repo." | applied (DEC-4 RESOLVED) |
| CMT-5 | change_request | DEC-5 | "I agree with Claude and Codex here, we don't to double filter an already sparse set …" | applied (DEC-5 RESOLVED) |
| CMT-6 | change_request | DEC-6 | "I think lets make HiSparse, Hicache, and PD-Disagg explicitly out of initial scope, and lets make HiSparse out of downstream scope as well … Hicache and PD-disagg are such commonly asked for features its good to at least keep it in mind …" | applied (DEC-6 RESOLVED with split: HiSparse out of all scope; HiCache + PD-Disagg deferred-with-mind-share; GLM-5.1 hard-deferred) |
| CMT-7 | change_request | After DEC-7, applies to DEC-8 | "They can co-exist but lets remove all currrent plans to integrate them together right now … make them mutually exclusive for now …" | applied (DEC-8 RESOLVED) |
| CMT-8 | change_request | DEC-9 | "Legacy SGL and Twilight both appear to do local/per-head selection without a TP rank-agreement path … Implementation clarification: do not all-gather page signatures …" | applied (DEC-9 RESOLVED with detailed implementation contract) |
| CMT-9 | change_request | DEC-10 | "Considering that GLM 5.1 is a hard requirement that has just been deferred … we should still at least use a MLA-capability validator …" | applied (DEC-10 RESOLVED) |

## Answers

*No raw `question`-type comments were extracted. All comments were decision-resolutions on PENDING DECs.*

## Research Findings

*No primary `research_request` comments. CMT-8 contains implementation research (legacy-SGL / Twilight precedent on local per-head selection) but presents a concrete decision rather than a research ask; it was applied as a change-request.*

## Plan Changes Applied

### CMT-1 → DEC-1 RESOLVED (hardware spec)

**Original Comment:**
```
The hardware is actually 2 nodes of H200. h200-10-220-51-6: 8 GPU(s), h200-10-220-51-8: 8 GPU(s). We can do cross-node 16-way TP or 2x replicas 8-way TP behind SMG. Client explicitly request H200s so we are sticking to H200s. 30 is referring to per-request TPS, which can also be estimated with 1000/TPOT is TPOT is in ms. Also should also be similar to output tok/s / concurrency.
```

**Changes Made:**
- DEC-1's Decision Status changed from `PENDING` to RESOLVED with full hardware spec (2-node H200, named instances, 8-way TP / 2× SMG default vs 16-way cross-node alternative).
- AC-7 cites the exact hardware ("`h200-10-220-51-6` or `-51-8`; 8-way TP default; optionally 16-way cross-node; or 2× 8-way replicas behind SMG").
- AC-8 cites the hardware and notes 16-way cross-node TP is acceptable but expected slower (DEC-9 all-reduce cost).
- Added per-request TPS consistency check (`1000 / TPOT_ms` ≡ `output_tok_per_sec / concurrency`) to AC-8.

**Affected Sections:** DEC-1, AC-7, AC-8.

### CMT-2 → DEC-2 deferred to M1 checkpoint

**Original Comment:**
```
Let's circle back after Milestone 1 lands.
```

**Changes Made:**
- DEC-2's Decision Status preserved as `PENDING` with annotation: "(revisit after Milestone 1 lands). User explicitly chose to defer until M1's validator scaffolding is in place. Claude default remains in force in the meantime; M3-B's page-stability fixture is the engineering gate. Does not block any pre-M3 work."
- Milestone 0 Phase C updated to reflect that the original CMT-18 execution gate (DEC-1, DEC-2, DEC-3 blocking M5) now reduces to DEC-2 only.
- Milestone 5 preamble updated to acknowledge DEC-2's M1-land checkpoint as the practical resolution point.

**Affected Sections:** DEC-2, Milestone 0 Phase C, Milestone 5 preamble.

### CMT-3 → DEC-3 RESOLVED (quality thresholds)

**Original Comment:**
```
I think claude's position is good enough, but we can choose to loosen it if they are too restrictive. Stay with Claude's position right now.
```

**Changes Made:**
- DEC-3's Decision Status changed to RESOLVED with Claude defaults (NIAH ≤ 5 pp, MMLU ≤ 1.0 pp) and an explicit "loosen via follow-on amendment to AC-9 if M5 finds these too restrictive; do not pre-loosen" clause.
- AC-9 unchanged (Claude defaults already encoded as the contract).

**Affected Sections:** DEC-3.

### CMT-4 → DEC-4 RESOLVED (calibration ownership)

**Original Comment:**
```
Aggree with claude here, lets provide the calibration script, but the actuall channel mask file will not be tracked by the repo.
```

**Changes Made:**
- DEC-4 Decision Status changed to RESOLVED with Claude position confirmed: script in repo, channel mask file external.
- AC-5 production-recipe wording already aligned; no further AC edits required.

**Affected Sections:** DEC-4.

### CMT-5 → DEC-5 RESOLVED (DS replaces NSA selector)

**Original Comment:**
```
I agree with Claude and Codex here, we don't to double filter an already sparse set, we should leave nsa alone and we would ds an alternative.
```

**Changes Made:**
- DEC-5 Decision Status changed to RESOLVED: "DS is an alternative to NSA's selection role; do not stack DS after NSA; leave NSA quant/dequant/cache plumbing untouched."
- AC-2 hook-site spec already encodes this (single branch swap, no stacking); no AC edits required.

**Affected Sections:** DEC-5.

### CMT-6 → DEC-6 RESOLVED with scope split

**Original Comment:**
```
I think lets make HiSparse, Hicache, and PD-Disagg explicitly out of initial scope, and lets make HiSparse out of downstream scope as well to not blow up the milestone budget. If we have to redesign to integrate with Hisparse, lets do it later since Hi-sparse is in constant flux right now. Hicache and PD-disagg are such commonly asked for features its good to at least keep it in mind, but still keep them outside of the initial scope as customer have yet to ask for them.
```

**Changes Made:**
- DEC-6 Decision Status changed to RESOLVED with three-way split:
  - HiSparse integration: OUT of initial AND downstream scope; no future-work bullet promises it.
  - HiCache, PD-Disagg: deferred but "kept in mind" future-work.
  - GLM-5.1: deferred-but-hard requirement; schema + validator must be GLM-5.1-ready.
- Future-Work Notes rewritten: removed HiSparse adapter section entirely; added GLM-5.1 as load-bearing forward-compat target; kept PD-Disagg, HiCache, 128K ISL, FP4 weights, Twilight as future-work-keep-in-mind bullets.
- Lower Bound text in Path Boundaries updated to drop "HiSparse / PD-Disagg integration are deferred" framing.

**Affected Sections:** DEC-6, Future-Work Notes.

### CMT-7 → DEC-8 RESOLVED (mutual exclusion at runtime)

**Original Comment:**
```
They can co-exist but lets remove all currrent plans to integrate them together right now. Let's keep the scope limited. I agree with codex and remove future-work claims about integrating these two together. They can still both exist in the codebase, but I guess make them mutually exclusive for now, as the client as has no expectation of using HiSparse.
```

**Changes Made:**
- DEC-8 Decision Status changed to RESOLVED: Option 1 (startup error when both enabled) + "no plans to integrate" framing.
- AC-1 negative test rewritten: error message is now "Double Sparsity and HiSparse are mutually exclusive; there are no plans to integrate them" (per DEC-8 resolution) instead of "deliberately undesigned for v1; see DEC-8".
- Future-Work Notes (already updated under CMT-6) reinforces "no HiSparse adapter on the roadmap".

**Affected Sections:** DEC-8, AC-1 negative test, Future-Work Notes.

### CMT-8 → DEC-9 RESOLVED with implementation contract

**Original Comment:**
```
Legacy SGL and Twilight both appear to do local/per-head selection without a TP rank-agreement path, but that precedent does not carry over cleanly to this rewrite because the V3.2/FlashMLA path has a shared page-table metadata contract. For the standalone DS rewrite, global rank agreement is the correct v1 decision.

Implementation clarification: do not all-gather page signatures. Keep page signatures TP/head-sharded, compute scalar page scores locally on each rank, all-reduce those scores across the attention TP group, then run deterministic top-K independently on every rank from the same reduced scores. Add a rank-agreement test where per-rank local top-K would diverge but the all-reduced result agrees. Note that the all-reduce is bandwidth-small but latency-sensitive, especially for cross-node 16-way TP; prefer validating first on 8-way H200 TP / 2x replicas behind SMG. If performance is too bad, we can maybe pivot back to per-rank selection and verify that there is no serious perf degradation from doing so.
```

**Changes Made:**
- DEC-9 Decision Status changed to RESOLVED with a numbered implementation contract:
  1. Do NOT all-gather page signatures (they stay TP/head-sharded).
  2. Each rank computes scalar page scores locally from its head shard.
  3. All-reduce SCORES (SUM) across the attention TP group; the tensor is `[max_pages]`-shaped.
  4. Every rank runs deterministic top-K independently from reduced scores.
  5. Rank-agreement test in task16.
  6. Validation order: 8-way intra-node H200 TP first (NVLink only); cross-node 16-way second.
  7. Pivot fallback: if per-step all-reduce dominates latency, revert to per-rank with documented perf-vs-quality benchmark.
- Feasibility Hint #5 "TP rank synchronization" sub-bullet rewritten per the contract above (replacing the earlier "all-reduce of partial head scores" generic language with the score-not-signature distinction).
- Precedent note added to DEC-9 explaining that legacy SGL / Twilight's per-head local approach does not transfer because of FlashMLA's shared page-table metadata contract.

**Affected Sections:** DEC-9, Feasibility Hint #5 TP synchronization sub-bullet.

### CMT-9 → DEC-10 RESOLVED (MLA-capability validator)

**Original Comment:**
```
Considering that GLM 5.1 is a hard requirement that has just been deferred to get a initial working version done as soon as possible, we should still at least use a MLA-capability validator, so use capability-check rather than relying name-string special case
```

**Changes Made:**
- DEC-10 Decision Status changed to RESOLVED: capability-check (presence of `nsa.Indexer` on the attention layer), not a model-name string match.
- Title shortened from "V3.2-scope vs MLA-capability validator" to "MLA-capability validator" to match the resolution.
- The capability-check is documented as the generalization seam for GLM-5.1 (deferred-but-hard under DEC-6).
- task10 description previously specified "V3.2-specific model check per DEC-10 default"; meaning carries through as "capability check on `nsa.Indexer` presence" which works for both V3.2 today and GLM-5.1 future.

**Affected Sections:** DEC-10, task10 (implicit through capability check), Future-Work Notes (GLM-5.1).

### Cross-Reference Updates

- DEC numbering: DEC-1..DEC-10 preserved. Eight RESOLVED, two PENDING.
- AC numbering: AC-1..AC-12 preserved.
- Task numbering: 19 tasks (task1..task17, task19, task20; task18 already dropped in prior round).
- Convergence Status updated to reflect the new RESOLVED/PENDING distribution.
- Original draft appendix preserved verbatim.

## Remaining Decisions

Two PENDING decisions remain.

### DEC-2: Radix cache reconciliation (deferred to M1 land)

**Related Comments:** CMT-2.

**Context:** DS validator radix-cache permission policy. The user chose to defer the final answer until Milestone 1 lands and there's working validator code to anchor the decision on.

**Options:**
1. Default-on; gate on M3-B page-stability fixture passing (Claude default; in force during the deferral).
2. Default-off; require users to opt in.

**Recommendation:** Option 1. Revisit checkpoint: at Milestone-1 land time.

**Status:** PENDING.

### DEC-7: "Extensions as a general knob for the sglang engine" interpretation

**Related Comments:** none directly.

**Context:** The user did not comment on this DEC; Claude default holds.

**Options:**
1. Expose DS runtime knobs through `--double-sparsity-config` JSON; no plugin system (Claude default).
2. Introduce a generic plugin / extension system (separate design effort).

**Recommendation:** Option 1.

**Status:** PENDING (low-stakes; default ships).

## Refinement Metadata

- **Input Plan:** `/sgl-workspace/sglang/development/refined_plan_v2.md`
- **Output Plan:** `/sgl-workspace/sglang/development/refined_plan_v3.md`
- **QA Document:** `/sgl-workspace/sglang/.humanize/plan_qa/refined_plan_v2-qa.md`
- **Total Comments Processed:** 9
  - Questions: 0
  - Change Requests: 9
  - Research Requests: 0
- **Plan Sections Modified:** `## Acceptance Criteria` (AC-1 negative test wording, AC-7 hardware cite, AC-8 hardware + TPS-formula cite), `## Feasibility Hints and Suggestions` (Hint #5 TP rank synchronization sub-bullet, Future-Work Notes rewritten), `## Dependencies and Sequence` (Milestone 0 Phase C, Milestone 5 preamble), `## Claude-Codex Deliberation` (User DEC-resolution round appended), `## Pending User Decisions` (DEC-1, DEC-2, DEC-3, DEC-4, DEC-5, DEC-6, DEC-8, DEC-9, DEC-10 all updated).
- **Convergence Status:** `partially_converged` (DEC-2 deferred to M1 checkpoint; DEC-7 low-stakes default).
- **Refinement Date:** 2026-05-19
- **Mode:** `discussion`
