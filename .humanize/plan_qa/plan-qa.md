# Refine Plan QA

## Summary

Refined `development/loop6/plan.md` → `development/loop6/refined_plan_v1.md` in **discussion** mode. The input carried **12 review comments** added in a prior Linus-style pensieve taste-review (7 critiques) + a Codex agreement pass (which agreed-with-modification on all 7 and added 5 new). All 12 were classified as `change_request` except **CMT-11** (`research_request`, file-existence verification → change). No pure `question` comments. Three comments interpreted prior user decisions or changed scope/cost and were confirmed via `AskUserQuestion`:

- **Lever authority** → *AC-2 authoritative* (skip int8 → structural directly if the budget predicts int8 insufficient).
- **Hardening scope** → *keep AC-7/AC-8 in Loop 6, soft*.
- **int8 quality gate** → *require a real-mask NIAH non-regression*.

All comments were applied; cross-references (ACs, tasks, milestones, DECs, path boundaries) were propagated. No comment markers remain. **Convergence: `converged`** — no PENDING decisions. The `--- Original Design Draft Start ---` appendix is preserved byte-exact except for a required `NON-AUTHORITATIVE` banner (CMT-12).

> Note: this QA path (`.humanize/plan_qa/plan-qa.md`) is derived from the input *basename* (`plan.md`), which collides with the earlier Loop-5 refinement. The prior Loop-5 QA was preserved at `.humanize/plan_qa/plan-qa.loop5.md` before this file was written.

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1 | change_request | Goal Description | "…you call AC-5 'the single done-criterion' — then DEC-3 redefines that criterion as a 'directional trend'… A done-criterion you can pass while the thing still doesn't work is not a done-criterion; it's a euphemism." | applied |
| CMT-2 | change_request | Goal Description | "…why does DS carry a per-token signature table at all on a model that already has a trained indexer?" / Codex: "don't reopen DEC-2; AC-2 should justify the table as the minimum reversible opt-in fix." | applied |
| CMT-3 | change_request | DEC-4 framing | "DEC-4 makes int8-quant… the PRIMARY lever and demotes the page-level/two-stage redesign… backwards taste… Demand that AC-2 prove int8 is *sufficient* before committing." | applied |
| CMT-4 | change_request | AC-2 | "The ladder 'int8 first, escalate…' risks IMPLEMENTING twice… give AC-2 teeth: skip int8 entirely and select the structural lever directly." | applied |
| CMT-5 | change_request | AC-3.1 | "…three metrics and ZERO thresholds… A menu is not an acceptance criterion. Nail it down in AC-2: ONE primary metric, ONE threshold (e.g. top-k overlap@2048 ≥ 0.99)." | applied |
| CMT-6 | change_request | AC-3.1 | "int8 signatures mean you DEQUANTIZE at scoring… margin ≈ 11%… Add an early decode-TPS-non-regression micro-check… before the full hardware sweep." | applied |
| CMT-7 | change_request | AC-3.2 | "AC-3.1 protects int8 with only a SYNTHETIC test, while AC-3.2 requires NIAH non-regression… Require an int8 quality gate on REAL V3.2 / Loop-5-mask data… in ADDITION to the synthetic unit test." | applied |
| CMT-8 | change_request | AC-5 | "A strict P99 SLO cannot be accepted by 'median pass with the worst trial disclosed.' Split the rule: all trials must pass for a hard SLO claim; median… only for directional characterization." | applied |
| CMT-9 | change_request | AC-5 | "AC-5 allows 'attribution unavailable' and still treats the run as directionally useful… Make admission/prefill attribution REQUIRED for directional success; without it… do not call the spine validated." | applied |
| CMT-10 | change_request | AC-9 | "AC-9 is Loop-5 janitorial backlog… in your Lower Bound (required)… Ten ACs / eleven tasks… is scope sprawl… mark it explicitly as a gap-filler… re-ask whether AC-7/AC-8 belong in THIS loop." | applied |
| CMT-11 | research_request | Feasibility Hints | "VERIFIED: development/loop6/probe_64k.json does NOT exist, yet the plan routes AC-8 at it while Scope-OUT forbids new fixture code… point AC-8 at an existing payload, or name the JSON fixture as an explicit AC-8 deliverable." | researched → applied |
| CMT-12 | change_request | Draft appendix | "This file contains TWO executable-looking plans… stale instructions in the same file are… a live mis-execution hazard. Mark every draft section NON-AUTHORITATIVE / archived." | applied |

Disposition values used: `applied`, `researched`.

## Answers

No `question`-type comments were present — every critique was an actionable `change_request` (CMT-11 a research-then-change). A few carried rhetorical question framing (CMT-2 "why does DS carry a table at all?", CMT-10 "what is this doing in this loop?"); these were treated as their dominant change intent and answered through plan edits rather than QA prose, per the dominant-classification rule.

## Research Findings

### CMT-11: Does `development/loop6/probe_64k.json` exist?

**Original Comment:**
```
CRITIQUE (Codex / NEW — VERIFIED): development/loop6/probe_64k.json does NOT exist on
disk, yet the plan routes AC-8 (a hardware acceptance check) at it while Scope-OUT forbids
new fixture/scaffolding code. An acceptance check whose input is undefined cannot run.
Either point AC-8 at an existing 64K payload, or name the JSON fixture as an explicit AC-8
deliverable (a tiny probe payload is a reasonable exemption from the no-new-scaffolding rule).
```

**Research Scope:** `ls development/loop6/probe_64k.json` and a `grep`/`ls` for any `probe*` fixture under `development/loop6/` (tools: `Bash` ls/grep only).

**Findings:** No `probe_64k.json` (or any `probe*` file) exists under `development/loop6/`. The Feasibility Hint and AC-8 referenced a file that was never created, while Scope-OUT bans new fixture/scaffolding code — an internal contradiction that would block AC-8's hardware run.

**Impact on Plan:** AC-8 now states the ~70K-token probe input is **harness-generated or a small explicit AC-8 deliverable**, and Scope-OUT names a minimal 64K probe payload as the single fixture exemption. The Feasibility Hint item 5 was updated to drop the assumed file and note it does not yet exist.

**(CMT-12 corroboration):** confirmed the appendix's "Acceptance criteria (draft)" and "Pending user decisions" restate five `*(PENDING)*` items that contradict the resolved plan (which states none remain PENDING) — supporting the NON-AUTHORITATIVE banner.

## Plan Changes Applied

### CMT-1: Done-criterion framing (directional ≠ shippable)
**Changes Made:** Relabeled directional MVP outcomes as **accepted progress, explicitly not a shippable pass**; reserved "shippable"/"done" for an all-trials strict pass at every conc. DEC-3 (kept directional per the user's prior decision) was not reversed.
**Affected Sections:** Goal Description; Resolved framing (DEC-3 bullet); Acceptance Criteria (AC-5 Grading); Pending User Decisions (DEC-3 status).
**Cross-Reference Updates:** none (no ID changes).

### CMT-2: Justify the table as the minimum reversible opt-in fix (don't reopen DEC-2)
**Changes Made:** Added to AC-2 a requirement to justify any `TokenLabelTable` work as the **minimum reversible DS-opt-in fix** given DSA already wins recall. Did **not** reopen the "should DS exist" question (DEC-2 stands).
**Affected Sections:** Acceptance Criteria (AC-2 body + positive bullet); Task Breakdown (task2).

### CMT-3 + CMT-4: AC-2 authoritative over the lever (no throwaway int8) — **user decision**
**Changes Made:** Made **AC-2's lever selection binding** on AC-3: if the HBM fixed-point predicts int8 insufficient, AC-3 implements the page-level/two-stage lever **directly** (no throwaway int8). int8 is implemented only when predicted sufficient.
**Affected Sections:** Resolved framing (DEC-4 bullet, retitled "AC-2 is authoritative"); AC-2 body + positive bullet; AC-3.1 header; Path Boundaries (Lower Bound, Allowed Choices); Feasibility item 2; Task Breakdown (task2, task3); Pending User Decisions (DEC-4 status).
**Cross-Reference Updates:** AC-3 / task3 reworded from "int8" to "the lever AC-2 selected".

### CMT-5: One primary metric + numeric threshold (verifiable test)
**Changes Made:** AC-2 must name the **primary selection-equivalence metric and numeric fail threshold** (e.g. top-k overlap@2048 ≥ 0.99) before coding; AC-3.1 is held to it; secondary metrics are diagnostics only.
**Affected Sections:** AC-2 body + positive bullet; AC-3.1 positive/negative; Task Breakdown (task2, task4).

### CMT-6: Early decode-TPS microbench (protect the 33.9→30 margin)
**Changes Made:** Added an **early compact-vs-fp16 decode-scoring microbench** with a numeric overhead budget tied to the 33.9→30 TPS/req margin, before the full AC-5 sweep; negative test fails if the compact path fixes TTFT at the cost of the TPS SLO.
**Affected Sections:** AC-3.1 positive/negative; Feasibility item 2; Risk #1; Task Breakdown (task4).

### CMT-7: Real-mask NIAH non-regression for int8 — **user decision**
**Changes Made:** int8 (AC-3.1) now **requires** a real-mask NIAH non-regression (Loop-5 mask, via the AC-Q/`test_double_sparsity_v32.py` harness) in addition to the synthetic test.
**Affected Sections:** AC-3.1 positive/negative; Path Boundaries (Lower Bound); Feasibility item 2; Risk #1; Task Breakdown (task4); Pending User Decisions (new DEC-8).

### CMT-8: Split the trial-aggregation rule
**Changes Made:** A hard/strict SLO pass requires **all trials** to pass; median-with-worst-disclosed is acceptable **only** for the directional characterization.
**Affected Sections:** AC-5 intro.

### CMT-9: Attribution required for the spine to count as validated
**Changes Made:** Admission-wait vs prefill-compute attribution is **required**; if genuinely unavailable, the run is recorded but the spine is **not** called validated (removes the prior "attribution unavailable → still directionally useful" escape that contradicted the Goal's "must separate").
**Affected Sections:** AC-5 positive/negative; Resolved framing (DEC-3); Path Boundaries (Lower Bound); Pending User Decisions (DEC-3 status).

### CMT-10: AC-9 opportunistic; keep AC-7/AC-8 soft — **user decision**
**Changes Made:** AC-9 marked **opportunistic hardening** (code edit may land early, but the gate re-run needs a live server; must not precede/delay the spine); task10 dependency changed `-` → `task5`. AC-7/AC-8 kept in Loop 6 as soft (Lower Bound "may be characterized") per the user.
**Affected Sections:** AC-9 body; Dependencies (Milestone 5); Task Breakdown (task10 description + dependency); Pending User Decisions (new DEC-9).
**Cross-Reference Updates:** task10 `Depends On`: `-` → `task5`.

### CMT-11: AC-8 input fixture (see Research Findings)
**Changes Made:** AC-8 no longer assumes the nonexistent `probe_64k.json`; the probe is harness-generated or a named small AC-8 deliverable; Scope-OUT names the one fixture exemption.
**Affected Sections:** AC-8 body; Feasibility item 5; Scope-OUT.

### CMT-12: Mark the draft appendix NON-AUTHORITATIVE
**Changes Made:** Added a `⚠ NON-AUTHORITATIVE — ARCHIVED ORIGINAL DRAFT` banner immediately after `--- Original Design Draft Start ---`, stating the appendix's `AC-L6-*` labels, `*(PENDING)*` decisions, draft ACs, and critical-path commands are superseded. Draft body text left byte-exact.
**Affected Sections:** Original Design Draft appendix (boundary banner only).

## Remaining Decisions

**None.** All 12 comments are resolved and no decision remains `PENDING`. The three discussion-mode decisions and two derived records are captured in the refined plan's `## Pending User Decisions`:

- **DEC-4 (refined):** AC-2 authoritative — evaluate int8 first on paper; structural directly if int8 predicted insufficient. *(RESOLVED)*
- **DEC-8 (new):** int8 requires a real-mask NIAH non-regression in addition to the synthetic test. *(RESOLVED)*
- **DEC-9 (new):** keep AC-7/AC-8 in Loop 6 as soft; AC-9 opportunistic. *(RESOLVED)*
- **DEC-3 (refined):** directional = accepted progress, not shippable; attribution required. *(RESOLVED)*

## Refinement Metadata

- **Input Plan:** `development/loop6/plan.md`
- **Output Plan:** `development/loop6/refined_plan_v1.md`
- **QA Document:** `.humanize/plan_qa/plan-qa.md`
- **Total Comments Processed:** 12
  - Questions: 0
  - Change Requests: 11
  - Research Requests: 1 (CMT-11)
- **Plan Sections Modified:** Goal Description; Resolved strategic & product framing (DEC-3, DEC-4); Acceptance Criteria (AC-2, AC-3.1, AC-5, AC-8, AC-9); Path Boundaries (Lower Bound, Allowed Choices); Feasibility Hints (items 2 & 5, Risk #1); Dependencies and Sequence (Milestone 5); Task Breakdown (task2, task3, task4, task10); Claude-Codex Deliberation (new refinement subsection + Convergence Status); Pending User Decisions (DEC-3, DEC-4 updated; DEC-8, DEC-9 added); Original Design Draft appendix (NON-AUTHORITATIVE banner)
- **Convergence Status:** converged
- **Refinement Date:** 2026-05-30
