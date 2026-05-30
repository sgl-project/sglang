# Loop 6 — Make Double Sparsity Shippable for the Client SLO on V3.2

## Goal Description

Make **Double Sparsity (DS)** serve the immediate client workload on **DeepSeek-V3.2 FP8** at the client SLO, on real 8×H200 hardware, at the fixed "Option B" operating point — and decide how far past the engineering wins to invest given that V3.2 already ships a trained native sparse indexer (DSA).

The single done-criterion: **DS serves the client workload (4096 ISL / 512 OSL / conc 16–64 / ~55% cache) at strict `P99 TTFT < 22 s` AND per-request `≥ 30 TPS/req`, measured as an absolute pass/fail against the client SLO (not a DS-vs-DSA ratio).** Per the user's DEC-3 clarification, for *this* loop (the MVP) that target is graded as a **directional trend** — measured movement toward the strict numbers vs the Loop-5 baseline counts as success — while it becomes a **hard pass/fail blocker downstream** as work with the client continues. The absolute strict numbers are still measured and recorded.

<comment>CRITIQUE (Linus / pensieve taste-review): The loop's TITLE is "Make DS Shippable" and you call AC-5 "the single done-criterion" — then DEC-3 redefines that criterion as a "directional trend" you can satisfy while DS still misses the 22 s bar by 3×. A done-criterion you can pass while the thing still doesn't work is not a done-criterion; it's a euphemism. "Directional" is a fine MVP *expectation*, but state it plainly: a directional result means NOT shippable yet. Don't let "moved 292 s → 60 s at conc 64" get filed as success under a heading that promises shipping. Bind the loop's PASS to the absolute bar; record directional movement as progress, not as the done-criterion.

— CODEX REVIEW (#1, AGREE-WITH-MODIFICATION): The plan uses "success" for two different states — absolute client-SLO pass vs directional progress while still missing the SLO. Keep the directional MVP gate if the client asked for it, but name it "progress accepted, NOT shippable"; reserve the words "done-criterion"/"shippable" for strict P99 TTFT < 22.0 s AND ≥ 30 TPS/req at all 16/32/64 conc.</comment>

The client-facing blocker is **not speed** (DS per-request generation already beats 30 TPS: Loop-5 measured 34.0 / 33.9 / 33.9 tok/s p50 at conc 16/32/64). It is an **admission/queue** problem: DS reserves a per-rank `TokenLabelTable` (~8 GB/rank fp16) on top of the ~84 GB/rank V3.2 FP8 weights, forcing `mem_fraction_static=0.6` (DSA runs 0.85). The small KV pool admits only 14.5 / 24.6 / 35.7 of the nominal 16 / 32 / 64 concurrency, so requests queue and P99 TTFT explodes (Loop-5: 57.7 / 132.9 / 292.0 s). Raising `mem_fraction_static` past 0.6 currently **OOMs DS during generation**.

**The spine of this loop:** shrink the per-rank `TokenLabelTable` footprint so DS can boot at a higher `mem_fraction_static` *without* generation-time OOM, restoring admission so TTFT falls toward the SLO. A central refinement from cross-review: the real target is **enough admitted KV-pool capacity / concurrency with HBM headroom to move TTFT toward `< 22 s`** — *not* `mem_fraction_static = 0.8` as a number in itself. Raising `mem_fraction_static` also grows the KV pool, which grows the table and the score scratch, so the footprint→admission→SLO chain must be validated as a memory fixed-point with a pre-coding feasibility budget, and the SLO claim must separate **admission-wait** from **prefill-compute** (TTFT may be prefill-bound at conc 64 even after admission is fixed).

Alongside the spine, the loop hardens the existing harnesses (64K servability, the AC-12 within-budget gate asserted from real token counts), refreshes the AC-11 directional DS-vs-DSA re-sweep at the lifted operating point, and resolves a strategic gate (DEC-1) on long-context recall R&D.

<comment>CRITIQUE (Linus / data-structures-first): Step back. This loop spends real kernel-craft — int8 quant, per-slot scales, CUDA-graph-safe dequant — to COMPRESS a TokenLabelTable that backs DS's *offline* selector, a selector your own DEC-1 finding calls strictly inferior to the trained DSA indexer the model already ships (NIAH 75/5/0 vs 100). "Bad programmers worry about the code; good programmers worry about data structures." The highest-taste question isn't "how do I shrink this 8 GB table," it's "why does DS carry a per-token signature table at all on a model that already has a trained indexer?" The recall gate (DEC-1) is correctly deferred — but that same finding should also inform the FOOTPRINT approach, not only the R&D scope. At minimum, name the irony in AC-2: you are investing engineering to compress a structure whose selection quality you just declared second-best.

— CODEX REVIEW (#2, AGREE-WITH-MODIFICATION): The smell is real, but don't reopen DEC-2 by asking whether DS should exist at all — this plan already says DS must meet the SLO as an opt-in while DSA stays default. The actionable, narrower critique: AC-2 should explicitly justify any TokenLabelTable work as the *minimum reversible* DS-opt-in fix on a DSA-native model whose trained indexer already wins recall (75/5/0 vs 100). That keeps "compress the inferior selector" from silently growing into more Tier-1 architecture than the client SLO needs.</comment>

### Resolved strategic & product framing (from the gen-plan discussion)

- **DEC-1 (Tier-2 gate):** Tier-2 long-context recall R&D **is pursued, but strictly *after* the full Tier-1 spine lands.** The chosen direction is a **custom sparse-matmul kernel that mirrors the native NSA/DSA decode kernel but exposes an adjustable `top_k`** (relaxing the hard `indices.shape[-1] == dsa_index_topk` cap), with a learned/query-aware selector as a secondary alternative.
- **DEC-2 ("shippable" = Both):** DS must **meet the client SLO itself** AND ship as an **opt-in knob while DSA remains the production default** that meets the SLO. Both properties are required.
- **DEC-3 (SLO hardness):** For this MVP the SLO is a **directional trend**; it becomes a **hard blocker downstream**. Measurement rigor (strict `<22.0`, all conc, radix-on proven, attribution) is preserved regardless.
- **DEC-4 (footprint lever):** A **two-step ladder** — run the feasibility budget, then implement **int8-symmetric signatures at the same `label_dim` (per-(layer/slot/head) scales applied at scoring), target `mem_fraction_static=0.8`, reusing the Loop-5 mask unchanged** first; **escalate to a page-level / two-stage table only if the budget shows int8 cannot restore nominal admission.** (Narrowing `label_dim` was explicitly *not* chosen.)

<comment>CRITIQUE (Linus / "worry about data structures, not code"): DEC-4 makes int8-quant-at-same-`label_dim` the PRIMARY lever and demotes the page-level/two-stage redesign to "escalate only if int8 is insufficient." That is backwards taste. int8-same-`label_dim` is a *code* bolt-on — quantize-on-write + per-(layer/slot/head) scales + apply-at-scoring + a tolerance test + a CUDA-graph-safe dequant — for a win the plan itself budgets at only ~1.6–1.8× (line 124). The table's size is set by its SHAPE (`num_layers · max_tokens · num_heads · label_dim`); the structural lever is the one that redesigns that shape. You chose the lever with the MOST new branches and the LEAST structural payoff. This also collides with the recorded user preference "prefer the theoretically correct implementation; don't justify the inferior choice by engineering cost" — int8-to-save-under-2× is precisely an engineering-cost-justified compromise. Demand that AC-2 prove int8 is *sufficient* before committing, not merely "try int8 first."

— CODEX REVIEW (#3, AGREE-WITH-MODIFICATION): int8 is not automatically bad taste — preserving the selector shape may be the lowest-quality-risk lever. The bad part is that DEC-4, the Lower Bound, AND task3 still hard-code int8 before AC-2 has proven the fixed-point works, even though the plan budgets only ~1.6–1.8× and says the table grows with the KV pool. Make AC-2 AUTHORITATIVE: if int8 does not predict nominal conc-64 admission with headroom at mem_fraction_static=0.8, skip it and select the structural lever directly.</comment>
- **DEC-5 (topology):** **Single-node TP=8 only** for this loop; multi-node TP is **deferred to a future loop** (the SLO numbers do not change). Tracked as a downstream requirement in `development/roadmap.md` §5.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Each AC drops its acceptance artifact under `runs/<date>_dsv32_loop6/` (e.g. `runs/20260530_dsv32_loop6/`), copying it there even when the underlying tool writes to its native output directory (e.g. `development/results/`). Draft cross-references to the original `AC-L6-*` labels are noted in parentheses for traceability.

- **AC-1 (Strategic gate — DECIDE FIRST; draft AC-L6-0)** — *analyze*. A decision doc `ds_on_v32_decision.md` records the DEC-1 resolution: pursue Tier-2 recall R&D **after** the Tier-1 spine lands, via a custom adjustable-`top_k` sparse-matmul kernel (NSA/DSA-mirrored), with the `index_topk` / shared-kernel / offline-selector rationale and the explicit consequence/sequencing for Tier-2 (AC-10).
  - Positive Tests (expected to PASS):
    - The doc states the decision (Tier-2 = pursue-after-Tier-1, custom adjustable-`top_k` kernel direction), its rationale, and its Tier-2 consequence/sequencing.
    - The doc exists and is committed before any Tier-2 (AC-10) code is written.
  - Negative Tests (expected to FAIL):
    - Any Tier-2 kernel/selector work committed before this doc exists, or before the Tier-1 spine lands, is out of order and fails the loop discipline.
    - A doc that omits the index_topk/shared-kernel/selector rationale or the Tier-2 sequencing is incomplete.

- **AC-2 (Footprint feasibility budget — PRE-CODING; new, cross-review required)** — *analyze*. Before any footprint code, compute the HBM fixed-point for each candidate lever on the ladder: required freed HBM, scale-storage overhead, the resulting target `max_total_num_tokens`, and the expected achieved concurrency at conc 64, at `mem_fraction_static=0.8`. Pick the minimum lever **predicted** to restore nominal admission with headroom.
  - Positive Tests (expected to PASS):
    - A budget artifact (`footprint_feasibility.md`) records, per lever (int8-same-`label_dim` primary; page-level/two-stage escalation), the freed-HBM math, scale overhead, target `max_total_num_tokens`, and predicted achieved-conc@64.
    - The artifact names the chosen lever and why it is the minimum predicted to restore nominal admission.
  - Negative Tests (expected to FAIL):
    - Starting footprint coding (AC-3) before this budget artifact exists.
    - A budget that omits scale-storage overhead or the larger-pool feedback (raising `mem_fraction_static` grows the table) — i.e. treats int8 as a flat "save 4 GB."

  <comment>CRITIQUE (Linus / don't build it twice): The ladder "int8 first, escalate to structural if insufficient" risks IMPLEMENTING twice — ship a marginal int8 path, discover 1.6–1.8× doesn't reach 0.8, then build the structural table anyway and throw the int8 work away. "Design it twice" (Ousterhout) means THINK twice before coding, not code twice. AC-2 is the right gate — so give it teeth: if the budget *predicts* int8 cannot restore nominal admission at the target, AC-2 must be permitted to skip int8 entirely and select the structural lever directly. As written, AC-3 hard-codes "int8 primary," pre-committing the implementation before AC-2's math exists.

— CODEX REVIEW (#4, AGREE): This is the core fix to DEC-4/AC-2. "int8 first" must mean "evaluate int8 first IN THE BUDGET," not "implement int8 first no matter what." AC-2 must be allowed to choose page-level/two-stage immediately when the fixed-point math says int8 cannot restore nominal admission with headroom; otherwise the plan is explicitly scheduling throwaway kernel work.</comment>

- **AC-3 (TokenLabelTable footprint reduction; draft AC-L6-1)** — *coding*. Reduce the per-rank table bytes via the lever chosen in AC-2, preserving DS selection numerics; the compact path is **flag-gated with fp16 as the default** until it has unit + hardware evidence, and is **CUDA-graph-safe** (pre-allocated, static-shaped scales; no host sync or Python dtype dispatch inside any captured path). The compact flag is an **explicit allowed config field** (added to `DoubleSparsityConfig` / the `extra` dict), not an unknown-field bypass.
  - AC-3.1 (Quantization class — int8 same `label_dim`, primary lever):
    - Positive: a unit test shows the compact table's selected-token set matches the fp16 baseline within an **explicit tolerance** (top-k overlap@2048 / selected-token recall / score-error distribution) on a synthetic shape; a measured per-rank byte count drops by the budgeted factor; a config/unit test shows the compact flag parses and that a **DSA-default boot allocates no DS table and does not alter decode** (DSA non-regression).

    <comment>CRITIQUE (Linus / a test with no number is not a test): "within an **explicit tolerance** (top-k overlap@2048 / selected-token recall / score-error distribution)" offers THREE metrics and ZERO thresholds. The negative test "selection divergence beyond tolerance" is unverifiable until someone picks the metric AND the number — and "we'll set the tolerance during implementation" is exactly how a real regression slips through behind a green check. Nail it down in AC-2 (pre-coding): ONE primary metric, ONE threshold (e.g. top-k overlap@2048 ≥ 0.99 on the synthetic shape), justified against the NIAH recall you are protecting. A menu is not an acceptance criterion.

    — CODEX REVIEW (#5, AGREE-WITH-MODIFICATION): A tolerance menu is not a test. AC-3.1 must name the PRIMARY equivalence metric and its numeric FAIL threshold before implementation, then use any secondary score-error distribution only as diagnostics. Without a concrete metric, threshold, and test shape, the negative test "selection divergence beyond tolerance" cannot fail deterministically.</comment>
    - Negative: any selection divergence beyond tolerance; the compact path becoming the default before hardware validation; the compact scoring path performing host sync / dynamic allocation under CUDA-graph capture.

    <comment>CRITIQUE (Linus / measure the hot path you're about to slow down): int8 signatures mean you DEQUANTIZE at scoring time inside the Triton selection kernel on every decode step — extra memory traffic and FLOPs on the decode hot path. Your TPS margin is thin: Loop-5 measured 33.9 tok/s vs the 30 floor ≈ 11% headroom. If dequant-at-scoring costs more than that, you FIX admission/TTFT and BREAK the TPS SLO — trading one failure for another. AC-5 catches it, but only end-to-end, after all the kernel work is sunk. Add an early, isolated decode-TPS-non-regression micro-check on the compact scoring path (in AC-3 or AC-4), so this surfaces BEFORE the full hardware sweep. As written, the one genuinely hard kernel risk (dequant under graph capture) is buried in a CUDA-graph-safety parenthetical.

    — CODEX REVIEW (#6, AGREE-WITH-MODIFICATION): The risk is not just CUDA-graph safety; it is spending the ONLY TPS margin you have. Loop-5 DS decode is 33.9 tok/s against a 30 TPS/req floor, while AC-3 adds scale reads and dequant/scale math inside scoring. Add an early compact-vs-fp16 scoring/decode microbench with a NUMERIC overhead budget tied to that 33.9→30 margin, before the full AC-5 hardware sweep.</comment>
  - AC-3.2 (Structural class — page-level / two-stage, only if AC-2 escalates):
    - Positive: a regenerated/saved mask or table artifact (if the selector shape changes), plus a NIAH quality **non-regression** vs the Loop-5 DS baseline, plus the measured byte drop.
    - Negative: a NIAH recall regression vs the Loop-5 DS baseline; a structural change held to bitwise "preserve numerics" it cannot honestly meet.

  <comment>CRITIQUE (Codex / NEW): Inconsistent rigor by lever class. AC-3.1 protects int8 with only a SYNTHETIC selected-token equivalence test, while AC-3.2 requires NIAH non-regression for structural changes — but int8 perturbs the very scores the already-weak DS selector relies on, and DEC-1 puts DS recall at 75/5/0 vs DSA 100 at the same 2048 budget. There is no headroom to lose. Require an int8 quality gate on REAL V3.2 / Loop-5-mask data (an AC-Q / NIAH non-regression artifact) IN ADDITION to the synthetic unit test, not instead of it.</comment>
  - AC-3.3 (Tier-1 ABI lock — applies to the whole spine):
    - Negative: any Tier-1 change that touches or bypasses the FlashMLA `indices.shape[-1] == dsa_index_topk` decode assert (Tier-1 keeps `top_k == dsa_index_topk == 2048`; relaxing it is AC-10 only).

- **AC-4 (mem-fraction lift + no-OOM validation; draft AC-L6-2)** — *coding* (hardware-run). With the compact table, boot DS at the lifted `mem_fraction_static` (DEC-4 target 0.8) and survive a sustained long `/generate`.
  - Positive Tests (expected to PASS):
    - A mem-fraction sweep log (`0.6 → … → 0.8`) shows `max_total_num_tokens` rising; the **full HBM budget is logged including NVML/torch reserved+allocated residual** (not only named tensors): weights + KV pool + table + scales + `written` + score scratch + FlashMLA metadata + CUDA-graph pool + headroom.
    - A long-context `/generate` completes with **no generation-time OOM** and **no monotonic memory growth** across the run; `/get_server_info` is recorded.
  - Negative Tests (expected to FAIL):
    - A generation-time OOM at the target mem fraction (the table is still too big — iterate the lever per the AC-2 ladder, or reconsider the admission model).
    - Hidden/monotonic memory growth over the run, even if no single OOM fires.

- **AC-5 (⭐ Direct client-SLO validation — the done-criterion; draft AC-L6-3)** — *coding* (hardware-run). Run the full client workload `development/benchmark.sh` at `NUM_PROMPTS=320`, **all** conc 16 / 32 / 64, full 4096 ISL / 512 OSL / ~55% cache, **single-node TP=8**, with **radix-on proven** from server args / `.meta.json` sidecars (`RADIX_FIXTURE_ARTIFACT` present; radix is off by default in `serve_double_sparsity.sh`). A **trial-aggregation rule is fixed before running** (all trials pass, or median pass with the worst trial disclosed — no failed trial hidden behind a summary).

  <comment>CRITIQUE (Codex / NEW): A strict P99 SLO cannot be accepted by "median pass with the worst trial disclosed." Disclosure helps analysis, but if any predeclared trial misses P99 TTFT < 22.0 s or 30 TPS/req, the strict SLO did NOT pass for that run. SPLIT the rule: ALL trials must pass for a hard SLO claim; median-plus-worst-disclosed is acceptable only for directional characterization.</comment>
  - Grading (DEC-3): for this MVP the criterion is **directional** — success is **measured movement toward** strict `P99 TTFT < 22.0 s` AND `≥ 30 TPS/req` at every conc vs the Loop-5 baseline (57.7 / 132.9 / 292.0 s; 34 / 33.9 / 33.9 TPS). The absolute strict numbers are recorded as the target and become a hard blocker downstream.
  - Positive Tests (expected to PASS):
    - `client_slo_report.md` records the **absolute** numbers vs the SLO (asserting strict `< 22.0`, even though `benchmark_compare.py` gates `<= 22.0`) at conc 16 / 32 / 64, with valid `.meta.json` sidecars and radix-on proof.
    - The report contains a **measured admission-wait vs prefill-compute attribution** — or an explicit "attribution unavailable" statement, in which case it makes **no root-cause claim** — plus a directional-improvement statement vs Loop-5.

    <comment>CRITIQUE (Codex / NEW): The Goal section says the SLO claim must separate admission-wait from prefill-compute, but AC-5 allows "attribution unavailable" and still treats the run as directionally useful. For this loop's SPINE that is not optional: if TTFT still misses, no attribution means you cannot tell whether TokenLabelTable compaction fixed admission or merely exposed a prefill bottleneck — which is the whole question the loop exists to answer. Make admission/prefill attribution REQUIRED for directional success; without it, record the run but do NOT call the spine validated.</comment>
  - Negative Tests (expected to FAIL):
    - A report claiming SLO-pass without the strict-`<22.0` numbers, the radix-on proof, or the attribution.
    - A report that hides a failed trial, or asserts a root cause (admission vs prefill) without attribution data. (A genuine miss of 22 s / 30 TPS is *not* a loop failure for the MVP, but it must be recorded as a follow-up *with* the breakdown.)

- **AC-6 (Opt-in DS while DSA remains the default; new, from DEC-2 "Both")** — *coding*. DS ships as an opt-in knob; DSA stays the production default that meets the SLO.
  - Positive Tests (expected to PASS):
    - A DSA-default boot (no DS flags) meets the SLO unchanged and allocates **no** DS `TokenLabelTable`.
    - Enabling the DS opt-in flag activates the compact DS path; a test/doc records the opt-in mechanism and that DSA-default behavior/perf is unaffected.
  - Negative Tests (expected to FAIL):
    - The DS path active by default; DSA-default decode or perf regressed by the DS code; or the opt-in flag not actually toggling DS.

- **AC-7 (AC-11 directional re-sweep at the lifted point; draft AC-L6-4, DEC-7)** — *coding* (hardware-run). Re-run the 3-trial DS+DSA sweep (conc 16/32/64, 120 s warmup / 600 s window) at the new operating point, radix-on **both** sides (proven), per-side `mem_fraction_static` consistency enforced.
  - Positive Tests (expected to PASS):
    - DS achieved concurrency tracks nominal (≈100%); the comparator emits an updated TPS/TTFT summary and `ac11_analysis.md` / `ac11_resweep.md` is refreshed from the new artifacts.
  - Negative Tests (expected to FAIL):
    - A sweep that hides queue-dominated admission (achieved ≪ nominal without disclosure) is invalid.

- **AC-8 (64K servability; draft AC-L6-5)** — *coding* (hardware-run). At the lifted mem fraction, confirm a ~70K-token `/generate` no longer returns HTTP 400.
  - Positive Tests (expected to PASS):
    - A ~70K-token `/generate` returns 200 at the lifted mem fraction, with the served `max_total_num_tokens` recorded and no OOM/instability; OR a documented/**characterized** new admission ceiling if 64K still does not fit.
  - Negative Tests (expected to FAIL):
    - Silently re-recording the Loop-5 HTTP 400 without the lifted-mem retry. (A characterized ceiling is acceptable; a silent re-record is not.)

- **AC-9 (AC-12 within-budget gate from real token counts; draft AC-L6-6)** — *coding*. Change `test/manual/test_double_sparsity_v32.py` to assert `within_budget` from the **actual** `usage.prompt_tokens` (or tokenized chat length), not the 1024/1536 **word-count** proxy. Rename `length_tokens`→`length_words` or add an `input_tokens` field. The DECIDED DS-fair gate definition is **unchanged**.
  - Positive Tests (expected to PASS):
    - The harness records `usage.prompt_tokens` per NIAH prompt and asserts `within_budget` from it, **failing closed** if `usage` is missing/inconsistent.
    - The re-run gate still PASSES (DS-fair definition unchanged) and a diff shows the word-count proxy was safe (or it is corrected); the artifact is copied into `runs/<date>_dsv32_loop6/` (the harness natively writes `development/results/`).
  - Negative Tests (expected to FAIL):
    - Any change that alters the DS-fair gate thresholds/definition.
    - A silent fallback to the word-count proxy when `usage.prompt_tokens` is absent.

  <comment>CRITIQUE (Linus / what is this doing in this loop?): AC-9 renames `length_tokens`→`length_words` and asserts `within_budget` from real token counts. Correct fix — but it is Loop-5 janitorial backlog (draft handoff #4) with ZERO connection to the footprint→admission→SLO spine, and it sits in your Lower Bound (required). Ten ACs / eleven tasks for a loop whose ONE done-criterion is AC-5 is scope sprawl; every non-spine AC dilutes the round and invites the stall you yourself warn against ("two code-only rounds in a row with no hardware artifact is a stall"). AC-9 is cheap and code-only so it can ride — but mark it explicitly as a gap-filler that must NOT precede or delay the spine, and re-ask whether AC-7 (re-sweep) and AC-8 (64K) belong in THIS loop or the next. Keep the loop's mass on the spine.

  — CODEX REVIEW (#7, AGREE-WITH-MODIFICATION): AC-9 is a correct harness fix, but task10 has NO dependency while the dependency text admits its re-run needs a live server, and the Lower Bound makes it REQUIRED even though it does not restore admission or prove the client SLO. Mark AC-9 as opportunistic hardening that runs AFTER a hardware artifact, or give task10 a dependency on the lifted-server validation it actually needs. Don't let a code-only Loop-5 cleanup consume the next round before AC-3→AC-5.</comment>

- **AC-10 (Tier-2 DS long-context recall R&D — GATED; draft AC-L6-7)** — *coding* (hardware-run), **only after AC-1 is recorded AND the full Tier-1 spine (AC-3–AC-9) has landed.** Implement the custom adjustable-`top_k` sparse-matmul kernel (NSA/DSA-mirrored, relaxing `indices.shape[-1] == dsa_index_topk`) and/or a learned/query-aware DS selector; measure NIAH 4K/16K/64K recall delta vs the Loop-5 DS baseline 75% / 5% / 0%.
  - Positive Tests (expected to PASS):
    - A kernel-variant (or selector) change with a NIAH 4K/16K/64K recall-delta artifact showing movement vs DS 75 / 5 / 0, and the TPS/TTFT cost recorded.
  - Negative Tests (expected to FAIL):
    - Starting this before AC-1 or before the Tier-1 spine lands; or letting it block or regress the Tier-1 spine.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
All Tier-1 criteria (AC-1 through AC-9) pass on single-node TP=8 hardware with full HBM accounting and measured admission-wait-vs-prefill-compute attribution: the compact int8 (or, if escalated, page-level/two-stage) `TokenLabelTable` lands flag-gated with fp16 default and selection-equivalence evidence; DS boots at `mem_fraction_static=0.8` with no generation OOM; the full client-SLO benchmark shows DS moving decisively toward (or reaching) strict `P99 TTFT < 22 s` and `≥ 30 TPS/req` at all conc; DS ships opt-in with DSA preserved as default; the AC-11 re-sweep, 64K servability, and the real-token-count within-budget gate all land; and — *after* the Tier-1 spine lands — the gated AC-10 custom adjustable-`top_k` kernel produces a NIAH recall-delta artifact vs 75/5/0 with its TPS/TTFT cost.

### Lower Bound (Minimum Acceptable Scope)
The strategic gate (AC-1) and feasibility budget (AC-2) are recorded; the int8 footprint reduction (AC-3) lands flag-gated with selection-equivalence and DSA-non-regression evidence; DS boots at the lifted mem fraction with no generation OOM (AC-4); the client-SLO benchmark (AC-5) runs the full workload at all conc and records the absolute numbers with a **directional improvement** vs Loop-5 and the admission/prefill attribution (a genuine miss is recorded as a follow-up, not a loop failure); DS is opt-in with DSA preserved as default (AC-6); and the real-token-count within-budget gate (AC-9) lands. The AC-11 re-sweep (AC-7) and 64K servability (AC-8) may be **characterized** (documented ceiling / root-cause) rather than fully passing if hardware reveals a deeper bottleneck (e.g. prefill-bound TTFT at conc 64). Tier-2 (AC-10) is **deferred to its own loop** if the Tier-1 spine consumes Loop 6.

### Allowed Choices
- **Can use:** the two-step footprint ladder — int8-symmetric signatures at the same `label_dim` with per-(layer/slot/head) scales applied at scoring (primary), escalating to a page-level / two-stage label table (only if the AC-2 budget shows int8 is insufficient); `mem_fraction_static=0.8` as the validation target (0.7 acceptable as a more conservative first step); reuse of the Loop-5 mask (`/models/dsv32-fp8-channel-mask.safetensors`), the serve/bench scripts, the comparator, and the quality harnesses; an explicit new allowed config field (or the `extra` dict) for the compact flag with fp16 as the default.
- **Cannot use:** narrowing `label_dim` (explicitly not chosen in DEC-4); changing the DECIDED DS-fair AC-12 gate thresholds/definition; new fixture/scaffolding/serve/bench code (reuse Loop-5's); touching or bypassing the FlashMLA `indices.shape[-1] == dsa_index_topk` assert anywhere in Tier-1 (that relaxation is AC-10, and only after the spine lands); multi-node topology for this loop's SLO claim; plan-process markers (`AC-`, `DEC-`, `Tier`, `Option B`, `Round N`) in implementation code or comments.

> **Note on Deterministic Designs**: The operating point is highly deterministic — TP=8, fp8 KV, page 64, `flashmla_kv` prefill+decode, overlap-schedule + piecewise-cuda-graph disabled, radix-on via the fixture artifact, single-node — and is *fixed* by the `serve_*.sh` / `benchmark*.sh` scripts. `mem_fraction_static` is the one lever the loop deliberately moves. The footprint lever is constrained to the DEC-4 ladder. Within those fixed points the upper and lower bounds differ only in how far past the directional SLO the loop reaches (Tier-1 hardening + gated Tier-2).

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
1. **Strategic gate + feasibility (AC-1, AC-2):** write `ds_on_v32_decision.md` first; then compute the HBM fixed-point. Sketch:
   - `weights/rank (~84 GB) + kv_pool(mem_fraction) + table(mem_fraction) + scales + written + score_scratch + flashmla_meta + cuda_graph_pool + headroom ≤ HBM/rank (~141 GB usable on H200)`.
   - `table_bytes = num_layers_local · max_tokens(mem_fraction) · num_heads_local · label_dim · elem_bytes`. int8 takes `elem_bytes` 2→1 but adds per-(layer/slot/head) scale storage; net win ≈ 1.6–1.8×, *and* the larger pool at 0.8 grows `max_tokens`. Solve for the achieved concurrency at conc 64, not for a mem number.
2. **int8 compact path (AC-3):** quantize-on-write in `token_label_write.py` (symmetric int8 of the gathered channel labels, store per-(layer/slot/head) scale); apply the scale at scoring in `selection_kernel.py` / its Triton kernel (`_compute_token_scores_kernel` reads `sig_ptr`; add a static-shaped scale tensor read entirely on device). Keep fp16 the default; the compact path is a config flag. Add a selection-equivalence unit test (top-k overlap@2048 vs fp16) and a DSA-default-boot test (no DS table allocated).
3. **mem lift + no-OOM (AC-4):** sweep `MEM_FRACTION_STATIC` via `serve_double_sparsity.sh`; log the full HBM budget incl. `torch.cuda.memory_reserved/allocated` and NVML; fire a long `/generate` to flush generation-time OOM; watch for monotonic growth.
4. **client-SLO (AC-5):** `NUM_PROMPTS=320 MEM_FRACTION_STATIC=0.8 MODE=double_sparsity CONCURRENCIES="16 32 64" bash development/benchmark.sh`, radix-on via `RADIX_FIXTURE_ARTIFACT`; assert strict `<22.0`; attribute admission-wait vs prefill-compute (from server queue metrics / bench arrays, or declare unavailable).
5. **AC-11 re-sweep (AC-7), 64K (AC-8), within-budget (AC-9):** reuse `benchmark_baseline.sh` + `benchmark_compare.py --ac11`; probe `development/loop6/probe_64k.json`; edit the manual test to assert from `usage.prompt_tokens`, fail-closed.

<comment>CRITIQUE (Codex / NEW — VERIFIED): `development/loop6/probe_64k.json` does NOT exist on disk, yet the plan routes AC-8 (a hardware acceptance check) at it while Scope-OUT forbids new fixture/scaffolding code. An acceptance check whose input is undefined cannot run. Either point AC-8 at an existing 64K payload, or name the JSON fixture as an explicit AC-8 deliverable (a tiny probe payload is a reasonable exemption from the no-new-scaffolding rule — say so).</comment>
6. **Tier-2 (AC-10), only after the spine lands:** the custom kernel mirrors the NSA/DSA sparse matmul but accepts `top_k > 2048` (relax `indices.shape[-1] == dsa_index_topk` at `dsa_backend.py:~2148`); measure NIAH delta vs 75/5/0.

### Relevant References
- `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` — `TokenLabelTable`, `allocate_token_label_table`, `bytes_per_rank()`, `estimate_hbm_bytes()`; where the footprint/dtype lives.
- `python/sglang/srt/layers/attention/double_sparsity/token_label_write.py` — `token_label_write` / `invalidate_token_label_slots`; the quantize-on-write site.
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` — `compute_token_scores` + `_compute_token_scores_kernel`; the apply-scales-at-scoring site.
- `python/sglang/srt/layers/attention/double_sparsity/config.py` — `DoubleSparsityConfig` rejects unknown fields; the compact flag must be an explicit allowed field (or go in `extra`).
- `python/sglang/srt/layers/attention/dsa_backend.py` — DSA+DS decode; the `_ds_token_label_table` binding (~line 497) and the `indices.shape[-1] == dsa_index_topk` assert (~line 2148, the Tier-2 hard cap).
- `development/serve_double_sparsity.sh` — Option B launcher; radix-off by default unless `RADIX_FIXTURE_ARTIFACT` is set; `MEM_FRACTION_STATIC` lever.
- `development/benchmark.sh`, `development/benchmark_baseline.sh`, `development/benchmark_compare.py` — workload runner, DSA baseline, comparator (already has `SLO_TTFT_P99_S=22.0`, `SLO_PER_REQUEST_TPS_P50=30.0`; gates with `<=`, so the AC-5 report asserts strict `<`).
- `test/manual/test_double_sparsity_v32.py` — NIAH/within-budget harness (the word-count proxy `length_tokens`).
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — registered unit tests (selection-equivalence belongs here).
- `development/loop6/runbook.md`, `development/roadmap.md` (§4–§5), `development/CLIENT_SLOS.md`, `CLUSTER.md` — loop process, roadmap, SLO, hardware.

### Risks and Likely Failure Modes (from the draft)
1. **Footprint shrink regresses selection (int8 quant).** Mitigation: selection-equivalence unit test vs fp16; fp16 default behind a flag until hardware-validated; re-run AC-Q / AC-9 after the change.
2. **mem-fraction lift still OOMs at the target.** The table may not be the only lever; the admission model may need adjusting. Mitigation: read the OOM verbatim; if footprint alone doesn't reach the target after ~2 rounds, reconsider the admission/KV-budget model (escalate the AC-2 ladder; see runbook Step 8 stagnation signals).
3. **P99 TTFT still > 22 s at full concurrency even after admission is fixed.** At conc 64 the bottleneck may shift from admission-queue to prefill compute (4096 ISL × 64). Mitigation: AC-5 must break down admission-wait vs prefill-compute; if prefill-bound, surface it honestly as a different (chunked-prefill / scheduling) follow-up, not hidden.
4. **DS-vs-DSA TPS parity at conc 16/32 still misses.** The client SLO is absolute 30 TPS/req (already met), so this is secondary — record it as a DEC-7 directional follow-up; don't let it block the client-SLO MVP claim.
5. **Tier-2 recall R&D started before AC-1 / before the spine lands.** The expensive trap. Mitigation: AC-1 + a landed Tier-1 spine are hard prerequisites; a closed/deferred gate is a legitimate outcome.

## Dependencies and Sequence

### Milestones
1. **Strategic gate & feasibility (AC-1, AC-2)** — analyze-only; gates the footprint lever and Tier-2.
   - Phase A: write `ds_on_v32_decision.md` (DEC-1 resolution).
   - Phase B: compute the HBM fixed-point feasibility budget; pick the minimum lever.
2. **Footprint compaction (AC-3)** — code + unit tests.
   - Phase A: int8 quantize-on-write + apply-scales-at-scoring, flag-gated, fp16 default, CUDA-graph-safe.
   - Phase B: selection-equivalence unit test + DSA-non-regression config test; (escalate to page-level/two-stage only if AC-2 says so).
3. **Admission restoration on hardware (AC-4)** — mem-fraction lift + full HBM budget + no-OOM long generate.
4. **Client-SLO MVP (AC-5, AC-6)** — the headline directional result + the opt-in / DSA-default property.
5. **Tier-1 hardening (AC-7, AC-8, AC-9)** — AC-11 re-sweep, 64K servability, real-token-count within-budget gate.
6. **Tier-2 recall R&D (AC-10)** — GATED; strictly after AC-1 and the full Tier-1 spine; custom adjustable-`top_k` kernel + NIAH recall delta.

Dependencies (relative, not temporal): M1 precedes everything (the lever choice and Tier-2 gate flow from it). M2 depends on M1's feasibility budget. M3 depends on M2 (needs the compact table). M4 depends on M3 (needs the lifted mem fraction); AC-6's DSA-non-regression overlaps M2's tests but is validated as a product property in M4. M5 depends on M3 (AC-7/AC-8 need the lifted point; AC-9 is a code-only harness edit that can land alongside but its re-run needs a live server). M6 depends on **all** of M1–M5. A code-only round (e.g. AC-3 + unit tests, or the AC-9 harness edit) is acceptable **if** the next round validates on hardware; two code-only rounds in a row with no `runs/<date>_dsv32_loop6/` artifact is a stall.

## Task Breakdown

Each task includes exactly one routing tag (`coding` = implemented by Claude incl. driving the cluster serve/bench scripts; `analyze` = executed via Codex / `/humanize:ask-codex`). The draft/runbook's third "hwrun" category maps to `coding`, since Claude drives the hardware scripts and Codex never touches hardware.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Write `ds_on_v32_decision.md` recording the DEC-1 resolution (Tier-2 after Tier-1; custom adjustable-`top_k` kernel) + rationale | AC-1 | analyze | - |
| task2 | Compute the HBM fixed-point feasibility budget per lever (int8 primary, page-level/two-stage escalation); pick the minimum lever | AC-2 | analyze | task1 |
| task3 | Implement compact int8 `TokenLabelTable`: quantize-on-write + per-slot scales, apply-scales-at-scoring, flag-gated, fp16 default, CUDA-graph-safe | AC-3 | coding | task2 |
| task4 | Selection-equivalence unit test (top-k overlap@2048 vs fp16) + compact-flag config test + DSA-default-boot non-regression test | AC-3, AC-6 | coding | task3 |
| task5 | Hardware: mem-fraction sweep (0.6→0.8) with full HBM budget log (incl. NVML/torch residual) + sustained long `/generate` no-OOM | AC-4 | coding | task3, task4 |
| task6 | Hardware: full client-SLO benchmark (NUM_PROMPTS=320, conc 16/32/64, radix-on proven) + `client_slo_report.md` with strict-`<22.0` numbers + admission/prefill attribution + directional vs Loop-5 | AC-5 | coding | task5 |
| task7 | Verify opt-in/DSA-default product property on hardware (DSA boot meets SLO, no DS table; DS flag toggles compact path) | AC-6 | coding | task5 |
| task8 | Hardware: 3-trial DS+DSA AC-11 re-sweep at the lifted point (radix-on both sides) + refresh `ac11_analysis.md`/`ac11_resweep.md` | AC-7 | coding | task5 |
| task9 | Hardware: ~70K-token `/generate` 64K servability probe at the lifted mem fraction (200, or characterized ceiling) | AC-8 | coding | task5 |
| task10 | Edit `test_double_sparsity_v32.py`: assert `within_budget` from `usage.prompt_tokens` (fail-closed), rename `length_tokens`→`length_words`/add `input_tokens`; re-run gate + proxy diff | AC-9 | coding | - |
| task11 | (GATED) Custom adjustable-`top_k` sparse-matmul kernel (NSA/DSA-mirrored, relax `indices.shape[-1]==dsa_index_topk`) and/or learned selector + NIAH recall-delta vs 75/5/0 | AC-10 | coding | task1, task3, task4, task5, task6, task7, task8, task9, task10 |

## Claude-Codex Deliberation

### Agreements
- The reframe is correct: success is **admitted KV capacity + HBM headroom sufficient to move the absolute client SLO**, not `mem_fraction_static=0.8` as a goal in itself; the chain must be validated as a memory fixed-point.
- Tier-2 is properly gated. Given the FlashMLA `indices.shape[-1] == dsa_index_topk` hard assert, a `top_k > 2048` relaxation is not a small config tweak — it needs a custom kernel (the user's chosen direction), pursued strictly after Tier-1.
- Keeping the compact path behind a flag with fp16 default, DSA non-regression as a negative criterion, and CUDA-graph safety as a hard requirement is sound.
- AC-9 is the right fix direction: actual `usage.prompt_tokens` replaces the word-count proxy, failing closed, with the DS-fair gate definition unchanged.
- Reuse Loop-5's mask, serve/bench scripts, comparator, and quality harnesses; build no new scaffolding.

### Resolved Disagreements
- **SLO concurrency coverage:** the draft's AC-L6-3 wrote "conc 16 and 64"; Codex required all of 16/32/64 (a conc-32 miss would still fail the SLO claim). Resolution: AC-5 asserts at **all** conc 16/32/64 (a superset that preserves the draft's endpoints and matches the client workload + the `CONCURRENCIES="16 32 64"` command).
- **Strict vs boundary comparison:** `benchmark_compare.py` gates `ttft_p99_s <= 22.0`; the client SLO is strict `< 22 s`. Resolution: AC-5 asserts strict `< 22.0` in the report (optionally tighten the comparator later).
- **A "production-flags" decision (a transient DEC-6 Claude floated):** Codex argued the target is fixed Option B and production-flag variation cannot be used to satisfy this goal. Resolution: removed; Option B is the fixed target, production-flag variation is post-MVP characterize-later only.
- **AC-2 conflating compaction classes:** Codex required splitting validation — quantization (same `label_dim`) is held to selected-token equivalence; structural levers (page-level/two-stage) change the selector and are held to NIAH non-regression instead. Resolution: AC-3.1 vs AC-3.2.
- **Vague "target factor":** Codex required a pre-coding feasibility budget (freed HBM, scale overhead, target `max_total_num_tokens`, achieved-conc@64). Resolution: added AC-2.
- **Radix-on under-enforcement:** verified `serve_double_sparsity.sh` defaults DS radix-**off** unless `RADIX_FIXTURE_ARTIFACT` is supplied. Resolution: AC-5/AC-7 must prove radix-on from server args/sidecars.
- **Weak attribution / hidden trials / artifact routing:** Resolution: AC-5 requires measured (or explicitly-unavailable) admission-vs-prefill attribution, a pre-declared trial-aggregation rule, NVML/torch residual in the HBM budget, and copying acceptance artifacts into `runs/<date>_dsv32_loop6/`.
- **Tier-1 ABI lock:** Resolution: AC-3.3 — no Tier-1 change may touch the FlashMLA decode assert (`top_k == dsa_index_topk == 2048`).

### Convergence Status
- Rounds executed: 2 (plus the first-pass analysis). Round 1 produced the required changes above; Round 2 confirmed **no remaining DISAGREE / REQUIRED_CHANGES**.
- Final Status: `converged`. Remaining open items are the user-decision items below — all now **resolved** in discussion mode.

## Pending User Decisions

All decisions from the draft were resolved in the gen-plan discussion; none remain `PENDING`.

- **DEC-1: Pursue Tier-2 DS long-context recall R&D on V3.2?**
  - Claude Position: cap at Tier-1 (defer Tier-2), since DS cannot match DSA at the shared 2048 budget.
  - Codex Position: Tier-2 properly gated; the `top_k > 2048` relaxation is not one-loop-feasible without a custom kernel.
  - Tradeoff Summary: investing GPU-hours on recall R&D vs shipping the engineering wins that pay off regardless.
  - Decision Status: **RESOLVED** — pursue Tier-2, but **strictly after the Tier-1 spine lands**, via a **custom sparse-matmul kernel mirroring NSA/DSA with an adjustable `top_k`** (relaxing the `indices.shape[-1] == dsa_index_topk` cap); learned selector secondary. Drives AC-1 and AC-10.

- **DEC-2: "Shippable" definition.**
  - Claude Position: DS meets the SLO itself (matches the draft objective).
  - Codex Position: shipping default may remain DSA with DS opt-in, but it must not weaken the MVP success definition.
  - Tradeoff Summary: a strict "DS meets the SLO" bar vs an opt-in-while-DSA-default framing.
  - Decision Status: **RESOLVED — Both.** DS must meet the SLO itself AND ship opt-in while DSA stays the default. Drives AC-5 + AC-6.

- **DEC-3: TTFT target source / hardness.**
  - Claude Position: hard absolute strict `P99 TTFT < 22 s` at `NUM_PROMPTS=320`.
  - Codex Position: require strict `< 22.0`, all conc, full workload; define trial aggregation before running.
  - Tradeoff Summary: a hard pass/fail gate vs a directional MVP target.
  - Decision Status: **RESOLVED** — for this MVP a **directional trend** (movement toward strict `<22 s` / `≥30 TPS` counts), becoming a **hard blocker downstream**; measurement rigor preserved. Drives AC-5 grading.

- **DEC-4: Footprint approach + target mem fraction.**
  - Claude Position: feasibility budget first, then int8 same-`label_dim` at 0.8, escalate if needed.
  - Codex Position: constrain the lever by the feasibility calculation; split validation by lever class; int8 net win may be only ~1.6–1.8×.
  - Tradeoff Summary: lowest-quality-risk int8 vs higher-win structural redesign.
  - Decision Status: **RESOLVED** — two-step ladder: feasibility budget → **int8 same-`label_dim` @ 0.8 first**, **escalate to page-level/two-stage** if insufficient; **narrower `label_dim` not chosen**. Drives AC-2/AC-3.

- **DEC-5: Deployment topology.**
  - Claude Position: single-node TP=8 for the client deliverable.
  - Codex Position: topology matters for execution logistics; the SLO claim still uses the fixed TP=8 Option B target.
  - Tradeoff Summary: single-node simplicity vs multi-node scaling.
  - Decision Status: **RESOLVED** — **single-node TP=8 only this loop**; multi-node TP **deferred to a future loop** and tracked as a downstream requirement in `development/roadmap.md` §5; the SLO numbers are unchanged. Drives AC-5.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must **NOT** contain plan-process terminology such as `AC-`, `DEC-`, `Milestone`, `Step`, `Phase`, `Tier`, `Option B`, `Round N`, or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase. Use descriptive, domain-appropriate naming in code (e.g. a config flag named for the compact-signature behavior, not for an AC number).

### Operational landmarks (verified)
- **Anchor:** `loop6-base` at the Loop-5 final commit (`989975625` or later) on `dev/double-sparsity-standalone`. Reuse the Loop-5 mask at `/models/dsv32-fp8-channel-mask.safetensors`; regenerate only if a recipe field changes (only the AC-3.2 structural escalation would change mask shape).
- **Compact flag:** `DoubleSparsityConfig` (`config.py`) rejects unknown fields — add the compact-path flag as an explicit allowed field or via the `extra` dict; fp16 stays the default until hardware-validated.
- **Radix-on:** not an env override — `serve_double_sparsity.sh` is radix-off by default and enables radix-on only with a fixtures-passed `RADIX_FIXTURE_ARTIFACT`. AC-5/AC-7 prove radix-on from server args/sidecars.
- **Comparator boundary:** `benchmark_compare.py` already encodes the SLO (`SLO_TTFT_P99_S=22.0`, `SLO_PER_REQUEST_TPS_P50=30.0`) but gates with `<=`; AC-5 asserts strict `<22.0`.
- **Killing servers between bench runs:** `pkill -f sglang_router` does NOT catch the Rust process (renamed `sglang::router`). Use `pkill -f 'sglang::router'` (or match the worker `python -m sglang.launch_server` pattern) so the old router doesn't hold the port across the DS↔DSA swap.
- **Loop discipline:** one mainline objective per round from the Tier-1 spine first (gate → feasibility → footprint → mem lift → client-SLO → AC-11 re-sweep → 64K → within-budget); Tier-2 only after the gate and the spine. Each hardware-validated step drops an artifact under `runs/<date>_dsv32_loop6/`. Push to `jimmy` at every round boundary (cluster pre-emptions). Do not re-open the DECIDED DS-fair AC-12 re-scope.

### Scope — OUT (from the draft)
No new fixture/scaffolding code; no re-opening the AC-12 DS-fair re-scope (beyond-budget recall may be *characterized* but the gate definition is fixed); Tier-2 recall R&D is out unless AC-1 opens it (it does, but strictly after the spine); deferred client requirements are out (128k ISL / 1024 OSL, nvfp4/mxfp4 weights, the performant-knobs × DS matrix, GLM-5.1 — their own downstream loops); productionization passes are out unless trivially in the way (page-size flexibility beyond 64, removing dev-only `SGLANG_DS_*` env gates, CI registration of manual gates, upstreaming/PR hygiene, **multi-node TP scaling (deferred per DEC-5; see `development/roadmap.md` §5)** — tracked in `development/roadmap.md` §5 for Loops 6–8).

<comment>CRITIQUE (Codex / NEW): This plan file contains TWO executable-looking plans. The "Original Design Draft" section below restates old ACs (`AC-L6-*`), five `*(PENDING)*` decisions, hard-done language, and concrete critical-path commands that CONTRADICT the resolved main plan above (which states none remain PENDING). In an agent-run loop, stale instructions in the same file are not harmless context — they are a live mis-execution hazard. Mark every draft section NON-AUTHORITATIVE / archived (or move it out of the plan file), so the implementer cannot act on a superseded AC-L6-* or re-litigate a resolved DEC.</comment>

--- Original Design Draft Start ---

# Loop 6 Draft — Make Double Sparsity Shippable for the Client SLO on V3.2

## Objective

Make **Double Sparsity (DS) itself** pass the immediate client SLO in
`development/CLIENT_SLOS.md` on the production workload, and decide whether to
invest further in DS on a model that already ships a native sparse indexer.

The one thing that, done, makes this loop a success:

> **DS serves the client workload (4096 ISL / 512 OSL / conc 16–64 / ~55% cache)
> at `P99 TTFT < 22 s` AND `≥ 30 TPS/req`, on real hardware, at the locked Option B
> operating point — measured as an absolute pass/fail against the client SLO, not a
> DS-vs-DSA ratio.**

Everything else in this loop is either a small hardening, a strategic decision, or
explicitly-gated recall R&D.

## Why this loop exists

Loop 5 shipped the smoke MVP + the loop4-compatible MVP and closed AC-0/1/1.1/4/6/8/9/10/1b/Q,
plus AC-11 (executed, recorded directional miss) and AC-12 (MET under the user-authorized
DS-fair re-scope). DS demonstrably serves V3.2 FP8 at TP=8, page 64, fp8 KV, radix-on.

But Loop 5 graded "MVP" against **DS-vs-DSA parity** (the internal loop-4 bar). Re-graded
against the **client's** bar (`CLIENT_SLOS.md`), the picture is sharper and the remaining gap
is small:

| Client SLO (immediate) | Target | DS measured (Loop 5 AC-11, conc 16/32/64) | Verdict |
|---|---|---|---|
| Per-request throughput | **≥ 30 TPS/req** | 34.0 / 33.9 / 33.9 tok/s (p50) | ✅ already MET |
| Tail latency | **P99 TTFT < 22 s** | 57.7 / 132.9 / 292.0 s | ❌ MISS (hard) |
| Model / knobs | V3.2 FP8, TP, CUDA graphs, radix | all enabled and recorded | ✅ MET |
| Workload | 4096 ISL / 512 OSL / conc 16–64 / ~55% cache | `benchmark.sh` shape matches exactly | ✅ MET |

**The single client-facing blocker is P99 TTFT, and it is not a speed problem** — DS per-request
generation already beats the 30 TPS SLO. It is an **admission/queue** problem: DS reserves a
per-rank `TokenLabelTable` (~8 GB/rank, fp16) on top of the ~84 GB/rank V3.2 FP8 weights, so it
must run at `mem_fraction_static=0.6` (DSA runs at 0.85). The small KV pool admits only
**14.5 / 24.6 / 35.7** of the nominal 16 / 32 / 64 concurrency, so requests queue and TTFT
explodes. Raising mem past 0.6 currently **OOMs DS during generation**.

➡️ **Shrinking the `TokenLabelTable` footprint is the lever that converts DS from "fails the
client SLO" to "shippable."** That is the spine of this loop.

A second, strategic finding governs how far past the SLO to invest: **DS cannot beat V3.2's
native DSA on long-context recall** — the shared `flashmla_kv` decode kernel hard-caps DS
selection at the model's `index_topk=2048`, and DS's offline channel-mask selector is inferior to
V3.2's *trained* DSA indexer at that budget (Loop-5 NIAH 4K/16K/64K = 75% / 5% / 0% vs DSA 100%).
DS's value is clearest on models **without** a trained sparse indexer. Decide this before spending
GPU-hours on recall R&D (DEC-1).

## How this loop differs from Loop 5

Loop 5's failure mode was building CPU scaffolding instead of *running* code; its discipline was
"every round drops a hardware artifact." Loop 6 is a **research loop with a decision gate**:

1. **Answer the strategic gate (DEC-1) first.** It gates only the expensive Tier-2 recall R&D —
   the Tier-1 engineering wins pay off regardless. Do not spend rounds on a learned selector or a
   kernel variant until the gate is decided.
2. **The engineering wins are the safe mainline.** The footprint → admission → SLO chain is the
   spine. A code-only round (e.g. the footprint change + unit tests) is fine **if** the next round
   validates it on hardware; two code-only rounds in a row with no hardware artifact is a stall.
3. **Do NOT re-litigate the DS-fair AC-12 re-scope.** It was DECIDED in Loop 5 (user-authorized).
   Loop 6 may *characterize* beyond-budget recall further, but the gate definition is settled.

**Anchor:** `loop6-base` at the Loop-5 final commit (`989975625` or later) on
`dev/double-sparsity-standalone`. The Loop-5 mask `/models/dsv32-fp8-channel-mask.safetensors`
already exists — reuse it; regenerate only if a recipe field changes. The no-env-override radix-on
path is done.

---

## Strategic decision gate (DEC-1 — decide FIRST, gates Tier 2)

**Is Double Sparsity worth pursuing past the engineering wins on a model that already ships a
trained native sparse indexer (DSA)?**

- On V3.2, DS is capped at the native `index_topk=2048` budget by the *shared* decode kernel AND
  uses an inferior *offline* selector — so it cannot match (let alone beat) DSA long-context recall
  at the shared budget.
- DS's value proposition is clearer on models WITHOUT a trained sparse indexer (relevant to the
  deferred GLM-5.1 and 128k requirements).
- This gate determines whether **Tier 2 (recall R&D)** is in scope at all. Capture the answer as a
  `DEC-1` decision doc dropped under `runs/<date>_dsv32_loop6/ds_on_v32_decision.md`.

If the gate **closes** Tier 2, that is a **legitimate Loop 6 outcome** — the loop ships the Tier-1
client-SLO MVP and explicitly records "recall R&D not pursued on V3.2 because DSA dominates at the
shared budget; revisit on a no-native-indexer model." A closed gate is not a stall.

---

## Scope — IN

### Tier 1 — engineering wins (the client-SLO spine; pay off regardless of DEC-1)

1. **TokenLabelTable footprint reduction** *(handoff #2 — THE client-SLO blocker)*.
   Shrink the per-rank `TokenLabelTable`
   (`python/sglang/srt/layers/attention/double_sparsity/token_label_table.py`, ~8 GB/rank fp16
   today) so DS can serve at a higher `mem_fraction_static` without the generation-time OOM seen at
   0.7. Candidate levers (pick the minimum that works; do not over-engineer): int8-symmetric
   signatures with per-layer/slot/head scales applied at scoring, a narrower `label_dim`, or a
   tighter slot model. **Selection numerics must be preserved** — the change must not regress DS
   selection/recall. Keep fp16 as the default behind a flag until the compact path has unit + hardware
   evidence.

2. **mem_fraction lift + no-OOM validation** *(handoff #2)*. With the smaller table, boot DS at a
   higher `mem_fraction_static` (target decided in DEC-4; e.g. 0.8) and prove `max_total_num_tokens`
   rises with **no generation-time OOM** under sustained long `/generate`.

3. **⭐ Direct client-SLO validation** *(NEW — the loop's done-criterion)*. Run the **full** client
   workload (`benchmark.sh` at `NUM_PROMPTS=320`, conc 16 / 32 / 64) and assert **absolute
   `P99 TTFT < 22 s` and `≥ 30 TPS/req`** for DS. This is the artifact that says "DS is shippable for
   the client." (Loop 5's AC-11 used `NUM_PROMPTS=64` and reported only DS/DSA ratios — insufficient
   for an absolute SLO claim.)

4. **AC-11 directional re-sweep at the lifted mem fraction** *(handoff #3, DEC-7 from Loop 5)*.
   Re-run the 3-trial DS+DSA sweep (conc 16/32/64, 120 s warmup / 600 s window) at the new operating
   point; confirm DS achieved concurrency now tracks nominal; update `ac11_analysis.md` verdict.
   (Expected to improve once admission is fixed; the DS-vs-DSA *parity* miss at conc 16/32 shares the
   same root cause as the TTFT miss.)

5. **64K servability** *(handoff #2; also unblocks the deferred 128k requirement)*. At the lifted
   mem fraction, confirm a 64K-context `/generate` no longer returns HTTP 400 (or document the new
   admission ceiling). This is a *servability* win; 64K recall accuracy is separate (Tier 2).

6. **AC-12 within-budget gate from real token counts** *(handoff #4 / Codex queued #1)*. Change the
   harness to assert `within_budget` from the **actual** `usage.prompt_tokens` (or tokenized chat
   length), not the 1024/1536 **word-count** proxy. Rename `length_tokens`→`length_words` or add an
   `input_tokens` field. Must **not** change the DECIDED DS-fair gate definition; re-run the gate and
   diff against the word-count proxy to show it was safe (or correct it).

### Tier 2 — DS long-context recall R&D (GATED on DEC-1; expect to defer to a dedicated loop)

7. **DS long-context recall** *(handoff #1)* — **only if DEC-1 opens it.** A `flashmla_kv` decode-kernel
   variant accepting `top_k > index_topk` (today asserts `indices.shape[-1] == dsa_index_topk` in
   `dsa_backend.py`) AND/OR a query-aware / learned DS selector that places the needle in the 2048
   budget. Measure NIAH 4K/16K/64K recall delta vs the Loop-5 baseline DS 75% / 5% / 0%. This is
   GPU- and engineering-heavy — a new kernel variant or a distilled selector is not a one-round task.

## Scope — OUT

- **No new fixture/scaffolding code.** Loop 5's harnesses, comparator, and serve/bench scripts are
  the tools; use them, don't rebuild them.
- **Do NOT re-open the AC-12 DS-fair re-scope.** Settled in Loop 5. Beyond-budget recall may be
  *characterized* further but the gate definition is fixed.
- **Tier 2 recall R&D is OUT unless DEC-1 explicitly opens it.**
- **Deferred client requirements are OUT of this loop:** 128k ISL / 1024 OSL, nvfp4/mxfp4 weights,
  the performant-knobs × DS matrix (DP Attention, MTP/EAGLE, EP, explicit/mixed chunked prefill,
  overlap scheduling, piecewise CUDA graph), and GLM-5.1. They are their own downstream loops.
- **Productionization passes are OUT** unless trivially in the way: page-size flexibility beyond 64,
  removing dev-only `SGLANG_DS_*` env gates, CI registration of the manual gates, upstreaming/PR
  hygiene, multi-node TP scaling. Tracked in `development/roadmap.md` §5 for Loops 6–8.

---

## Acceptance criteria (draft — gen-plan will formalize positive/negative tests)

Each AC drops an artifact under `runs/<date>_dsv32_loop6/` (e.g. `runs/20260530_dsv32_loop6/`).

- **AC-L6-0 (DEC-1 strategic gate)** — *analyze*. A decision doc `ds_on_v32_decision.md` records:
  pursue Tier-2 recall R&D on V3.2 or not, with the `index_topk`/shared-kernel/selector rationale,
  and the explicit consequence for Tier 2. **Positive:** the doc states a decision and its Tier-2
  consequence. **Negative:** starting any Tier-2 R&D before this doc exists is out of order.

- **AC-L6-1 (TokenLabelTable footprint)** — *coding*. The per-rank table memory is reduced by the
  target factor, and DS selection numerics are preserved. **Positive:** a unit test shows the compact
  table's selected-token set matches the fp16 baseline within tolerance on a synthetic shape, and a
  measured per-rank byte count drops by the target. **Negative:** any selection divergence beyond
  tolerance, or the compact path becoming the default before hardware validation, fails.

- **AC-L6-2 (mem-fraction lift, no OOM)** — *hwrun*. DS boots at the higher `mem_fraction_static`
  (DEC-4 target) and survives a sustained long `/generate`. **Positive:** a mem-fraction sweep log
  (0.6 → … → target) shows `max_total_num_tokens` rising and a long-context generation completing
  with **no generation-time OOM**; `/get_server_info` recorded. **Negative:** a generation-time OOM
  at the target mem fraction fails (the table is still too big — iterate or reconsider the admission
  model).

- **AC-L6-3 (⭐ client-SLO validation — the done-criterion)** — *hwrun*. Full client workload
  (`benchmark.sh`, `NUM_PROMPTS=320`, conc 16 / 32 / 64). **Positive:** DS **absolute
  `P99 TTFT < 22 s`** AND **`≥ 30 TPS/req`** at conc 16 and 64; a `client_slo_report.md` records the
  absolute numbers vs the SLO with valid `.meta.json` sidecars. **Negative:** any conc with
  `P99 TTFT ≥ 22 s` or `< 30 TPS/req` fails the client-SLO MVP claim (record it as a follow-up with
  the admission/compute breakdown).

- **AC-L6-4 (AC-11 directional re-sweep)** — *hwrun*. 3-trial DS+DSA sweep at the lifted operating
  point, radix-on both sides, per-side `mem_fraction_static` consistency enforced. **Positive:** DS
  achieved concurrency tracks nominal (≈100%); comparator emits an updated TPS/TTFT summary and
  `ac11_analysis.md` is refreshed from the new artifacts. **Negative:** a sweep that hides
  queue-dominated admission (achieved ≪ nominal without disclosure) is invalid.

- **AC-L6-5 (64K servability)** — *hwrun*. **Positive:** a ~70K-token `/generate` returns 200 (no
  HTTP 400) at the lifted mem fraction, with the served `max_total_num_tokens` recorded; OR a
  documented new admission ceiling if 64K still doesn't fit. **Negative:** silently re-recording the
  Loop-5 HTTP 400 without the lifted-mem retry fails.

- **AC-L6-6 (AC-12 within-budget from real token counts)** — *coding*. **Positive:** the harness
  records `usage.prompt_tokens` per NIAH prompt and asserts `within_budget` from it; the re-run gate
  still PASSES (DS-fair definition unchanged) and a diff shows the word-count proxy was safe (or is
  corrected). **Negative:** any change that alters the DS-fair gate thresholds/definition fails.

- **AC-L6-7 (Tier 2 recall R&D — GATED)** — *coding/hwrun*, **only if AC-L6-0 opened it.**
  **Positive:** a selector or `top_k > index_topk` kernel-variant change with a NIAH 4K/16K/64K
  recall delta artifact showing movement vs DS 75% / 5% / 0%, and the TPS/TTFT cost recorded.
  **Negative:** starting this before AC-L6-0, or letting it block the Tier-1 spine, fails the loop
  discipline.

---

## Pending user decisions (resolve in gen-plan discussion mode)

- **DEC-1 — Strategic gate (above).** Pursue Tier-2 recall R&D on V3.2, or cap at Tier-1 engineering
  wins? Gates AC-L6-7 and the downstream 128k/GLM framing. *(PENDING — decide early.)*
- **DEC-2 — "Shippable" definition.** Is the deliverable "DS meets the client SLO *itself*," or "DS
  available as an opt-in knob while DSA is the default that meets the SLO"? (DSA already meets both
  SLOs trivially.) *(PENDING.)*
- **DEC-3 — TTFT target source.** Confirm the client SLO is **absolute `P99 TTFT < 22 s`** at the
  client workload (not a DS-vs-DSA ratio), validated at full `NUM_PROMPTS=320`. *(PENDING.)*
- **DEC-4 — Footprint approach + target mem fraction.** Which compaction lever (int8 signatures /
  narrower `label_dim` / tighter slot model), the target `mem_fraction_static` to validate
  (0.7 / 0.8?), and the OOM-safety bar. *(PENDING.)*
- **DEC-5 — Deployment topology.** Single-node TP=8 vs multi-node for the client deliverable.
  *(PENDING — affects how the SLO is validated.)*

---

## Hardware

See `CLUSTER.md` (2-node 8×H200; node 0 local `h200-10-220-51-16`, node 1
`h200-10-220-51-5` via `ssh double-sparsity` / `rx devbox exec double-sparsity --no-tmux --rank 1`).
DSv3.2 FP8 weights at `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`. Loop-5 mask already on
disk at `/models/dsv32-fp8-channel-mask.safetensors` (regenerate only if a recipe field changes).
Default ports: workers 30001, router 30000, prometheus 29000.

---

## Critical path (concrete commands)

All runs use the **Option B operating point** encapsulated by the `serve_*.sh` / `benchmark*.sh`
scripts (TP=8, fp8 KV, page 64, `flashmla_kv` prefill+decode, overlap-schedule + piecewise-cuda-graph
disabled, radix-on via the config-bound fixture state). `mem_fraction_static` is the lever Loop 6
deliberately moves. Don't hand-roll `launch_server`.

```bash
# 0. Sanity
nvidia-smi --query-gpu=index,name,memory.free --format=csv
ls -la /models/dsv32-fp8-channel-mask.safetensors        # reuse Loop-5's mask

# 0a. DEC-1 strategic gate — write runs/<date>_dsv32_loop6/ds_on_v32_decision.md FIRST.

# 1. TokenLabelTable footprint (handoff #2) — edit
#    python/sglang/srt/layers/attention/double_sparsity/token_label_table.py
#    (+ token_label_write.py quantize-on-write, selection_kernel.py apply-scales),
#    then prove selection equivalence + reduced bytes:
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q

# 2. mem-fraction lift + no-OOM sweep (handoff #2)
for MF in 0.6 0.7 0.8; do
  MEM_FRACTION_STATIC=$MF bash development/serve_double_sparsity.sh \
    2>&1 | tee development/logs/ds_memfrac_${MF}_$(date +%Y%m%d-%H%M%S).log
  # check /get_server_info -> max_total_num_tokens; fire a long /generate to flush OOM
done

# 3. 64K servability probe at the lifted mem fraction (handoff #2)
curl -s -X POST http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d @development/loop6/probe_64k.json | python -c "import sys,json;print(json.load(sys.stdin).get('meta_info',{}))"

# 4. ⭐ DIRECT CLIENT-SLO VALIDATION — full workload, absolute P99 TTFT < 22s, >= 30 TPS/req
NUM_PROMPTS=320 MEM_FRACTION_STATIC=<lifted> \
MODE=double_sparsity CONCURRENCIES="16 32 64" bash development/benchmark.sh
#   -> write runs/<date>_dsv32_loop6/client_slo_report.md with the ABSOLUTE numbers vs the SLO.

# 5. AC-11 directional re-sweep at the lifted mem fraction (handoff #3, DEC-7)
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 MEM_FRACTION_STATIC=<lifted> \
MODE=native_nsa CONCURRENCIES="16 32 64" bash development/benchmark_baseline.sh
TRIALS=3 WARMUP_SECONDS=120 MEASUREMENT_WINDOW_S=600 MEM_FRACTION_STATIC=<lifted> \
MODE=double_sparsity CONCURRENCIES="16 32 64" bash development/benchmark.sh
python development/benchmark_compare.py --ac11 \
  --baseline development/results/native_nsa_gsp_isl4096_osl512_c64_t3.jsonl \
  --ds       development/results/double_sparsity_gsp_isl4096_osl512_c64_t3.jsonl \
  --output   runs/$(date +%Y%m%d)_dsv32_loop6/ac11_resweep.md

# 6. AC-12 within-budget gate from REAL token counts (handoff #4) — edit the harness, then re-run:
DS_BASE_URL=http://127.0.0.1:30000 DSA_BASE_URL=http://127.0.0.1:30001 \
  pytest test/manual/test_double_sparsity_v32.py -v

# 7. (Tier 2, ONLY if DEC-1 opened it) recall R&D
#    flashmla_kv asserts indices.shape[-1] == dsa_index_topk in
#    python/sglang/srt/layers/attention/dsa_backend.py — that is the hard cap to relax for a
#    top_k > index_topk variant. A learned selector instead reshapes the offline channel mask.
#    Measure NIAH 4K/16K/64K recall delta vs DS 75/5/0.
```

> **Killing servers between bench runs:** `pkill -f sglang_router` does NOT catch the Rust process
> (renamed to `sglang::router`). Use `pkill -f 'sglang::router'` (or match the worker
> `python -m sglang.launch_server` pattern) so the old router doesn't hold the port across the DS↔DSA swap.

---

## Acceptance evidence — what "Loop 6 done" looks like

A directory `runs/<date>_dsv32_loop6/` containing:

- `ds_on_v32_decision.md` — the DEC-1 strategic-gate decision.
- The TokenLabelTable footprint change with a per-rank byte-count measurement + selection-equivalence
  unit-test result.
- A `mem_fraction_static` sweep log showing higher `max_total_num_tokens` and **no generation-time OOM**.
- **`client_slo_report.md`** — the headline: DS absolute `P99 TTFT` and `TPS/req` at the full client
  workload vs the `< 22 s` / `≥ 30 TPS` SLO, with bench JSONLs + `.meta.json` sidecars.
- `ac11_resweep.md` — the refreshed DS-vs-DSA directional verdict at the lifted operating point.
- A 64K servability result (served, or a documented admission ceiling).
- The AC-12 within-budget re-run asserting from real token counts + a diff vs the word-count proxy.
- If DEC-1 opened Tier 2: a NIAH 4K/16K/64K recall-delta artifact vs DS 75% / 5% / 0%.

**Tier-1 done (client-SLO MVP):** "DS now serves the client workload at `P99 TTFT < 22 s` and
`≥ 30 TPS/req` after the TokenLabelTable footprint reduction lifted `mem_fraction_static` and
restored full admission; 64K is servable (or characterized); the AC-12 within-budget gate asserts
from actual token counts; the strategic gate on Tier-2 recall R&D is decided."

**Tier-1 + Tier-2 done (only if DEC-1 opened it):** the above plus "DS recall at the
widened/learned budget moved from the Loop-5 baseline DS 75% / 5% / 0%, and the TPS/TTFT cost is recorded."

---

## Risks + likely failure modes

1. **Footprint shrink regresses selection (int8 quant of signatures).** The compact table could
   change the top-k selection vs fp16 and drop recall/quality. Mitigation: a selection-equivalence
   unit test against the fp16 baseline; keep fp16 the default behind a flag until hardware-validated;
   re-run AC-Q / AC-12-within-budget after the change.
2. **mem-fraction lift still OOMs at the target.** The table size may not be the *only* lever — the
   admission model itself may need adjustment. Mitigation: read the OOM verbatim; if footprint alone
   doesn't reach the target after ~2 rounds, reconsider the admission/KV-budget model (see Step 8
   stagnation signals in `development/loop6/runbook.md`).
3. **P99 TTFT still > 22 s at full nominal concurrency even after admission is fixed.** If at conc 64
   the bottleneck shifts from admission-queue to prefill compute (4096 ISL × 64), TTFT may still miss.
   Mitigation: AC-L6-3 must break down admission-wait vs prefill-compute; if prefill-bound, that's a
   different (chunked-prefill / scheduling) follow-up, surfaced honestly rather than hidden.
4. **DS-vs-DSA TPS parity at conc 16/32 still misses** even with full admission. The client SLO is
   **absolute 30 TPS/req (met)**, so this is secondary — record it as a DEC-7 directional follow-up,
   don't let it block the client-SLO MVP claim.
5. **Tier-2 recall R&D gets started before DEC-1.** The expensive trap. Mitigation: AC-L6-0 is a hard
   prerequisite; a closed gate is a legitimate outcome, not a stall.

## Loop-runner notes

- One mainline objective per round, taken from the Tier-1 spine first (gate → footprint → mem lift →
  client-SLO validation → AC-11 re-sweep → 64K → AC-12 token-count); Tier 2 only after the gate opens.
- A code-only round (footprint change + unit tests; AC-12 harness edit) is acceptable **if** the next
  round validates on hardware. Two code-only rounds in a row with no `runs/<date>_dsv32_loop6/`
  artifact is a stall.
- Reuse the Loop-5 mask, serve/bench scripts, comparator, and quality harnesses — don't rebuild them.
- Implementation code/comments must NOT contain plan-process markers (`AC-`, `DEC-`, "Tier",
  "Option B", "Round N") — use behavior-based naming; those markers live in this plan doc only.
- Push to `jimmy` at every round boundary (cluster pre-emptions).
</content>

--- Original Design Draft End ---
