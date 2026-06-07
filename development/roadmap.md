# Double Sparsity (DS) on DeepSeek-V3.2 — Master Roadmap

> Written 2026-05-30, after Loop 5 (`.humanize/rlcr/2026-05-28_10-17-12`) closed.
> Source material: Loop 5 plan/draft/goal-tracker/round-summaries, the Loop-5 handoff
> `runs/20260528_dsv32_mvp/next_loop_issues.md`, the AC-11/AC-12 analyses, and
> `development/CLIENT_SLOS.md`. This is the single checklist we mark off going forward.
>
> **Loop numbering note:** Loop 5 is **complete**. The next loop is **Loop 6** (a stub
> already exists at `development/loop6/`). Where the original ask said "loop 5 MVP," it
> means *the next loop* — captured here as **Loop 6 = Client-SLO MVP**.

---

## 0. TL;DR — the one finding that reframes everything

Loop 5 measured "MVP" as **DS-vs-DSA parity** (the internal loop-4 bar). But the **client's**
bar is `development/CLIENT_SLOS.md`, and against *that* bar DS is much closer than the
Loop-5 "AC-11 FAIL / AC-12 re-scoped" framing implies:

| Client SLO (immediate) | Target | DS measured (Loop 5, AC-11, conc 16/32/64) | Verdict |
|---|---|---|---|
| Per-request throughput | **≥ 30 TPS/req** | **34.0 / 33.9 / 33.9 tok/s** (p50) | ✅ **MET** at all conc |
| Tail latency | **P99 TTFT < 22 s** | **57.7 / 132.9 / 292.0 s** | ❌ **MISS (hard)** |
| Model | V3.2 FP8 | running, TP=8, page 64, fp8 KV, radix-on | ✅ MET |
| Workload | 4096 ISL / 512 OSL / conc 16–64 / ~55% cache | `benchmark.sh` shape matches exactly | ✅ MET |
| Knobs | TP, CUDA graphs, radix cache | all enabled and recorded | ✅ MET |

**The single client-facing blocker is P99 TTFT**, and it is **not a generation-speed problem**
(per-token TPS already beats the SLO). It is an **admission/queue** problem: DS reserves a
per-rank `TokenLabelTable` on top of the ~84 GB/rank FP8 weights, so it must run at
`mem_fraction_static=0.6` (vs DSA's 0.85). The small KV pool admits only **14.5 / 24.6 / 35.7**
of the nominal 16 / 32 / 64 concurrency, so requests queue and TTFT explodes. Raising mem past
0.6 currently **OOMs DS during generation**.

➡️ **Shrinking the `TokenLabelTable` footprint is the lever that converts DS from "fails the
client SLO" to "shippable."** It is already the #2 carried-over item; this roadmap promotes it
to the **mainline of the next loop**. Everything else for the immediate client deliverable is
already done.

A second, strategic finding (does not block the SLO but governs how much further to invest):
**DS cannot beat V3.2's native DSA on long-context recall**, because the shared `flashmla_kv`
decode kernel hard-caps DS selection at the model's `index_topk=2048` and DS's offline
channel-mask selector is inferior to V3.2's *trained* DSA indexer at that same budget. DS's
value proposition is strongest on models that **don't** ship a trained sparse indexer. See §7.

---

## 1. State of the Accuracy Test

Two accuracy harnesses exist. Both currently **PASS** at their (current) gate definitions.

### AC-Q — paired quality smoke (TIER 1) — ✅ PASS
- `test/manual/test_dsv32_quality_smoke.py` (sequential: capture DSA refs, then DS), 20+5 prompts.
- Result (concise measurement, user-approved): `all_pass=true` — prefix-match **0.95**,
  mean ROUGE-L **0.944**, NIAH-mini **5/5**, first-8-token divergence **0**.
- Artifact: `runs/20260528_dsv32_mvp/dsv32_quality_smoke_concise.json`.

### AC-12 — full quality gate (TIER 2) — ✅ PASS (DS-fair re-scope) / ❌ original long-context FAIL (characterized)
- `test/manual/test_double_sparsity_v32.py`. **HARD gates (DS-fair, Round-14 user-authorized re-scope):**
  - MMLU 5-shot: DS **89.00%** == DSA **89.00%** (Δ 0.00 pp; ≤ 1 pp) ✅
  - NIAH within selection budget (≤ `index_topk`=2048 tok): @1024 words **100%==100%**, @1536 words **100%==100%** ✅
- **Beyond-budget (original AC-12 NIAH 4K/16K/64K) — recorded `verdict=FAIL`, characterized, NOT gated:**
  - 4K: DS **75%** vs DSA 100% · 16K: DS **5%** vs 100% · 64K: DS **0%** (HTTP 400, unservable at mem 0.6)
  - Root cause: top_k kernel-locked at 2048 + offline selector inferior to trained DSA indexer (NOT a decode bug — DS recalls 100% when its selection is dense; MMLU == DSA).
- Artifacts: `runs/20260528_dsv32_mvp/ac12_analysis.md`, `ac12_results/` (+ `superseded_prerescope/`).

**Accuracy gap to client SLO:** none for the immediate deliverable (4096 ISL). The long-context
recall gap only matters for the **deferred** 128k-ISL requirement (§6, Loop 7).

---

## 2. State of the Performance Test

### AC-11 — 3-trial directional comparator (radix-on) — ❌ recorded directional FAIL (DEC-7)
Shape: conc 16/32/64, 3 trials, 120 s warmup, 600 s window, GSP ISL≈4096/OSL 512, ~55% cache,
`NUM_PROMPTS=64`, both sides radix-on. Artifact: `mvp_compare_ac11.md`, `ac11_analysis.md`.

| Conc | DSA TPS p50 | DS TPS p50 | TPS ratio (gate ≥0.95) | DSA P99 TTFT | DS P99 TTFT | TTFT gate (≤1.1×) |
|---|---|---|---|---|---|---|
| 16 | 46.88 | **34.04** | 0.726 ❌ | 0.73 s | **57.7 s** | ❌ |
| 32 | 37.64 | **33.88** | 0.900 ❌ | 1.37 s | **132.9 s** | ❌ |
| 64 | 29.60 | **33.92** | 1.146 ✅ | 2.04 s | **292.0 s** | ❌ |

Effective vs nominal concurrency (the cause): DS achieves **91% / 77% / 56%** of nominal; DSA ~100%.

**Interpretation:**
- DS per-request **generation rate is competitive-to-better** (beats DSA at conc 64). The 30 TPS/req
  client SLO is **already met**.
- The DS-vs-DSA TPS *parity* miss at conc 16/32 and the **catastrophic TTFT miss at all conc** are
  the **same root cause**: mem-0.6 admission starvation from the `TokenLabelTable` footprint.
- This is recorded per DEC-7 as a directional follow-up, not a build break.

**Not yet measured against the client SLO directly:** the AC-11 run used `NUM_PROMPTS=64`, not the
plan-locked `320`. A clean client-SLO validation should re-run at the full prompt count and report
**absolute** P99 TTFT vs the 22 s bar (not just the DS/DSA ratio).

---

## 3. Completed milestones (Loops 1–5) — ✅ checked off

### Loop 1 — Decide & restart
- [x] Captured client requirements; chose to **restart DS from scratch** on `dev/double-sparsity-standalone` (vs resuming legacy/rewrite PRs).
- [x] Established source lineage (Twilight, legacy SGL DS, original DoubleSparse, the paper; HiSparse as a shipped-sparsity reference).

### Loop 2 — Structural plumbing (M1-C)
- [x] Page-table adapter (selector tuple → FlashMLA `block_table`); removed `NotImplementedError` DS branch + `SGLANG_DS_ALLOW_NO_ADAPTER` gate.
- [x] Scheduler `customized_info` glue → DS metrics surface in per-request `meta_info`.
- [x] M3-B page-stability fixture + CI hook; operator RUNBOOK. **150 unit tests.** (Hit circuit breaker at R9 — remaining items hardware-gated.)

### Loop 3 — "Actually serve" (planned, **never ran**)
- [x] Scoped: live PageSignatureTable population, per-request ownership mask, end-to-end bench. *(Carried into Loop 4; loop itself produced no hardware run.)*

### Loop 4 — Architecture rotation + MVP scaffolding (**never executed on hardware**)
- [x] Rotated page-level → **token-level signatures at page_size=64** (`token_label_table.py`, `token_label_write.py`, `compute_token_scores`, thin page-table adapter); `top_k` default 2048 == model `index_topk`.
- [x] Built comparator validation, bench_serving timing path, M3-B fixtures, AC-12 harness, validator helpers. *(Drifted: all-CPU, no hardware run; the channel mask never existed on disk — the true root blocker for Loop 5.)*

### Loop 5 — Hardware execution: TIER-1 smoke + loop4-compatible MVP
- [x] **AC-0** — `forward_batch` threaded into `_write_token_labels` (4 call sites); radix-capture extend-snapshot publishes on hardware; producer regression.
- [x] **AC-4** — Calibration redesigned to **native-FP8 device-sharded** load (deepseek_v32→v3 remap; DeepGEMM-hub bypass; fail-closed dry-run). **Mask generated:** `/models/dsv32-fp8-channel-mask.safetensors` (SHA `7b3207ca…`, L=61 H=128 label_dim=16).
- [x] **AC-1 / AC-1.1** — DS boots TP=8 with mask; `/get_server_info` + `/generate` verified; genuine sparsity (`0<rate<1`, `dense_fallback=0`) on a >2048-tok prompt; invalid-mask fail-closed. *(Fixed decode degeneration: `req_to_token` resolve + decode label-write `kv_b_proj`/`head_width`.)*
- [x] **AC-6** — Regular CUDA-graph capture recorded at first boot (52 batch sizes); piecewise disabled (distinct).
- [x] **AC-8 / AC-9** — Smoke DS+DSA benchmarks (TRIALS=1, radix-off both, labeled non-AC-11) + comparator.
- [x] **AC-Q** — Sequential paired quality smoke PASS (concise measurement; 4 gates, hardened matcher).
- [x] **AC-10** — No-env-override radix flip (`--double-sparsity-radix-fixture-artifact` + config-bound state file); both fixtures pass; `--disable-radix-cache` removed; DS boots radix-on.
- [x] **AC-1b** — Chunked-prefill probe PASS at radix-on point (8192+2432 genuine multi-chunk; needle recalled); kept on both sides.
- [x] **AC-11** — 3-trial radix-on sweep + comparator EXECUTED; directional MISS recorded (DEC-7) with effective-vs-nominal accounting (§2).
- [x] **AC-12** — Re-scoped DS-fair gate PASS; beyond-budget degradation characterized (§1).
- [x] **Evidence bundle** assembled (`runs/20260528_dsv32_mvp/evidence_bundle.md`); 411 CPU tests green; calibration provenance recorded.

**Net:** DS demonstrably serves V3.2 FP8 at the Option B operating point with comparable quality
and competitive per-request throughput. **Outstanding for the *client SLO*: P99 TTFT only.**

---

## 4. LOOP 6 — Client-SLO MVP ("make DS shippable") — ✅ DONE (Minimum Acceptable Scope, 2026-05-31)

**Goal:** make DS *itself* (not just DSA) pass `CLIENT_SLOS.md` on the immediate workload, and
decide whether to invest further. The spine is the **admission/TTFT fix**; everything else here is
a small hardening or a strategic decision. Executed as RLCR loop `.humanize/rlcr/2026-05-30_06-27-19`
(plan `development/loop6/refined_plan_v1.md`).

> **Loop-6 outcome (Lower Bound met).** The Tier-1 engineering spine **landed** and the admission/TTFT
> blocker is **fixed at conc-16**: a **compact int8 TokenLabelTable** (≈1.78× smaller; ~6.48 GB/rank) lets DS
> boot+serve at **`mem_fraction_static`=0.7** (KV pool `max_total_num_tokens`=396096) with generation headroom.
> At the full-context Option-B point DS hits **conc-16 P99 TTFT 13.13 s < 22** (the Loop-5 hard blocker —
> was 57.7 s). Per-request **TPS** is the characterized structural decode-batch ceiling (24.9/19.5/17.3 at
> conc 16/32/64; DS ≤ DSA; **conc-64 ≥30 unattainable even for DSA at 29.4**), and an R24 microbench proved
> **no full-context top-k design reaches conc-16 ≥30**. AC-5 therefore closed **directional (DEC-3)** — spine
> validated + strict misses recorded with measured attribution, **not a strict/shippable all-conc pass**. The
> **strict all-concurrency SLO** (≥30 TPS at every conc) and **Tier-2 recall R&D (§4.4 → Loop 7)** are the
> deferred downstream/own-loop work. As-built detail:
> [`development/past_implementations/study/08-current-system-architecture.md`](past_implementations/study/08-current-system-architecture.md).
> Evidence: `runs/20260530_dsv32_loop6/`.

### 4.0 Strategic gate (decide FIRST — gates Tier-2 R&D for this and later loops) — ✅ DECIDED
- [x] **DEC: Is DS worth pursuing on a DSA-native model (V3.2)?** **RESOLVED: pursue Tier-2 recall R&D, but
  strictly AFTER the Tier-1 spine lands** — recorded in `runs/20260530_dsv32_loop6/ds_on_v32_decision.md`
  (Loop-6 AC-1). Rationale confirmed on hardware: DS is capped at native `index_topk=2048` by the shared
  kernel AND uses an inferior offline selector, so it cannot match DSA long-context recall at the shared
  budget; DS's value is clearer on models without a trained indexer. **The Tier-1 spine has landed → this
  gate is now OPEN for Loop 7 (§6).**

### 4.1 ⭐ THE client-SLO blocker — TokenLabelTable footprint → mem fraction → admission (handoff #2) — ✅ DONE (directional)
- [x] **Shrank the per-rank `TokenLabelTable` via int8-symmetric signatures** (+ per-`(layer,slot,head)` fp16
  scales applied at scoring): ≈1.78× smaller, ~6.48 GB/rank at the lifted point. Implemented across
  `token_label_table.py` (storage), `token_label_write.py` (quantize-on-write), `selection_kernel.py`
  (scale-aware scoring), `serve_double_sparsity.sh` (`SIGNATURE_DTYPE`), + radix-fixture fail-closed on dtype.
  `label_dim` narrowing was NOT chosen (DEC-4); page-level escalation not needed (int8 sufficient).
- [x] **mem-fraction lift to 0.7 with no generation-time OOM** — full HBM budget closed; `max_total_num_tokens`
  rises to 396096; NVML plateau + no-OOM long-generate proven (`runs/20260530_dsv32_loop6/`, AC-4).
- [x] **Admission restored / client-SLO validated at conc-16**: at the **full-context** point DS hits
  **P99 TTFT 13.13 s < 22** (conc-16), with a fail-closed verifier (recompute-from-raw) + measured
  admission-vs-prefill attribution (`ac5_fullctx/`). conc-32/64 TTFT and all-conc **TPS** miss strict and are
  **characterized** as the structural decode-batch ceiling (DS ≤ DSA; conc-64 ≥30 unattainable even for DSA).
  **AC-5 closed directional (DEC-3) — not a strict/shippable all-conc pass.** The strict all-conc SLO is
  carried downstream (§6 note).

### 4.2 64K servability (side-effect of 4.1; also unblocks deferred 128k) — (handoff #2) — ✅ DONE
- [x] At the lifted mem fraction a 64K-context `/generate` **serves** (no HTTP 400) — servability win recorded
  (`runs/20260530_dsv32_loop6/ac8_servability/ac8_64k_servability.md`, AC-8). Recall at 64K is separate (Tier 2).

### 4.3 Accuracy-harness hardening — (handoff #4 / Codex queued #1) — ✅ DONE
- [x] AC-12 within-budget gate asserts `within_budget` from **actual** token counts (not the word-count proxy);
  the DECIDED DS-fair gate definition was **not** changed.

### 4.4 Tier-2 — DS long-context recall R&D (GATED on 4.0) — ⏩ DEFERRED to Loop 7 (gate OPEN, high priority)
- [→] **Deferred to its own loop** (owner R24 close; plan Lower Bound: Tier-2 moves to its own loop if the
  Tier-1 spine consumes Loop 6 — it did). The strategic gate (§4.0) is **open**. The work — a `flashmla_kv`
  decode-kernel variant accepting `top_k > index_topk` (relaxing `indices.shape[-1] == dsa_index_topk`)
  **and/or** a query-aware / learned DS selector — is the **high-priority Loop-7 mainline**, drafted at
  [`development/loop7/draft.md`](loop7/draft.md). Target: NIAH 4K/16K/64K recall delta vs DS baseline
  75% / 5% / 0%. *(GPU- and engineering-heavy.)*

**Loop 6 done = client-SLO MVP (Minimum Acceptable Scope met):** §4.1 landed the admission/TTFT spine with
**conc-16 P99 TTFT 13.13 s < 22** at full context (AC-5 **directional**, DEC-3 — TPS the characterized
structural ceiling); §4.0 strategic gate **decided** (and now open); §4.2/§4.3 **done**; Tier-2 §4.4
**explicitly deferred to Loop 7** (a deferral with an open gate + a written draft, not a stall). The **strict
all-concurrency SLO** (≥30 TPS at every conc) is carried downstream — structurally DS ≤ DSA and conc-64 ≥30
is unattainable even for DSA.

---

## 5. Cross-cutting / productionization / tech-debt (schedule across Loops 6–8)

These are not tied to one client requirement but must land before "shippable to the client's prod":

- [ ] **Page-size flexibility.** Client says the impl "should support different page sizes" (64 strongly
  preferred but not hard). Everything to date is locked at page 64 — add a probe/test that DS boots and
  serves at ≥1 other page size, or document the constraint.
- [ ] **Remove dev-only scaffolding / env overrides from the production path.** Audit `SGLANG_DS_*` env
  gates (`SGLANG_DS_ALLOW_TOPK_MISMATCH`, `SGLANG_DS_RADIX_FIXTURE_CAPTURE`, any remaining override) —
  keep only the no-env-override radix path (AC-10) for production; fence the rest behind clearly-labeled
  ablation/debug flags.
- [ ] **CI registration of the hardware/manual gates.** `test/manual/test_double_sparsity_v32.py` and
  `test_dsv32_quality_smoke.py` are manual (hardware-gated); decide what a GPU-CI smoke looks like so
  regressions are caught without a full hand-run. (411 CPU unit tests already in CI.)
- [ ] **Upstreaming / PR hygiene.** Plan the merge to mainline SGLang: squash the loop scaffolding,
  remove `development/loop*/` and `runs/*` from the shippable diff, write the PR/reviewer guide
  (loop1 has a `PR_DESCRIPTION.md`/`REVIEWER_GUIDE.md` template to follow).
- [ ] **Calibration / mask provenance for re-runs and other models.** The mask is V3.2-specific and
  carries no provenance field (caught only indirectly via AC-1.1). Add a provenance field or a
  documented re-calibration recipe so the next model isn't a from-scratch archaeology dig.
- [ ] **Multi-node / TP scaling story (deferred downstream requirement — dedicated future loop).**
  DECIDED in Loop 6 (DEC-5): the client deliverable is validated **single-node TP=8 only**; **multi-node
  TP scaling is deferred to a dedicated future loop** and tracked here as a downstream requirement. All
  Loop-5 serving was single-node TP=8 (node 1 used only for the cross-node AC-12). That future loop
  validates DS multi-node (e.g. TP=8 × N replicas behind the router/SMG, or a larger TP world size)
  against the **same unchanged** client SLO — the SLO numbers do not change across topologies.
- [ ] **Comparator per-side `mem_fraction_static` check** is in (Round 13); keep it green when 4.1 moves
  the mem fraction.

---

## 6. Downstream loops — deferred CLIENT requirements (ordered by client priority)

From `CLIENT_SLOS.md` "Deferred Client requirements ordered from most important to least." Each is a
candidate RLCR loop; sizing/dependencies noted.

> **Re-prioritization (2026-06-07):** with Loop 7 landed, the client pulled **GLM-5.1 (Loop 10 below,
> deferred client #1)** forward as the **next active loop** — ahead of the original Loops 8/9 (nvfp4, knob
> compat), which are deferred behind it. The GLM-5.1 loop is drafted (out of roadmap order) on disk at
> [`development/loop8/draft.md`](loop8/draft.md) — i.e. **disk `loop8` = roadmap Loop 10**.

### Loop 7 — ✅ LANDED — DS long-context RECALL R&D (Tier-2 / AC-10) (closed 2026-06-02)
*Outcome:* gap **rigorously characterized + partially closed** — M0 oracle attributed the regimes (4K
budget-limited / 16K budget-partial ~46% cap / 64K scorer-limited); the Tier-2.B hybrid scorer lifted 16K
**6%→38%** (decode-free), the opt-in production-ready Tier-2.A lifted-budget path recovered 4K **75%→95%**,
DSA default + Tier-1 op-point non-regressed. Final decision record supersedes the gate's Tier-2.A-primary
ordering (Tier-2.B primary long-context, Tier-2.A bounded 4K lever); 64K residual → learned-selector
follow-on (Loop 11). Process: `.humanize/rlcr/2026-06-01_09-27-07/`; decision: `development/loop7/m12_final_decision.md`.
The strategic gate (§4.0) **resolved to pursue this AFTER the Tier-1 spine** — which had landed — so the gate was **open**. Draft: [`development/loop7/draft.md`](loop7/draft.md).
The core problem (established on hardware in Loop 6): DS recall **4K 75% / 16K 5% / 64K 0%** vs DSA **100%** at
the same 2048 budget + same kernel; dense DS = 100% and DS MMLU == DSA, so the gap is **selection quality +
the `index_topk=2048` kernel lock**, not a decode bug.
- [ ] **PRIMARY — adjustable-`top_k` sparse decode kernel:** a `flashmla_kv`-style decode kernel that relaxes
  `indices.shape[-1] == dsa_index_topk` as a **new opt-in DS path** (do not weaken the default DSA assert), so
  DS can spend > 2048 budget on hard prompts. CUDA-graph-safe, R23 deterministic tie-break carried over.
- [ ] **SECONDARY — query-aware / learned DS selector** (or pull top-p/Twilight forward, roadmap Loop 11):
  better needle placement inside the existing 2048 budget — **no kernel change**, cheaper to try first.
- [ ] **Measure NIAH 4K/16K/64K recall delta** vs the DS baseline 75% / 5% / 0% (DS-vs-DSA, real hardware).
- [ ] **Secondary engineering scope — 128k servability** (deferred client #2): KV-budget / admission to
  **serve** 128k (extends the §4.2 64K servability work); 128k/1024 SLO definition + benchmark shape (current
  `benchmark.sh` is 4096/512 only). Servability is separate from recall; the 128k deliverable needs both.
- **Note:** the **strict all-concurrency client SLO** (≥30 TPS/req at every conc) is a *separate downstream*
  concern (structurally DS ≤ DSA; conc-64 ≥30 unattainable even for DSA) — not this loop's recall goal unless
  the owner merges them.

### Loop 8 — nvfp4 / mxfp4 quantized weight support (deferred client #3)
- [ ] DS calibration + serving on nvfp4 and mxfp4 weights (today: FP8 e4m3 only). Channel-mask
  calibration assumes the FP8 dequant path; new quant formats need their own dequant in
  `token_label_write` / the calibration load.
- [ ] Validate the DS selection numerics survive the lower-precision weights (quality gate re-run).

### Loop 9 — performant knobs × DS compatibility (deferred client #4)
Each knob must be verified compatible with DS (several are known-incompatible with the DSv3.2+DSA
stack today — see `project_dsv32_dsa_incompat` memory: a2a backends, MoE runners, torch-compile,
NGRAM, pdmux all crash at scheduler init). Treat as a matrix, not one task:
- [ ] DP Attention × DS
- [ ] MTP / EAGLE speculative decode × DS
- [ ] Expert Parallel (EP) × DS
- [ ] **Explicit** chunked prefill × DS (Loop 5 only *probed* implicit support)
- [ ] Mixed chunked prefill × DS
- [ ] Overlap scheduling × DS (currently disabled under Option B)
- [ ] Piecewise CUDA graph × DS (currently disabled under Option B)

### Loop 10 — ⭐ NOW ACTIVE — GLM-5.1 (deferred client #1; pulled forward 2026-06-07 → disk `development/loop8/`)
*Highest client priority; a whole new model bring-up.* Draft: [`development/loop8/draft.md`](loop8/draft.md).
Weights already staged at `/cluster-storage/models/models--zai-org--GLM-5.1-FP8/`.
- [x] **§4.0 question ANSWERED — GLM-5.1 ships a native trained DSA indexer.** Config is
  `GlmMoeDsaForCausalLM` / `model_type: glm_moe_dsa` (MLA + DSA indexer + 256-expert MoE, FP8 e4m3 block-quant,
  78 layers, ~198k max ctx) with `self_attn.indexer` / `indexers_proj` modules. **`is_deepseek_dsa()` returns
  True for it** (`model_config.py:111`, gated on `index_topk`), so GLM-5.1 routes through the **same
  `dsa_backend.py`** as V3.2. ⇒ same posture as V3.2: **DSA-native is the default; DS is the opt-in fallback**,
  most valuable only where the trained indexer underperforms.
- [ ] **Wire DS into the preexisting GLM DSA backend** (NOT a standalone GLM DS path): reuse the existing
  DS↔`dsa_backend.py` wiring pattern (bind site + `TokenLabelTable` + selection/label hooks); generalize the
  DS model-forward hooks (today DeepSeek-specific in `deepseek_v2.py`) onto the GLM model forward. DSA-native
  default untouched.
- [ ] Calibrate a GLM-5.1 channel mask; bring up the DS serving path (TP=8, FP8) on the GLM architecture.
- [ ] Re-run the full accuracy + SLO gates on GLM-5.1.

---

## 7. Downstream loops — post-client-deliverable requirements

From `CLIENT_SLOS.md` "Downstream requirements after client deliverables."

### Loop 11 — Twilight (top-p selection instead of top-k)
- [ ] Replace/augment the fixed top-k=2048 selection with **top-p** (nucleus) selection over the
  signature scores (per the Twilight source). **Note:** this is also a candidate *fix for the
  long-context recall cap* — top-p can spend more budget on hard prompts — so it may be pulled
  forward into the §4.4/Loop-7 recall R&D if the strategic gate opens it.

### Loop 12 — Extensions as a general engine knob
- [ ] Generalize DS from a bespoke path into a first-class **"extension"** mechanism in the SGLang
  engine (so other sparsity/selection schemes plug in without re-plumbing the scheduler/forward path).

### Loop 13 — Integration into the rest of SGLang
- [ ] **PD-Disaggregation** × DS (prefill/decode split).
- [ ] **HiSparse** × DS coexistence (the serve script currently *excludes* HiSparse).
- [ ] General-feature integration sweep (whatever the §6 Loop-9 matrix didn't already cover).

---

## 8. Open decisions to resolve before/while planning the above

- [x] **DEC (strategic, §4.0) — RESOLVED (Loop 6, `ds_on_v32_decision.md`):** **pursue** DS recall R&D on
  DSA-native V3.2, but strictly **after** the Tier-1 spine (now landed → gate open, Loop 7 §6). Primary
  direction = adjustable-`top_k` decode kernel; secondary = learned/query-aware selector.
- [x] **DEC (SLO scope) — RESOLVED (Loop 6, AC-6):** DS ships as an **opt-in knob with DSA as the default**;
  the compact DS path is flag-gated (fp16/DSA default unchanged, non-regression proven). Whether DS *itself*
  meets the strict all-conc SLO is a separate downstream question (DS ≤ DSA on decode TPS).
- [ ] **DEC (TTFT target source):** confirm the client SLO is **absolute P99 TTFT < 22 s** at the
  client workload (not a DS-vs-DSA ratio) — and re-validate at full `NUM_PROMPTS=320` (§4.1).
- [x] **DEC (deployment topology) — RESOLVED (Loop 6, DEC-5):** the client deliverable is validated
  **single-node TP=8** for this loop; **multi-node TP scaling is deferred to a dedicated future loop**,
  tracked as a downstream requirement in §5. The SLO numbers are unchanged across topologies.
- [ ] **DEC (AC-12 long-context disposition):** the DS-fair re-scope is DECIDED for Loop 5; confirm
  whether the *original* 4K/16K/64K parity is a hard client requirement (it is **not** in the
  immediate SLO — only in the deferred 128k item).

---

## 9. Key artifacts & files (re-derivation index)

- **Client bar:** `development/CLIENT_SLOS.md`
- **Loop-5 evidence:** `runs/20260528_dsv32_mvp/` — `evidence_bundle.md`, `ac11_analysis.md`,
  `ac12_analysis.md`, `mvp_compare_ac11.md`, `next_loop_issues.md`, `ac12_results/`
- **Loop-5 process record:** `.humanize/rlcr/2026-05-28_10-17-12/` — `goal-tracker.md`, `stop-state.md`,
  round summaries/reviews
- **Loop-6 (DONE):** plan `development/loop6/refined_plan_v1.md`; process record
  `.humanize/rlcr/2026-05-30_06-27-19/` (goal-tracker, round summaries); evidence `runs/20260530_dsv32_loop6/`
  (`ds_on_v32_decision.md`, `footprint_feasibility.md`, `ac5_fullctx/`, `ac5_topk_design/`, `ac8_servability/`)
- **As-built system state (read first):** `development/past_implementations/study/08-current-system-architecture.md`
- **Loop-7 draft (Tier-2 recall R&D, high priority):** `development/loop7/draft.md`
- **The lever (§4.1):** `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py`
- **The kernel cap (§4.4):** `indices.shape[-1] == dsa_index_topk` assert in
  `python/sglang/srt/layers/attention/dsa_backend.py`
- **Serve/bench:** `development/serve_double_sparsity.sh`, `serve_native_nsa.sh`, `benchmark.sh`,
  `benchmark_baseline.sh`, `benchmark_compare.py`
- **Quality gates:** `test/manual/test_double_sparsity_v32.py`, `test/manual/test_dsv32_quality_smoke.py`
- **Mask (on disk, untracked):** `/models/dsv32-fp8-channel-mask.safetensors`
- **Weights:** `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`
</content>
</invoke>
