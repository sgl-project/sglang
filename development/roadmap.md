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

## 4. LOOP 6 — Client-SLO MVP ("make DS shippable") — the next loop

**Goal:** make DS *itself* (not just DSA) pass `CLIENT_SLOS.md` on the immediate workload, and
decide whether to invest further. The spine is the **admission/TTFT fix**; everything else here is
a small hardening or a strategic decision. A stub plan exists at `development/loop6/` — flesh out
`development/loop6/draft.md` from `runs/20260528_dsv32_mvp/next_loop_issues.md` before `gen-plan`.

### 4.0 Strategic gate (decide FIRST — gates Tier-2 R&D for this and later loops)
- [ ] **DEC: Is DS worth pursuing on a DSA-native model (V3.2)?** Capture as a `DEC-N` decision doc
  `runs/<date>_dsv32_loop6/ds_on_v32_decision.md`. Rationale on the table: DS is capped at native
  `index_topk=2048` by the shared kernel AND uses an inferior offline selector, so it cannot match
  DSA long-context recall at the shared budget; DS's value is clearer on models without a trained
  indexer. **This decision gates §6 Loop 7 (128k) and the Tier-2 recall R&D below.**

### 4.1 ⭐ THE client-SLO blocker — TokenLabelTable footprint → mem fraction → admission (handoff #2)
- [ ] Shrink the per-rank `TokenLabelTable` (`python/.../double_sparsity/token_label_table.py`,
  ~8 GB/rank fp16 today) so DS can serve at a higher `mem_fraction_static` **without the
  generation-time OOM seen at 0.7**. Candidate levers: int8-symmetric signatures (+ per-layer/slot/head
  scales applied at scoring), narrower `label_dim`, or a tighter slot model.
- [ ] Sweep `mem_fraction_static` 0.6 → 0.7 → 0.8; record `max_total_num_tokens` rising with **no OOM**
  under a long sustained `/generate`. Artifact: mem-fraction sweep log.
- [ ] **Re-run AC-11 at the lifted mem fraction** (conc 16/32/64, 3 trials, 120 s/600 s) and confirm
  achieved concurrency tracks nominal; update `ac11_analysis.md` verdict (handoff #3 / DEC-7).
- [ ] **Direct client-SLO validation** (NEW — the actual acceptance test): run `benchmark.sh` at the
  **full** `NUM_PROMPTS=320` client shape and assert **absolute P99 TTFT < 22 s** and **≥ 30 TPS/req**
  at conc 16 and 64. This is the artifact that says "DS is shippable for the client."

### 4.2 64K servability (side-effect of 4.1; also unblocks deferred 128k) — (handoff #2)
- [ ] At the lifted mem fraction, confirm a 64K-context `/generate` no longer returns HTTP 400
  (or document the new ceiling). This is a *servability* win; recall accuracy at 64K is separate (Tier 2).

### 4.3 Accuracy-harness hardening — (handoff #4 / Codex queued #1)
- [ ] AC-12 within-budget gate: assert `within_budget` from **actual** `usage.prompt_tokens`
  (or tokenized chat length), not the 1024/1536 **word-count** proxy. Rename `length_tokens`→`length_words`
  / add `input_tokens`. Must **not** change the DECIDED DS-fair gate definition.

### 4.4 Tier-2 — DS long-context recall R&D (GATED on 4.0; likely DEFER to Loop 7) — (handoff #1)
- [ ] **Only if the strategic gate opens it:** a `flashmla_kv` decode-kernel variant accepting
  `top_k > index_topk` (today asserts `indices.shape[-1] == dsa_index_topk`) **and/or** a query-aware /
  learned DS selector that places the needle in the 2048 budget. Measure NIAH 4K/16K/64K recall delta
  vs the Loop-5 baseline DS 75% / 5% / 0%. *(GPU- and engineering-heavy; do not start before 4.0.)*

**Loop 6 done = client-SLO MVP:** §4.1 lands and the full-shape benchmark shows **P99 TTFT < 22 s
and ≥ 30 TPS/req** for DS; §4.2/§4.3 recorded; strategic gate (§4.0) decided. Tier 2 (§4.4) closed-or-opened
explicitly (a closed gate is a legitimate outcome, not a stall).

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
- [ ] **Multi-node / TP scaling story.** All Loop-5 serving was single-node TP=8 (node 1 used only for
  the cross-node AC-12). Document/validate whether the client deployment is single- or multi-node.
- [ ] **Comparator per-side `mem_fraction_static` check** is in (Round 13); keep it green when 4.1 moves
  the mem fraction.

---

## 6. Downstream loops — deferred CLIENT requirements (ordered by client priority)

From `CLIENT_SLOS.md` "Deferred Client requirements ordered from most important to least." Each is a
candidate RLCR loop; sizing/dependencies noted.

### Loop 7 — 128k ISL / 1024 OSL long-context (deferred client #2 — but engineering-gated first)
*Most technically continuous with Loop 6.* Depends on §4.2 (admission/KV budget) **and** the §4.4
recall R&D if the strategic gate opened it.
- [ ] KV-budget / admission sufficient to **serve** 128k context (extends the 64K servability work).
- [ ] Long-context **recall** at 128k: requires the query-aware/learned selector and/or `top_k>index_topk`
  kernel variant (else DS recall collapses as in the Loop-5 4K/16K/64K characterization).
- [ ] 128k/1024 SLO definition + benchmark shape (the current `benchmark.sh` is 4096/512 only).
- [ ] **Strategic dependency:** if §4.0 decided DS is *not* worth pursuing on DSA-native V3.2, this loop
  may instead be "serve 128k via DSA, DS off" — surface that explicitly.

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

### Loop 10 — GLM-5.1 (deferred client #1 — highest client priority, largest lift)
*Listed as most important deferred requirement, but it's a whole new model bring-up.*
- [ ] Determine whether GLM-5.1 ships a native sparse indexer (the strategic §4.0 question recurs — DS
  is most valuable here **if GLM-5.1 has no trained indexer**).
- [ ] Calibrate a GLM-5.1 channel mask; bring up the DS serving path on the GLM architecture.
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

- [ ] **DEC (strategic, §4.0):** pursue DS recall R&D on DSA-native V3.2, or cap DS at "engineering
  wins only" and lean on DSA for long context? Governs Loops 6 Tier-2, 7, and the GLM-5.1 framing.
- [ ] **DEC (SLO scope):** is "shippable" = DS meets the client SLO *itself*, or = DS available as an
  opt-in knob while DSA is the default that meets the SLO? (DSA already meets both SLOs trivially.)
- [ ] **DEC (TTFT target source):** confirm the client SLO is **absolute P99 TTFT < 22 s** at the
  client workload (not a DS-vs-DSA ratio) — and re-validate at full `NUM_PROMPTS=320` (§4.1).
- [ ] **DEC (deployment topology):** single-node TP=8 vs multi-node for the client deliverable (§5).
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
- **Loop-6 stub:** `development/loop6/draft.md` (DUMMY — flesh out), `development/loop6/runbook.md`
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
