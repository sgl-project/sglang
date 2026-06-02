# M12 — Loop 7 Final Decision Record (strategic-gate supersession + loop close)

**Status: Loop 7 closes here.** This is the decision artifact required by the plan's M4
(`refined_plan_v1.md:165-167`, task20): it **supersedes** the Loop-6 strategic gate's
Tier-2.A-primary ordering with the measured M0 evidence, citing exactly what changed, and
consolidates the per-AC evidence into the loop-close decision. The prior gate rationale was
**sound when written**; the oracle data is what changed.

---

## 1. The decision (one line)

**Tier-2.B (a better non-learned selector) is the primary path to DS long-context recall;
Tier-2.A (a wider decode budget) is a bounded, opt-in, default-off 4K lever — NOT the
primary long-context path.** The landed deliverable is **DS-default + the opt-in Tier-2.B
hybrid scorer**, with the production-ready opt-in Tier-2.A lifted-budget decode available
for the 4K budget-limited regime. The DSA production default and the Loop-6 Tier-1 operating
point are not regressed.

---

## 2. What the strategic gate said, and what changed

**The Loop-6 gate** (`runs/20260530_dsv32_loop6/ds_on_v32_decision.md`, Loop-6 AC-1)
resolved to pursue Tier-2 recall R&D **after** the Tier-1 spine landed, and named
**Tier-2.A (the "heavy" adjustable-budget decode kernel) as the primary/selected
direction**. That ordering was a reasonable prior: if the DS selector were ranking the
needle just outside the fixed 2048 budget, simply widening the budget would be the most
direct recall fix.

**What the M0 measure-first oracle changed.** The diagnostic measured *why* DS misses the
needle — budget-limited (the 2048 cap is the wall) vs scorer-limited (DS ranks the needle
far below 2048 regardless of budget) — on the live all-reduced score tensor, fail-closed,
binding N=20. The score-only recall@K (`oracle_stride_reference.json`,
`m0_oracle_finding_r4.md`) is decisive:

| length | recall@2048 | recall@4096 | recall@8192 | regime |
|---|---|---|---|---|
| 4K  | 0.44 | 0.86 | **1.00** | **budget-limited** (needle rank min/median/max 44/2417/4406, clustered just past 2048) |
| 16K | 0.23 | 0.31 | **0.46** | **budget-partial** (caps ~46% even at 8192; needle rank ≈ position) |
| 64K | 0.15 | 0.20 | **0.24** | **scorer-limited** (no feasible budget ≤8192 recovers it) |

**The long-context regimes — 16K and 64K, which ARE the goal — are scorer-limited /
budget-partial, not budget-limited.** A wider budget (Tier-2.A) provably cannot close
16K/64K: at 16K even an 8192 budget caps at 46% score-only recall, and 64K barely moves.
Only **4K** is genuinely budget-limited (recall@8192 = 100%, recall@4096 ≈ 86%). Therefore the gate's
Tier-2.A-primary ordering is **superseded**: Tier-2.B (raising the needle's *rank* via a
better selector) is the primary long-context lever, and Tier-2.A is justified only as a
bounded opt-in 4K improvement. This is the supersession the plan's DEC-1 and M4 require.

---

## 3. Evidence chain (per AC → committed artifact)

### AC-1 — Measure-first oracle diagnostic + separated baseline (MET, R0–R8)
- Flag-gated debug-oracle on the live all-reduced token scores, **fail-closed** (explicit
  keyed failure records, no silent guessing), config-borne to TP workers, shared-FS sink
  (`m0_oracle_finding_r4.md`, `oracle_budget_vs_scorer_r4.json`): binding N=20 for
  4K/16K/64K, 0 failures, invariant `recall@2048 == selected_contains_needle` preserved.
- **Oracle-off is byte-identical + zero-alloc under CUDA-graph replay**
  (`oracle_off_graph_replay_alloc.json`, R8): 120 replays, byte-identical
  `selected_indices`/`valid_lengths` vs eager, **0 alloc bytes** under
  `assert_no_alloc_in_region` — "zero hot-path cost" demonstrated, not asserted.
- **Separated baseline** (`m0_baseline.md`): served-recall vs admission-status distinct at
  mem 0.7 — the old "64K 0%" was an HTTP-400 admission failure at mem 0.6, not a served
  miss; at mem 0.7 64K serves 20/20.
- **Stride / oracle provenance (cited explicitly for this record).** The oracle samples
  score-only recall over ALL needle tokens densely — `stride == 1` — never subsampling.
  `oracle_stride_reference.json` records `hardcoded_stride: 1` for **all 14,640** success
  records (`emitted_stride_value_counts: {"1": 14640}`, `default_equals_stride1: true`),
  with `source: selection_kernel.py::_maybe_record_recall_oracle ->
  oracle_payload_for_row(stride=1)`. The raw per-record sink
  (`.sglang_ds_oracle/sink.jsonl`) is gitignored, so the durable provenance is the
  **committed aggregate `oracle_stride_reference.json` + the hardcoded `stride=1` call site
  in `selection_kernel.py::_maybe_record_recall_oracle`** — both checked into the tree;
  this is the evidence relied on above and resolves the queued stride-provenance item.
- **AC-1.1**: dense-within-window force by post-topK logical-index replacement (evict
  lowest-ranked non-needle, insert needle, preserve exactly 2048).

### AC-2 — Recall uplift measured, DS-vs-DSA same node (MET, R5–R7)
- `ds_vs_dsa_recall_matrix_graph_n50.json`, `m4_ac3_nonregression_finding.md`: all variants
  under CUDA graph, same session, **N=50**, Clopper-Pearson 95% CIs, directional
  materiality rule (a variant point counts as material only if it exceeds the DS-default
  baseline CI **upper** bound):
  - within-budget 1024w: DSA / default / hybrid all **100%** (parity premise holds);
  - 4K: default 80%, hybrid 80% (parity, NOT material — within CI);
  - **16K: default 6% [1.3, 16.5] → hybrid 38% [24.7, 52.8], +32 pp MATERIAL**;
  - 64K: floor noise (scorer-limited, per M0).
- DSA same-node reference is the 100% ceiling at every length (`niah_*` artifacts). A
  legitimate negative result (64K) is recorded and characterized, which AC-2's floor allows.

### AC-3 — Tier-2.B selector uplift, non-learned, flag-gated, non-regression (MET, R6–R9)
- Three independently flag-gated non-learned variants — channel weighting/normalization
  (`scorer_norm`), head aggregation (`head_agg`), anchor budget (`anchor_mode`) — ported to
  the **graph-safe** Triton selector; **bit-identical eager-vs-graph** over the full
  24-combo matrix on fp16+int8; default channel-mask byte-identical when off
  (`m3_graphsafe_scorer_finding.md`, `m6_anchor_graphsafe_finding.md`).
- **MMLU 5-shot N=200** (same questions): DSA 89.0% / default 88.5% / hybrid 88.5% →
  **−0.5 pp ≤ 1.0 pp PASSED**; dense-DS / within-budget 100%; no 4K regression; **TP=8
  cross-rank selected-index determinism** holds (`test_ds_scorer_tp_determinism.py`).

### AC-4 — Tier-2.A opt-in adjustable-budget decode, production-ready (MET, R10–R18)
- New opt-in ABI `enable_lifted_budget_decode` + `lifted_budget_top_k` (NOT `max_top_k` /
  Twilight, NOT `SGLANG_DS_ALLOW_TOPK_MISMATCH`); the default DSA `flashmla_kv`
  `dsa_index_topk` assert is untouched; default-off byte-identical.
- **Production-ready, graph-safe** (`m9_tier2a_disposition.md`, `m10_lifted_graph_finding.md`):
  alloc-free `dequantize_k_cache_paged_out` into caller scratch + fixed-shape tensorized
  compact builder + `DSGraphState` scratch + q-pad scratch; physical→`page_table_1_flattened`
  →compact remap; `-1`/pad masked before dequant; fixed budget + padding; R23 tie-break.
- **Served 4K recall recovery** (R14 eager N=20 same-node, isolating budget — distinct from
  the graph N=50 default 80% above): DS-lifted-4096 **95% (19/20)** vs DS-default-2048 **75%
  (15/20)**, +20 pp material; **live production CUDA-graph 95%** recall (3.4× faster than
  eager); graph-captured TP=8 lifted-width determinism via the composed (a) single-rank
  CUDAGraph zero-alloc + bit-identical + (b) eager 8-rank all-reduce equality + (c) live
  TP=8 graph server. Lifted+speculative fails closed (validator guard).
- **DEC-4 close-gate SATISFIED**: the production-ready landing disposition exists, so the
  close leaves no dangling pursued-hardening item. Per M0, Tier-2.A is correctly scoped as
  a **bounded 4K lever** (4K is the only budget-limited regime).

### AC-5 — Tier-2.C 64K servability, separated (MET, R0)
- `ds_niah_baseline_mem07.json`, `m0_baseline.md`: 64K `/generate` **served 20/20, 0
  admission failures** at mem 0.7 (served vs admission reported distinctly). 128k is
  explicitly its own loop (DEC-3) and is not conflated with recall.

### AC-6 — No Tier-1 regression + perf guardrails, conc-1/16 (MET, R19–R21)
- `m11_perf_consolidation.md` (now with exact commit provenance + per-run `run_provenance`
  in each `ttft_*.json`): the full guardrail set — **TTFT, decode TPS/req, GPU memory,
  graph-replay, admission** — at conc-1/16, all under CUDA graph:
  - **Decode-free Tier-2.B**: DS-hybrid == DS-default decode TPS (27.6 == 27.6 conc-16) AND
    TTFT (within noise) — the landed long-context recall winner costs nothing on the hot
    path;
  - DS ≈ 0.48–0.49× DSA structurally (the known channel-mask selector cost, not a Loop-7
    regression); DS-default conc-16 27.6 matches the Loop-6 closed-batch 27.1 (Tier-1 spine
    intact); every measured TTFT far below the P99 22 s ceiling (heaviest ≈ 1.4 s);
    DSA/fp16 defaults behavior-unchanged.

---

## 4. Ultimate-Goal outcome (close OR rigorously characterize)

The Ultimate Goal was to **close — or rigorously characterize — the DS long-context recall
gap**, without regressing the DSA default or the Loop-6 Tier-1 op-point. Outcome:

- **Rigorously characterized**: the M0 oracle attributes the gap by regime — 4K
  budget-limited, 16K budget-partial (~46% cap), 64K scorer-limited — the root-cause map the
  Loop-6 carryover lacked.
- **Partially closed, materially**: 16K served recall **6% → 38% (+32 pp material)** via the
  default-off Tier-2.B hybrid scorer (MMLU within 0.5 pp, decode-free); 4K **75% → 95%**
  (R14 eager N=20 same-node default-2048 vs lifted-4096) via the opt-in production-ready
  Tier-2.A lifted budget. Within-budget recall is at 100% parity
  with DSA.
- **Characterized residual**: 64K remains scorer-limited — a legitimate, measured negative
  result (AC-2's floor explicitly allows this). The frontier is the **offline channel-mask
  scorer's discrimination at extreme length**, which is exactly the **learned/distilled
  selector** territory deferred behind DEC-5 to its own loop.
- **Non-regression**: DSA default + fp16 defaults byte-identical; Loop-6 Tier-1
  admission/decode spine intact; every new code path opt-in / default-off / reversible.

This satisfies the Ultimate Goal under its "close OR rigorously characterize" standard.

---

## 5. Disposition & follow-on

- **Landed**: DS-default + the opt-in Tier-2.B hybrid scorer (the long-context deliverable);
  the production-ready opt-in Tier-2.A lifted-budget decode (the bounded 4K lever). All
  flag-gated, default-off, DSA default untouched.
- **Deferred to its own loop (DEC-5, approved-but-deferred)**: a **learned/distilled
  selector** for the 16K/64K scorer-limited regime — the only lever the M0 evidence points
  to for the residual long-context gap, and out of this loop's non-learned scope.
- **Queued before final merge (not loop-blocking)**: remove plan/workflow markers
  (`AC-*`, `task*`, `Tier-2`, `DEC-`) from production code comments/tests (the
  implementation-note ban; documentation-only, no runtime effect).

---

## 6. Provenance of this record
Synthesized from the committed Loop-7 artifacts: `m0_baseline.md`,
`m0_oracle_finding_r4.md` / `oracle_budget_vs_scorer_r4.json`, `oracle_stride_reference.json`,
`oracle_off_graph_replay_alloc.json`, `m2_recall_matrix_finding.md` /
`ds_vs_dsa_recall_matrix_graph_n50.json`, `m4_ac3_nonregression_finding.md`,
`m3_graphsafe_scorer_finding.md`, `m6_anchor_graphsafe_finding.md`,
`m9_tier2a_disposition.md`, `m10_lifted_graph_finding.md`, `m11_perf_consolidation.md`, and
the per-run `run_provenance` in `ttft_*.json`. It supersedes the prior strategic gate
`runs/20260530_dsv32_loop6/ds_on_v32_decision.md` per the plan's M4/DEC-1; `m0_decision.md`
was the M0 A-vs-B draft this record finalizes. GPU: 8× NVIDIA H200, TP=8. No production-code
change in the consolidation rounds (R19–R21 touched only `development/loop7/`).
