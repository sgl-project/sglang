# Loop 7 — AC-3 binding non-regression matrix (graph-mode, production op-point)

The Tier-2.B hybrid scorer is now landed on the production CUDA-graph path (R6).
This is its **binding AC-3 non-regression evidence**, measured entirely under
CUDA graph (the production path — per
`BL-20260602-eager-vs-graph-recall-differs-despite-identical-scorer`, recall must
be measured under graph, not eager), same session, 8×H200 TP=8, int8 / mem 0.7.

## Served recall — DS-default vs DS-hybrid (Tier-2.B) vs DSA, N=50, 95% Clopper–Pearson CI

| length | role | DSA (ceiling) | DS-default | DS-hybrid (Tier-2.B) | uplift | material? |
|--------|------|---------------|------------|-----------------------|--------|-----------|
| **1024w** | **dense-DS / within-budget (≤2048 tok)** | 100% [92.9,100] | **100% [92.9,100]** | **100% [92.9,100]** | 0 pp | **dense-DS parity ✓** |
| **4K** | beyond-budget (raw regime) | 100% [92.9,100] | 80% [66.3,90.0] | 80% [66.3,90.0] | 0 pp | == default (no regression) |
| **16K** | beyond-budget (cosine regime) | 100% [92.9,100] | 6% [1.3,16.5] | **38% [24.7,52.8]** | **+32 pp** | **YES** (38% ≫ default CI high 16.5%) |

`niah_dsa_graph_n50.json`, `niah_default_graph_n50.json`,
`niah_hybrid_graph_n50.json`, `ds_vs_dsa_recall_matrix_graph_n50.json`.

## MMLU re-anchor — 5-shot, N=200, SAME questions (deterministic seed), graph-mode

| config | MMLU acc | hits |
|--------|----------|------|
| DSA (re-anchored) | **89.0%** | 178/200 |
| DS-default | 88.5% | 177/200 |
| DS-hybrid (Tier-2.B) | **88.5%** | 177/200 |

**DS-hybrid is −0.5 pp vs the re-anchored DSA (≤ 1.0 pp gate PASSED)**; it is 0 pp
vs DS-default (identical — MMLU prompts are within-budget, so the hybrid uses its
raw regime = default). `mmlu_dsa_graph.json`, `mmlu_default_graph.json`,
`mmlu_hybrid_graph.json` (fast runner: 5-shot "Answer:" prompt, `max_new_tokens=4`).

## Findings — AC-3 non-regression SATISFIED for the hybrid scorer
- **Material long-context uplift (binding).** 16K served recall 6% → **38%**
  (+32 pp), DS-hybrid point well outside the DS-default CI [1.3, 16.5] → material
  by the directional rule, now at **N=50** (the R6 N=20 graph sample read 25% —
  a low draw; N=50 binds it at 38%, close to the eager-research 40%).
- **MMLU non-regression.** 88.5% vs re-anchored DSA 89.0% = −0.5 pp, within the
  ≤1.0 pp gate.
- **Within-budget / dense-DS parity.** At 1024w (≤2048 tokens, the whole context
  fits the budget so DS selects densely) all three are 100% — the explicit
  dense-DS reference; the hybrid does not regress within-budget recall.
- **No 4K regression.** 4K hybrid 80% == default 80% (4K ≤ the 8192 hybrid
  threshold ⇒ raw regime ⇒ identical to default).
- **TP=8 determinism** + bit-identical eager-vs-graph selection were established in
  R3/R6 (`test_ds_scorer_tp_determinism.py`, `TestGraphSafeScorerEqualsEager`).

## Net (AC-2 + AC-3)
The Tier-2.B hybrid scorer, on the **production CUDA-graph path**, delivers a
**binding material 16K recall uplift (6% → 38%)** with **MMLU within 0.5 pp of
DSA**, **dense-DS/within-budget parity**, and **no 4K regression** — a
production-ready, non-regressing Tier-2.B improvement. The long-context gap to
DSA's 100% is reduced but not closed (16K 38% vs 100%; 64K remains scorer-limited
per the oracle) — a recorded, characterized result.

## Remaining for AC-3/AC-6 (queued, task #15)
- **graph-vs-eager scorer perf delta** (AC-6, conc-1/16 TTFT / decode-TPS / mem).
- **anchor_mode graph-safe port** (still eager-only).
- 64K hybrid at N=50 (R5 eager showed 0%; scorer-limited — low priority).
