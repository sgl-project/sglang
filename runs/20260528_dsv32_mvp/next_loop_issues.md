# Loop 5 — issues to knock out in the next RLCR loop

Loop 5 shipped the TIER-1 smoke MVP and the full evidence bundle; the only unmet criterion is
**AC-12** (NIAH long-context recall), now characterized precisely (see `ac12_topk_sweep/analysis.md`).
This file lists the carried-over work for a future loop, ordered by impact.

## 1. AC-12 disposition (decision + work)
AC-12 NIAH is a **hard fail** that is **not fixable on the current DS + `flashmla_kv` path**:
DS's selection budget is kernel-locked to V3.2's native DSA `index_topk=2048`, and at that budget
DS's offline channel-mask selection is inferior to V3.2's trained DSA indexer for needle recall
(DS 5% vs DSA 100% at 16K, same budget). DS decode itself is sound (dense ≤2048 → 100% recall;
MMLU == DSA). Decide one of:
- **(a) Accept the TIER-1 smoke milestone as final** (the plan's stated fallback) and close the loop.
- **(b) Re-scope AC-12** to a DS-fair gate (recall within the selectable budget / NIAH ≤ index_topk /
  a recall-vs-length characterization), which a real DS quality gate should arguably measure.
- **(c) R&D to genuinely meet AC-12** — see items 2-3.

## 2. DS long-context recall R&D (to approach DSA selection quality)
- A **query-aware / learned DS selector** that places the needle in the 2048 budget (vs the current
  offline Method-1 channel-mask), i.e., closes the selection-quality gap to V3.2's DSA indexer.
- A **`flashmla_kv` decode-kernel variant that accepts `top_k > index_topk`**, so DS could trade
  compute for recall by widening the selection budget (today the kernel asserts
  `indices.shape[-1] == dsa_index_topk`, hard-capping DS at 2048).

## 3. DS KV-budget / TokenLabelTable footprint (admission)
- DS at `mem_fraction_static=0.6` has `max_total_num_tokens≈53K` (vs DSA's ≈910K at 0.85), which
  (a) makes 64K context **unservable** (AC-12 64K HTTP 400) and (b) caps effective concurrency
  (the AC-11 directional TTFT miss). Both share one lever: **shrink the per-rank TokenLabelTable**
  so DS can serve at a higher mem fraction without the generation-time OOM seen at 0.7. Orthogonal
  to recall (a 64K needle still misses the 2048 selection), but required for 64K servability and
  for closing the AC-11 admission gap.

## 4. AC-11 directional TTFT follow-up (DEC-7)
- Re-sweep once item 3 lifts DS effective concurrency; the recorded directional TTFT miss
  (admission-bound at mem 0.6) should be re-evaluated against DSA. (Filed in `ac11_analysis.md`.)

## 5. Strategic question (flagged by the Round-13 investigation)
- **Is Double Sparsity worthwhile on a model that already ships native DSA?** On V3.2, DS is capped
  at the native `index_topk` budget by the shared decode kernel AND uses an inferior offline
  selector — so it cannot match (let alone beat) DSA's long-context recall at the shared budget.
  DS's value proposition is clearer on models WITHOUT a trained sparse indexer. Worth an explicit
  decision before further DS investment on V3.2.

## 6. Cosmetic
- Pre-existing "Locked Option B operating point (plan §13 / DEC-1)" header lines in
  `development/serve_*.sh` still carry plan-specific terms (predate Round 11); reword to
  behavior-based language next time those headers are edited.
