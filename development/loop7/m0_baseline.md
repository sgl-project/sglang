# Loop 7 M0 — DS NIAH served-recall baseline (separated), mem 0.7

**Op-point:** DS int8 compact table / `mem_fraction_static=0.7` / TP=8 / page_size 64 / fp8 KV / `flashmla_kv` prefill+decode / radix-off (M0 recall measurement; radix-on is a Tier-1/AC-6 perf concern, not recall). Server `max_total_num_tokens=396096`. `index_topk=2048`.
**Method:** reuses the AC-12 harness helpers (`_make_niah_prompt`, `_niah_needle`, `_generate` chat, `_niah_recall_hits`) via `development/loop7/niah_ds_baseline.py`. N=20 trials/length, fixed seeds (1000+idx), `max_new_tokens=64`. Raw: `development/loop7/ds_niah_baseline_mem07.json`.
**Materiality rule (AC-2):** Clopper–Pearson 95% binomial CI per length; a recall uplift counts as "material" only if it clears the baseline's CI upper bound.

| Length | tokenized (max) | budget class | served | admission_fail | served recall | 95% CI |
|--------|-----------------|--------------|--------|----------------|---------------|--------|
| 1024 w | 1119 | within (≤2048) | 20/20 | 0 | **100%** | [0.832, 1.0] |
| 1536 w | 1672 | within (≤2048) | 20/20 | 0 | **100%** | [0.832, 1.0] |
| 4096 w | 4408 | beyond | 20/20 | 0 | **75%** (15/20) | [0.509, 0.913] |
| 16384 w | 17603 | beyond | 20/20 | 0 | **5%** (1/20) | [0.001, 0.249] |
| 65536 w | 70102 | beyond | 20/20 | 0 | **5%** (1/20) | [0.001, 0.249] |

## Findings

1. **Within-budget recall = 100%** (1K/1.5K, both ≤ 2048 tokens) → DS decode is sound and within-budget parity with DSA holds. The recall gap is purely a beyond-budget *selection* phenomenon.
2. **Beyond-budget reproduces the historical curve**: 4K = 75% (exactly 15/20, matching the Loop-5/6 baseline), 16K = 5%.
3. **64K is now SERVED, not unservable.** At mem 0.7 the int8 compact table admits the ~70K-token prompt (served 20/20, **0 admission failures**) and recalls **5%** — whereas the old `ac12` "64K = 0%" was an *admission failure* (HTTP 400) at mem 0.6, not a served miss. This:
   - confirms Codex's served-vs-admission correction (the separated baseline is the right frame);
   - effectively satisfies **task18 / AC-5** (64K `/generate` servability at the lifted op-point);
   - **corrects the loop's working baseline from "75 / 5 / 0" to a served "100 / 100 / 75 / 5 / 5"** with all five lengths admitted.
4. **Scorer-limited signal (pre-oracle):** at 4K (~4.4K tokens, budget 2048 ⇒ DS selects ~47% of tokens) recall is only 75% — selecting nearly half the tokens still misses 25% of needles. This is consistent with the gap being selection-quality, not raw budget — but the **definitive** budget-vs-scorer call is the oracle's score-only recall@4096/8192 vs recall@2048 (next).

## Materiality bars (for AC-2 uplift claims)
- 4K: must exceed **0.913** to be material (already high; little headroom).
- 16K: must exceed **0.249** (i.e. > ~5/20) — a 1–2 needle move does NOT count.
- 64K: must exceed **0.249**.

## Status
- This is the **served-recall + admission** half of the M0 baseline (task6). DSA re-confirmation (sequential boot; DSA is documented 100% at every length) and the **oracle score-only recall@K diagnostic** (the A-vs-B decider) are the remaining M0 evidence. MMLU re-anchor (AC-3) pending.
