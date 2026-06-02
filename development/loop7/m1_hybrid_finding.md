# Loop 7 M1 — length-conditional hybrid scorer: best of both regimes

The AC-3 landed candidate. `scorer_norm=hybrid` uses the **raw** channel-dot scorer for context ≤ `scorer_norm_hybrid_threshold` tokens (default 8192) and **cosine** above, decided per-request by `seq_len`. Config-borne (reaches TP workers); routes decode through the eager logical scorer (B1 fail-fast prevents a non-default scorer entering the graph-safe raw path). Measured on 8×H200 via `LOOP7_MEASUREMENT=1` (pins int8 / mem 0.7), N=20, Clopper–Pearson 95% CI. Artifact: `recall_hybrid.json`.

| length | tokenized | raw (prod) | uniform cosine | **hybrid** | hybrid path |
|--------|-----------|------------|----------------|-----------|-------------|
| **4K** | ~4.4K | 75% [.51,.91] | 25% [.09,.49] | **85% [.62,.97]** | raw (≤8K) |
| **16K** | ~17.5K | 5% [.00,.25] | 40% [.19,.64] | **40% [.19,.64]** | cosine (>8K) |

## Finding

The hybrid achieves the **best of both regimes** — it is the per-length max:
- **4K recovered**: 85% (vs uniform cosine's 25%) — the raw scorer is kept where the needle is high-magnitude-salient. Statistically indistinguishable from the prod baseline (75%, overlapping CIs); cosine's 4K regression is gone (hybrid 85% CI [.62,.97] vs cosine 25% CI [.09,.49] — non-overlapping).
- **16K kept**: 40% (matches uniform cosine) — cosine surfaces the needle where background magnitude dominates.

This makes the hybrid the production-shaped Tier-2.B candidate: it improves the long-context regime (the loop's goal) without the moderate-context regression that disqualified uniform cosine.

## Status / remaining (NOT AC-3 closure yet)
- This is the served-recall result for the hybrid on the **eager** path (DEC-6 research path). **Remaining for binding AC-3 closure** (queued): port the hybrid scorer into the **graph-safe Triton path** for production perf (currently the non-default scorer fails fast under capture and must serve eager); the full non-regression matrix — **MMLU re-anchor** at the op-point, **dense-DS / within-budget parity**, **N≥50** for a binding 16K number (the 16K lower CI 0.19 dips under the 24.9% bar), **DSA same-node reference**, and **TP=8 cross-rank selected-index equality** for every scorer/anchor flag.
- head-aggregation and anchor-budget variants are implemented + unit-tested but not yet measured (queued for the per-variant matrix).
- The hybrid threshold (8192) is a first cut chosen from the 4K-vs-16K regime split; it should be swept once the matrix runs.
