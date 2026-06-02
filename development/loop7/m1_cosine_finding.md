# Loop 7 M1 — cosine scorer measured: regime-dependent, materially helps 16K

First measured Tier-2.B result. `scorer_norm=cosine` unit-normalizes the query
projection and each token signature per head before the dot, so selection scores
by direction (relevance) not magnitude. Config-borne (in `--double-sparsity-config`)
so it reaches the TP worker processes; routes decode through the eager logical
scorer (`_compute_logical_token_scores`) when `scorer_norm != off`. Measured on
8×H200, int8 / mem 0.7, N=20, Clopper–Pearson 95% CI. Raw artifacts:
`recall_cosine_cfg.json`, `recall_forceeager_raw.json`, `ds_niah_baseline_mem07.json`.

| length | raw (prod, graph-safe) | **cosine** | Δ | significance |
|--------|------------------------|-----------|---|--------------|
| **4K** (~4.4K tok) | 75% [.509,.913] | **25%** [.087,.491] | **−50pp** | non-overlapping CIs → cosine **hurts** 4K |
| **16K** (~17.5K tok) | 5% [.001,.249] | **40%** [.191,.639] | **+35pp (8×)** | point clears the 24.9% materiality bar; CIs touch at the edge (N=20) |

## Finding

**Cosine normalization is a real, regime-dependent lever:**
- **16K (scorer-limited): 5% → 40% recall (8×).** This directly confirms the M0 diagnosis — the long-context gap is scorer-limited — AND that it is *fixable by a better scorer*. Cosine surfaces the needle that raw scoring buried under per-token background magnitude. This clears the M0 materiality bar (24.9%) on the point estimate.
- **4K (budget-limited): 75% → 25% recall.** Cosine *hurts* the moderate-length regime: there the needle is high-magnitude-salient, and discarding magnitude throws away the signal raw scoring uses.

This is physically coherent and matches the M0 oracle picture: where the needle was rank≈position (16K, magnitude-dominated), direction-only scoring helps; where the needle was just-past-budget and salient (4K), it hurts.

## Implication (M1 direction)
A *uniform* cosine scorer is not a clean win — it trades 4K for 16K. The natural next lever is a **length-/budget-conditional or hybrid scorer** (e.g. cosine only when the selected fraction is small / context is long, raw otherwise; or a magnitude+direction blend). That should keep 4K's 75% while capturing 16K's 40%. M1 should also firm the 16K number to N≥50 (the N=20 lower CI, 0.191, dips just below the 0.249 bar) before a binding recall claim, and re-confirm 4K/64K + MMLU non-regression under the chosen scorer.

## Caveats
- N=20: the 16K gain is large (8×) and the 4K loss is significant, but the 16K lower CI (0.191) is below the materiality bar — firm to N≥50.
- The raw baseline here is the production graph-safe path; cosine runs on the eager fallback path (DEC-6 research path). A landed cosine/hybrid scorer must be ported to the graph-safe Triton scorer for production perf (or accepted as opt-in slow). Per-variant TP cross-rank determinism + MMLU non-regression still required (AC-3).
