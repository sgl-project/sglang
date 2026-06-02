# Loop 7 M0 — oracle budget-vs-scorer finding (the A-vs-B decider)

> **SUPERSEDED (R4) for the 64K row and N.** This run recorded **nothing for 64K**
> (the oracle env did not reach TP workers; see the caveat at the bottom) and used
> N=3–4. The binding re-run — fail-closed, config-borne, N=20, with **64K measured**
> — is `m0_oracle_finding_r4.md` (artifact `oracle_budget_vs_scorer_r4.json`). The
> A-vs-B *decision* below is unchanged and confirmed; only the 64K evidence moved
> from inferred to measured and 16K is now characterized as budget-partial.

**What this measures:** for each NIAH trial, the needle's score rank on the **live all-reduced DS token-score tensor** (after `all_reduce_token_scores`, before `select_topk_sequence_order`) and the **score-only** recall@K — i.e. would a budget of K have selected every needle token, judged purely from the score ranking (no decode). Needle logical positions located via raw-prompt offset mapping with `token_match` verified against the server's `prompt_tokens`. Server: DS int8 / mem 0.7 / TP=8, **eager** (`--disable-cuda-graph`; the Python oracle hook does not re-run under graph replay). Raw artifacts: `oracle_sink.jsonl`, aggregate `oracle_budget_vs_scorer.json`.

| length | tokenized | trials × layers | r@2048 | r@4096 | r@8192 | needle rank (min/med/max) | **verdict** |
|--------|-----------|-----------------|--------|--------|--------|----------------------------|-------------|
| **4K** | ~4.4K | 4 × ~57 | 0% | **100%** | 100% | 2105 / 2208 / 2580 | **budget-limited** |
| **16K** | ~17.5K | 3 × ~48 | 0% | **0%** | **0%** | 8832 / 10218 / 10306 | **scorer-limited** |
| 64K | ~70K | (records absent — see note) | — | — | — | needle planted at pos ~35.4K–41.2K | scorer-limited by construction |

## The finding — the gap is REGIME-DEPENDENT

- **4K is budget-limited.** The needle ranks just *past* the 2048 budget (median 2208) and is recovered 100% at budget 4096. A wider-budget decode (Tier-2.A) genuinely recovers recall here — and the oracle-uplift gate (score-only recall@4096 materially > recall@2048) is **met at 4K**.
- **16K is scorer-limited.** The needle ranks ~10.2K out of 17.5K — roughly its sequence position, i.e. the offline channel-mask scorer barely discriminates the needle from arbitrary neighbors. **No feasible budget (≤ 8192) recovers it**; only a better selector (Tier-2.B) that ranks the needle higher would. The oracle-uplift gate is **NOT met at 16K**.
- **64K is scorer-limited by construction**: the needle sits at logical position ~35K–41K in ~70K tokens, far beyond any feasible budget, so unless the scorer strongly discriminates it (it does not even at 16K), no budget recovers it.

## Implication for DEC-1 (A-vs-B)

This **refines** the plan's "measure-first → B → A-if-evidence":
- There **is** real, bounded budget-limited evidence (4K) — Tier-2.A has a legitimate, *length-bounded* use (it recovers the regime where the needle ranks in `(2048, ~budget]`).
- But the loop's actual goal is **long-context** recall (16K/64K), and there the gap is **scorer-limited** — Tier-2.B (a better selector) is **necessary**; a wider budget alone is insufficient.
- Net: **both levers matter, in different length regimes.** Lead with Tier-2.B (it's the only lever for the long-context goal); pursue Tier-2.A as a bounded win for the moderate-length regime if cheap. This is the evidence task7 (Codex) should adjudicate into the final A-vs-B ordering + the strategic-gate supersession.

## Caveats / follow-ups
- Small N (3–4 trials/length after needle-span filtering); rank ranges are tight and the r@K split is clean (0%/100%), but firm up with N≥20 before the binding decision.
- **Side issue (queued):** the oracle records nothing for 64K (very-long sequences) though the request runs and `token_match=true` — the best-effort hook likely swallows an exception on long-seq score tensors. The served-recall baseline already covers 64K; the budget-vs-scorer call for 64K rests on the needle-position argument. Worth a fix before a 64K oracle claim.
- recall@K is per (trial, layer); the verdict is uniform across the 61 layers (r@K is cleanly 0% or 100%), so layer aggregation does not change it.
