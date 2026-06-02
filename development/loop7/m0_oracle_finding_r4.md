# Loop 7 M0 — oracle budget-vs-scorer finding, BINDING re-run (R4)

**Supersedes** the inferred-64K reading in `m0_oracle_finding.md` (which had 64K
records *absent* — see that file's caveat). This run records oracle samples for
**all three lengths including 64K**, fail-closed, at N=20.

## What changed vs the old run (why this is binding)
The old oracle run silently recorded **nothing for 64K** and used N=3–4. R4 fixed
three stacked reasons the oracle under-recorded (all in
`BL-20260602-ds-oracle-decode-only-and-shared-fs`):
1. **Config-borne activation** (`recall_oracle` in `--double-sparsity-config`) so
   the hook reaches the TP workers (env does not).
2. **Shared-FS cross-process files** (`./.sglang_ds_oracle/`, the repo mount) —
   the old `/dev/shm` default is a per-sandbox tmpfs the worker can't see.
3. **Forced decode steps** (`ignore_eos`, `decode_steps+1` tokens) — DS selection
   runs ONLY in decode; the instruction-style NIAH prompts emit immediate EOS on
   raw `/generate` (prefill-only), so `max_new_tokens=1` did zero decode → zero
   selection → zero oracle samples.
The oracle is now **fail-closed**: out-of-range needle spans, missing trials, and
payload exceptions emit explicit `failure` records, and the sweep asserts every
issued trial produced records and aborts on any missing length.

## Method
Per NIAH trial: locate the needle's logical token span (raw-prompt offset
mapping), register it via the cross-process trial file, then force
`decode_steps=4` decode forwards (`ignore_eos`). The server-side oracle records,
per (trial, layer, decode-step), the needle's **worst score rank** and
**score-only recall@K** on the live all-reduced DS score tensor (after
`all_reduce_token_scores`, before top-K). Server: DS int8 / mem 0.7 / TP=8,
**eager** (`--disable-cuda-graph`), `recall_oracle: true` (config-borne). N=20 per
length, ~4,880 (layer × decode-step) samples each. Raw: `.sglang_ds_oracle/sink.jsonl`,
aggregate `oracle_budget_vs_scorer_r4.json`.

`recall@K` = ALL needle tokens ranked `< K` (`needle_worst_rank < K`), averaged
over the (layer, decode-step) samples. K=2048 is the locked `index_topk`; K>2048
is **score-only** (a ranking property, not a decode outcome).

## Results

| length | tokenized | trials × samples | r@2048 | r@4096 | r@8192 | needle rank min/med/max | uplift (r4096−r2048) | **verdict** |
|--------|-----------|------------------|--------|--------|--------|--------------------------|----------------------|-------------|
| **4K**  | ~4.4K  | 20 × 4880 | 44% | **86%** | **100%** | 44 / 2417 / 4406   | **+56 pp** | **budget-limited** |
| **16K** | ~17.6K | 20 × 4880 | 23% | 31% | **46%** | 80 / 9318 / 17599  | +23 pp | **budget-partial** (a wider budget helps but caps at ~46%) |
| **64K** | ~70K   | 20 × 4880 | 15% | 20% | 24% | 104 / 32668 / 70096 | **+9 pp** | **scorer-limited** |

## The finding — the gap is REGIME-DEPENDENT, and long-context is scorer-bound

- **4K is budget-limited.** Needle median rank ~2417 (just past the 2048 budget);
  recall climbs 44% → 86% → **100%** as budget widens to 8192. A wider-budget
  decode (Tier-2.A) genuinely recovers recall here. The oracle-uplift gate
  (score-only recall@4096 ≫ recall@2048) is strongly **met** (+56 pp).
- **16K is budget-PARTIAL.** A wider budget helps (23% → 46% at 8192, +23 pp) but
  **caps at ~46%** — over half the needle-tokens are still mis-ranked even at 4×
  the budget. So 16K needs **both** a wider budget **and** a better scorer; budget
  alone is insufficient.
- **64K is scorer-limited.** Needle median rank ~32.7K of ~70K (≈ its sequence
  position — the channel-mask scorer barely discriminates it). A 4× budget moves
  recall only 15% → 24% (+9 pp, below the materiality gate). **No feasible budget
  recovers 64K**; only a better selector (Tier-2.B) ranks the needle higher. This
  is the binding 64K evidence the old run could only infer.

## Implication for DEC-1 (A-vs-B) — confirms and sharpens the M0 decision

This **strengthens** the plan's "lead Tier-2.B; pursue Tier-2.A only where the
oracle proves budget recovers recall":
- Tier-2.A (wider budget) has a **real but length-bounded** win: it recovers 4K
  and partially helps 16K — exactly the regime where the needle ranks in
  `(2048, ~8192]`.
- The loop's actual goal is **long-context** recall (16K/64K). There the gap is
  **scorer-bound**: 64K is firmly scorer-limited and 16K is only ~46%-recoverable
  by budget. **Tier-2.B (a better selector) is necessary**; a wider budget alone
  is insufficient for the long-context goal.
- Net: **both levers matter, in different length regimes** — lead with Tier-2.B
  (the only lever for 64K and the binding lever for 16K), pursue Tier-2.A as a
  bounded win for ≤16K. Unchanged from the M0 decision; now binding at N=20 with
  64K measured rather than inferred.

## Caveats / evidence hygiene
- **Decode-query measurement.** With `ignore_eos`, the decode queries are
  continuation tokens, not the needle question — so these recall numbers
  characterize the needle's retrievability under *generic* decode-step selection
  (the production sparsity point), and likely **under**-estimate a
  question-conditioned answer-step recall. The relative pattern (4K recovers, 64K
  does not) is robust to this; the absolute recall is a lower bound on
  question-conditioned recall.
- recall@K is per (layer, decode-step); the verdict (budget vs scorer) is stable
  across the 61 layers and the 4 decode steps (the rank medians are length-monotone
  and the uplift ordering 4K > 16K > 64K is clean).
- N=20 per length, fail-closed (every issued trial recorded; 0 failure markers);
  this is the binding replacement for the old N=3–4 inferred-64K reading.
