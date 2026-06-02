# Loop 7 — DS-vs-DSA same-node served-recall matrix (AC-2, binding)

**The binding recall evidence the loop rests on.** DS-default and DS-hybrid
(Tier-2.B) served NIAH recall vs the **DSA same-node reference**, all measured on
the same 8×H200 node at the Loop-7 op-point, with **Clopper–Pearson exact 95%
CIs** and an up-front materiality rule.

## Method
Server-agnostic NIAH driver (`niah_ds_baseline.py`): `_make_niah_prompt` +
`_niah_needle`, chat transport (`use_chat=True`, the instruction-style recall
prompt elicits the needle answer → real decode steps so DS selection actually
runs), `_niah_recall_hits` scoring, **served-recall separated from
admission/HTTP failure**, N=20 per length, `max_new_tokens=64`. DS engagement
verified live via the `double_sparsity` response meta (16K: `sparsity_rate 0.88,
selected_tokens 2048, dense_fallback 0` — sparse, not an error-contained dense
fallback). Configs:
- **DSA reference**: `serve_native_nsa.sh` (no double-sparsity, same
  `flashmla_kv` prefill/decode backends, page 64, fp8-KV), mem 0.7,
  `niah_dsa_reference.json`.
- **DS-default**: int8 / mem 0.7 / TP=8, default channel-mask scorer (CUDA graph
  on) — the AC-1 served baseline `ds_niah_baseline_mem07.json`.
- **DS-hybrid (Tier-2.B)**: int8 / mem 0.7 / TP=8, `scorer_norm=hybrid`
  (`scorer_norm_hybrid_threshold=8192`: raw ≤8192 tokens, cosine above), eager
  (`--disable-cuda-graph`), `niah_ds_hybrid.json`.

**Materiality rule (stated before any uplift claim):** a DS-variant uplift over
the DS-default baseline at a length is **material** only when the variant recall
point lies **outside the baseline's 95% Clopper–Pearson CI**. A one-/two-needle
move at N=20 is within the CI and is NOT material. DSA is the recall ceiling; the
gap = DSA − DS.

## Matrix (N=20, served recall % [95% Clopper–Pearson CI])

| length | tokenized | DSA (ceiling) | DS-default (baseline) | DS-hybrid (Tier-2.B) | hybrid uplift | material? |
|--------|-----------|---------------|------------------------|-----------------------|---------------|-----------|
| **1024w** (within-budget ≤2048) | ~1.1K | 100% [83.2,100] | 100% [83.2,100] | **100%** [83.2,100] | 0 pp | parity ✓ |
| **4K**  | ~4.4K  | 100% [83.2,100] | 75% [50.9,91.3] | 85% [62.1,96.8] | +10 pp | **NO** (within baseline CI) |
| **16K** | ~17.5K | 100% [83.2,100] | 5% [0.1,24.9]   | **40% [19.1,63.9]** | **+35 pp** | **YES** (40% > baseline CI hi 24.9%) |
| **64K** | ~70K   | 100% [83.2,100] | 5% [0.1,24.9]   | 0% [0,16.8] | −5 pp | **NO** (floor noise: default 1/20 vs hybrid 0/20) |

## Findings
- **Within-budget parity holds (AC-3).** At 1024w (≤2048 tokens, the whole
  context fits the budget) all three are 100% — the Tier-2.B scorer does not
  regress within-budget recall; this is the dense-DS-equivalent parity point
  (when context ≤ budget, DS selects all).
- **DSA is a clean 100% ceiling** at every length (0 admission failures at mem
  0.7), so the gap is purely DS selection quality, not servability.
- **Tier-2.B delivers a MATERIAL 16K uplift**: DS-default 5% → DS-hybrid 40%
  (+35 pp), and 40% lies **outside** the baseline CI [0.1, 24.9] → material by
  the up-front rule. This is the long-context regime that is the loop's goal.
- **4K is a non-material move**: 75% → 85% (+10 pp) lies **within** the baseline
  CI [50.9, 91.3] — recorded, but not claimed as material (per the rule).
- **64K stays scorer-limited**: DS-default 5% (1/20) vs DS-hybrid 0% (0/20) — a
  single-needle difference at the floor, within sampling noise (NOT a material
  regression; both CIs sit at ~0). No scorer reaches the 64K needle, matching the
  oracle's 64K scorer-limited verdict. The −5 pp is reported, not claimed.

## Relation to the M0 oracle (consistency)
The oracle (`m0_oracle_finding_r4.md`) found 16K **budget-partial** (score-only
recall caps ~46% even at 8192) and 64K **scorer-limited**. The served-recall
matrix is consistent: the Tier-2.B scorer lifts 16K materially (toward the
budget-partial ceiling) but cannot reach DSA's 100% — closing the rest needs
either a wider budget (Tier-2.A, bounded) or a still-better scorer. This is a
**recorded, characterized, non-regressing** result, which closes AC-2 even where
the uplift is partial.

## Artifacts
`niah_dsa_reference.json`, `ds_niah_baseline_mem07.json` (DS-default baseline),
`niah_ds_hybrid.json`, consolidated `ds_vs_dsa_recall_matrix.json`.
