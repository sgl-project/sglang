# Loop 7 M0 — A-vs-B decision (Codex-adjudicated, evidence-based)

The measure-first M0 milestone is complete. Evidence: `m0_baseline.md` (served-recall, mem 0.7, N=20 + CIs) and `m0_oracle_finding.md` (oracle score-only recall@K, the budget-vs-scorer decider). Codex adjudicated the A-vs-B call (task7) on this evidence.

## Decision (refines DEC-1)

**Lead Tier-2.B (better non-learned selector); Tier-2.A is a bounded secondary / moderate-context win — NOT the main path to long-context recall.**

### Oracle-uplift gate
- **Met at 4K only** (budget-limited): score-only recall@2048=0% → recall@4096=100%, needle rank 2105–2580 (just past the 2048 budget). A 4096-budget decode genuinely recovers 4K.
- **NOT met at 16K** (scorer-limited): recall@2048/4096/8192 all 0%, needle rank ~8.8K–10.3K (≈ its sequence position — the channel-mask scorer barely discriminates the needle at length). No feasible budget (≤8192) helps.
- **64K**: inferred scorer-limited (needle at pos ~35–41K of ~70K; oracle records absent — known bug). 
- Small N blocks a *production-binding statistical* claim but not the *directional engineering* call (the 4K and 16K rank margins are decisive within their trials).

### Strategic-gate supersession (for task20's decision record)
The gate (`ds_on_v32_decision.md`) named **Tier-2.A primary**. M0 evidence **supersedes** that ordering: the oracle did NOT show broad budget recoverability — only at 4K, while 16K stayed unrecovered even at 8192. Corrected ordering: **Tier-2.B is Loop-7 primary for long-context recall; Tier-2.A is justified only as an opt-in moderate-context improvement / measurement aid, not the main path to 16K/64K recall.** The prior rationale was sound when written; the oracle data is what changed.

### Tier-2.B direction (M1 / task8)
Codex's concrete first non-learned lever: **length-/channel-normalized scoring before top-K**, with head/layer aggregation audited separately. Rationale: at 16K the needle rank ≈ position ⇒ the score is dominated by positional/background magnitude, not needle salience; a normalization pass that removes length/position/channel scale bias is more likely to raise the needle's rank than adding budget.

### M1 measurement contract
Report served recall AND score-rank recall on the SAME prompt path; recall@2048 + needle-rank distributions at 4K/16K/64K. **Materiality (per the M0 CIs):** 16K and 64K must exceed the **24.9%** baseline-CI upper bound (the 5% baseline's CI) to count as material, AND the needle-rank must move upward (proving the selector — not decode luck — caused the gain).

## Codex risks to carry
Small oracle N may overstate regime separation; eager vs graph/runtime score path may differ; per-layer/head aggregation may hide a discriminative layer; 64K oracle data absent (inferred); baseline (chat) vs oracle (raw) prompt-path difference; score-only recall ≠ decoded-answer recall until a changed selector is served-measured. Codex confidence: **medium-high**.

## Status
M0 DONE (instrumentation + baseline + oracle + adjudicated decision). Next: task20 (write the gate-supersession decision record from this), the oracle firm-up + 64K fix (queued, before any binding recall claim), then **M1 = task8 (length/channel-normalized scorer)**. DSA re-confirm (documented 100%) + MMLU re-anchor remain for AC-2/AC-3 closure.
