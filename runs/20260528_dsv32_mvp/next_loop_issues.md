# Loop 5 — issues to knock out in the next RLCR loop

Loop 5 shipped the TIER-1 smoke MVP and the full evidence bundle, and **AC-12 is MET under the
DS-fair re-scope** (the AC-12 disposition was decided in this loop — see below). This file lists
the carried-over R&D / follow-up work for a future loop, ordered by impact.

## 0. AC-12 disposition — DECIDED (no longer open)
The AC-12 disposition was resolved in this loop, **not** carried forward. The user authorized a
**DS-fair re-scope** of AC-12 (option (b) of the earlier choice). DS is dense-prefill /
sparse-decode with a fixed selection budget = the model's native DSA `index_topk=2048`
(kernel-locked); testing needle recall at 4K/16K/64K tested DS beyond its budget (a selection-quality
limit vs V3.2's trained DSA indexer, not a decode bug — DS recalls 100% when its selection is dense,
and MMLU == DSA). The re-scoped AC-12 HARD gates (MMLU within 1pp + NIAH within the selection budget
within 5pp of DSA) **PASS** on hardware; the beyond-budget 4K/16K/64K recall is transparently
**characterized** (DS 75%/5%/0%, artifacts keep `verdict=FAIL`). See `ac12_analysis.md`,
`evidence_bundle.md`, and the Round-14 Plan Evolution in the goal-tracker. The items below are the
R&D that would lift the beyond-budget limits if DS long-context recall is pursued further.

## 1. DS long-context recall R&D (to approach DSA selection quality)
- A **query-aware / learned DS selector** that places the needle in the 2048 budget (vs the current
  offline Method-1 channel-mask), i.e., closes the selection-quality gap to V3.2's DSA indexer.
- A **`flashmla_kv` decode-kernel variant that accepts `top_k > index_topk`**, so DS could trade
  compute for recall by widening the selection budget (today the kernel asserts
  `indices.shape[-1] == dsa_index_topk`, hard-capping DS at 2048).

## 2. DS KV-budget / TokenLabelTable footprint (admission)
- DS at `mem_fraction_static=0.6` has `max_total_num_tokens≈53K` (vs DSA's ≈910K at 0.85), which
  (a) makes 64K context **unservable** (the beyond-budget 64K HTTP 400) and (b) caps effective
  concurrency (the AC-11 directional TTFT miss). Both share one lever: **shrink the per-rank
  TokenLabelTable** so DS can serve at a higher mem fraction without the generation-time OOM seen at
  0.7. Orthogonal to recall (a 64K needle still misses the 2048 selection), but required for 64K
  servability and for closing the AC-11 admission gap.

## 3. AC-11 directional TTFT follow-up (DEC-7)
- Re-sweep once item 2 lifts DS effective concurrency; the recorded directional TTFT miss
  (admission-bound at mem 0.6) should be re-evaluated against DSA. (Filed in `ac11_analysis.md`.)

## 4. AC-12 within-budget gate: assert from actual token counts (Codex queued #1)
- The within-budget hard gate currently sizes prompts by WORD count (1024/1536) and derives
  `within_budget` from `length_tokens <= INDEX_TOPK`. Current evidence is validated safe (tokenizer
  sanity: 1024w→1097 tokens, 1536w→1658 tokens, both < 2048), but the next substantive harness touch
  should record the **actual chat input token count** (e.g. `usage.prompt_tokens`) and assert the
  within-budget gate from that, not from the word-count proxy.

## 5. Strategic question (flagged by the Round-13 investigation)
- **Is Double Sparsity worthwhile on a model that already ships native DSA?** On V3.2, DS is capped
  at the native `index_topk` budget by the shared decode kernel AND uses an inferior offline
  selector — so it cannot match (let alone beat) DSA's long-context recall at the shared budget.
  DS's value proposition is clearer on models WITHOUT a trained sparse indexer. Worth an explicit
  decision before further DS investment on V3.2.

## 6. Cosmetic
- The `serve_*.sh` operating-point headers and the `serve_double_sparsity.sh` HiSparse-exclusion
  comment have been reworded to behavior-based language (no plan markers). The only remaining
  plan-process references are the deliberate `# Round 26-29 …` MMLU-history comments in
  `test/manual/test_double_sparsity_v32.py` (historical rationale, intentionally kept).
