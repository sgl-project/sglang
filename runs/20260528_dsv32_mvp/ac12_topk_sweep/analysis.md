# AC-12 NIAH selection-budget investigation (Round 13)

**User question:** "look into whether if we significantly increase selection budget we can pass
NIAH. if not, then there is a serious issue."

**Answer:** (1) DS's selection budget **cannot** be significantly increased — it is structurally
locked to the model's native DSA `index_topk=2048` by the shared decode kernel. (2) This is **not a
serious decode bug**: when DS's selection is *complete* (seq ≤ 2048 → dense) it recalls **100%**,
identical to a dense model; short-context MMLU already equals DSA. The AC-12 NIAH failure is a
**selection-quality** limitation at the fixed 2048 budget — V3.2's *trained* DSA indexer puts the
needle in its 2048 while DS's *offline channel-mask* heuristic does not.

## Finding 1 — the DS selection budget is hard-locked to `index_topk=2048`

Booting DS with `top_k=8192` fails in two layers (`boot_evidence_topk_locked.txt`):

- The startup validator refuses: *"Double Sparsity top_k=8192 does not match the model's DSA
  index_topk=2048 … set SGLANG_DS_ALLOW_TOPK_MISMATCH=1 to override."*
- With the override, the **decode kernel itself** asserts during CUDA-graph capture
  (`dsa_backend.py:2148`, `_forward_flashmla_kv`): `assert indices.shape[-1] == self.dsa_index_topk`.

DS runs on DeepSeek-V3.2's **native DSA** attention backend and reuses its `flashmla_kv` sparse-decode
kernel, which consumes **exactly** `index_topk=2048` indices. So DS cannot select more than 2048
tokens per decode step without a different decode kernel. "Significantly increasing the selection
budget" is therefore not possible on the current DS + flashmla_kv path — the override bypasses the
config check but not the kernel contract. **This is an architectural constraint, not a bug.**

## Finding 2 — DS decode is sound; the gap is selection quality at the fixed 2048 budget

DS-only NIAH recall at the locked `top_k=2048` (chat transport, 20 prompts/length,
`ds_recall_vs_length_topk2048.json`), versus the established DSA reference (100% — DSA uses the
*same* `flashmla_kv` kernel and the *same* 2048 budget, via its learned indexer):

| Context (words) | Selected fraction | DS recall | DSA recall |
|-----------------|-------------------|-----------|------------|
| 1024 | 100% (dense, budget ≥ seq) | **100% (20/20)** | 100% |
| 1536 | 100% (dense) | **100% (20/20)** | 100% |
| 4096 | ~50% | 75% (15/20) | 100% |
| 16384 (AC-12) | ~12.5% | 5% (1/20) | 100% |
| 65536 (AC-12) | unservable at mem 0.6 | HTTP 400 | served |

Reading the curve:

- **DS decode is correct.** At dense selection (seq ≤ 2048) DS recalls 100%, identical to a dense
  model; and AC-12 MMLU (short prompts) already had DS == DSA (89.0%). So there is no decode/serving
  bug — the "serious issue" check is negative.
- **Recall degrades purely with the selected fraction** (100% → 75% → 5% as 100% → ~50% → ~12.5% of
  tokens are selected). The needle's KV must be among the 2048 selected to be attended.
- **The DS-vs-DSA gap at the SAME 2048 budget is selection quality.** DSA recalls 100% at 16K because
  its *trained, query-aware* DSA indexer reliably places the needle in its top-2048; DS's
  *offline Method-1 channel-mask* importance scoring does not. The deficit is *which* 2048 tokens are
  chosen, not *how many*.

## Implication for AC-12 (informs the next-loop decision, does not change this loop's verdict)

On a model that **already has a native trained sparse indexer (DSA)**, Double Sparsity is (a) capped
at the native `index_topk` by the shared decode kernel and (b) selects with an offline heuristic that
is inferior to the model's learned indexer for long-context needle recall. So DS cannot match DSA's
long-context recall at the shared budget, and cannot widen the budget without a new decode kernel.
AC-12 NIAH at 4K/16K/64K therefore remains a **hard fail** at the locked operating point — now
understood precisely as a selection-quality + kernel-budget-cap limitation, **not** a decode bug and
**not** something a larger `top_k` could fix on this backend.

Paths that *could* close AC-12 NIAH (next-loop R&D, none in scope here):
1. A better DS selector (query-aware / learned indexer) that places the needle in the 2048 budget —
   i.e., approach DSA's selection quality.
2. A decode kernel that accepts `top_k > index_topk` (so DS could trade compute for recall by
   widening the budget) — a new flashmla_kv variant.
3. Raising the DS KV budget (shrink the per-rank TokenLabelTable) to make 64K *servable* — orthogonal
   to recall (a 64K needle would still miss the 2048 selection), but needed for the 64K admission gap.

## Artifacts
- `ds_recall_vs_length_topk2048.json` — the DS recall curve above (top_k=2048, chat transport).
- `boot_evidence_topk_locked.txt` — the validator refusal + the `flashmla_kv` kernel assertion that
  lock the budget to `index_topk=2048`.
