# AC-12 full quality gate — result (Round 11)

**Verdict: HARD FAILURE.** AC-12 is a hard pass/fail gate (plan §10; DEC-7 directional
handling applies to AC-11 ONLY). NIAH fails at 4K/16K/64K; MMLU 5-shot passes. Therefore the
**loop4-compatible MVP is NOT complete** — the deliverable is a TIER-1 smoke milestone plus a
substantially-complete TIER-2 (AC-10 radix-on, AC-11 comparator, AC-6 graph status, AC-1b
chunked-prefill, and AC-12 MMLU all done) with a **recorded AC-12 long-context NIAH quality
gap**. This is the expected outcome of the documented `top_k=2048` recall limit, now quantified.

## Setup (two-node, both at the locked Option B point)

Two TP=8 DeepSeek-V3.2 FP8 servers cannot co-reside on one 8-GPU node, so the paired gate ran on
both H200 nodes (Codex Round-10 directive):

- **DS** — node 0, `serve_double_sparsity.sh`, **radix-on via the config-bound fixture artifact**
  (`ds_radix_fixture_state.json`, no env override → AC-10 mechanism intact at this operating
  point), `enable_double_sparsity=true`, `mem_fraction_static=0.6`, top_k=2048. `localhost:31020`.
- **DSA baseline** — node 1, `serve_native_nsa.sh` (`HOST=0.0.0.0`), radix-on, `mem 0.85`.
  `10.220.51.5:31010`, reached cross-node.
- Both matched on the locked Option B fields: `tp_size=8`, `page_size=64`,
  `kv_cache_dtype=fp8_e4m3`, `disable_radix_cache=false`, `flashmla_kv` prefill/decode backends,
  `disable_overlap_schedule=true`, `disable_piecewise_cuda_graph=true`, `chunked_prefill_size=8192`.
  They differ only by DS-enablement and the sanctioned `mem_fraction` asymmetry
  (BL-20260529-ds-vs-dsa-memfraction-admission-asymmetry).
- `/get_server_info` captured: `ac12_ds_server_info.json`, `ac12_dsa_server_info.json`.

### Measurement transport (harness fix this round)

The harness queried raw `/generate` for everything. On the instruction-tuned V3.2 checkpoint that
returns an **immediate-EOS empty string** for the long instruction-style NIAH prompts (verified
live), which would make paired recall a vacuous 0/0 on BOTH servers and falsely "pass" the gate
(BL-20260529-dsv32-quality-smoke-needs-chat-template). Fixed `_generate` so:

- **NIAH → `/v1/chat/completions`** (chat template applied; the model answers the instruction).
  Verified: DS NIAH-4K recalls the exact needle via chat vs empty via raw.
- **MMLU 5-shot → raw `/generate`** (unchanged). MMLU 5-shot is a genuine few-shot *completion*
  benchmark; the chat template makes the model answer conversationally and the gold letter is no
  longer the leading token (verified on DS: raw parsed 10/10 correct, chat 0/10). Locked MMLU
  transport stays raw to match `benchmark/mmlu/bench_sglang.py`.

This changes neither the thresholds nor the prompt fixtures — it fixes the transport so the model
actually answers. 47 AC-12 helper CPU regressions still pass.

## Results

| Gate | DSA | DS | Δ (DSA−DS) | Threshold | Verdict |
|------|-----|-----|-----------|-----------|---------|
| MMLU 5-shot (200 ex) | 89.00% | 89.00% | 0.00 pp | ≤ 1.0 pp | **PASS** |
| NIAH @ 4K (20) | 100% (20/20) | 75% (15/20) | 25.0 pp | ≤ 5 pp | **FAIL** |
| NIAH @ 16K (20) | 100% (20/20) | 5% (1/20) | 95.0 pp | ≤ 5 pp | **FAIL** |
| NIAH @ 64K (20) | served 20/20 | **HTTP 400 (unservable)** | — | ≤ 5 pp | **FAIL** |

Artifacts: `ac12_results/ac12_mmlu_5shot_*.json`, `ac12_niah_4096_*.json`, `ac12_niah_16384_*.json`
(64K wrote no artifact — DS rejected the request before generation). pytest summary:
`ac12_results/ac12_pytest_summary.txt` (`3 failed, 1 passed, 2 skipped` — the 2 skips are the
optional corrupt-mask / zero-signature negative-sensitivity servers, not booted).

## Two distinct, both-real failure mechanisms

1. **DS sparse-decode recall is bounded by `top_k=2048` (NIAH 4K/16K).** DS selects only 2048 KV
   tokens per decode step. As context grows past a few× top_k the needle's KV is increasingly not
   selected, so the model cannot attend to it: recall degrades monotonically 75% → 5% from 4K →
   16K. This is an inherent DS sparsity tradeoff, NOT a serving bug — DSA (V3.2's native sparse
   attention, designed for long context) recalls 100% throughout, and DS MMLU (short prompts,
   seq ≤ top_k → effectively-dense selection) matches DSA exactly. Confirms
   BL-20260529-ds-longcontext-needle-recall-vs-topk with hard numbers.

2. **DS cannot admit a 64K context at the radix-on mem-0.6 operating point (NIAH 64K).** The 64K
   NIAH prompt tokenizes to **69,970 tokens**, but DS's KV pool at `mem 0.6` is only
   `max_total_num_tokens=53,056` (the per-rank TokenLabelTable consumes most of the static budget),
   so the server returns `HTTP 400: Input length (69970) exceeds the maximum allowed (53050)`.
   DSA at `mem 0.85` has `max_total_num_tokens=910,784` (≈17×) and served all 20 64K prompts. This
   is the same TokenLabelTable / KV-budget lever behind the AC-11 admission asymmetry — DS at mem
   0.6 has both a smaller concurrency ceiling (AC-11) and a smaller max context (AC-12 64K). Even
   if DS could admit 64K, recall would extrapolate to ≈0% from the 16K result (top_k 2048/65536 ≈
   3% selection).

## What this means for the goal

Per the Ultimate Goal's own framing: *"If AC-10 radix, AC-11 comparator, or AC-12 full quality are
missing, the result is a useful smoke milestone, not the minimal viable working version requested
by loop4."* AC-12 full quality is **not met** (NIAH hard-fails). The honest claim is:

- **TIER 1 smoke MVP: complete** (AC-0/4/1/1.1/8/9/Q + DS short-context quality identical to DSA).
- **TIER 2 loop4-compatible MVP: incomplete** — AC-10/AC-11/AC-6/AC-1b done and AC-12 MMLU passes,
  but **AC-12 long-context NIAH fails** (DS recall is top_k-bounded and DS cannot serve 64K at the
  mem-0.6 operating point). This is recorded as a hard AC-12 failure, **not** reclassified as
  directional.

## Follow-up (filed; does NOT change the AC-12 verdict)

These are inherent-design / operating-point tradeoffs, not bugs to "fix" for a green AC-12:

1. DS long-context needle recall requires a larger `top_k` (a sparsity/perf tradeoff that erodes
   DS's reason for existing) — there is no DS code change that makes a needle in 16K/64K reliably
   selectable at top_k=2048.
2. Serving 64K on DS needs a larger KV budget, which needs the per-rank TokenLabelTable footprint
   reduced (mem 0.7 OOMs during generation today) — the same lever as the AC-11 admission
   follow-up. Until then DS's max admissible context at this operating point is ≈53K tokens.
