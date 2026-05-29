# AC-Q #H — Round 7: reviewable DS-selection evidence + verdict

Round 6 was STALLED for not capturing DS `meta_info["double_sparsity"]` on the failing
prompts. Round 7 closes that gap with reviewable raw JSON and answers the blocking question:
**is DS dropping context (a selection/label bug), or is the `17*23` loop greedy
degeneration?**

## Why the meta was `None` (root of the Round-6 gap)

`deepseek_v2.py::_publish_ds_request_summary` is gated by
`if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()): publish`
(deepseek_v2.py:2349-2359). Under CUDA-graph **replay** (default decode), the Python publish
never executes, so `meta_info["double_sparsity"]` is `None` for graph-mode decode. It only
populates in **eager** mode. Round 6 probed a graph-mode server → `None`. Round 7 probes an
**eager** (`--disable-cuda-graph`) DS server, where the publish runs every decode step.

## DS selection metadata on the failing prompts (eager, raw JSON committed)

| artifact | prompt_tok + completion_tok | selected_tokens | sparsity_rate | dense_fallback | answer |
|----------|-----------------------------|-----------------|---------------|----------------|--------|
| `ds_meta_eager_chat_1723.json` | 14+120 | 132 | 0.0075 | **0** | (loop-ish) |
| `ds_meta_eager_raw_1723.json` | 11+120 | 129 | 0.0077 | **0** | degenerate `#` (no chat template) |
| `ds_meta_eager_chatthink_1723.json` | 15+250 | 263 | 0.0038 | **0** | **reaches 391** |
| `ds_meta_eager_chat_primes.json` | 14+200 | 212 | 0.0047 | **0** | coherent |
| `ds_meta_eager_concise_1723.json` | 19+2 | 19 | 0.050 | **0** | **391** |
| `ds_meta_eager_concise_primes.json` | 23+8 | 29 | 0.033 | **0** | **53, 59, 61** |
| `ds_meta_eager_sampling_1723.json` (temp 0.5) | 14+160 | 172 | 0.0058 | **0** | **reaches 391** |
| `dsa_ref_1723.json` (DSA temp-0 reference) | — | — | — | — | **391** |

In every case `selected_tokens ≈ seq_len` (the small residual sparsity_rate 0.0038–0.05 is the
1–2 in-flight decode tokens not yet counted in `valid_lengths`) and **`dense_fallback == 0`**.
That is exactly Codex's "healthy seq ≤ top_k" shape: full-context selection, no dropping, no
fallback. **There is no DS selection/label bug.**

## So why does AC-Q's `17*23` fail?

It is fragile temperature-0 greedy decoding, hypersensitive to the exact prompt rendering:
- The sglang `/v1/chat/completions` render (prompt_tokens=15) **deterministically loops** and
  never emits `391` (reproduced again this round at `repetition_penalty` 1.0).
- A manual `<｜begin▁of▁sentence｜><｜User｜>…<｜Assistant｜><think>` render (also prompt_tokens=15)
  **reaches 391**.
- Concise-answer phrasing → `391`; mild sampling (temp 0.5) → `391`.
- A repetition penalty does NOT rescue the exact chat render: rep_penalty 1.1 and 1.3 both
  still fail to reach `391` in 160 tokens (they diverge into different non-answering text).

So under the *exact* AC-Q config (this chat render + greedy + 256 tokens), DS's decode
trajectory on `17*23` diverges into a non-answering loop while DSA's reaches `391`. DS's
selection is healthy; the divergence is in DS's decode-attention numerics tipping greedy into
a bad trajectory on this specific prompt. DS answers correctly under any of: concise framing,
the `<think>` render, or sampling.

## Verdict

- **Not a CUDA-graph bug** (Round 6: eager == graph loop).
- **Not a DS selection/label bug** (this round: full-context selection, `dense_fallback=0`
  across all prompts/lengths, reviewable JSON).
- It is a **narrow greedy-decoding quality gap**: under the exact AC-Q render+greedy config,
  DS fails to answer 1–2 open-ended prompts that DSA answers, because DS's decode numerics
  differ. There is no DS-attention code fix (DS cannot be bit-identical to DSA, and a
  repetition penalty does not fix the exact render).

## Measurement-change request (for approval; nothing changed unilaterally)

AC-Q's ROUGE-L gate measures DS-vs-DSA lexical trajectory identity on greedy 256-token
open-ended generation, which is confounded by this greedy fragility. The answer-agreement
gates (prefix-match, first-8, NIAH) pass. Recommended, in order:
1. **Constrain the AC-Q prompts to concise answers** (most are already "Output only X"; the
   open-ended ones invite the divergence). With concise answers DS == DSA — verified here for
   `17*23`→391 and primes→53,59,61. This tests answer quality, AC-Q's actual intent.
2. **Absolute DS-quality gate** on the known-answer prompts (DS output contains the expected
   answer / is non-degenerate) instead of DS-vs-DSA lexical ROUGE.
NOT recommended: a repetition penalty (tested, does not fix the exact render); a bounded-token
ROUGE (divergence is already < 16 tokens on open-ended prompts).

This requires plan-owner approval per the Round-5/6 review directive. No threshold, prompt
fixture, or decoding default was changed in the harness.
