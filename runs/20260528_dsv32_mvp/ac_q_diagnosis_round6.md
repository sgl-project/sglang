# AC-Q failure (#H) — Round 6 diagnosis

Investigating why the AC-Q ROUGE-L gate failed (0.726 < 0.85) and whether the DS decode
failures Codex flagged (`17 * 23` loops without emitting `391`; primes truncates) are a DS
bug or something else. All probes on 8x H200, DS booted from cluster weights, radix-off,
`/v1/chat/completions`, temperature 0.

## Controls run

1. **Eager vs CUDA-graph (decisive).** DS booted graph-mode (default) and again with
   `--disable-cuda-graph`. Both produce the **identical** `17 * 23` repetition loop and the
   identical dropped-`17` (`\( \times 23 \)`). → **Not a CUDA-graph bug.** (Matches the
   prior decode lesson's "eager failed identically" note.)
   Artifacts: `ds_diag_graph_chat_1723.json`, `ds_diag_eager_chat_1723.json`.

2. **Does DS know the answers? (short-answer framing).** Same DS server:
   - "What is 17 times 23? Output only the number" → **`391`** ✓
   - "List three prime numbers between 50 and 80. Output only the three numbers" →
     **`53, 59, 61`** ✓
   - "Give the SI unit of electric current" (short) → ampere ✓
   → DS's arithmetic/recall is **intact**; the failure is not a knowledge/correctness loss.

3. **Greedy vs sampling.** The same `Compute 17 * 23 …` prompt at temperature 0.5 → DS
   **reaches `391`**. → The loop is a **greedy (temperature-0) decode degeneration**, not a
   deterministic inability to compute.

4. **DS coherent on most prompts.** Pythagorean (a passing AC-Q prompt) → DS matches DSA;
   all 11 short factual AC-Q answers were exact in Round 5; NIAH recall 5/5.

## Why ROUGE-L is low — trajectory divergence, early

Offline over the Round-5 outputs, truncating BOTH DSA and DS to the first N whitespace
tokens:

| first-N tokens | mean ROUGE-L | ≥0.85 |
|----------------|--------------|-------|
| 8 | 0.894 | yes |
| 16 | 0.815 | no |
| 32 | 0.790 | no |
| 64 | 0.751 | no |
| full (256) | 0.726 | no |

DS and DSA agree for the first few tokens, then **diverge in generation trajectory within
~16 tokens** on the 7 open-ended prompts (list/sequence/explanation/arithmetic): DS tends to
be markedly more verbose (e.g. SI-unit: DSA 13 tokens vs DS 163; 17*23: DSA 45 vs DS 178).
So a simple bounded-token comparison does NOT rescue the gate — the divergence is early, not
a late tail.

## Root cause

The DS decode attention is numerically different from DSA's (DS routes decode through the
channel-mask sparse-decode path / `flashmla_kv`; DSA uses the native NSA indexer). Under
**temperature-0 greedy** decoding, that small numerical difference makes the two models
follow **different (both individually valid) generation trajectories** on open-ended
prompts, and on `Compute 17 * 23` DS's greedy trajectory falls into a well-known repetition
loop that never reaches `391`. This is:

- **NOT a CUDA-graph bug** (eager == graph),
- **NOT a DS correctness/knowledge loss** (DS gives 391, the primes, the SI unit when asked
  concisely; escapes the loop under mild sampling),
- but **a real DS-amplified greedy-degeneration on a minority of long-CoT prompts** under
  the exact AC-Q decoding config (temp 0, 256 tokens, no repetition penalty).

There is no DS *code* fix for temperature-0 greedy degeneration: DS attention cannot be made
bit-identical to DSA's, and greedy looping is a property of the decoding config, not of the
DS selection/label path (which is correct — short answers and NIAH are perfect).

## Consequence for AC-Q

The AC-Q ROUGE-L gate measures DS-vs-DSA **lexical trajectory identity** on temp-0 256-token
free-form generations. Two different attention mechanisms cannot satisfy that on open-ended
prompts (divergence by ~16 tokens). The other three gates — prefix-match (answer start),
first-8-token overlap, and NIAH recall (answer correctness) — measure **answer agreement**,
and DS **passes** all three. The gate that fails is the one most confounded with greedy
trajectory chaos.

A measurement change is required and is proposed (NOT applied unilaterally) in the
round-6 summary's Goal Tracker Update Request, with these candidate options:
1. Make AC-Q an **absolute DS-quality** gate on the known-answer prompts (DS output contains
   the expected answer / is non-degenerate) instead of DS-vs-DSA lexical ROUGE — but note
   `17 * 23` still fails absolutely under temp-0/256-token greedy, so this also needs (2).
2. Change the AC-Q **decoding config** to one that is not greedy-degenerate (a small
   repetition penalty, or a low fixed temperature with a fixed seed) applied identically to
   DS and DSA, keeping determinism via the seed.
3. Constrain the open-ended prompts to request concise answers (where DS == DSA).

All three are plan/measurement changes that require Codex/user approval per the Round-5
review directive; no threshold, prompt, or decoding default was changed in the harness this
round.
