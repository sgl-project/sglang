# DS decode degeneration — Round 2 diagnosis (well-localized; fix is next round)

## Symptom
DS `/generate` on V3.2: prefill correct (" Paris."), decode degenerates to a
repeated token. `meta_info["double_sparsity"]` (surfaced in eager mode) shows the
mechanism:
- 1st probe: `selected_tokens=5`, `sparsity_rate=0.82` (seq≈28) — selected far too few.
- short-8tok probe: `selected_tokens=19`, `sparsity_rate=-0.58` (seq≈12) —
  **negative sparsity: selected MORE tokens than exist in the sequence.**

## Controls (what it is NOT)
1. **Not environmental / not the V3.2 stack.** DSA baseline (`serve_native_nsa.sh`,
   same model + `dsa` backend + fp8 KV + flashmla_kv, NO Double Sparsity) produces
   perfect output: " Paris. 法国的首都是巴黎。 The capital of Italy is Rome. ...".
2. **Not CUDA-graph-related.** Booting DS with `--disable-cuda-graph` (eager decode)
   degenerates identically — so the bug is in the core DS selection, not the
   graph-safe capture/replay path.
3. **Not single-step.** A single decode step over a clean 26-token prefill is
   correct (" Rome"). The corruption accumulates over successive decode steps.

## Localization
The eager decode path is `_select_topk_indices` → `selector.retrieve_topk` →
`retrieve_topk_via_labels` (logical-domain mode: `req_pool_indices`+`req_to_token`+
`seq_lens` provided) → `_compute_logical_token_scores` (logical scorer, bounded by
`seq_len` at selection_kernel.py:90 `pos_valid = in_range & (tok_offs < seq_len_i)`)
→ `select_topk_sequence_order` (valid_lengths = count of non -inf scores).

`valid_lengths` should be ≤ seq_len, but is observed > seq_len. So one of:
- the `seq_lens` passed to the selector during DECODE is wrong (larger than the
  true per-request length), or
- the logical scorer's seq_len bound / valid count is wrong (e.g., counts written
  physical slots beyond the request), possibly aggravated by **radix cache off →
  KV slots reused across requests with stale `written` flags**, or
- `select_topk_sequence_order` produces duplicate or invalid positions inflating
  the count.

The over-selection feeds `logical_to_physical` → wrong/duplicate physical slots →
decode attention reads garbage → repetition.

## Next-round fix plan (focused)
1. Instrument `retrieve_topk_via_labels` (eager, env-gated) to log per first few
   decode steps: `seq_lens`, `max_seq_len`, `scores.shape`, `valid_lengths`, and the
   written-count within `req_to_token[pool, :seq_len]`. One DS eager boot.
2. Confirm which of the three causes; fix at the source (seq_lens plumbing vs. the
   logical scorer's bound vs. stale-written invalidation across reused slots).
3. Re-probe: expect `0 < sparsity_rate < 1` with `selected_tokens ≈ seq_len` for a
   short prompt, and coherent decode → unblocks AC-1.1 (task6) and AC-Q.

## Artifacts
- ds_generate_probe.json (graph DS), ds_eager_generate_probe.json (eager DS),
  dsa_baseline_generate_probe.json (baseline control), decode_degeneration_control.md.

## Round-2 code-read refinement
`_compute_logical_token_scores` (the eager logical scorer) DOES mask positions >= seq_len to -inf (selection_kernel.py:495-496), so the over-count is NOT a missing seq_len bound in that scorer. The bug must be in its INPUTS (the `seq_lens` actually passed during decode) or in `select_topk_sequence_order`'s valid count. Round-3 MUST instrument the live values (seq_lens, max_seq_len, finite-score count, valid_lengths) rather than blind-patch the scorer, which reads correct.
