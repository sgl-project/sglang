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

## RESOLUTION UPDATE — #16 is TWO bugs (bug #1 fixed, bug #2 open)

On-hardware instrumentation (env-gated DS_DEBUG_SELECT / DS_DEBUG_WRITE, since removed)
decomposed the degeneration into two independent bugs:

### Bug #1 — req_to_token None during decode → wrong selection domain (FIXED, commit 2af5f4e65)
`_select_topk_indices` read `req_to_token` from `forward_batch.req_to_token_pool`, which
is None on the decode path. Effects: (a) the selector fell into physical-domain mode
(returned physical slot indices, e.g. sel_head=[64,65,66,67,68] instead of logical
[0,1,2,3,4]); (b) `logical_to_physical` was skipped (`ds_out.fill_(-1)`) so decode
attention had no valid slots. Fix: resolve `req_to_token` from the ForwardContext
attention backend (caches model_runner.req_to_token_pool at init). Validated:
sel_head→[0,1,2,3,4,...]; output "Paris. DDDD" → "Paris. The capital of France...".

### Bug #2 — decode tokens are never label-written (OPEN)
After bug #1, `written_in_seq` stays frozen at the prompt length (e.g. 5) even as
seq_len grows to 28, so DS can only ever select the prompt tokens; the model can't see
its own generated tokens and loops. Write-side probe (DS_DEBUG_WRITE in
dsa_backend._write_token_labels) showed EVERY call is `mode=EXTEND`; there are ZERO
`mode=DECODE` label writes. So `_write_token_labels` is not invoked on the decode path.
On H200/CUDA the aiter/tilelang fused branches are off, so `forward_absorb_core` calls
`attn_mqa` (→ forward_decode) with k not None and save_kv_cache True — yet the label
write still does not fire on decode. Next step: instrument `forward_decode` directly
(log reached?/k is None?/save_kv_cache) for a decode step, then add the decode-token
label write (the projected K-noPE is available as `k_nope` in forward_absorb_core).

## Bug #2 — COMPLETE root cause (two layers); correct fix scoped for next round

Bug #2 is itself two-layered, both confirmed on hardware via env-gated probes:

(2a) `kv_b_proj` is lazily attached only to `attn_mha` (prefill), never to `attn_mqa`
     (decode) — deepseek_v2.py ~1724. So decode's `_write_token_labels(layer=attn_mqa)`
     hit the `kv_b_proj is None` guard and returned early (DS_DEBUG_WRITE_GUARD:
     mode=DECODE kv_b_proj_none=True vs mode=EXTEND kv_b_proj_none=False).

(2b) DEEPER: even with kv_b_proj attached, the decode path passes the WRONG tensor.
     `forward_absorb_core` (forward_mla.py:513) calls
     `attn_mqa(q_nope_out, k_nope, k_nope, ...)`, so `forward_decode` receives
     `k = k_nope` — the ALREADY-PROJECTED K-noPE — not the 512-d latent key.
     `_write_token_labels` assumes the latent and re-projects through kv_b_proj,
     giving `RuntimeError: shape '[1, 16, 640]' is invalid for input of size 4096`
     (decode k_nope is [T, H_local, *], last dim != kv_lora_rank=512).

So merely attaching kv_b_proj to attn_mqa (attempted, then REVERTED) is wrong — it
feeds an already-projected tensor into the projection. The correct fix writes decode
token-labels from the already-projected `k_nope` directly (available in
`forward_absorb_core` as a param, and as the selector's expected
`[T, H_local, nope_dim]` input to `token_label_write`), bypassing the kv_b_proj
re-projection. Candidate approaches for next round:
  - Add a decode-path label write in `forward_absorb_core` using the in-hand `k_nope`
    + the selector's `token_label_table`/`channel_mask` (skip kv_b_proj), OR
  - Make `_write_token_labels` accept pre-projected k_nope (detect last-dim !=
    kv_lora_rank) and skip the projection.
Must validate: after the fix, `written_in_seq` grows with seq_len (not frozen at
prompt_len), `0 < sparsity_rate < 1` on a >top_k prompt, and coherent decode.
