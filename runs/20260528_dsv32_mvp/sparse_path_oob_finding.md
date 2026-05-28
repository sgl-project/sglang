# #18 — DS genuine-sparse path crashes (CUDA illegal address) when seq_len > top_k

## Symptom
After the #16 decode-degeneration fix, DS serves coherently for seq_len < top_k=2048,
but ANY request with seq_len > top_k=2048 crashes the server with
`CUDA error: an illegal memory access was encountered` (cudaErrorIllegalAddress),
which then aborts all 8 ranks via the NCCL watchdog. Crash site:
`deepseek_v2.py:2433 _select_topk_indices` (the genuine sparse top-k selection),
reached from `forward_absorb_prepare`.

## Bisection (mem_fraction=0.6, eager, top_k=2048)
| prompt_tokens | seq vs top_k | result |
|---------------|--------------|--------|
| ~28           | < 2048       | OK, coherent (#16 fix) |
| 1376          | < 2048       | OK, sparsity_rate≈0, dense_fallback=0, selected≈seq |
| 1933          | < 2048       | OK, sparsity_rate≈0, dense_fallback=0, selected≈seq |
| ~2316         | > 2048       | CRASH (cudaErrorIllegalAddress) |
| ~3500         | > 2048       | CRASH (cudaErrorIllegalAddress) |

So the trigger is **seq_len crossing top_k=2048** — i.e. the first time DS actually
sparsifies (selects a subset rather than all tokens). For seq < top_k the selector
returns all tokens (effectively dense) and the buggy sparse indexing path is not hit.

## Impact
Blocks AC-1.1 (genuine `0 < sparsity_rate < 1` requires seq > top_k), and any
real-workload benchmark/quality run (AC-8/9/Q/11/12) at the locked operating point.
#16 (coherent DS output) is unaffected and remains fixed for the dense-equivalent regime.

## Next-round investigation (compute-sanitizer)
Re-run a >2048-token request under `compute-sanitizer --tool memcheck` (or
`CUDA_LAUNCH_BLOCKING=1` + a python-level bisect of the selection kernels) to localize
the offending kernel/index. Prime suspects in the sparse path:
- `selection_kernel.select_topk_sequence_order` / the Triton logical-score kernel when
  max_seq_len > top_k (scratch/index bounds sized by top_k or device_buffer_size=4096),
- `page_table_adapter.logical_to_physical` / the FlashMLA sparse decode page table when
  the selected-index count is exactly top_k over a longer sequence,
- DS graph-state scratch buffers (scratch_scores etc.) sized for a max that the
  >top_k sequence exceeds.

## DEEPER root cause (continued investigation) — DS prefill selection has bad batch semantics

Localized with CUDA_LAUNCH_BLOCKING=1 (cheap, no compute-sanitizer) and a force-eager
toggle:
- The crash fires during the LONG-PROMPT PREFILL (forward_extend → forward_absorb_prepare
  → `_select_topk_indices`), at layer 0 — NOT decode.
- It is NOT graph-safe-Triton-specific: forcing the eager pure-torch selection
  (`_compute_logical_token_scores`, confirmed no Triton) ALSO fails, but with a telling
  error: `double_sparsity error cls=bad_adapter_input message=464 bad req_pool_indices
  in batch at layer 0`. `logical_to_physical` found hundreds of out-of-range
  `req_pool_indices` in the batch → bad physical slots → CUDA illegal access downstream
  (the graph-safe path manifests the same as a Triton OOB).
- Short prompts avoid this entirely via the MHA_ONE_SHOT bypass (`use_mha` →
  `_select_topk_indices` returns None). Long prefills (> the MHA one-shot threshold,
  ≈ top_k/2048 tokens here) use the MLA absorb prefill, which DOES run DS selection —
  and the selection's per-extend-token batch / `req_pool_indices` handling is wrong.

## Real fix (substantial, next round)
DS sparse selection during PREFILL/extend mishandles the extend-token batch layout
(req_pool_indices count/values don't match the extend-token rows). Options:
1. Correctly expand/index `req_pool_indices` (and seq_lens) per extend token for the
   prefill selection (match how DSA's indexer handles prefill queries), OR
2. Determine the intended DS prefill-selection semantics vs DSA and align them.
Note: simply skipping DS selection during prefill would diverge from DSA (which DOES
select for prefill queries), so verify against DSA semantics before choosing.
This is the blocker for AC-1.1 (genuine sparsity needs seq>top_k) and real-shape runs.
