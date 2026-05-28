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
