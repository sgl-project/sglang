# Experimental LiteTopK decode and verify on B200

This opt-in path selects the exact FP32 top-2048 KV slots of every query token for DSA decode (`next_n = 1`) and
speculative verify (`next_n` up to 4) in two kernels. SGLang's default dispatch is unchanged; there is no serving
hook.

1. DeepGEMM's `fp8_fp4_paged_mqa_logits` writes the dense scores and, given `histogram=`, also counts every live
   score of a token into 1024 ordered coarse bins. This needs the DeepGEMM change that adds the `histogram`
   argument; without it the call fails.
2. [select.cuh](select.cuh), compiled on first use by SGLang's JIT, reads each token's histogram, finds the bin
   holding its 2048th largest score and scans the scores once: scores above that bin are selected outright, the
   bin's own scores are ranked exactly. From 16 to 63 tokens a CTA has 1024 threads, of which only long parts use
   all 32 warps. Batches with 149–192 or 297–444 token rows use three 512-thread CTAs per SM to shorten the final
   scan wave. Any disagreement between the histogram and the scores (for example a stale
   histogram) or a crossing bin larger than the candidate capacity falls back to an exact radix select over the whole
   row, so the result never depends on the histogram being right. Ties on a score go to the lower physical slot.

The selector clears the histogram and candidate buffer and balances its persistent hand-off counters, so no reset
kernel runs between calls or CUDA graph replays.

## API

```python
from sglang.kernels.experimental.litetopk_decode.fused import FusedDecodePlan

plan = FusedDecodePlan(batch, next_n, device)  # allocates buffers and compiles the selector
# Same arguments as deep_gemm.fp8_fp4_paged_mqa_logits: q FP8 [B,next_n,32,128], KV cache [pages,64,1,132],
# weights FP32 [B*next_n,32], context_lens int32 [B,next_n], block_table int32 [B,pages] shared by a request's
# tokens, schedule metadata from deep_gemm.get_paged_mqa_logits_metadata
physical_topk = plan(q, kv_cache, weights, context_lens, block_table, schedule_metadata, max_context_len)
# int32 [B*next_n, 2048], unsorted, padded with -1; overwritten by the next call
```

For speculative verify, `context_lens[b, j]` must be token `j`'s causal length; build the schedule from that same
tensor. SGLang's producer-only verify metadata can repeat the final length across all tokens, so it cannot be
passed here unchanged: the histogram and selection must count only each token's visible keys.

`candidate_capacity` (default 8192) bounds the crossing-bin scores kept per token; the plan allocates
`B*next_n*(16 + 8*capacity)` bytes of workspace, whose layout depends on the row count: the selector refuses any
other size. Create the plan before CUDA graph capture. The grid (148, 296 or 444 CTAs) and the thread-count
thresholds are tuned for the 148-SM B200; on another part they only cost speed.

The plan does not change DeepGEMM's PDL setting. Standalone DeepGEMM defaults to PDL off; call
`deep_gemm.set_pdl(True)` before warmup and capture to enable it. SGLang's existing DeepGEMM wrapper enables it
by default through `SGLANG_DEEPGEMM_PDL`, but this experimental module imports DeepGEMM directly.
The selector itself enables PDL on launch, waits for its producer before reading data and then lets its
dependent launch early (a PDL dependent waits for the selector to finish before reading its output). Its score
loads use ordinary global memory instructions, including the fallback paths. This also works with DeepGEMM PDL
off.

The plan does not create a CUDA graph. Warm it up, capture a call with `torch.cuda.graph`, then replay the graph;
keep tensor addresses and shapes fixed, and copy updated lengths and their corresponding schedule into the
captured buffers. The smoke below exercises this flow. Serving integration can reuse SGLang's existing graph
runner once a dispatch hook and per-shape plans are added.

## Validation

[smoke_fused.py](smoke_fused.py) compares every token's selected slots with the exact top-2048 of DeepGEMM's own
scores for decode and verify batches, in eager calls and graph replays, and checks that the histogram and candidates
return to zero and the hand-off counters balance, including after a stale histogram and a candidate overflow:

```sh
python3 python/sglang/kernels/experimental/litetopk_decode/smoke_fused.py --batches 1,3,33,128 --next-n 1,2,3,4 --pdl
# Long verify rows also exercise the 1024-thread scan and transitions between long and short rows on replay.
python3 python/sglang/kernels/experimental/litetopk_decode/smoke_fused.py --batches 4,8,16,21 --next-n 2,3,4 --max-len 1048576 --pdl
```

Omit `--pdl` to check the standalone DeepGEMM default as well. Both commands test eager calls and CUDA graphs.
