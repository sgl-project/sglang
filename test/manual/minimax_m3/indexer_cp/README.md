# MiniMax-M3 indexer context partitioning on gfx950

This opt-in path partitions **index-cache reads** across four tensor-parallel
ranks during ordinary decode. The cache remains replicated, so this does not
reduce cache capacity. It applies to index-only sparse layers that need a new
top-k selection; layers using index values or reusing a selection retain their
existing paths.

## Algorithm

1. Gather the four existing post-RoPE index-query heads through the attention TP
   communicator. The model projections and checkpoint layout stay the same.
2. Rank `r` scores blocks `r, r+4, r+8, ...` for all four heads, reusing each
   loaded K tile across heads. Each block contains 128 tokens.
3. Keep 16 candidates per head **on each rank**. Pack scores and block IDs into
   64-bit keys with the native ROCm selector's score-descending, ID-ascending
   ordering. Initial/recent block priorities are applied before selection.
4. Gather the candidates through SGLang's communicator. Each rank merges the
   64 candidates for its own head and emits 16 ascending block IDs, with `-1`
   padding for short/empty rows. The main sparse attention consumes these IDs.

Keeping 16 candidates on every shard preserves global top-16 selection: a
candidate discarded by a shard already has 16 better candidates on that shard.
Keeping only four per shard would fail when the winners concentrate on one shard.

The scorer and selectors are Triton kernels. SGLang's communicator selects AITER
custom all-gather when supported, with its existing collective fallback and
graph-buffer registration. Candidate keys are bitcast to FP32 views for the
copy-only gather; they are never numerically converted. Both collective paths
are warmed before capture.

## Enable

Add this to the environment of an existing MiniMax-M3 launch:

```bash
export SGLANG_MINIMAX_M3_INDEXER_CP=1
```

The flag defaults to `0`. The initial scope requires:

- AMD gfx950, TP4, attention TP4, and attention CP/DP size 1.
- Four global index heads and four main KV heads, dimension 128, block size 128,
  top-k 16, max scoring, and the native ROCm radix top-k selector enabled.
- Ordinary decode without speculation, TBO, HiSparse, FP8 queries, or dense
  sparse decode. Configured context length must be at most 16,384 blocks.
- One local query head and one replicated index-K head per rank. The shape gate
  accepts BF16/FP16 queries with a matching or FP8 cache; the measurements below
  cover BF16 queries and BF16/FP8 E4M3 caches only.

Unsupported run configurations log a reason and retain the existing path.
Prefill and main sparse attention are unchanged. The implementation does not
automatically choose a context/batch threshold: the short-context regressions
below make workload validation necessary before enabling the flag.

## Reproduce the indexer checks and timings

Use this SGLang checkout in a ROCm environment with its dependencies installed,
including AITER and the native ROCm radix selector. Run on four gfx950 GPUs:

```bash
SGLANG_USE_AITER=1 SGLANG_USE_AITER_AG=1 \
SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX=1 \
HIP_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  test/manual/minimax_m3/indexer_cp/bench_cp.py \
  --output /tmp/minimax-indexer-cp.json
```

`--quick` retains all eight correctness cases and reduces the timing grid to
batches 1 and 16. The harness calls the CP helper directly, so the feature flag
is not needed for this command. It does **not** exercise the server's feature
gate, backend dispatch, model accuracy, or serving throughput.

### Correctness checked on all four ranks

- BF16 and FP8 E4M3 caches, each with random scores, tied scores, winners
  concentrated on one shard, and mixed lengths: eight cases passed.
- Exact selected-ID comparison against both native SGLang and an independent
  FP32 PyTorch reference, including partial blocks and an empty row.
- Lengths changed after graph capture to include short contexts, a partial
  block, and an empty row; replay matched both references.
- Every timed shape additionally passed native-versus-CP ID parity in eager
  execution and graph replay.

### Measurement method

- Four MI355X GPUs; PyTorch `2.10.0+rocm7.2.4.git3d3aa833`, Triton
  `3.7.0+amd.rocm7.2.0.git89002410`, AITER
  `94dca7bc674650b6de2609eeae2bcdd58e970bf3`.
- Identical inputs for both arms: BF16 Q, FP8 E4M3 K, dimension 128, top-16
  blocks, one initial/two recent forced blocks, shuffled physical pages.
- Native scorer plus radix top-k versus the CP chain, including query gather,
  shard scoring, local selection, candidate gather, and merge. AITER custom
  gather was available. Projections and main sparse attention are excluded.
- Eight calls per HIP graph, 100 replays per round, seven rounds alternating
  arm order. Each sample uses the slowest rank; the table reports the median
  of seven samples.
- Repeated inputs and cache are warm. No L2 flush or rotating-layer protocol
  was used. These are indexer microbenchmarks, **not model throughput gains**.

## Results: 2026-09-28

Regenerate the raw samples with the command above; the environment is recorded under
Methodology. Context lengths use K = 1,024 tokens.
A negative latency change is an improvement.

| Context | Batch | Native TP (µs) | CP (µs) | CP latency change | Native / CP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 1 | 9.40 | 15.40 | +63.8% | 0.611× |
| 8K | 2 | 9.48 | 17.60 | +85.7% | 0.538× |
| 8K | 8 | 7.72 | 18.32 | +137.3% | 0.421× |
| 8K | 16 | 8.70 | 18.80 | +116.1% | 0.463× |
| 8K | 32 | 12.12 | 19.58 | +61.5% | 0.619× |
| 32K | 1 | 11.56 | 15.55 | +34.5% | 0.743× |
| 32K | 2 | 13.86 | 18.06 | +30.4% | 0.767× |
| 32K | 8 | 14.24 | 18.81 | +32.1% | 0.757× |
| 32K | 16 | 20.80 | 20.04 | -3.7% | 1.038× |
| 32K | 32 | 30.41 | 24.38 | -19.8% | 1.248× |
| 128K | 1 | 12.71 | 15.93 | +25.4% | 0.797× |
| 128K | 2 | 15.60 | 18.76 | +20.3% | 0.831× |
| 128K | 8 | 32.85 | 24.02 | -26.9% | 1.368× |
| 128K | 16 | 51.71 | 32.08 | -38.0% | 1.612× |
| 128K | 32 | 109.84 | 41.63 | -62.1% | 2.639× |
| 1 × 128K + remaining × 1K | 16 | 40.76 | 27.82 | -31.7% | 1.465× |
| 1 × 128K + remaining × 1K | 32 | 68.89 | 34.62 | -49.7% | 1.990× |

CP loses at all tested 8K shapes and at batches 1–2 even with 128K context.
The 3.7% improvement at 32K/batch 16 needs independent repetition. At batch
32/128K, it saves 68.21 µs per active indexer invocation; this cannot be directly
translated into output tokens/s because other model work and top-k reuse remain.

## Remaining validation

Full-model accuracy, runtime dispatch through a loaded server, serving TPS,
TTFT/TPOT, and performance with the fallback collective remain unmeasured.
A serving comparison should hold the attention backend and index-sharing
frequency fixed, alternate CP off/on on the same GPUs, and include short-context
controls. Query-projection replication and speculative decoding are future work.
