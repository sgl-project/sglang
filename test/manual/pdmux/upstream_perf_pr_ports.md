# DSV4.1 PDMux upstream performance ports

These ports target H100/SM90, TP8/DP8/EP8, DSPARK, HiCache, `dsv4` attention
and `flashinfer_mxfp4` MoE. They retain the existing PDMux lane submission and
communication order. New SM90 paths are opt-in pending GPU acceptance.

## PR #42365: fused prefill indexer scoring

Source: [sgl-project/sglang#42365](https://github.com/sgl-project/sglang/pull/42365),
head `71bbd5402845f6651080339814c266d505dd9bab`.

Fuses QK, ReLU, head weighting/reduction and the causal mask. The port preserves
CPU request geometry and uses the calling stream, without a helper stream or
shared mutable scratch. Dispatch requires SM90, BF16 operands and head dim 128;
other inputs keep the existing scorer. Head reduction rounding can change
cutoff ties, so top-k overlap alone does not establish model accuracy.

Add before the existing serve command:

```bash
export SGLANG_OPT_USE_FUSED_DSV41_INDEXER_SCORES=1
```

Set it to `0` to restore existing scoring. `SGLANG_DSV41_TORCH_PREFILL_INDEXER=1`
also forces the Torch prefill oracle. No new serve CLI argument is needed.

GPU kernel checks:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/attention/dsv4/test_fp4_indexer_prefill.py \
  test/registered/unit/layers/test_dsv41_prefill_submission.py
```

## Combination acceptance

Run each optimization alone and then together, with the same checkpoint,
DSPARK block, SM allocation and workload. Include zero-work DP ranks, graph
padding, short/long prompts, cold/warm HiCache and split continuations. Confirm
greedy output/quality, peak memory, TTFT, ITL and throughput, and sample a trace
to verify the lanes still overlap.

Repeat the historical chunk1024 / `max_split_forward_layers: 2` hang case at
c10 for at least 30 minutes before c24. The user's current YAML requests 62
prefill SMs (the driver may round); its omitted layer cap and chunk size are
not that historical configuration. CPU tests do not verify CUDA numerics,
multi-rank liveness or throughput.
