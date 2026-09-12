# Compact linear speculative verification

Opt in with `SGLANG_ENABLE_COMPACT_SPEC_VERIFY=1`. The qualified route is
single-node TP4 on SM100, BF16 GLM vocabulary 154880, linear rejection sampling,
temperature 1 and no penalties, grammar, top-k/top-p/min-p or requested logprobs.
Unqualified builds and request features use the original verifier. The engine
checks the exact Torch revision and binary checksum before using this route.

ASIS
```text
local target logits [R,V/TP]
    -> full gather [R,V] -> full target probabilities [R,V]
    -> acceptance and recovery
```

PR
```text
local target logits [R,V/TP]
    -> selected probabilities and error intervals [R]
    -> uncertain rows: original softmax in bounded peer-read chunks
    -> acceptance -> one recovery/bonus row per request
```

## Numerical contract

The input to the reference exponential is the FP32 shifted logit. The SM100
instruction sequence in `exp_dag.py` was compared exhaustively with the pinned
Torch exponential for all 1,115,684,865 negative FP32 bit patterns from -0 to
-64, plus positive zero. Profiling confirmed the corresponding Torch kernel
variants; reference SASS uses the same normal-range exponential operations.
Power-of-two exponent scaling remains exact in this normal range even where
the reference fuses it into an accumulation FMA.

For V154880 the reference per-thread sum plus block reduction has depth less
than 2048. With FP32 unit roundoff u=2^-24, gamma_2048 is about 0.000122085.
FP64 local/global accumulation has gamma_(V+4) below 1.8e-11. Including final
division and interval endpoint rounding, the relative deviation from the
FP64 center is below 0.000122160 under this qualified operation contract.
The interval radius 2^-10 is conservative. FP32 shifted ranges wider than 64,
or nonfinite ranges, are recomputed through original row softmax.

A decision is used directly only outside the interval. Every ambiguous row
uses the original FP32 softmax, and recovery uses the original sampler's CDF
kernel and random coins. This is a pinned-build argument, not a portability
guarantee for arbitrary Torch/CUDA releases or devices.

## Graph and ownership contract

Local target inputs use stable symmetric storage. The draft probability tensor
is not copied: a device pointer is rebound before replay, with `record_stream`
protecting its lifetime. Idle graph caches do not retain that full tensor.
All ranks share the root repair descriptor. NCCL stays outside conditional
bodies; peer reads and original row softmax perform repairs inside them.

The common branch repairs up to eight rows. A second branch processes all
remaining rows in bounded chunks, so no invalid acceptance count can escape
because of queue overflow. This finishes before SGLang updates KV/Mamba state.
Both branches reuse one scratch pair. Captured graphs must be reset before
their communication groups are destroyed; the distributed cleanup hooks do so.

`SGLANG_COMPACT_SPEC_VERIFY_SHADOW=1` enables the expensive same-logits reference
comparison. Use it for validation only, not performance measurements.

## Tests

```bash
PYTHONPATH=python pytest test/registered/unit/sampling/test_compact_verify_config.py
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python COMPACT_TEST_OUTPUT=/tmp/compact_verify.json \
  torchrun --standalone --nproc-per-node=4 test/manual/test_compact_spec_verify.py
```

The manual test requires the qualified runtime and four GPUs. Acquire their
lease before running it. It covers live q-pointer changes, index permutation,
the source NaN-q rule, repair overflow, actual `eagle_sample`, and clean shutdown.
