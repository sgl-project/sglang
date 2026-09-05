Status: implemented

# Attention input contracts

TL;DR: Attention backends preserve each model's positional-encoding policy and
use the cache layout and sequence boundaries supplied by serving.

## Behavior and invariants

- TokenSpeed MLA prefill quantizes Q, K, V and the latent KV cache to FP8.
  Models with rotary embeddings apply them before quantization; models without
  rotary embeddings retain their positional components unchanged.
- NPU vision graphs distinguish equal-length inputs with different cumulative
  sequence lengths. Capture and replay use the same graph key, and workspace
  allocation uses the input token count.
- Lean decode attention receives a three-dimensional `[slot, head, dim]` KV
  buffer. Paged tests retain shuffled slot indices and compare Lean with SplitK
  at each tested page size.

```text
model Q/K/V + position policy --> FP8 prefill tensors + latent KV cache
vision tokens + boundaries   --> graph key --> capture or replay
KV slots + page size         --> slot addressing --> Lean / SplitK
```

[Regression reproducer and verification evidence](evidence/core-00-attention-inputs.md).

## Boundaries

These contracts do not add serving options, alternate cache layouts, or relaxed
accuracy thresholds. Hardware graph execution and end-to-end distributed
attention still require their corresponding CI runners.
