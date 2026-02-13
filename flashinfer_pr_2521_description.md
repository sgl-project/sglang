## 📌 Description

This PR adds pool-indexed (indirect) state access to the GDN decode kernel, enabling zero-copy integration with SGLang's state pool architecture.

### Background: SGLang's State Pool Architecture

In SGLang, when serving linear attention models (like Qwen3-Next using Gated Delta Rule), we maintain a **state pool** to store recurrent states for all active requests:

`ssm_states: [num_layers, pool_size, num_heads, head_dim, head_dim]`

where `pool_size` = `max_num_reqs` (maximum concurrent requests).

Each active request has a `req_pool_idx` that maps it to a slot in this pool. The mapping is **not contiguous** - requests come and go, so indices can be scattered (e.g., a batch of 4 requests might have pool indices `[3, 7, 12, 25]`).

### Motivation

The current GDN decode kernel expects state with shape `[B, H, K, V]` where B equals batch size and there's a 1:1 mapping (batch index i → state index i). To use it with SGLang's pool, we would need to:

1. **Gather** states from pool indices before kernel call
2. Run kernel on contiguous `[B, H, K, V]` state
3. **Scatter** updated states back to pool indices

This adds 2 extra memory copy operations per decode step.

### Changes

This PR adds a `state_indices` parameter for **zero-copy pool access**:

```python
def gated_delta_rule_decode_pretranspose(
    q, k, v, beta,
    state,           # Can be [pool_size, H, K, V] instead of [B, H, K, V]
    state_indices,   # NEW: int32 tensor [B] mapping batch_idx -> pool_idx
    ...
)
```

When `state_indices` is provided:
- Kernel uses indirect addressing: `state[state_indices[batch_idx]]` instead of `state[batch_idx]`
- Negative indices (padding slots for CUDA graph) skip computation and write zeros to output
- Eliminates gather/scatter overhead + host-side `torch.where` for padding (~37μs/call)

### Performance

Combined with K-last layout, the pool indexing optimization delivers **4-5.6% speedup** for decode at batch sizes >= 4.

End-to-end benchmark results from SGLang integration:

**Model:** Qwen3-Next-80B-A3B-Instruct, 8x H20, TP=8, EAGLE speculative decoding

#### Latency (seconds, lower is better)

| Batch | V-last | K-last | Change |
|-------|--------|--------|--------|
| 1 | 0.405 | 0.375 | **-7.5%** |
| 4 | 0.504 | 0.481 | **-4.5%** |
| 16 | 1.051 | 0.960 | **-8.6%** |
| 32 | 1.527 | 1.483 | **-2.9%** |

#### Prefill Throughput (tok/s, higher is better)

| Batch | V-last | K-last | Change |
|-------|--------|--------|--------|
| 1 | 9,179 | 10,705 | **+16.6%** |
| 4 | 32,530 | 35,055 | **+7.8%** |
| 16 | 47,720 | 49,365 | **+3.4%** |
| 32 | 49,177 | 50,229 | **+2.1%** |

## 🔍 Related Issues

- [sgl-project/sglang#18361](https://github.com/sgl-project/sglang/pull/18361) - FlashInfer K-last GDN integration into SGLang

## 🚀 Pull Request Checklist

Thank you for contributing to FlashInfer! Before we review your pull request, please make sure the following items are complete.

### ✅ Pre-commit Checks

- [x] I have installed `pre-commit` by running `pip install pre-commit` (or used your preferred method).
- [x] I have installed the hooks with `pre-commit install`.
- [x] I have run the hooks manually with `pre-commit run --all-files` and fixed any reported issues.

> If you are unsure about how to set up `pre-commit`, see [the pre-commit documentation](https://pre-commit.com/).

## 🧪 Tests

- [ ] Tests have been added or updated as needed.
- [x] All tests are passing (`unittest`, etc.).

## Reviewer Notes

This PR is required for integrating FlashInfer's K-last GDN kernels into SGLang. The pool indexing feature allows SGLang to directly use its state pool without gather/scatter overhead.
