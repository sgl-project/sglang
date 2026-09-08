# V4.1 C1/C2 Compress on Ascend

This is a **Compress-only bring-up API**, based on the GPU V4.1 implementation.
It does not enable end-to-end V4.1 NPU inference. The old C4/C128 fused
compressor ABI and cache publication path are unchanged.

## Entry point

```python
result = attn_backend.forward_low_ratio_compressor(
    compressor=layer.compressor,
    x=x,                         # BF16 [tokens, hidden_size]
    positions=positions,         # absolute token positions
    forward_batch=forward_batch,
    layer_id=layer.layer_id,     # KV source layer, not a sharing consumer
)
```

`forward_core_compressor` also dispatches C1/C2 to this API and **returns** the
result. Unlike C4/C128, it does not write any cache. Only KV source layers have
a compressor and C2 pair state. Idle returns `None` without reading state.

The shared `layers/attention/dsv4/dsv41_compressor.py` preserves the GPU module's
weights, parameter names and computation. `dsv41_sparse.py` re-exports the moved
symbols for existing GPU callers. The Ascend batching implementation uses only
torch operations and imports no CUDA/Triton or fused C4/C128 compressor kernel.

## Computation and result contract

| Ratio | Projection and pooling | State |
| --- | --- | --- |
| C1 | BF16 `wkv(x)`, then RMSNorm | None; no gate |
| C2 | FP32 `wkv(x.float())` and `wgate(x.float())`; per-channel softmax over tokens `(2k, 2k+1)` and weighted sum; cast to BF16, then RMSNorm | FP32 pending KV and score per source layer/request slot |

RMSNorm computes statistics and the weight multiply in FP32, then returns BF16.
The result has four tensor fields:

- `latent`: normalized BF16, **before RoPE and FP4 quantization**.
- `positions`: absolute group-start positions (C1: `p`; completed C2: `p - 1`).
- `out_loc`: `full_token_loc // ratio`, or `-1` for an invalid row. The NPU
  allocation bundle's `out_full_loc` takes precedence over `out_cache_loc`;
  SWA locations are never used to derive compressed addresses.
- `valid_mask`: which result rows may be published. Extend returns only
  completed, valid groups. Decode retains the input row count, including
  incomplete C2 pairs and graph padding; their latent values are unspecified
  and MUST NOT be published as valid groups.

C2 state comes from `token_to_kv_pool.c2_pair_kv_state[layer_id]` and
`c2_pair_score_state[layer_id]`. The existing shared pool allocates it on the
selected device. Each has shape `[num_req_slots + 1, head_dim]`; the last row
absorbs padding without modifying a live request 0. The old C4/C128 ring state
pool is not involved.

## Supported input/state contract

- Request-major, globally ordered EXTEND/MIXED; one token per live request in
  DECODE. Live request slots are unique in decode, in range, and do not use the
  reserved padding row. Slot 0 in the full token pool denotes padding.
  Metadata uses int32/int64. Full token groups are physically ratio-aligned,
  as in the shared allocator; `out_loc` is not an arbitrary-address remapper.
- Ragged prefill, chunked prefill and request reordering between calls are
  supported. Chunks must be contiguous for each request. An odd-start C2
  chunk must find the preceding even token in that request's pending state.
- A newly reused request slot starts at position 0 (which overwrites state).
  Prefix-cache resume at an odd position, PD state transfer and speculative
  rollback require a separate state restore/recompute protocol; this API does
  not supply one. Compressed KV alone cannot reconstruct an unfinished pair.
- CP/DCP-sharded input and speculative/nonstandard forward modes are rejected
  before state mutation. Decode uses fixed-shape operations, but NPU graph
  capture/replay still requires device validation.

## Remaining full-model integration

Do not substitute this API for `forward_low_ratio_sources`: that method also
publishes index keys, compressed KV and top-k indices. The integration order is
`Compress -> index K from pre-RoPE latent -> latent RoPE/FP4 -> compressed KV`.
NPU low-ratio Indexer, cache layout/writers, attention dispatch, state lifecycle
and model multi-stream entry still require adaptation. In particular, low-ratio
results must not be sent to the existing C4/C128 cache writer or quantized as
the old C4 indexer format. No new cache layout or allocator is introduced here.

## Tests

From the repository root, with PyTorch installed:

```bash
python test/registered/unit/npu/attention/test_npu_dsv41_compressor.py
```

The tests load the dependency-light implementation directly, run CPU numerical
and state regressions, and repeat device cases on NPU when `torch_npu` and a
device are available. Backend dispatch tests isolate the actual methods from
the heavy backend imports; they are not a full server initialization test.
Before deployment, run these tests on the target Ascend software stack, then
validate graph replay, full-model precision and latency after the remaining
integration. CPU results alone establish neither NPU kernel support nor speedup.
