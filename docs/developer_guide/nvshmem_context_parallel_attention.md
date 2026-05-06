# NVSHMEM Context-Parallel Attention Prototype

This developer note describes the experimental helpers in
`sglang.srt.layers.attention.nvshmem_context_parallel`.

The goal is to separate context-parallel attention semantics from the transport
used to read remote K/V blocks.  That lets us validate sharding, causal masking,
LSE merging, and gradient ownership before wiring the path into a production
attention backend.

## Scope

The current implementation provides:

- sequence chunk metadata for contiguous and head-tail context parallel layouts;
- local K/V chunk construction for deterministic tests;
- an NVSHMEM peer-tensor adapter that slices symmetric local or remote K/V views;
- a torch reference implementation for forward and autograd correctness;
- a Triton forward path that computes partial attention per K/V chunk and merges
  chunk outputs with log-sum-exp state.

The current implementation does not yet provide:

- a production SGLang attention backend registration;
- a handwritten Triton/CUDA backward kernel;
- cross-rank gradient synchronization for peer-owned dK/dV;
- CUDA graph integration or paged-KV metadata integration.

## Tensor Contract

The helper functions use batched, head-major tensors:

```python
q: [batch, heads, q_len, head_dim]
k: [batch, heads, k_chunk_len, head_dim]
v: [batch, heads, k_chunk_len, head_dim]
```

`SequenceChunk.global_start` and `SequenceChunk.global_end` describe the chunk's
global sequence interval.  `SequenceChunk.local_start` and
`SequenceChunk.local_end` describe where that interval lives inside the owning
rank's local symmetric K/V tensor.

Head-tail layouts can give one rank multiple non-contiguous global intervals
that are concatenated in local memory.  Query positions are therefore passed
explicitly with `query_positions_for_chunks(...)`; causal masking is based on
global positions, not local offsets.

## NVSHMEM Peer Tensor Path

`nvshmem_peer_kv_chunks(...)` expects local symmetric K/V tensors and an optional
`get_peer_tensor` callback.  For remote owners it lazily calls:

```python
get_peer_tensor(local_k, owner_rank)
get_peer_tensor(local_v, owner_rank)
```

The returned peer views are sliced with each chunk's local offsets.  The Triton
forward path can consume those chunk views directly, avoiding materialization of
the full K/V sequence on every rank.

## Forward Merge Contract

Each chunk produces:

- partial output: `[batch, heads, q_len, head_dim]`
- partial LSE: `[batch, heads, q_len]`

Chunk states are merged with:

```python
merged_lse = logaddexp(prefix_lse, suffix_lse)
merged_out = prefix_out * exp(prefix_lse - merged_lse)
           + suffix_out * exp(suffix_lse - merged_lse)
```

This is the same normalization contract used by split attention paths: chunks
can be processed independently as long as each partial result carries its LSE.

## Current Verification

`test/registered/attention/test_nvshmem_context_parallel.py` covers:

- multi-head contiguous context-parallel attention vs full attention;
- head-tail load-balanced chunks vs full attention;
- causal and non-causal masking;
- LSE output parity;
- autograd dQ/dK/dV parity with owner-local K/V slices;
- Triton forward parity against the torch reference on CUDA.

The backward path is intentionally still validated through torch autograd.  A
production training path should add explicit kernels for dQ/dK/dV and define how
peer-owned dK/dV are accumulated or synchronized.
