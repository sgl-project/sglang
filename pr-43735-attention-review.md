# Attention backend review for PR 43735

Recommendation: request changes. Three blocking correctness findings remain in the reviewed implementation, plus one conditional issue in the native NoPE path.

- PR: https://github.com/sgl-project/sglang/pull/43735
- Reviewed head: `07453a5e4ffd43021c90d4f7ab37e2aeb680fbcf`
- Base: `e6072935bfa1036d4e20e8ed042ce1ff09872f57`
- Context: `/Users/khoa.pham/sglang-workspace/sglang/dsa-dcp-design.md`
- Review date: October 11, 2026

The review covers sparse attention dispatch, global and local slot addressing, target and draft cache geometry, MTP prefill, shared top-k reuse, graph metadata, folded query heads, TRT-LLM buffers, zero-KV rows, LSE merging, and launch validation. Findings below refer to the pinned head rather than later changes.

## P1 Target index cache misses valid widened slots

Location: [index_key_cache.py line 19](https://github.com/sgl-project/sglang/blob/07453a5e4ffd43021c90d4f7ab37e2aeb680fbcf/python/sglang/srt/mem_cache/index_key_cache.py#L19).

The target index cache reserves `index_buf_size + pool.page_size`. For the target, these are `C * c` and `64`, respectively, where `C` is physical token capacity and `c` is DCP width. However, the allocator reserves page zero at width `64 * c` and can return slots through `(C + 64) * c - 1`.

The target index cache therefore lacks `64 * (c - 1)` entries. Allocating the final widened page lets indexer writes and reads access beyond the buffer. This is an addressing bug, separate from the design's intentional omission of reserve-page bytes from memory accounting.

Executing the production `IndexKeyCache` constructor with shape-only allocation stubs and physical capacity `C = 1024` gives:

| DCP width | Target index capacity | Required slot capacity | Missing entries |
| --- | --- | --- | --- |
| 1 | 1088 | 1088 | 0 |
| 2 | 2112 | 2176 | 64 |
| 4 | 4160 | 4352 | 192 |
| 8 | 8256 | 8704 | 448 |

The replicated draft covers its slot space correctly because its pool page is already widened.

Reserve the widened page for the target index cache while retaining the draft's already widened reserve. Add a regression test that allocates the last page and checks every emitted slot against index capacity at DCP2, DCP4, and DCP8.

## P1 Draft one shot prefill uses the global DCP prefix reader

Locations: [dsa_backend.py line 488](https://github.com/sgl-project/sglang/blob/07453a5e4ffd43021c90d4f7ab37e2aeb680fbcf/python/sglang/srt/layers/attention/dsa_backend.py#L488), [forward_mha.py line 269](https://github.com/sgl-project/sglang/blob/07453a5e4ffd43021c90d4f7ab37e2aeb680fbcf/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py#L269), and [deepseek_v2.py line 3447](https://github.com/sgl-project/sglang/blob/07453a5e4ffd43021c90d4f7ab37e2aeb680fbcf/python/sglang/srt/models/deepseek_v2.py#L3447).

The draft backend enables one-shot MHA because its effective `dcp_size` is one. Eagle's initial draft prefill retains `EXTEND` mode, so short batches can select this path, including batches with a cached prefix.

The MHA prefix reader still checks `get_parallel().dcp_enabled`, which is true for the target topology. It consequently tries to gather sharded prefix KV, despite the draft pool being replicated. The PR makes DSA's DCP metadata preparation return `None`, and the prefix reader dereferences that missing metadata:

```text
AttributeError: 'NoneType' object has no attribute 'dcp_local_prefix_kv_indices'
```

Executing the production selector and prefix branch with CPU stubs reproduces the exception for a BF16 draft prefill with 256 total tokens and a 128-token prefix on an SM100-shaped configuration.

Disable draft one-shot MHA under global DCP, or make the complete prefix-read path respect the layer's effective width and replicated storage. Add a short cached-prefix MTP prefill regression; the existing high-concurrency accuracy test does not directly exercise this trigger.

## P1 Unsupported HiCache and PD combinations pass launch validation

Location: [deepseek_v2.py model overrides line 25](https://github.com/sgl-project/sglang/blob/07453a5e4ffd43021c90d4f7ab37e2aeb680fbcf/python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py#L25).

The new DSA guard rejects prefill CP and HiSparse but leaves HiCache and PD available. The generic guards do not enforce the DSA-specific exclusions stated in the design and documentation.

HiCache L1/L2 without speculative decoding passes both the actual DSA guard and `resolve_hicache_dcp_compatibility`. The index host pool inherits the sharded MLA host pool's physical capacity, while its transfers address replicated index pages using widened indices. Those indices can address beyond host storage. The replicated index needs its own capacity and transfer geometry before this combination can be enabled.

PD also passes with Mooncake/NIXL. In `_send_kv_chunk`, the prefill DSA state payload computes index-page IDs using the widened allocator page size. Decode computes them using the device page size. At DCP2, one 128-token allocator page contains two 64-token index pages; the sender supplies only one page ID. Mooncake's lenient DSA state-length truncation can then leave index keys missing or copy the wrong page.

Add DSA-specific rejection for HiCache and PD until their replicated index transfer layouts are implemented. The generic MLA transfer guards are insufficient for these two storage spaces.

## P2 Conditional NoPE chunk metadata mismatch

Location: [dsa_dcp.py line 97](https://github.com/sgl-project/sglang/blob/07453a5e4ffd43021c90d4f7ab37e2aeb680fbcf/python/sglang/srt/layers/attention/dsa/dsa_dcp.py#L97).

This issue is outside the documented RoPE model scope, but affects the helper if native NoPE DSA remains accepted. The chunk loop slices query, block tables, sequence lengths, and LSE, but forwards `sparse_mla_top_k_lens` unchanged through `attention_kwargs`.

With 32,768 query rows and two head groups, the folded batch has 65,536 rows. The first call contains 65,534 rows but receives all 65,536 top-k lengths. FlashInfer's native NoPE validation requires one length per query row and rejects the call. A shape-only execution of the production helper reproduces this boundary: 32,767 rows pass; 32,768 rows fail.

Slice the row-dependent top-k lengths for each call if this path stays supported. Otherwise, reject native NoPE explicitly. The shape requirement is enforced in [FlashInfer v0.7.0.post1](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/flashinfer/mla/_core.py#L3689).

## Attention path checks

The core sparse path otherwise appears consistent with the design:

- Ownership filtering preserves top-k order, compacts owned slots, and maps widened slots to local KV rows.
- Query head folding matches the repeated page-table and count rows.
- Persistent counter sizing preserves the product of query rows and heads after folding, matching FlashInfer's counter-size formula.
- Zero-KV fixup precedes LSE merging, and the denominator clamp preserves finite live-row normalization while keeping all-empty finite-zero partials at zero.
- Shared top-k reuse retains the global indices needed by MTP and state capture. Sequential layer IDs prevent reuse across the normal layer-cycle boundary.
- Target DSA extend uses gathered query heads and sparse local KV rather than gathering the prefix KV.

## Validation and remaining coverage

Validation consisted of source tracing, CPU stub executions of the production constructor and selected attention branches, Python syntax checks, and `git diff --check`. CUDA kernels, distributed serving, accuracy, and performance were not executed because the local environment lacks PyTorch/CUDA. The CPU reproductions establish shape, capacity, and dispatch failures; they do not establish GPU numerical parity.

The current tests miss the target index allocation boundary, short cached-prefix draft prefill, the DSA-specific HiCache and PD rejection cases, and the conditional NoPE chunk boundary. GPU acceptance should also cover chunked and prefix extend, target verification, folded-head configurations, graph replay, and the selected communication backends.

No implementation files were changed and no GitHub review was posted.
