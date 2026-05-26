# DSA Attention Capability Matrix

This folder tracks DeepSeek Sparse Attention style unit tests. The existing
registered/model tests exercise DSA at a higher level; this unit matrix covers
small deterministic backend slices with independent PyTorch references.

## Current Matrix

| Backend family | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `dsa` MHA_ONE_SHOT dense prefill fallback | 6 dense-fallback extend layouts: no-prefix ragged, no-prefix exact-page, prefix ragged, cross-page-boundary, prefix-exact-page, and total-exact-page | Not implemented | Not implemented | Uses real `DSATokenToKVPool`, DSA backend metadata, and an independent dense PyTorch reference. |
| DSA sparse/indexer prefill/decode paths | 7 sparse top-k layouts: long-prefix bsz=1 prefill, long-prefix multi-token prefill, multi-request long-prefix prefill, decode with bsz=2 trailing-topk, decode with sub-topk prefix padding, ragged 3-request decode, long-prefix decode | Not implemented | Not implemented | Uses production-shaped DSA dimensions, synthetic top-k rows, real `DSATokenToKVPool`, and an independent sparse PyTorch reference. |

## Input And Config Coverage

- DSA page-size-64 extend and decode batches.
- Dense fallback: no-prefix ragged, no-prefix exact-page, prefix ragged,
  cross-page-boundary, prefix-exact-page, and total-exact-page.
- Sparse top-k: uses `qk_nope=512`, `qk_rope=64`, and `topk=128` to match
  local FlashMLA kernel constraints.
- Sparse prefill coverage spans single-request, multi-token extend, and
  multi-request long-prefix layouts above the dense one-shot threshold so the
  backend selects `flashmla_sparse`.
- Sparse decode coverage spans (key_count < topk), (key_count == topk), and
  (key_count >> topk) so the per-request topk slicing varies across the
  batch, plus long-prefix decode that walks the trailing topk window deep
  into the KV cache.

## Current Progress

- Phase 2 eager coverage is enabled for the locally runnable DSA dense
  prefill fallback selected by short MHA_ONE_SHOT batches, with page-aligned
  and cross-page-boundary variants.
- Phase 2 eager coverage is enabled for sparse top-k prefill (`flashmla_sparse`)
  and sparse decode (`flashmla_kv`) across the layouts above.
- Higher-level registered/model coverage should stay separate from this
  folder's small deterministic unit matrix.

## Required Fixture Work

- Extend the sparse reference to additional block/index layouts that diverge
  from the trailing-`topk` row builder (e.g., non-trailing or interleaved
  index patterns).
- Decide hardware gates for TileLang/FA/FlashMLA-sparse paths before enabling
  default tests.

## Next Work

- Add CUDA graph and PCG/BCG coverage once metadata parity is clear.
- Add non-trailing index-layout sparse cases when a representative
  index-construction path is identified.
