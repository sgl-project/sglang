# DSA Attention Capability Matrix

This folder tracks DeepSeek Sparse Attention style unit tests. The existing
registered/model tests exercise DSA at a higher level; this unit matrix covers
small deterministic backend slices with independent PyTorch references.

## Current Matrix

| Backend family | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `dsa` MHA_ONE_SHOT dense prefill fallback | No-prefix ragged, no-prefix exact-page, and prefix ragged extend coverage | Not implemented | Not implemented | Uses real `DSATokenToKVPool`, DSA backend metadata, and an independent dense PyTorch reference. |
| DSA sparse/indexer prefill/decode paths | Sparse prefill through `flashmla_sparse`; sparse decode through default `flashmla_kv` | Not implemented | Not implemented | Uses production-shaped DSA dimensions, synthetic top-k rows, real `DSATokenToKVPool`, and an independent sparse PyTorch reference. |

## Input And Config Coverage

- DSA page-size-64 extend batches.
- No-prefix ragged extend, no-prefix exact-page extend, and prefix ragged extend.
- Sparse top-k coverage uses `qk_nope=512`, `qk_rope=64`, and `topk=128` to match
  local FlashMLA kernel constraints.

## Current Progress

- Phase 2 eager coverage is enabled for the locally runnable DSA dense prefill
  fallback selected by short MHA_ONE_SHOT batches.
- Phase 2 eager coverage is enabled for sparse top-k prefill/decode paths.
- The next split is runner metadata coverage once the sparse method fixture is
  stable.
- Higher-level registered/model coverage should stay separate from this folder's
  small deterministic unit matrix.

## Required Fixture Work

- Create a tiny DSA-shaped attention module with real DSA KV/indexer pools.
- Extend the sparse reference to additional block/index layouts.
- Decide hardware gates for TileLang/FA/FlashMLA-sparse paths before enabling default tests.

## Next Work

- Add CUDA graph and PCG/BCG coverage once metadata parity is clear.
