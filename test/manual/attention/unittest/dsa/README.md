# DSA Attention Capability Matrix

This folder tracks DeepSeek Sparse Attention style unit tests. The existing
registered/model tests exercise DSA at a higher level; this unit matrix covers
small deterministic backend slices with independent PyTorch references.

## Current Matrix

| Backend family | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `dsa` MHA_ONE_SHOT dense prefill fallback | No-prefix ragged, no-prefix exact-page, and prefix ragged extend coverage | Not implemented | Not implemented | Uses real `DSATokenToKVPool`, DSA backend metadata, and an independent dense PyTorch reference. |
| DSA sparse/indexer prefill/decode paths | Not implemented | Not implemented | Not implemented | Needs indexer top-k fixture and sparse-reference implementation. |

## Input And Config Coverage

- DSA page-size-64 extend batches.
- No-prefix ragged extend, no-prefix exact-page extend, and prefix ragged extend.
- Dense MHA_ONE_SHOT fallback only; sparse top-k/indexer behavior is not covered
  by this file.

## Current Progress

- Phase 2 eager coverage is enabled for the locally runnable DSA dense prefill
  fallback selected by short MHA_ONE_SHOT batches.
- The next split is sparse/indexer method-level eager correctness first, then
  runner metadata coverage once that fixture is stable.
- Higher-level registered/model coverage should stay separate from this folder's
  small deterministic unit matrix.

## Required Fixture Work

- Create a tiny DSA-shaped attention module with real DSA KV/indexer pools.
- Implement an HF-style sparse/block reference that does not call SGLang DSA kernels.
- Cover dense-fallback and sparse paths as separate config axes.
- Decide hardware gates for TileLang/FA/FlashMLA-sparse paths before enabling default tests.

## Next Work

- Add an indexer-driven DSA sparse prefill/decode fixture.
- Add CUDA graph and PCG/BCG coverage once metadata parity is clear.
