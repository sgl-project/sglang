# DSA Attention Capability Matrix

This folder is reserved for DeepSeek Sparse Attention style unit tests. The
existing registered/model tests exercise DSA at a higher level, but this unit
matrix still needs a small attention-method fixture with an independent reference.

## Planned Matrix

| Backend family | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| DSA prefill/decode backends | Not implemented | Not implemented | Not implemented | Needs DSATokenToKVPool/indexer fixture and sparse-reference implementation. |

## Required Fixture Work

- Create a tiny DSA-shaped attention module with real DSA KV/indexer pools.
- Implement an HF-style sparse/block reference that does not call SGLang DSA kernels.
- Cover dense-fallback and sparse paths as separate config axes.
- Decide hardware gates for TileLang/FA/FlashMLA-sparse paths before enabling default tests.

## First Test Target

- `dsa/test_<backend_name>.py` for one locally runnable representative backend.
- Start with Phase 2 eager correctness, then add CUDA graph and PCG/BCG once metadata parity is clear.
