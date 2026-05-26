# DSA Attention Capability Matrix

This folder tracks DeepSeek Sparse Attention style unit tests. The existing
registered/model tests exercise DSA at a higher level; this unit matrix covers
small deterministic backend slices with independent PyTorch references.

## Coverage Matrix

Columns are runner modes; rows are the two DSA sub-paths exercised through the
`dsa` backend (selection is by case shape, not backend choice). Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable / not exercised
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| DSA sub-path | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dsa` MHA_ONE_SHOT dense prefill fallback | ✓ 6 dense-fallback extend layouts: no-prefix ragged, no-prefix exact-page, prefix ragged, cross-page-boundary, prefix-exact-page, total-exact-page | deferred: graph metadata parity not scoped | deferred | deferred | deferred | deferred | deferred | deferred | deferred | deferred | deferred | — |
| `dsa` sparse top-k (`flashmla_sparse` prefill + `flashmla_kv` decode) | ✓ 7 sparse top-k layouts: long-prefix bsz=1 prefill, long-prefix multi-token prefill, multi-request long-prefix prefill, decode with bsz=2 trailing-topk, decode with sub-topk prefix padding, ragged 3-request decode, long-prefix decode | deferred | deferred | deferred | deferred | deferred | deferred | deferred | deferred | deferred | deferred | — |

## Input And Config Coverage

- DSA page-size-64 extend and decode batches.
- Dense fallback: no-prefix ragged, no-prefix exact-page, prefix ragged,
  cross-page-boundary, prefix-exact-page, total-exact-page.
- Sparse top-k: uses `qk_nope=512`, `qk_rope=64`, and `topk=128` to match local
  FlashMLA kernel constraints.
- Sparse prefill spans single-request, multi-token extend, and multi-request
  long-prefix layouts above the dense one-shot threshold so the backend selects
  `flashmla_sparse`.
- Sparse decode spans (key_count < topk), (key_count == topk), and
  (key_count >> topk) so the per-request topk slicing varies, plus long-prefix
  decode that walks the trailing topk window deep into the KV cache.

## Production-Unsupported

- **Page size other than 1 (HIP legacy) or 64 (CUDA)** — the DSA indexer
  hard-asserts the page size: HIP legacy at `dsa/dsa_indexer.py:547-548,
  724-725` (`assert page_size == 1`); CUDA at `dsa/dsa_indexer.py:550, 727,
  946, 1095` and `dsa/index_buf_accessor.py:436` (`assert page_size == 64`).
  The `dsa/transform_index.py:53, 79, 100, 121` helpers also assert
  `page_size == 1`.
- **`Unsupported {forward_batch.forward_mode=}`** — `forward_extend`
  fall-through asserts `False` (`dsa_backend.py:629`) for anything not in
  `is_decode_or_idle` / `is_extend()` (incl. `MIXED`, `DRAFT_EXTEND`,
  `TARGET_VERIFY`, `SPLIT_PREFILL`, `DLLM_EXTEND`) / `is_draft_extend(include_v2=True)`.

## Required Fixture Work

- Extend the sparse reference to additional block/index layouts that diverge
  from the trailing-`topk` row builder (e.g., non-trailing or interleaved
  index patterns).
- Decide hardware gates for TileLang / FA / FlashMLA-sparse paths before
  enabling default tests.

## Next Work

- Add CUDA graph and PCG/BCG coverage once metadata parity is clear.
- Add non-trailing index-layout sparse cases when a representative
  index-construction path is identified.
