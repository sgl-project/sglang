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
| `dsa` MHA_ONE_SHOT dense prefill fallback | ✓ 6 dense-fallback extend layouts: no-prefix ragged, no-prefix exact-page, prefix ragged, cross-page-boundary, prefix-exact-page, total-exact-page | deferred: graph metadata parity not scoped | blocked: K-slice mismatch | blocked | — | — | — | — | — | — | — | — |
| `dsa` sparse top-k (`flashmla_sparse` prefill + `flashmla_kv` decode) | ✓ 7 sparse top-k layouts: long-prefix bsz=1 prefill, long-prefix multi-token prefill, multi-request long-prefix prefill, decode with bsz=2 trailing-topk, decode with sub-topk prefix padding, ragged 3-request decode, long-prefix decode | ✓ flashmla_kv | — | — | blocked: deep_gemm metadata fails on small `aligned_batch_size` | blocked | ✓ DRAFT_EXTEND eager | — | ✓ DRAFT_EXTEND_V2 eager | — | — | — |

## Implementation Variant Matrix (`--dsa-prefill-backend` / `--dsa-decode-backend`)

DSA has multiple kernel impls; `dsa_impl_capability(impl)` gates each per
hardware/SDK. The variant tests live in `test_dsa.py` as
`test_sparse_prefill_impl_variants`, `test_sparse_decode_impl_variants`, and
`test_sparse_cuda_graph_decode_impl_variants`.

| Impl | Prefill | Decode | CG decode | Hardware gate (test box: H200 SM9.0) |
|---|---|---|---|---|
| `flashmla_sparse` | ✓ | ✓ | ✓ | SM>=9.0 + `sgl_kernel.flash_mla` |
| `flashmla_kv` | ✓ | ✓ | ✓ | SM>=9.0 + `sgl_kernel.flash_mla` |
| `fa3` | ✓ | ✓ | ✓ | SM>=9.0 + `sglang.jit_kernel.flash_attention` |
| `tilelang` | skipped: topk=2048 contract | skipped: topk=2048 contract | skipped: topk=2048 contract | `tilelang_sparse_fwd` asserts `topk == 2048`; shared sparse fixture uses `topk=128`. Needs a topk=2048 fixture variant. |
| `trtllm` | skipped: SM<10 | skipped: SM<10 | skipped: SM<10 | TRT-LLM Gen FMHA/MLA requires Blackwell (SM>=10.0). |
| `aiter` | skipped: not HIP | skipped: not HIP | skipped: not HIP | AMD-only kernel library. |
| `flashmla_auto` (default) | ✓ (resolves to `flashmla_sparse` for bf16, `flashmla_kv` for FP8) | ✓ | ✓ | covered indirectly by all sparse cases |

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
- **PCG/BCG split-op extend on the MHA_ONE_SHOT dense fallback path** —
  structurally incompatible with `unified_attention_with_output`. DSA's
  dense fallback passes K as concatenated `prefix + extend` (shape
  `[sum(seq_lens), num_kv_heads, head_dim]`) to `module.attn(q, k, v,
  forward_batch, save_kv_cache=False)`, but `unified_attention_with_output`
  (`radix_attention.py:170-208`, which RadixAttention routes to under
  piecewise CG) slices K to `forward_batch.num_token_non_padded_cpu` (=
  live extend-token count) on the per-token K convention used by
  Triton/FlashInfer/FA. The slice removes the prefix portion, so a
  piecewise CG run diverges from the eager DSA dense fallback by ~50%
  mismatch (~0.35 max diff) vs the HF reference. Unblocking needs
  either (a) the DSA dense fallback rewritten to write K to cache
  (`save_kv_cache=True`) and pass extend-only K to `module.attn` (so
  the slicing is a no-op), or (b) a backend-hint on `RadixAttention` to
  skip the K-slice when the kernel expects prefix-concatenated K.

## Required Fixture Work

- Extend the sparse reference to additional block/index layouts that diverge
  from the trailing-`topk` row builder (e.g., non-trailing or interleaved
  index patterns).
- Decide hardware gates for TileLang / FA / FlashMLA-sparse paths before
  enabling default tests.
- Runner-mode integration is now plumbed at the fixture level:
  `DSAMockModelRunner` accepts `disable_cuda_graph`,
  `disable_piecewise_cuda_graph`, and `runner_batch_size` kwargs;
  `build_dsa_attention_fixture` passes them through; and
  `dsa_attention.py` exposes the standard adapter callbacks
  (`make_dsa_case_with_prefix_lens`, `dsa_fixture_inputs`,
  `make_dsa_random_inputs`, `make_dsa_token_padded_inputs`,
  `prepare_dsa_runner_inputs`, `run_dsa_forward`,
  `expected_dsa_output_from_inputs`, `dsa_attention_layers`,
  `_clone_dsa_cache`, `_restore_dsa_cache`). The dense fallback path
  still can't actually exercise piecewise CG (see
  "Production-Unsupported"); CG decode through the sparse fixture is
  the natural next target once the sparse-fixture topk-indices
  threading is added to the adapter contract.

## Next Work

- **FP8 KV cache path** — `DSATokenToKVPool.dsa_kv_cache_store_fp8`
  needs `dtype=torch.float8_e4m3fn` and a non-None `override_kv_cache_dim`;
  the fixture currently uses bf16. Adding it would unlock
  `TopkTransformMethod.RAGGED` coverage (`get_topk_transform_method`:
  RAGGED needs `dsa_kv_cache_store_fp8 AND dsa_prefill_impl == "flashmla_sparse"
  AND forward_mode == EXTEND`) and the FP8 `flashmla_auto → flashmla_kv`
  resolution branch. The prefix-write path also needs to pack BF16 K into
  the FP8 store layout (UE8M0 scales + packed quants) so the reference
  matches what the kernel reads from cache.
- **TARGET_VERIFY** — deep_gemm's `paged_mqa_logits_metadata` kernel
  fails to compile for small `aligned_batch_size`
  (`zero-sized variable "num_segs"`); the production speculative graph
  runner only hits large aligned shapes, so the synthetic fixture would
  need to mock the deep_gemm metadata path or raise the alignment.
- **tilelang topk=2048 fixture** — `tilelang_sparse_fwd` asserts
  `topk == 2048`; building a separate sparse fixture with
  `DSA_SPARSE_INDEX_TOPK=2048` would unlock the tilelang variant tests.
- Add non-trailing index-layout sparse cases when a representative
  index-construction path is identified.
