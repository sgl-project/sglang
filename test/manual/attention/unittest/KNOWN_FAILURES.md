# Known Failures — Attention Backend Unit Tests

This file catalogs:
1. Every **non-passing subtest result** (FAILED or SKIPPED-with-reason) that
   is expected at the current state of the test suite, grouped by root cause
   (§1-5).
2. Every **H200-observable production-side backend limitation** identified
   during fixture investigation that does not currently have a corresponding
   test in the suite, but for which a fixture probe confirmed the
   production-side wall (§6).

A failure is "expected" if it is caused by a missing container dependency, a
hardware gate, or a documented production limitation — not by a bug in the
test logic or attention backend that we intend to fix in this branch.

Anything that is not listed here and fails should be treated as a regression.

Last updated: 2026-05-27
Reference runs:
- H200 SM9.0 (Hopper), sglang-kernel current: **176 tests, 30 skipped,
  0 failures** in ~40 s.
- GB300 SM10.3 (Grace-Blackwell), sglang-kernel 0.4.3, nightly-dev-cu13
  container: **21 failed, 160 passed, 87 skipped, 436 subtests passed** in
  ~215 s. After the conditional skips landed in `cf482d662`, the §1/§2/§3
  failures on GB300 now skip cleanly with documented reasons.

---

## 1. Container-level: `flash_attn_varlen_func` missing on SM10.3

**Count**: 18 SUBFAILED
**Affected file**: `dual_chunk/test_dual_chunk_flash_attn.py`
**All test methods**:
- `test_projected_dual_chunk_attention_cases` (10 sub-cases)
- `test_runner_mode_cuda_graph_decode_cases` (1 sub-case)
- `test_sparse_dual_chunk_attention_cases` (3 sub-cases)
- `test_sparse_dual_chunk_threshold_gated_cases` (1 sub-case)
- `test_layout_robustness_cases` – `interleaved_pages` layout (2 sub-cases)

**Error**:
```
ImportError: cannot import name 'flash_attn_varlen_func' from 'flash_attn' (unknown location)
```

**Root cause**: The `flash_attn` wheel in the `lmsysorg/sglang:nightly-dev-cu13`
container is compiled for SM8x/SM9x (Ampere/Hopper). SM10.3 (GB300 / Grace-Blackwell
NVL2) is not a supported target in that build. `DualChunkFlashAttentionBackend` uses
`flash_attn_varlen_func` for every forward call, so all dual-chunk sub-tests fail
at import time inside the first forward.

**Fix**: Re-image the container with a `flash_attn` wheel compiled for `sm_103a`.
No test-code change is needed; the tests will run correctly once the wheel is present.

---

## 2. Container-level: tilelang `wait_wgmma` template missing on SM10.3

**Count**: 2 FAILED
**Affected file**: `dsa/test_dsa.py`
**Test methods**: `test_sparse_tilelang_prefill_case`, `test_sparse_tilelang_decode_case`

**Error**:
```
RuntimeError: #include <tl_templates/cuda/instruction/mma.h>
...
namespace "tl" has no member "wait_wgmma"
```

**Root cause**: The tilelang MMA template library in the container does not include
`wait_wgmma`, a Blackwell-specific WGMMA synchronization primitive that tilelang
generates for SM10.x targets. The tilelang JIT fails at PTX compilation.

**Fix**: Re-image the container with a tilelang version that ships SM10.x MMA
templates. No test-code change needed; the skip gate `dsa_impl_capability("tilelang")`
already returns `(True, "")` on SM10+ (the import succeeds), so the tests will
run once the template is present.

---

## 3. Backend limitation: FlashInfer MLA EAGLE draft CUDA-graph chain on Blackwell

**Count**: 1 SUBFAILED
**Affected file**: `mla/test_flashinfer.py`
**Test method**: `test_runner_mode_eagle_draft_cuda_graph_runner_cases`
**Sub-case**: `runner_eagle_draft_decode_mla_cuda_graph_chain` (backend=`flashinfer`, topk=1)

**Error**:
```
AssertionError: Tensor-likes are not close!
Greatest absolute difference: 22 at index (1, 1)  (up to 0.03 allowed)
```

**Root cause**: The FlashInfer MLA multi-step draft backend (`FlashInferMLAMultiStepDraftBackend`)
CUDA-graph capture/replay produces numerically wrong outputs on GB300 (SM10.3).
The eager path and the DRAFT_EXTEND path pass correctly; the regression is specific
to the CUDA-graph decode path for the EAGLE draft runner. Likely cause: the
FlashInfer version in the container targets SM9x (Hopper) for its MLA decode
kernel; on SM10.3 the kernel falls back to a generic path that does not restore
metadata buffers correctly under graph replay. This test passes on H200 (SM9.0).

**Fix**: Update FlashInfer to a version that ships an SM10.3-compiled MLA
multi-step decode kernel. No test-code change needed; the eager and DRAFT_EXTEND
paths are not affected.

---

## 4. Hardware gates (expected SUBSKIPPED or test-level SKIPPED on GB300)

These were previously failing with runtime errors on GB300 until hardware-gate
fixes were applied in commit `6615dd6439`. They are now correctly skipped.

| Test | Guard condition | Error if unguarded |
|---|---|---|
| `mla/test_cutlass_mla.py` — all methods | `sm == 10.0` exactly | `RuntimeError: cutlass_mla_decode is only supported on compute capability 10.0, but found sm version 103` |
| `mla/test_flashmla.py` — `test_runner_mode_cuda_graph_decode_cases`, `test_runner_mode_eagle_verify_cases`, `test_runner_mode_eagle_verify_cuda_graph_cases`, `test_runner_mode_eagle_draft_cuda_graph_runner_cases`; DECODE sub-tests in `test_tiny_deepseek_mla_attention_cases` | `major >= 10` (SM90a required) | `RuntimeError: Dense decode MLA is only supported on SM90a architecture` |
| `dsa/test_dsa.py` — `fa3` impl sub-tests in `test_sparse_prefill_impl_variants`, `test_sparse_decode_impl_variants`, `test_sparse_cuda_graph_decode_impl_variants` | `major < 9 or major >= 10` | `NotImplementedError: flash_attn at sgl-kernel is only supported on sm90 and above` |
| `dsa/test_dsa.py` — `trtllm` impl sub-tests (same three methods) | `major != 10 or minor != 0` | `tvm.error.InternalError: Missing TRTLLM-GEN kernel` (kernel compiled for SM10.0 only) |

**Note on SM10.3 vs SM10.0**: GB300 is SM10.3, not SM10.0 (B200 NVL). Any skip
condition that checks `major == 10` without also checking `minor == 0` will run
on GB300 but may fail if the underlying kernel was compiled only for SM10.0.
Current guards that require exactly SM10.0 (`cutlass_mla`, `dsa trtllm`) are
intentionally conservative until the container ships SM10.3-compiled binaries.

---

## 5. Documented layout limitations (SUBSKIPPED via `LAYOUT_KNOWN_FAILURES`)

These are real production limitations surfaced by the layout-robustness tests
(see PLAN.md "Latest verification — Layout-robustness arc"). They are not
container issues and will not be fixed by re-imaging. Each `LAYOUT_KNOWN_FAILURES`
entry in the test file records the production-side cause inline so the next
person can pick up fixes without re-discovering the bug.

All entries are exercised through `test_layout_robustness_cases` (one method
per affected backend) and skip cleanly with the documented reason when the
test runs. Three layouts are tested per case: `interleaved_pages`,
`non_monotonic_extend`, and (the default everywhere) `shuffled_pages` — the
last is exercised by every test in the suite, so a regression in shuffled_pages
shows up immediately in any test, not just `test_layout_robustness_cases`.

### FlashAttention dense family

| Test | Case | Layout | Root cause |
|---|---|---|---|
| `dense/test_fa3.py::test_layout_robustness_cases` | `layout_extend_two_request_ragged` | `non_monotonic_extend` | FA3 prefill metadata assumes `out_cache_loc` is monotonic within an extend; a fragmented allocator could trip this |
| `dense/test_fa4.py::test_layout_robustness_cases` | `layout_extend_two_request_ragged` | `non_monotonic_extend` | FA4 inherits FA3's prefill metadata assumption |

### MLA family

| Test | Case | Layout | Root cause |
|---|---|---|---|
| `mla/test_flashinfer.py::test_layout_robustness_cases` | `layout_mla_extend_prefix_exact_page` | `interleaved_pages` | FlashInfer MLA paged-prefill metadata assumes a tidy page-table layout; interleaved pages trip an illegal memory access inside the kernel |
| `mla/test_flashinfer.py::test_layout_robustness_cases` | `layout_mla_extend_prefix_exact_page` | `non_monotonic_extend` | FlashInfer MLA paged-prefill metadata assumes monotonic `out_cache_loc` within an extend; scattered extend slots trip an illegal memory access |
| `mla/test_flashinfer.py::test_layout_robustness_cases` | `layout_mla_decode_page_boundary` | `interleaved_pages` | FlashInfer MLA paged-decode metadata raises `CUBLAS_STATUS_EXECUTION_FAILED` on interleaved-page layouts |
| `mla/test_flashmla.py::test_layout_robustness_cases` | `layout_mla_extend_prefix_exact_page` | `interleaved_pages` | FlashMLA extend path raises CUDA illegal memory access on interleaved-page layouts; the kernel assumes a tidy page-table layout |
| `mla/test_flashmla.py::test_layout_robustness_cases` | `layout_mla_extend_prefix_exact_page` | `non_monotonic_extend` | FlashMLA extend path raises CUDA illegal memory access on non-monotonic `out_cache_loc` within an extend |
| `mla/test_flashmla.py::test_layout_robustness_cases` | `layout_mla_decode_page_boundary` | `interleaved_pages` | FlashMLA decode path raises a shape mismatch (`shape '[-1, 64, 1, 32]' is invalid for input of size N`) on interleaved-page layouts |

### Dual-chunk

| Test | Case | Layout | Root cause |
|---|---|---|---|
| `dual_chunk/test_dual_chunk_flash_attn.py::test_layout_robustness_cases` | `layout_dual_chunk_extend_two_request` | `non_monotonic_extend` | `_dual_chunk_flash_attn_prefill_func` uses `cu_seqlens_*` indexing into contiguous K slots (`dual_chunk_flashattention_backend.py:834+`); scattered extend-token slots break that contiguity |

Total: **9 documented layout failures** across 5 backend test files. See per-method
READMEs for engineering paths to fix.

---

## 6. H200-observable production limitations (no corresponding test in suite)

These are production-side backend bugs / structural rejects identified during
fixture investigation on H200/SM9.0. They do **not** correspond to any active
test in the suite — the test was probed, found to hit a production-side wall,
and the case was either deferred or omitted with the cause recorded inline.
Listed here so that anyone touching the affected production path knows there
is recoverable test coverage waiting on a fix.

### 6a. Dual-chunk vertical+slash sparse production bugs

Surfaced while attempting a sub-context-window sparse smoke test (PLAN.md
"Deferred follow-ups"; `dual_chunk/README.md`). Both bugs block landing a
genuine sub-window pruning test today.

| Citation | Symptom | Trigger |
|---|---|---|
| `dual_chunk_flashattention_backend.py:1110-1132` | `RuntimeError: The size of tensor a (4) must match the size of tensor b (5)` at `vertical_buffer.copy_()` | `vertical_size ≤ 5`: fallback `torch.arange(0, intra_K_size, max(1, intra_K_size/5))` returns up to 5 elements into a `vertical_size=4` slot buffer when `intra_vertical_indices.nelement() == 0` |
| `_vertical_slash_sparse_attention` (`convert_vertical_slash_indexes` block math) | `cudaErrorIllegalAddress` deep inside the kernel | `vertical_size=8` with `seq_len ≥ 128`: unstated invariant that `vertical_size + slash_size >= chunk_len_blocks` |

The smoke helper `run_dual_chunk_sparse_sub_window_case` is wired through
`common/attention_methods/dual_chunk_attention.py` for when these production
bugs are fixed; no test method invokes it today. See `dual_chunk/README.md`.

### 6b. Speculative-kind production limitations

| Backend | Mode / spec_kind | Cause | Test status |
|---|---|---|---|
| FlashInfer MLA | non-EAGLE chain verify (frozen_kv_mtp / dflash / ngram) | `forward_extend` reads EAGLE-specific `spec_info` attrs and trips CUDA illegal-memory access on non-EAGLE attrs | Test omitted; `mla/README.md` documents |
| FlashMLA | non-EAGLE chain verify | Same as FlashInfer MLA (inherits) | Test omitted; `mla/README.md` documents |
| FlashInfer SWA | non-EAGLE chain verify | `FlashInferIndicesUpdaterPrefill.update_sliding_window` rejects `prefix_lens=None` which the non-EAGLE paths supply (`flashinfer_backend.py:742,754,1316`) | Test omitted; `swa/README.md` documents |
| KDA | non-EAGLE chain verify | 1/384 elements at ~0.11 max diff vs `KDA_ATOL=0.1`; needs kind-specific reference tolerance adjustment | Per-case `atol=0.2` override applied where attempted; `kda/test_triton.py` |
| Mamba2 | tree verify (`topk > 1`) | SSM kernel ignores tree masks and processes drafts linearly regardless of tree shape | `skipTest("Mamba2 tree verify (topk>1) is structurally unsupported")` in `speculative_target_verify_runner.py:1214,1276` |
| Lightning | tree verify (`topk > 1`) | `linear/seg_la.py` has no parent-indices / retrieve-index plumbing; processes drafts as chain regardless of tree shape | Test omitted; `lightning/README.md` documents |
| FA3 / FA4 | EAGLE tree verify (`topk = 2`) | ~0.16 abs-diff bf16 drift on the **eager** path (kernel-level numerical drift, not CG mechanic) | Test omitted; `dense/README.md` documents |

### 6c. Graph-capture / runner-mode production limitations

| Backend | Mode | Cause | Test status |
|---|---|---|---|
| FlashMLA | MLA `DRAFT_EXTEND` CUDA-graph replay | Capture falls through to `FlashInferMLAAttnBackend.init_forward_metadata_capture_cuda_graph` which reads 1D `cuda_graph_kv_indices`; FlashMLA decode uses 2D `[max_bs, (max_context + PAGE_SIZE) // PAGE_SIZE]` layout. Buffer mismatch | Test omitted; `mla/README.md` "Next Work" |
| Triton dense | `DRAFT_EXTEND` (non-V2) eager | Fixture/reference semantic mismatch even on narrow accepted-token layouts | Test omitted; `dense/README.md` documents |
| GDN, KDA, Lightning, Mamba2 | `DRAFT_EXTEND` and `DRAFT_EXTEND_V2` graph capture | `HybridLinearAttnBackend` raises `ValueError("Invalid forward mode")` at `hybrid_linear_attn_backend.py:509,572` | Eager-only paths covered; CG paths omitted with production citation |
| DSV4 | EAGLE draft_extend with `compress_ratio != 0` | `DeepseekV4ModelNextN` hardcodes `compress_ratio_override=0`, making C4/C128 draft_extend production-unreachable | Test asserts `case.compress_ratio == 0` at the call site to make this loud |
| DSV4 | tree verify (`topk > 1`) | `assert self.topk in [0, 1]` at `deepseek_v4_backend.py:369` | Test omitted; `dsv4/README.md` documents |

### 6d. Split-op (PCG/BCG) production limitations

| Backend | Cause | Test status |
|---|---|---|
| Lightning | Backend returns flat `[T, num_heads * head_dim]` at `lightning_backend.py:335`; `RadixAttention` piecewise writes per-head (`radix_attention.py:124-137`). Shape mismatch when comparing eager vs piecewise output | Adapter wired (`run_lightning_split_op_extend_case`), test omitted; `lightning/README.md` |
| Mamba2 | `MambaMixer2.forward` projects ALL rows of `hidden_states` before per-layer `num_token_non_padded_cpu` slicing (`mamba.py:467`); trips assert under token-padding | Adapter wired, test omitted; `mamba/README.md` |
| DSV4 | `flash_mla.flash_mla_with_kvcache` asserts `indices.shape == (b, s_q, topk)`; metadata buffers sized for live batch, q is static-token-padded — shape mismatch only manifests inside the kernel | Test omitted; `dsv4/README.md` Coverage Matrix shows `—` |
| DSA MHA_ONE_SHOT dense fallback | DSA passes K as concatenated `prefix + extend` to `module.attn(save_kv_cache=False)`; `unified_attention_with_output` (`radix_attention.py:170-208`) slices K to `num_token_non_padded_cpu` (= live extend-token count), dropping the prefix portion — piecewise CG diverges from eager by ~50% mismatch (~0.35 max diff) | Test omitted; `dsa/README.md` "Production-Unsupported" |

### 6e. DSA-specific structural limitations

| Item | Cause | Test status |
|---|---|---|
| DSA EAGLE tree draft (`topk > 1`) | Synthesizes `topk_indices` on-GPU from `batch.seq_lens` (trailing-topk); tree draft needs parent-indices plumbing through the synthesis that production sources from the DSA indexer | Chain-only (`topk=1`) covered; tree omitted; `dsa/README.md` documents |
| DSA HiSparse coordinator path | `set_dsa_prefill_impl` forces `use_mha=False` when `hisparse_coordinator is not None`. Mocking the coordinator needs to mirror the fast-drifting production page-table contract | Test omitted; `dsa/README.md` "Next Work" |
| DSA page sizes other than `{1 (HIP), 64 (CUDA)}` | Indexer hard-asserts (`dsa/dsa_indexer.py:547-548, 550, 724-725, 727, 946, 1095`; `dsa/index_buf_accessor.py:436`; `dsa/transform_index.py:53, 79, 100, 121`) | Cases omitted; `dsa/README.md` "Production-Unsupported" |

### 6f. MLA backend page-size hard-pins

These are not bugs but **production-design constraints** documented for completeness — they make many "natural" page-size cases (e.g., `page_size=1`) impossible to test for those backends.

| Backend | Required page size(s) | Citation |
|---|---|---|
| FlashMLA | `64` only | `server_args.py:2767-2770` |
| Cutlass MLA | `128` only | `server_args.py:2776-2779`, `cutlass_mla_backend.py:31` |
| TRT-LLM MLA | `{32, 64}` | `server_args.py:2790-2794` |
| Tokenspeed MLA | `{32, 64}` | `server_args.py:2809-2813`, `tokenspeed_mla_backend.py:111-113` |
| TRT-LLM MHA | `{16, 32, 64}` | `server_args.py:2849-2853` |
| FA4 (non-MLA) | `128` when default-selected | `server_args.py:2862-2870` |
| DSV4 | `256` only | `deepseek_v4_backend.py:355`, `dsv4/metadata.py:134` |
| Intel XPU MLA decode | `{16, 32, 64, 128}` | `server_args.py:2906` |
| Intel XPU non-MLA decode | `{64, 128}` | `server_args.py:2909` |

---

## Summary table

| # | File | Sub-test count | Category | Fix |
|---|---|---|---|---|
| 1 | `dual_chunk/test_dual_chunk_flash_attn.py` | 18 | Container: flash_attn missing SM10.3 | Re-image |
| 2 | `dsa/test_dsa.py` | 2 | Container: tilelang missing SM10.3 MMA | Re-image |
| 3 | `mla/test_flashinfer.py` | 1 | Backend: FlashInfer MLA draft CG on Blackwell | Update FlashInfer |
| 4 | Various | 0 (now correctly SKIPPED) | Hardware gate: SM90a/SM10.0 kernels | Fixed in test gates |
| 5 | `dense/test_fa3.py`, `dense/test_fa4.py`, `mla/test_flashinfer.py`, `mla/test_flashmla.py`, `dual_chunk/test_dual_chunk_flash_attn.py` | 9 (SUBSKIPPED) | Production limitation: backend metadata assumes tidy `(req_to_token, out_cache_loc)` layout | Backend fix (see per-method READMEs) |
| 6 | No test in suite (deferred) | ~25 production limitations | H200-observable production-side backend bugs / structural rejects identified during fixture investigation | Backend fix (see per-method READMEs and PLAN.md "Deferred follow-ups") |
