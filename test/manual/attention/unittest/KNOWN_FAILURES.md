# Known Failures — Attention Backend Unit Tests

This file catalogs every backend issue (production-side bug, structural
reject, container gap, or hardware-architecture gate) that affects the
unit-test suite, **organized by the action needed to address it**.

Anything failing that is not listed here should be treated as a regression.

Last updated: 2026-05-27

## Reference runs

| Host | Hardware | Result |
|---|---|---|
| H200 | SM 9.0 (Hopper) | **176 tests, 30 skipped, 0 failures** in ~40 s |
| GB300 | SM 10.3 (Grace-Blackwell) | After `cf482d662`: all §A/§B/§C.3-Blackwell failures now skip cleanly with documented reasons. Previously: 21 failed, 160 passed, 87 skipped, 436 subtests passed in ~215 s. |

## Top-level structure

| § | Category | Action needed |
|---|---|---|
| **A** | Container dependency missing | **Re-image** with SM10.x-compatible wheels |
| **B** | Hardware-architecture gate | None — tests skip cleanly when SM doesn't match; correctly designed |
| **C** | Backend production-side bug or structural reject | **Production code change** in `python/sglang/srt/layers/attention/` |
| **D** | Production-design constraint | None — these are intentional rejects (page-size pins, topk limits) |

Within **C**, sub-sections by bug category (layout / speculative / graph-runner /
split-op / sparse-kernel / DSA-specific). Each entry tags its current test
status: `[gated]` (skipTest gate fires today), `[no test]` (no test attempts
it; documented in per-method README), or `[gated on X]` (gate fires only on
hardware/version X).

---

# A. Container re-image required

## A.1. `flash_attn` SM10.x wheel missing

**Affected**: `dual_chunk/test_dual_chunk_flash_attn.py` (entire class — 5
test methods, ~18 subtests)

**Symptom on GB300**:
```
ImportError: cannot import name 'flash_attn_varlen_func' from 'flash_attn'
```

**Root cause**: `DualChunkFlashAttentionBackend` calls `flash_attn_varlen_func`
via `sglang.jit_kernel.flash_attention`. On SM 8.x / 9.x that resolves to
sgl-kernel's FA3 build (works on H200). On other SMs, the JIT kernel falls
back to the upstream `flash_attn` (FA2) wheel — but the
`lmsysorg/sglang:nightly-dev-cu13` container's `flash_attn` package on
SM10.x is missing `flash_attn_varlen_func`.

**Gate**: `_dual_chunk_fa_supported()` in
`dual_chunk/test_dual_chunk_flash_attn.py` skips the whole class on the
fallback-broken path. Hopper passes through unchanged.

**Fix**: Re-image with an SM10.x-compiled `flash_attn` wheel.

## A.2. tilelang `wait_wgmma` template missing on SM10.x

**Affected**:
- `dsa/test_dsa.py::test_sparse_tilelang_prefill_case` (1 test)
- `dsa/test_dsa.py::test_sparse_tilelang_decode_case` (1 test)
- `tilelang` rows in `test_sparse_{prefill,decode,cuda_graph_decode}_impl_variants`

**Symptom on GB300**:
```
RuntimeError: namespace "tl" has no member "wait_wgmma"
```

**Root cause**: tilelang JIT generates `wait_wgmma` (a Blackwell WGMMA-sync
intrinsic) on SM10.x, but the container's MMA template library is missing
it. PTX compilation fails.

**Gate**: `dsa_impl_capability("tilelang")` in
`common/attention_methods/dsa_attention.py` skips on `major >= 10`. Override
with `SGLANG_TEST_DSA_TILELANG_FORCE=1` after re-imaging.

**Fix**: Re-image with an SM10.x-compatible tilelang version.

---

# B. Hardware-architecture gates (no action needed)

These tests skip cleanly when the running SM doesn't match the backend's
required architecture. The gates are correct as designed; the table is here
so that "skipped: ..." results have a quick lookup.

| Backend | Required SM | Gate location | Error if unguarded |
|---|---|---|---|
| `cutlass_mla` | exactly SM 10.0 (B200) | `mla/test_cutlass_mla.py::_supported` | `cutlass_mla_decode is only supported on compute capability 10.0, but found sm version 103` |
| `flashmla` decode/verify | SM 9.0 (Hopper) only | `mla/test_flashmla.py:_DECODE_REQUIRES_SM90A` | `Dense decode MLA is only supported on SM90a architecture` |
| `trtllm_mla` | SM 12.0a / 12.1a | `mla/test_trtllm_mla.py::_supported` | FlashInfer XQA MLA dispatch reject |
| `tokenspeed_mla` | SM ≥ 10.0 + FP8 KV + pkg | `mla/test_tokenspeed_mla.py::_supported` | `tokenspeed_mla` import or kernel dispatch |
| `trtllm_mha` prefill | SM ≥ 10.0 | `dense/test_trtllm_mha.py` decode-only matrix | FlashInfer TRT-LLM Gen FMHA reject (`Unsupported architecture`) |
| `dsa` `fa3` impl | SM 9.x only | `dsa_impl_capability("fa3")` | `flash_attn at sgl-kernel is only supported on sm90 and above` |
| `dsa` `trtllm` impl | exactly SM 10.0 | `dsa_impl_capability("trtllm")` | `Missing TRTLLM-GEN kernel` (compiled for SM10.0) |
| `fa3` (non-MLA) | SM 80 or SM 90 | `_is_fa3_supported` in `flash_attention_v3.py` | `attention_registry.py:177-180` reject |

**SM10.3 vs SM10.0**: GB300 is SM10.3. Gates that require exactly SM10.0
(cutlass_mla, dsa trtllm) intentionally skip on GB300 because the kernel
binaries in the container aren't compiled for sm_103. Flip the gates to
`major == 10` (drop the `minor == 0`) once GB300-compiled binaries land.

---

# C. Backend bugs needing production code fixes

## C.1. Layout-handling bugs (gated via `LAYOUT_KNOWN_FAILURES`)

Surfaced by the layout-robustness arc. The default layout for every test
is now `shuffled_pages`; the more aggressive `interleaved_pages` and
`non_monotonic_extend` are exercised by per-backend `test_layout_robustness_cases`
methods that record each backend's failure mode inline as
`LAYOUT_KNOWN_FAILURES`. Each entry below `[gated]` and skips cleanly.

### FA dense

| Test | Layout | Root cause |
|---|---|---|
| `dense/test_fa3.py::test_layout_robustness_cases` (extend) | `non_monotonic_extend` | FA3 prefill metadata assumes `out_cache_loc` is monotonic within an extend. |
| `dense/test_fa4.py::test_layout_robustness_cases` (extend) | `non_monotonic_extend` | FA4 inherits FA3's assumption. |

### MLA

| Test | Mode / layout | Root cause |
|---|---|---|
| `mla/test_flashinfer.py::test_layout_robustness_cases` (extend) | `interleaved_pages` | FlashInfer MLA paged-prefill metadata assumes tidy page-table layout; trips illegal memory access. |
| `mla/test_flashinfer.py::test_layout_robustness_cases` (extend) | `non_monotonic_extend` | FlashInfer MLA paged-prefill metadata assumes monotonic `out_cache_loc`; trips illegal memory access. |
| `mla/test_flashinfer.py::test_layout_robustness_cases` (decode) | `interleaved_pages` | FlashInfer MLA paged-decode raises `CUBLAS_STATUS_EXECUTION_FAILED`. |
| `mla/test_flashmla.py::test_layout_robustness_cases` (extend) | `interleaved_pages` | FlashMLA extend raises CUDA illegal memory access. |
| `mla/test_flashmla.py::test_layout_robustness_cases` (extend) | `non_monotonic_extend` | FlashMLA extend raises CUDA illegal memory access. |
| `mla/test_flashmla.py::test_layout_robustness_cases` (decode) | `interleaved_pages` | FlashMLA decode raises `shape '[-1, 64, 1, 32]' is invalid for input of size N`. |

### Dual-chunk

| Test | Layout | Root cause |
|---|---|---|
| `dual_chunk/test_dual_chunk_flash_attn.py::test_layout_robustness_cases` (extend) | `non_monotonic_extend` | `_dual_chunk_flash_attn_prefill_func` uses `cu_seqlens_*` indexing into contiguous K slots (`dual_chunk_flashattention_backend.py:834+`); scattered extend-token slots break that contiguity. |

**Total**: 9 layout-handling production bugs documented.

## C.2. Speculative-mode rejects

Mix of `[gated]` (skipTest fires today) and `[no test]` (probed during
fixture investigation; no test in the suite).

| Backend | Spec mode/kind | Status | Root cause |
|---|---|---|---|
| Mamba2 | tree verify (`topk > 1`) | `[gated]` `speculative_target_verify_runner.py:1214,1276` | SSM kernel ignores tree masks and processes drafts linearly |
| FlashInfer MLA | non-EAGLE chain verify (frozen_kv_mtp / dflash / ngram) | `[no test]` (`mla/README.md`) | `forward_extend` reads EAGLE-specific `spec_info` attrs; trips CUDA illegal-memory access on non-EAGLE attrs |
| FlashMLA | non-EAGLE chain verify | `[no test]` (`mla/README.md`) | Same as FlashInfer MLA (inherits) |
| FlashInfer SWA | non-EAGLE chain verify | `[no test]` (`swa/README.md`) | `FlashInferIndicesUpdaterPrefill.update_sliding_window` rejects `prefix_lens=None` which non-EAGLE paths supply (`flashinfer_backend.py:742,754,1316`) |
| KDA | non-EAGLE chain verify | `[no test]` (`kda/test_triton.py`, per-case `atol=0.2` attempted) | 1/384 elements at ~0.11 max diff vs `KDA_ATOL=0.1`; needs kind-specific reference tolerance |
| Lightning | tree verify (`topk > 1`) | `[no test]` (`lightning/README.md`) | `linear/seg_la.py` has no parent-indices / retrieve-index plumbing |
| FA3 / FA4 | EAGLE tree verify (`topk = 2`) | `[no test]` (`dense/README.md`) | ~0.16 abs-diff bf16 eager-path drift; kernel-level numerical |
| DSV4 | tree verify (`topk > 1`) | `[no test]` (`dsv4/README.md`) | `assert self.topk in [0, 1]` at `deepseek_v4_backend.py:369` |

## C.3. Graph-runner / CG-capture rejects

| Backend | Mode | Status | Root cause |
|---|---|---|---|
| FlashInfer MLA | EAGLE draft CG, chain | `[gated on SM≥10]` `mla/test_flashinfer.py::test_runner_mode_eagle_draft_cuda_graph_runner_cases` | FlashInfer MLA decode kernel in container targets SM9x; on Blackwell falls back to a generic path that doesn't restore metadata buffers under graph replay (~22 abs-diff vs reference) |
| FlashMLA | MLA `DRAFT_EXTEND` CUDA-graph replay | `[no test]` (`mla/README.md` Next Work) | Capture falls through to `FlashInferMLAAttnBackend.init_forward_metadata_capture_cuda_graph` (1D `cuda_graph_kv_indices`); FlashMLA decode uses 2D `[max_bs, (max_context + PAGE_SIZE) // PAGE_SIZE]` layout — buffer mismatch |
| GDN / KDA / Lightning / Mamba2 | `DRAFT_EXTEND` and `DRAFT_EXTEND_V2` graph capture | `[no test]` for CG; eager-only paths covered | `HybridLinearAttnBackend` raises `ValueError("Invalid forward mode")` at `hybrid_linear_attn_backend.py:509,572` |
| DSV4 | EAGLE draft_extend with `compress_ratio != 0` | `[no test]` (runner asserts `case.compress_ratio == 0`) | `DeepseekV4ModelNextN` hardcodes `compress_ratio_override=0`, making C4/C128 draft_extend production-unreachable |

## C.4. Split-op (PCG / BCG) rejects

All four have the adapter helpers wired so the test can be enabled the
moment production is fixed; no test method invokes them today.

| Backend | Status | Root cause |
|---|---|---|
| Lightning | `[no test]` (`lightning/README.md`) | Backend returns flat `[T, num_heads * head_dim]` at `lightning_backend.py:335`; `RadixAttention` piecewise writes per-head (`radix_attention.py:124-137`). Shape mismatch eager vs piecewise |
| Mamba2 | `[no test]` (`mamba/README.md`) | `MambaMixer2.forward` projects ALL rows of `hidden_states` before per-layer `num_token_non_padded_cpu` slicing (`mamba.py:467`); trips assert under token-padding |
| DSV4 | `[no test]` (`dsv4/README.md`) | `flash_mla.flash_mla_with_kvcache` asserts `indices.shape == (b, s_q, topk)`; metadata sized for live batch, q is static-token-padded |
| DSA MHA_ONE_SHOT dense fallback | `[no test]` (`dsa/README.md`) | DSA passes K as concatenated `prefix + extend` to `module.attn(save_kv_cache=False)`; `unified_attention_with_output` (`radix_attention.py:170-208`) slices K to `num_token_non_padded_cpu`, dropping the prefix portion — piecewise CG diverges from eager ~50% mismatch (~0.35 max diff) |

## C.5. Sparse-kernel production bugs

| Citation | Symptom | Trigger | Status |
|---|---|---|---|
| `dual_chunk_flashattention_backend.py:1110-1132` | `RuntimeError: The size of tensor a (4) must match the size of tensor b (5)` at `vertical_buffer.copy_()` | `vertical_size ≤ 5`: fallback `torch.arange(0, intra_K_size, max(1, intra_K_size/5))` returns up to 5 elements into `vertical_size=4` buffer when `intra_vertical_indices.nelement() == 0` | `[no test]` (`dual_chunk/README.md`); smoke helper `run_dual_chunk_sparse_sub_window_case` wired but not invoked |
| `_vertical_slash_sparse_attention` (`convert_vertical_slash_indexes` block math) | `cudaErrorIllegalAddress` deep inside the kernel | `vertical_size=8` with `seq_len ≥ 128`: unstated invariant that `vertical_size + slash_size >= chunk_len_blocks` | `[no test]` (same smoke helper) |
| Triton dense `DRAFT_EXTEND` (non-V2) | Eager fixture/reference mismatch on narrow accepted-token layouts | Test omitted | `[no test]` (`dense/README.md`) |

## C.6. DSA-specific structural gaps

| Item | Status | Root cause |
|---|---|---|
| DSA EAGLE tree draft (`topk > 1`) | `[no test]` (`dsa/README.md`); chain-only (`topk=1`) covered | `_DSAEagleDraftForward.__call__` synthesizes `topk_indices` on-GPU (trailing-topk in token-position space); tree draft needs parent-indices plumbing through that synthesis (production sources them from the DSA indexer that lives outside attention) |
| DSA HiSparse coordinator path | `[no test]` (`dsa/README.md` Next Work) | `set_dsa_prefill_impl` forces `use_mha=False` when `hisparse_coordinator is not None`. Mocking the coordinator needs to mirror the fast-drifting production page-table contract |

---

# D. Production-design constraints (intentional, not bugs)

These are documented for context — they make many "natural" test shapes
impossible because production rejects the combination at construction time.
No action needed; just useful for fixture authors to know what shapes will
fail at backend init.

## D.1. Backend page-size hard-pins

| Backend | Required page size(s) | Citation |
|---|---|---|
| FlashMLA | `64` only | `server_args.py:2767-2770` |
| Cutlass MLA | `128` only | `server_args.py:2776-2779`, `cutlass_mla_backend.py:31` |
| TRT-LLM MLA | `{32, 64}` | `server_args.py:2790-2794` |
| Tokenspeed MLA | `{32, 64}` | `server_args.py:2809-2813`, `tokenspeed_mla_backend.py:111-113` |
| TRT-LLM MHA | `{16, 32, 64}` | `server_args.py:2849-2853` |
| FA4 (non-MLA) | `128` when default-selected | `server_args.py:2862-2870` |
| DSV4 | `256` only | `deepseek_v4_backend.py:355`, `dsv4/metadata.py:134` |
| DSA indexer | `1` (HIP) or `64` (CUDA) | `dsa/dsa_indexer.py:547-548, 550, 724-725, 727, 946, 1095` |
| Intel XPU MLA decode | `{16, 32, 64, 128}` | `server_args.py:2906` |
| Intel XPU non-MLA decode | `{64, 128}` | `server_args.py:2909` |

## D.2. Speculative `topk` hard-rejects

| Backend | Allowed `topk` | Citation |
|---|---|---|
| `flashinfer_mla` | `1` only | `flashinfer_mla_backend.py:910-913` |
| `flashmla` | `1` only | `flashmla_backend.py:555-558` |
| `trtllm_mla` | `1` only | `trtllm_mla_backend.py:1223-1229` (inherits) |
| `tokenspeed_mla` | `1` only | `tokenspeed_mla_backend.py:341-347` (inherits) |
| `dsv4` | `0` or `1` | `deepseek_v4_backend.py:369`, `:363` (HIP) |
| `trtllm_mha` (graph replay) | `1` only | `trtllm_mha_backend.py:459,492`; `server_args.py:2391-2392` |

## D.3. KV cache dtype restrictions

| Backend | Allowed dtype | Citation |
|---|---|---|
| `tokenspeed_mla` | `fp8_e4m3` only | `server_args.py:2814-2818` |
| `trtllm_mla` | `{fp8_e4m3, fp4_e2m1, bf16, auto}` | `server_args.py:2796-2799` |
| `fa3` | not `fp8_e5m2` (silently falls back to `triton`) | `server_args.py:2855-2860` |
| `dsv4` | Packed FP8/BF16 layout enforced by `DeepSeekV4TokenToKVPool` | `deepseek_v4_backend.py:363` |

---

# Quick lookup — by test file

| Test file | Failure type | Section |
|---|---|---|
| `dual_chunk/test_dual_chunk_flash_attn.py` | Container: `flash_attn` SM10.x wheel | §A.1 |
| `dual_chunk/test_dual_chunk_flash_attn.py::test_layout_robustness_cases` (non_monotonic_extend) | Layout-handling bug | §C.1 |
| `dsa/test_dsa.py::test_sparse_tilelang_*` | Container: tilelang `wait_wgmma` | §A.2 |
| `dsa/test_dsa.py::test_sparse_*_impl_variants` (tilelang row) | Container: tilelang `wait_wgmma` | §A.2 |
| `dsa/test_dsa.py::test_sparse_*_impl_variants` (fa3 / trtllm rows) | Hardware gate | §B |
| `mla/test_cutlass_mla.py` (all) | Hardware gate (SM 10.0 exactly) | §B |
| `mla/test_flashmla.py` (DECODE/verify subtests) | Hardware gate (SM 9.0 Hopper) | §B |
| `mla/test_flashinfer.py::test_runner_mode_eagle_draft_cuda_graph_runner_cases` | Backend bug gated on SM≥10 | §C.3 |
| `mla/test_flashinfer.py::test_layout_robustness_cases` | Layout-handling bug | §C.1 |
| `mla/test_flashmla.py::test_layout_robustness_cases` | Layout-handling bug | §C.1 |
| `dense/test_fa3.py::test_layout_robustness_cases` (non_monotonic_extend) | Layout-handling bug | §C.1 |
| `dense/test_fa4.py::test_layout_robustness_cases` (non_monotonic_extend) | Layout-handling bug | §C.1 |
| `mamba/test_mamba2.py` spec verify tree (topk>1) | Speculative reject | §C.2 |
