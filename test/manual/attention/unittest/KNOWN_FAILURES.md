# Known Failures — Attention Backend Unit Tests

This file catalogs every non-passing subtest result that is **expected** at the
current state of the test suite, grouped by root cause. A failure is "expected"
if it is caused by a missing container dependency, a hardware gate, or a
documented production limitation — not by a bug in the test logic or attention
backend that we intend to fix in this branch.

Anything that is not listed here and fails should be treated as a regression.

Last updated: 2026-05-27
Reference run: GB300 SM10.3, sglang-kernel 0.4.3, nightly-dev-cu13 container
Result: **21 failed, 160 passed, 87 skipped, 436 subtests passed** in ~215 s

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

These are real production limitations surfaced by the layout-robustness tests.
They are not container issues and will not be fixed by re-imaging.

| Test | Case / layout | Root cause |
|---|---|---|
| `dual_chunk/test_dual_chunk_flash_attn.py::test_layout_robustness_cases` | `layout_dual_chunk_extend_two_request` / `non_monotonic_extend` | `_dual_chunk_flash_attn_prefill_func` uses `cu_seqlens_*` indexing into contiguous K slots (backend.py:834+); scattered extend-token slots break that contiguity |

See `dual_chunk/README.md` for the engineering path to fix.

---

## Summary table

| # | File | Sub-test count | Category | Fix |
|---|---|---|---|---|
| 1 | `dual_chunk/test_dual_chunk_flash_attn.py` | 18 | Container: flash_attn missing SM10.3 | Re-image |
| 2 | `dsa/test_dsa.py` | 2 | Container: tilelang missing SM10.3 MMA | Re-image |
| 3 | `mla/test_flashinfer.py` | 1 | Backend: FlashInfer MLA draft CG on Blackwell | Update FlashInfer |
| 4 | Various | 0 (now correctly SKIPPED) | Hardware gate: SM90a/SM10.0 kernels | Fixed in test gates |
| 5 | `dual_chunk/test_dual_chunk_flash_attn.py` | 1 (SUBSKIPPED) | Production limitation: contiguous K assumption | Backend fix (see README) |
