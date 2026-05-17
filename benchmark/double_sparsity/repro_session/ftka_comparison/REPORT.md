# FTKA evaluation: integrate / defer / reject

**Scope.** Evaluate `tsinghua-ideal/flash-topk-attention` (FTKA, commit
`d8803b29961c44d77a747636ad4282bd7a9094af`) as an **implementation
substitute** for the DS native sparse-decode path's *score* and *top-k*
phases. No algorithm change: same K-channel calibration, same token
budget, same sink/recent retention, same selected-token contract. The
Twilight adaptive-budget / top-p pruner / weight estimator family is
explicitly out of scope for this pass.

**Method.** The microbench
(`benchmark/double_sparsity/repro_session/microbench_ftka_backends.py`)
times four paths at 76 DS shapes (3 contexts × 5 bs × 5 top_k + an
h_kv=8 stress row). For each (path, shape) entry it records mean µs/call,
selected-physical-id set parity vs the torch baseline, and CUDA-graph
capture/replay status under an isolated probe. Raw artifacts:
[`results.json`](./results.json), [`results.md`](./results.md).

The four paths:

| ID | Path | What it substitutes |
|----|------|---------------------|
| **P1** | `score_triton + torch.topk + build_selected_physical` | — (current production baseline) |
| **P2** | `score_triton + flashinfer.top_k_page_table_transform + append_sink_recent` | top-k step only |
| **P3** | `score_triton + ftka.cuda_ops.raft_topk + build_selected_physical` | top-k step only |
| **P4** | `ftka.cuda_ops.batched_sparse_gemv + ftka.cuda_ops.raft_topk + build_selected_physical` | score + top-k |

## Environment of this run

| Item | Value |
|------|-------|
| GPU | NVIDIA H200 (sm_90, 144 SMs) |
| torch | `2.11.0+cu130` |
| flashinfer | `0.6.11.post1` |
| ftka | installed editable from local clone of `flash-topk-attention@d8803b29961c44d77a747636ad4282bd7a9094af` |
| ftka build-glue patches | minimal raft-topk-only TU (see "FTKA build-glue port" below) |

## Headline verdict

| Component | Decision | Evidence |
|-----------|----------|----------|
| `torch.topk` selector (P1) | **KEEP — production default** | Clean at all 76 shapes; capture-safe in isolated probe; **fastest path at the DS production envelope** (`bs ≥ 16`, `ctx ≥ 64K`). |
| `flashinfer_topk_page_table` (P2) | **NO STATE CHANGE — measured but not production-safe** | 46 valid pairs (`top_k ≤ 2048`); median 1.22× speedup vs P1; isolated capture OK but still fails under SGLang's nested-stream graph capture (Triton `load_binary` illegal memory access at first captured forward). Documented in `selector_backends.py`. |
| **`ftka_raft_topk` (P3)** | **REJECT for DS production** | Parity ok (76/76), graph-safe (76/76), top_k up to 8192 supported. But: median speedup is **strongly context-dependent**. At `ctx = 32K` RAFT wins 1.97×–3.06× (torch under-utilizes for that workload); at `ctx = 64K` it's a wash (0.85×–1.19×); at `ctx = 128K` — **the DS production target** — RAFT is **0.47×–0.83×** of torch (i.e., 17–53% slower). Production envelope (`bs ≥ 16, ctx ≥ 64K`): of 20 shapes, **only 2 pass the ≥1.15× gate, 14 fail it outright (<0.95×)**. |
| **`ftka_gemv+ftka_topk` (P4)** | **REJECT — structural** | `csrc/src/ftka_ops.cu:70` instantiates `BatchedSparseGEMV<128, ...>` at compile time. The head_dim template parameter is the literal `128`. DS K-labels have `S = 32`. The kernel would crash or compute over uninitialized memory when invoked against DS's layout. To run honestly would require either recompiling FTKA with `HEAD_DIM = 32` (no Python knob exposed) or padding DS labels to 128 (4× memory + 4× compute hit — not "same algorithm"). The microbench surfaces this as `status=error` with the source citation in every shape (see `results.md`). |

**Bottom line: no FTKA component is fast enough to replace any current DS
selector at the production shape envelope.** Production default
`selector_backend=torch` stays unchanged. No e2e (README headline)
benchmarks were run — PLAN gate explicitly conditions e2e on P3
passing the microbench gate, which it does not.

## P3 — full grid, by context

The clearest single picture is binning by `max_ctx`:

| max_ctx | n | min ratio | median ratio | max ratio | verdict |
|---:|---:|---:|---:|---:|---|
| 32768 | 25 | 1.97× | **2.36×** | 3.06× | RAFT wins decisively (torch is launch-bound at this regime) |
| 65536 | 26 | 0.85× | **1.00×** | 1.19× | wash; RAFT marginal |
| 131072 | 25 | 0.47× | **0.63×** | 0.83× | RAFT loses everywhere; gap widens with top_k |

**P3 headline DS shapes** (`bs ∈ {16, 32}, top_k ∈ {1024, 2048}`, `h_kv = 1`):

| bs | ctx | top_k | torch µs | ftka µs | ratio | gate ≥1.15× |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 32K | 1024 | 88.6 | 34.2 | **2.59×** | pass |
| 16 | 32K | 2048 | 89.1 | 37.8 | **2.36×** | pass |
| 16 | 64K | 1024 | 88.1 | 81.9 | 1.08× | fail |
| 16 | 64K | 2048 | 88.4 | 90.3 | 0.98× | fail |
| 16 | **128K** | 1024 | 94.6 | 141.8 | **0.67×** | **fail hard** |
| 16 | **128K** | 2048 | 96.9 | 153.8 | **0.63×** | **fail hard** |
| 32 | 32K | 1024 | 88.7 | 34.7 | **2.56×** | pass |
| 32 | 32K | 2048 | 89.1 | 38.2 | **2.33×** | pass |
| 32 | 64K | 1024 | 90.7 | 84.9 | 1.07× | fail |
| 32 | 64K | 2048 | 92.4 | 91.6 | 1.01× | fail |
| 32 | **128K** | 1024 | 114.8 | 145.6 | **0.79×** | fail |
| 32 | **128K** | 2048 | 116.2 | 157.0 | **0.74×** | fail |

**P3 large-k gate** (top_k ∈ {4096, 8192}; PLAN requires "supports top_k ≥ 8192"):

- Top_k = 8192 runs cleanly at every shape: **gate passes on supportability**.
- But the speedup pattern is identical to top_k = 1024/2048 — wins at 32K, breaks even at 64K, large losses at 128K (down to **0.47×** at bs=1 / ctx=128K / k=8192).

**P3 graph capture**: ok at all 76/76 shapes under the isolated probe. **Gate passes on capture-safety.**

**P3 parity**: ok at all 76/76 shapes (set equality of selected physical ids vs torch.topk). **Gate passes on correctness.**

### Why RAFT loses at 128K

RAFT's `decode_select_k` uses the **one-block-per-batch** radix kernel
(`csrc/include/raft_topk.cuh::radix_topk_one_block_kernel`,
`BitsPerPass = 8`, `BlockSize = 512`). Each batch row gets one CUDA
block. For DS at `h_kv = 1` and `bs ∈ {1, 4, 8, 16, 32}`, that's at most
32 active blocks — far below H200's 144 SM count. Within a block, the
radix passes scan the entire seq_len multiple times (`calc_num_passes`
≈ 4 passes for fp32). Time scales roughly linearly with seq_len: in
the data, `ctx 32K → 128K` (4×) increases RAFT time by 4–5× (e.g.,
bs=16/k=1024: 34.2µs → 81.9µs → 141.8µs).

torch.topk's CUB-based radix appears to use a fixed wave of blocks that
covers many SMs simultaneously, so its time is dominated by global
memory bandwidth and stays roughly flat across batch sizes at a fixed
seq_len (88–116 µs across the entire grid for h_kv = 1).

The two cross between 32K and 64K. DS production runs at ≥64K context
(README headline operating points are 128K), so the cross is on the
wrong side.

### Why h_kv > 1 doesn't rescue P3

The microbench includes one `h_kv = 8` stress row (bs=4 / ctx=64K /
k=2048). At that shape RAFT ties torch exactly (91.8 µs vs 91.7 µs).
For DS this doesn't help in practice — production GQA reductions
typically collapse to `h_kv = 1` (max-abs over query heads), so the
"more rows → better RAFT utilization" lever isn't available.

## P2 vs P3 head-to-head (`top_k ≤ 2048`, the FlashInfer ceiling)

46 shape pairs ran both P2 and P3. **Median P3/P2 = 1.10×** (i.e., P2
is 10% faster than P3 on the median shape), range 0.40×–1.78×. At
all `bs ≥ 16, ctx ≥ 64K` shapes, P2 beats P3 outright. The
FlashInfer path is the better candidate among external selectors —
but capture-unsafe under SGLang's nested-stream graph as before.

## P4 — structural rejection, no measurement performed

**Verbatim from `csrc/src/ftka_ops.cu:70`:**

```cpp
cudaError_t status = BatchedSparseGEMV<128, c_type, c_type, c_type, int32_t>(
    static_cast<c_type*>(q.data_ptr()), paged_kv, nullptr, nullptr,
    static_cast<c_type*>(o.data_ptr()), num_heads);
```

The first template argument is the C++ literal `128`. This is the
`HEAD_DIM` template parameter of `flashinfer::BatchedSparseGEMV`
(declared `template <uint32_t HEAD_DIM, ...>` in
`csrc/include/gemv.cuh:435`). It dictates compile-time constants for
the shared-memory tile size, the per-thread vec_size, and the
block-dim x dimension. DS's K-label cache uses `S = 32` heavy channels
per head. Running the kernel against an `S = 32` K-label buffer
either:

1. **Crashes** because the kernel writes past the end of the `[T_pool,
   H_kv, 32]` cache cells (it expects 128 head_dim elements per cell).
2. **Computes over uninitialized memory** in the unallocated trailing
   96 head_dim slots, producing garbage scores and thus wrong
   selections.

Neither is "same algorithm." Acceptable bridges:

- **Recompile FTKA with `HEAD_DIM = 32`** — not exposed as a Python
  knob; would require editing every `BatchedSparseGEMV<128, ...>`
  instantiation in `csrc/src/ftka_ops.cu` (4 sites) and rebuilding.
- **Pad DS K-labels to 128** — 4× memory, 4× compute, 4× cache
  bandwidth for the score phase, and a non-trivial change to
  `K_label` allocation, the score kernel, the K-label append path,
  and calibration. That is not an "implementation substitute" — it's
  a structural cache-layout change.

The microbench's `_FtkaScoreAndSelectRunner.setup()` raises
`RuntimeError` with the source citation, which the harness records as
`status = error` with the explanatory message on every shape. The
dead-code path-runner that *would* build the page-table view is kept
for documentation: if the head_dim constraint were ever lifted, the
existing scaffolding times the (gemv + topk) pair and tracks
`layout_transform_us` separately so the per-step transform cost is
visible.

## FTKA build-glue port (no kernel-logic change)

FTKA commit `d8803b29` was authored against `flashinfer == 0.2.0.post1`
and `torch == 2.5.0` (per its `requirements.txt`). Running against the
SGLang env (`flashinfer 0.6.11.post1`, `torch 2.11.0+cu130`, CUDA 13)
exposed several build-glue mismatches:

1. **flashinfer JIT API moved.** `flashinfer.jit.has_prebuilt_ops` and
   `load_cuda_ops` are gone; replaced by
   `gen_jit_spec(...).build_and_load()`. Path constants
   (`FLASHINFER_INCLUDE_DIR`, `FLASHINFER_CSRC_DIR`) moved to
   `flashinfer.jit.env`.
2. **flashinfer 0.6.11 JIT builds against tvm_ffi, not torch pybind.**
   The 0.6.11 JIT toolchain does not add torch's `csrc/utils/pybind.h`
   to the include path. FTKA's wrappers use `PYBIND11_MODULE` directly
   so compilation fails with "torch/csrc/utils/pybind.h: No such file".
3. **`vec_t<uint8_t, N>` redefinition.** flashinfer 0.6.11's
   `vec_dtypes.cuh` now provides specializations that FTKA's
   `lowbit_dtypes.cuh` also declares — duplicate-definition errors at
   compile time.
4. **nvcc 13 strict mode rejects `half x = 0;`.** FTKA's `gemv.cuh`
   and `lowbit_dtypes.cuh` use nvcc-12-era int→half implicit
   conversions; nvcc 13 requires explicit `__float2half`.
5. **FTKA's `DISPATCH_PYTORCH_DTYPE_TO_CTYPE` macro only covers
   half/bf16/fp8.** DS's score buffer is fp32. The default FTKA build
   would fail dispatch at runtime.

(1)–(4) are only triggered by `gemv.cuh` / `lowbit_dtypes.cuh` /
`pytorch_extension_utils.h` — files that FTKA's GEMV / quantized paths
depend on, but `raft_topk` does not. (5) is in the dispatch macro at
the top of `ftka_ops.cu` and would block raft_topk on fp32 even after
the build-glue fixes.

Resolution: a single carved-out translation unit at
`csrc/src/ftka_topk_only.cu` (added in the local FTKA clone, not in
this repo) that includes only `raft_topk.cuh` and exposes `raft_topk`
via pybind11 with an explicit fp32-inclusive dispatch. The patched
`ftka/cuda_ops/module.py` builds this TU via
`torch.utils.cpp_extension.load`. **No kernel logic is changed.**

Patched FTKA files (local clone only, NOT in this repo):

- `/tmp/flash-topk-attention/ftka/cuda_ops/module.py` (rewritten to use
  `torch.utils.cpp_extension.load` against the minimal TU).
- `/tmp/flash-topk-attention/csrc/src/ftka_topk_only.cu` (new, ~70
  lines, includes only `raft_topk.cuh` + an fp32 dispatch macro).

Both files carry comment headers explaining the patch rationale and
linking to this report.

## How to reproduce on a fresh checkout

```bash
# 1. Clone FTKA at the pinned commit (read-only source audit).
git clone --depth 1 https://github.com/tsinghua-ideal/flash-topk-attention.git /tmp/flash-topk-attention
cd /tmp/flash-topk-attention
git fetch --depth 1 origin d8803b29961c44d77a747636ad4282bd7a9094af
git checkout d8803b29961c44d77a747636ad4282bd7a9094af

# 2. Apply the local build-glue patches described above
#    (module.py rewrite + ftka_topk_only.cu).

# 3. Install in editable mode without forcing dep rollback.
pip install --no-deps --no-build-isolation --break-system-packages -e .

# 4. Verify import.
python3 -c "import inspect; from ftka.cuda_ops import raft_topk; print(inspect.signature(raft_topk))"
#  -> (input, input_idx, output, output_idx, buf, k)

# 5. Run microbench (quick first ~30s, then full ~5min on H200).
cd /workspace/sglang
PYTHONPATH=python python3 benchmark/double_sparsity/repro_session/microbench_ftka_backends.py --quick
PYTHONPATH=python python3 benchmark/double_sparsity/repro_session/microbench_ftka_backends.py
```

## Production default — unchanged

`--double-sparsity-selector-backend=torch` remains the default. None of
the FTKA components pass the PLAN gates at the DS production envelope.

## Twilight-side ideas: explicit reject for this pass

For traceability, the algorithmic items in `tsinghua-ideal/Twilight`
that **change token-selection semantics** are not part of this
evaluation:

- Top-p pruner after the top-k step (`twilight/pyimpl/top_p.py`).
- Adaptive token budget / `HistoryBudgetInfo` accumulator.
- INT8 weight-estimator over K_label.
- The hierarchical elementwise-threshold variant.

These can be added later as offline oracle/control experiments labeled
`twilight_algorithmic_control`, never as "same algorithm" DS results.

## Invalidation conditions

This rejection re-opens for evaluation if any of the following change:

- FTKA upstream exposes a tunable `HEAD_DIM` for `batched_sparse_gemv`
  (P4 re-evaluable).
- DS adopts a paged K-label cache layout (P4 re-evaluable).
- FTKA upstream replaces the one-block radix in `decode_select_k` with
  a multi-block / many-SM variant that scales better with seq_len
  (P3 re-evaluable at long contexts).
- DS's score kernel becomes faster, shifting the balance against the
  current torch.topk timings (P3 might become competitive at lower
  contexts).
