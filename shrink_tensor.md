# Plan: Shrink Intermediate Tensors in `_run_masked_gemm`

## 1. Problem Statement

`_run_masked_gemm` allocates two full `(num_groups, m, n)` tensors regardless of how
many experts are actually active:

```python
# GEMM-0 output  (bfloat16)
gateup_output = torch.empty((num_groups, m, n), ...)   # n = 2 * ffn_dim

# GEMM-1 output  (bfloat16)
down_output   = torch.empty((num_groups, m, n), ...)   # n = hidden_dim  ← user focus
```

In decode mode the vast majority of expert slots are empty (`masked_m[g] == 0`).
For a representative Qwen-72B decode step (256 experts, m_max = 256, K = 7 168,
FFN-dim = 18 944, top_k = 6 active experts out of 256):

| Tensor | Shape | Size |
|--------|-------|------|
| `gateup_output` | (256, 256, 18 944) × bf16 | **~2.5 GB** |
| `down_input` (fp8) | (256, 256, 9 472) × fp8 | **~590 MB** |
| `down_output` | (256, 256, 7 168) × bf16 | **~940 MB** |

These three tensors are live simultaneously, peaking at **~4 GB for 6 active rows**.

---

## 2. Why the Current Layout Exists

`_run_masked_gemm` maps one expert per "group" in the `[G, m, K]` layout.
`post_permute_asym_gemm_to_standard` then calls `post_reorder_triton_kernel`, which reads
from `down_output[g, row, :]` (the masked per-group layout) and scatters into the final
flat `(num_tokens, K)` output using the `src2dst` permutation table.

The full `(G, m, K)` shape is only needed because `post_reorder` reads it that way.

---

## 3. Proposed Solution: Chunked Processing with Direct Scatter

Instead of keeping the full `(G, m, K)` output alive, process experts in groups of
`chunk_size` at a time, scatter each chunk's valid rows directly into the final
`(num_tokens, K)` output, and reuse the same small intermediate buffers on every
iteration.

### 3.1 Memory Layout After the Change

| Buffer | Old shape | New shape (reused) |
|--------|-----------|---------------------|
| `gateup_chunk` | `(G, m, 2N)` | `(chunk_size, m, 2N)` |
| `down_input_chunk` | `(G, m, N)` | `(chunk_size, m, N)` |
| `down_chunk` | `(G, m, K)` | `(chunk_size, m, K)` |
| `down_output` (final) | `(G, m, K)` → large | `(num_tokens, K)` → exact size |

`num_tokens` is the number of actual dispatched tokens (a small number in decode),
orders of magnitude smaller than `G * m`.

### 3.2 Choosing `chunk_size`

`chunk_size` should be a **compile-time constant** matching the `kNumGroups` template
parameter used when JIT-compiling the AsymGEMM kernel. This avoids re-compilation per
chunk (assuming `chunk_size` divides `num_groups`, or the last chunk is padded).

Recommended default: `chunk_size = 8` (balances reuse vs. kernel launch overhead).
Expose as an env-var `SGLANG_MASKED_GEMM_CHUNK_SIZE` for tuning.

### 3.3 Algorithm

```
Pre-allocate once (outside the chunk loop):
    gateup_chunk    = (chunk_size, m, 2N)  bf16   ← reused every iteration
    down_input_c    = (chunk_size, m, N)   fp8    ← reused every iteration
    down_chunk      = (chunk_size, m, K)   bf16   ← reused every iteration
    final_output    = (num_tokens, K)      bf16   ← the return value

For g_start in range(0, num_groups, chunk_size):
    g_end      = min(g_start + chunk_size, num_groups)
    c          = g_end - g_start

    masked_m_c = masked_m[g_start:g_end]       # (c,)
    if masked_m_c.sum() == 0: continue         # skip fully-inactive chunks

    hs_c       = hidden_states[g_start:g_end]  # (c, m, K)
    hs_scale_c = hidden_states_scale[g_start:g_end]

    # GEMM-0: hs_c × w13[g_start:g_end] → gateup_chunk[:c]
    grouped_gemm_nt_f8f8bf16_masked(
        (hs_c, hs_scale_c),
        (w13_weight[g_start:g_end], w13_scale[g_start:g_end]),
        gateup_chunk[:c],
        masked_m_c, expected_m,
    )

    # Act + quant: gateup_chunk[:c] → down_input_c[:c]
    act_and_quant(gateup_chunk[:c], down_input_c[:c], masked_m_c, ...)

    # GEMM-1: down_input_c[:c] × w2[g_start:g_end] → down_chunk[:c]
    grouped_gemm_nt_f8f8bf16_masked(
        (down_input_c[:c], down_input_scale[:c]),
        (w2_weight[g_start:g_end], w2_scale[g_start:g_end]),
        down_chunk[:c],
        masked_m_c, expected_m,
    )

    # Scatter valid rows → final_output (replaces post_reorder for this chunk)
    _scatter_chunk_to_output(
        down_chunk[:c],    # (c, m, K)
        final_output,      # (num_tokens, K)
        src2dst,           # permutation table
        masked_m_c,
        g_start,
        m,                 # per-group row stride
    )

return final_output
```

### 3.4 `_scatter_chunk_to_output` Kernel

A new lightweight Triton kernel (or reuse/extend `post_reorder_triton_kernel`):

```python
@triton.jit
def _scatter_chunk_kernel(
    down_chunk_ptr,    # (chunk_size, m, K) source
    output_ptr,        # (num_tokens, K) destination
    src2dst_ptr,       # int32[G * m]  — maps (g*m + row) → output row
    masked_m_ptr,      # int32[chunk_size]
    g_start,           # int  — expert offset for this chunk
    m,                 # int  — per-group row stride
    K,                 # int  — hidden dim
    BLOCK_K: tl.constexpr,
):
    gid    = tl.program_id(0)   # within-chunk expert index
    row    = tl.program_id(1)   # row within this expert's slice

    valid_rows = tl.load(masked_m_ptr + gid)
    if row >= valid_rows:
        return

    src_flat = (gid * m + row)                       # flat index in down_chunk
    dst_row  = tl.load(src2dst_ptr + (g_start + gid) * m + row)

    offs = tl.arange(0, BLOCK_K)
    src = tl.load(down_chunk_ptr + src_flat * K + offs, mask=offs < K)
    tl.store(output_ptr + dst_row * K + offs, src, mask=offs < K)
```

Launch grid: `(chunk_size, m)` — each thread block handles one row of one expert.

### 3.5 Integration with `running_state`

`_scatter_chunk_to_output` needs `src2dst`, which currently lives in `running_state`
and is used only in `post_permute_asym_gemm_to_standard`. Two options:

**Option A (preferred)**: Pass `src2dst` into `_run_masked_gemm` via `running_state`.
The runner already has access to `running_state`, so read `src2dst` from there and
pre-allocate `final_output = torch.zeros((num_tokens, K), ...)` using
`running_state["hidden_states_shape"]`.

**Option B**: Keep `down_output = (G, m, K)` and only shrink `gateup_output` by
chunking GEMM-0 + act. This halves peak memory with zero API changes but leaves
`down_output` large. A stepping stone, not the final solution.

### 3.6 Changes to `post_permute_asym_gemm_to_standard`

With Option A, the runner returns `final_output` of shape `(num_tokens, K)` directly.
`post_permute_asym_gemm_to_standard` currently calls `post_reorder` to convert from
`(G, m, K)` to `(num_tokens, K)`. This step is **no longer needed** because
`_run_masked_gemm` already produced `(num_tokens, K)`.

Add a flag (e.g., `runner_output.is_scattered = True`) so `post_permute` can detect
this and skip the `post_reorder` step.

---

## 4. Files to Change

| File | Change |
|------|--------|
| `asym_gemm.py::_run_masked_gemm` | Replace single large alloc with chunked loop + scatter |
| `asym_gemm.py` | Add `_scatter_chunk_kernel` and `_scatter_chunk_to_output` |
| `asym_gemm.py::post_permute_asym_gemm_to_standard` | Skip `post_reorder` when output is already scattered |
| `base.py::RunnerOutput` (or `AsymGemmRunnerOutput`) | Add `is_scattered: bool = False` |
| Env-var handling | `SGLANG_MASKED_GEMM_CHUNK_SIZE` with default 8 |

---

## 5. Constraints and Risks

### 5.1 `kNumGroups` JIT Template Parameter

The AsymGEMM kernel JIT-compiles with `kNumGroups` as a compile-time constant.
Processing chunks of size `chunk_size` means the kernel sees `num_groups = chunk_size`
instead of `num_groups = G`. This will trigger a new JIT compilation on first use —
acceptable since `chunk_size` is fixed.

**Action**: Ensure the weight slices `w13_weight[g_start:g_end]` are contiguous
(use `.contiguous()` if needed, or pre-layout weights with expert-first stride).

### 5.2 `down_gemm_overlap_args` / Pipeline Overlap

The current `_run_masked_gemm` supports a `down_gemm_overlap_args` path for overlapping
GEMM-1 with communication (DeepEP normal mode). Chunking serialises GEMM-1 over chunks,
potentially reducing overlap opportunities.

**Mitigation**: Keep the overlap path only for the single-chunk (non-chunked) case, or
limit chunking to cases where `down_gemm_overlap_args is None`.

### 5.3 CUDA Graph Safety

With the cuda-graph-rewrite already in place (`masked_m` passed directly, `gridDim.y =
num_groups`), processing a fixed-size chunk of `chunk_size` experts per call remains
graph-safe as long as:
- `chunk_size` is a compile-time constant
- `final_output` is pre-allocated at capture time (fixed shape `(num_tokens, K)`)
- The scatter kernel uses `masked_m[g_start:g_end]` (a fixed-offset view, graph-safe)

### 5.4 Correctness of Scatter

The existing `post_reorder` kernel applies `topk_weights` when scattering. The new
scatter kernel must also apply `topk_weights[g_start + gid, row]` to match semantics.

---

## 6. Expected Memory Reduction

Continuing the Qwen-72B decode example (6 active experts, chunk_size = 8):

| Tensor | Old peak | New peak |
|--------|----------|----------|
| `gateup_chunk` | 2 500 MB | **78 MB** (8 * 256 * 18944 * 2) |
| `down_input_chunk` | 590 MB | **18 MB** |
| `down_chunk` | 940 MB | **28 MB** |
| `final_output` | (part of post_permute) | **~0.1 MB** (6 tokens * 7168) |
| **Total intermediate** | **~4 GB** | **~125 MB** |

Reduction factor: **~32×** for peak intermediate memory during the MoE forward pass.

---

## 7. Step-by-Step Implementation Order

1. Add `_scatter_chunk_to_output` Triton kernel alongside existing kernels in
   `asym_gemm.py`.
2. Add `is_scattered: bool = False` to `AsymGemmRunnerOutput`.
3. Refactor `_run_masked_gemm` to the chunked loop. Gate with
   `SGLANG_MASKED_GEMM_CHUNK_SIZE` env-var (0 or unset = old behavior for easy
   rollback).
4. Update `post_permute_asym_gemm_to_standard` to skip `post_reorder` when
   `is_scattered`.
5. Test correctness: compare output of chunked vs. unchunked on a fixed random input
   for both decode (sparse `masked_m`) and prefill (dense `masked_m`).
6. Benchmark peak memory with `torch.cuda.max_memory_allocated()` before and after.
