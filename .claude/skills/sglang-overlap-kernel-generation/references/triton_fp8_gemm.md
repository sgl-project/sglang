# Block-wise FP8 GEMM Reference

> **On-demand reference.** Load this file when the user's compute kernel is a **GEMM / matmul** that operates on **FP8 quantized** inputs (e.g. batched GEMM, linear layers with FP8 weight, AllGather + FP8 GEMM overlap). If the compute kernel uses bf16/fp16 natively (not block-wise FP8), this reference is not needed.

> **Self-contained: do NOT import from `sglang`.** The generated kernel file must include the matmul kernel inline. Only standard imports (`torch`, `torch.distributed`, `triton`, `triton.language`) are allowed.

This reference documents the block-wise FP8 (W8A8) GEMM kernel used in production inference frameworks (SGLang, vLLM) and adapted for overlap kernels in triton_dist. It covers:

- §A — Block-wise FP8 matmul kernel (`_w8a8_block_fp8_matmul`)
- §B — Host-side launch helpers and configuration
- §C — Integration into overlap kernels: activation quantization strategies (separate kernel vs. on-the-fly), code patterns, and stride conventions

---

## (A) Block-wise FP8 Matmul Kernel

### A.1 Core algorithm

The kernel computes `C = (A_fp8 * A_scale) @ (B_fp8 * B_scale)` where scales are applied per-block after `tl.dot`:

```
for each K-tile:
    a_tile = load A_fp8[M_tile, K_tile]       # fp8
    b_tile = load B_fp8[K_tile, N_tile]       # fp8
    a_s    = load A_scale[M_tile, K_tile // group_k]   # float32
    b_s    = load B_scale[N_tile // group_n, K_tile // group_k]  # float32
    acc   += tl.dot(a_tile, b_tile) * a_s * b_s
```

Key insight: `tl.dot(fp8, fp8)` produces an `int32` result on SM90+ hardware (HMMAInstructions), which is implicitly cast to `float32` in Triton. The scale multiplication happens in `float32` after the dot product.

### A.2 Tensor layout

| Tensor | Dtype | Shape | Scale shape | Scale granularity |
|--------|-------|-------|-------------|-------------------|
| A (activation) | `float8_e4m3fn` | `[M, K]` | `[M, K // group_k]` | Per-token, per-`group_k` columns |
| B (weight) | `float8_e4m3fn` | `[N, K]` (column-major) | `[N // group_n, K // group_k]` | Per-`group_n` rows, per-`group_k` columns |

Typical block sizes: `group_n = 128`, `group_k = 128`.

### A.3 Full kernel

```python
@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers
    A, B, C, As, Bs,
    # Dimensions
    M, N, K,
    M_local,         # rows per rank (for compact AG layout row addressing)
    M_local_tiles,   # cdiv(M_local, BLOCK_SIZE_M)
    # Block quantization sizes
    group_n, group_k,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_As_m, stride_As_k,
    stride_Bs_k, stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_k_masking: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    # Swizzled PID for L2 locality
    pid = tl.program_id(axis=0)
    num_pid_m = M_local_tiles * world_size
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Rank-aware tile rotation (match AG signal arrival order)
    logical_src_rank = min(pid_m // M_local_tiles, world_size - 1)
    tile_in_rank = pid_m - logical_src_rank * M_local_tiles
    src_rank = (rank + logical_src_rank) % world_size

    # Row offsets — compact layout: src_rank * M_local + tile_in_rank * BLOCK_SIZE_M
    # Use % M to handle M not divisible by BLOCK_SIZE_M: tail tile rows wrap
    # around to valid indices; store mask prevents overwriting wrong rows.
    tile_row_start = src_rank * M_local + tile_in_rank * BLOCK_SIZE_M
    offs_am = (tile_row_start + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Scale pointer offsets (use % M for tail tiles)
    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    # Main loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_k_masking:
            k_mask = offs_k < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    # Epilogue: cast and store
    c = accumulator.to(tl.bfloat16)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_am[:, None] + stride_cn * offs_cn[None, :]
    # offs_am used % M wrapping, so re-compute raw row indices for the store mask
    offs_cm = tile_row_start + tl.arange(0, BLOCK_SIZE_M)
    rank_row_end = (src_rank + 1) * M_local
    c_mask = (offs_cm[:, None] < M) & (offs_cm[:, None] < rank_row_end) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

### A.4 Critical details

1. **Scale stepping**: The scale tensors have lower resolution than the data (one scale per `group_k` elements along K). The pointer only advances when we cross a scale-group boundary: `scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)`. This means `BLOCK_SIZE_K` must divide `group_k` (typically `BLOCK_SIZE_K = group_k = 128`).

2. **`BLOCK_SIZE_N` alignment with `group_n`**: `offs_bsn = offs_bn // group_n` computes which B-scale row to load. If `BLOCK_SIZE_N < group_n`, threads within a CTA load duplicate scale values — this is harmless but wastes bandwidth. Recommended: `BLOCK_SIZE_N = group_n` or `BLOCK_SIZE_N` is a multiple of `group_n`.

3. **B is column-major (transposed)**: The weight matrix B has shape `[N, K]` with `stride_bk` (row stride in the `[N, K]` view) being the fast dimension. This matches FP8 weight-quantized models in vLLM/SGLang where weights are stored transposed for column-major access.

4. **Swizzled PID**: The `GROUP_SIZE_M` swizzling pattern reorders CTAs for better L2 cache locality. `GROUP_SIZE_M = 32` is a good default.

5. **`needs_k_masking`**: A `tl.constexpr` flag for K-tail compilation fast/slow-path. `needs_k_masking = K % BLOCK_SIZE_K != 0`. For M-dimension tail tiles (when `M_local % BLOCK_SIZE_M != 0`), the kernel uses `% M` wrapping on row offsets instead of a separate compilation flag — the raw `tile_row_start` is stored separately for the store mask, while `offs_am = (tile_row_start + tl.arange(...)) % M` ensures A loads never go out of bounds. The store mask `(offs_cm[:, None] < M) & (offs_cm[:, None] < rank_row_end)` protects tail tiles from writing outside the current rank's shard. This follows the SGLang `consumer_bf16_a_block_fp8_matmul` pattern and avoids an extra JIT compilation path.

6. **Avoid `tl.constexpr` for frequently-changing values.** Triton JIT compiles a separate kernel binary for each unique combination of `tl.constexpr` arguments. Quantities derived from dynamic problem size (`M`, `M_per_rank`, `n_chunks`, etc.) should be regular (non-constexpr) parameters, or computed inside the kernel from other arguments:

   ```python
   # BAD — recompiles every time ntokens changes:
   #   M_per_rank: tl.constexpr,

   # GOOD (option A) — pass as regular (non-constexpr) parameter:
   #   M_per_rank,   # no tl.constexpr annotation

   # GOOD (option B) — compute inside kernel from runtime M:
   M_per_rank = M // world_size
   ```

---

## (B) Host-side Launch Helpers

### B.1 Launching the kernel

```python
def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,       # fp8 [M_local * world_size, K] (gathered A, compact layout)
    B: torch.Tensor,       # fp8 [N, K], column-major
    As: torch.Tensor,      # float32 scale for A
    Bs: torch.Tensor,      # float32 scale for B [N // group_n, K // group_k]
    block_size: list,       # [BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K]
    M_local: int,           # rows per rank
    rank: int,
    world_size: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert len(block_size) == 3
    block_m, block_n, block_k = block_size

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    C = A.new_empty(*A.shape[:-1], N, dtype=output_dtype)

    needs_k_masking = bool(K % block_k != 0)
    M_local_tiles = triton.cdiv(M_local, block_m)
    num_pid_m = M_local_tiles * world_size
    num_pid_n = triton.cdiv(N, block_n)
    num_tiles = num_pid_m * num_pid_n

    _w8a8_block_fp8_matmul[(num_tiles,)](
        A, B, C, As, Bs,
        M, N, K,
        M_local,
        M_local_tiles,
        block_n, block_k,
        A.stride(-2), A.stride(-1),
        B.stride(1), B.stride(0),
        C.stride(-2), C.stride(-1),
        As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=32,
        needs_k_masking=needs_k_masking,
        rank=rank,
        world_size=world_size,
        num_warps=4,
        num_stages=3,
    )
    return C
```

### B.2 Default tuning configuration

| Parameter | Default | Reason |
|-----------|---------|--------|
| `BLOCK_SIZE_M` | 64 | Balance SM occupancy vs. register pressure |
| `BLOCK_SIZE_N` | 128 | Match `group_n` for clean B-scale indexing |
| `BLOCK_SIZE_K` | 128 | Match `group_k` for clean A-scale stepping |
| `GROUP_SIZE_M` | 32 | L2 swizzle group; 32 is a good default |
| `num_warps` | 4 | Standard for **standalone** matmul kernels (compute-bound). When the FP8 GEMM is embedded **inside an overlap kernel** (intra-sm, without-sm), the outer kernel is memory-bound and should use `num_warps=32` per the SKILL.md Performance Checks. The two settings apply at different levels: the matmul inner loop benefits from 4 warps (more SM occupancy for tensor cores), while the overlap wrapper needs 32 warps (hide memory latency for signal polling / barriers). |
| `num_stages` | 3 | Standard for standalone matmul (software pipelining). When embedded in an overlap kernel, `num_stages=1` is used for the outer wrapper (simple load/store patterns, no pipelining). |

When `BLOCK_SIZE_K == group_k`, the scale stepping logic simplifies: scales advance every K-tile, and `n_tiles_k_per_group_k == 1`.

---

## (C) Integration into Overlap Kernels

### C.1 Activation quantization strategies

B (weight) and Bs (weight scale) are always quantized offline before the overlap begins — they are static and do not change across iterations.

A (activation) may arrive as either **fp8** (pre-quantized) or **bf16** (raw from communication). There are two strategies:

| Strategy | When to use | How it works | Pros | Cons |
|----------|-------------|--------------|------|------|
| **Separate quantize kernel** | A is pre-quantized, or you need exact per-`group_k` granularity | Run a `per_token_group_quant_fp8` kernel to produce `A_fp8` and `A_scale`, then feed them into `_w8a8_block_fp8_matmul` | Exact per-`group_k` quantization granularity; clean separation of concerns | Extra kernel launch + one global memory round-trip for `A_fp8` and `A_scale` |
| **On-the-fly quantization inside GEMM** | A arrives as bf16 from symmetric memory and you want to save the extra kernel launch | Inside the GEMM K-loop, load bf16 A tile → compute per-row `absmax` → `scale = absmax / fp8_max` → `A_fp8 = clamp(A_bf16 / scale, -fp8_max, fp8_max)` → `tl.dot(A_fp8, B_fp8) * scale * B_scale` | Saves one kernel launch and one global memory round-trip (no separate `A_fp8` / `A_scale` buffers) | Quantization granularity is per-`BLOCK_SIZE_K`-tile per M-tile-row; matches per-`group_k` only when `BLOCK_SIZE_K == group_k` |

When `BLOCK_SIZE_K == group_k == 128` (the standard setting), both strategies produce equivalent numerical results.

### C.2 On-the-fly quantization code pattern

To add on-the-fly A quantization inside the `_w8a8_block_fp8_matmul` K-loop, replace the A loading and scaling logic:

**Before** (pre-quantized A_fp8 + A_scale):
```python
# Load pre-quantized fp8 A and its scale
a = tl.load(a_ptrs, ...)           # fp8
a_s = tl.load(As_ptrs)             # float32 scale
accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
```

**After** (on-the-fly bf16→fp8 quantization inside GEMM):
```python
# Load bf16 A tile and quantize on the fly
fp8_max = 448.0  # float(torch.finfo(torch.float8_e4m3fn).max)
a_bf16 = tl.load(a_ptrs, ...)                           # bf16
a_absmax = tl.max(tl.abs(a_bf16), axis=1)              # [BLOCK_SIZE_M]
a_s = tl.maximum(a_absmax, 1e-12) / fp8_max             # [BLOCK_SIZE_M], per-row scale
a_rcp_s = fp8_max / tl.maximum(a_absmax, 1e-12)         # [BLOCK_SIZE_M], reciprocal for cast
a_fp8 = (a_bf16 * a_rcp_s[:, None]).to(B.dtype.element_ty)  # FMUL + saturating cast to fp8
accumulator += tl.dot(a_fp8, b) * a_s[:, None] * b_s[None, :]
```

Key differences when using on-the-fly quantization:
- The `As` pointer and stride parameters are removed from the kernel signature.
- A is loaded as bf16 instead of fp8, so `A.dtype` should be `torch.bfloat16`.
- The `a_s` scale is computed inline per K-tile instead of loaded from a separate tensor.
- The `fp8_max = 448.0` constant should be defined at the top of the kernel.

### C.3 Pattern: signal poll → GEMM

In the overlap scenario (e.g. AllGather → GEMM), the consumer GEMM kernel:

1. **Polls** a signal (e.g., `ld_sys(ag_signal_ptr + src_rank)`) to confirm that the corresponding A shard has arrived in symmetric memory.
2. **Loads** A from symmetric memory (either as pre-quantized fp8 or as bf16 for on-the-fly quantization).
3. **Performs** `tl.dot(A_fp8, B_fp8) * A_scale * B_scale`.
4. **Stores** the result to the output tensor.

### C.4 Stride conventions for overlap kernels

When A comes from symmetric memory, be especially careful with strides:

- `symm_ag_a` shape: `[M * world_size, K]`, contiguous row-major.
- Strides: `stride_am = K`, `stride_ak = 1`.
- B is pre-quantized fp8 with shape `[N, K]`, column-major (transposed): `B.stride(0) = K, B.stride(1) = 1`. In the kernel, `stride_bk` is the stride along K (fast dim = 1) and `stride_bn` is the stride along N (slow dim = K). So `stride_bk = B.stride(1) = 1, stride_bn = B.stride(0) = K`.
- Bs for B-scale is `[N // group_n, K // group_k]` with `Bs.stride(0) = K // group_k, Bs.stride(1) = 1`. So `stride_Bs_k = Bs.stride(1) = 1, stride_Bs_n = Bs.stride(0) = K // group_k`.

**CRITICAL: stride parameter ordering pitfall.** The kernel signature declares `stride_Bs_k, stride_Bs_n` (K-dim first, N-dim second), but `Bs` tensor shape is `[N // group_n, K // group_k]` (N-dim first, K-dim second). You must pass `Bs.stride(1), Bs.stride(0)` — NOT `Bs.stride(0), Bs.stride(1)`. Getting this wrong silently produces garbage results with large numerical errors.

### C.5 Signal tile_id mapping with swizzled PID

When the GEMM kernel uses `GROUP_SIZE_M` swizzling (§A.4 point 4), the `pid` (program_id) does NOT correspond to a linear `(tile_m, tile_n)` mapping. If a communication kernel (e.g., AllGather comm kernel) polls signals by linear tile_id (`tile_m * num_tiles_n + tile_n`), the GEMM kernel must signal using the **linear tile_id**, not the raw `pid`:

```python
# WRONG: signal with swizzled pid
_send_signal(signal_ptr + pid, "release")

# CORRECT: signal with linear tile_id derived from swizzled (pid_m, pid_n)
linear_tile_id = pid_m * num_pid_n + pid_n
_send_signal(signal_ptr + linear_tile_id, "release")
```

If the GEMM signals `signal_ptr + pid` but the comm kernel waits on `signal_ptr + tile_id` (linear), the comm kernel will read stale/wrong tile data, causing either a hang (waiting for a signal that never fires) or silent data corruption.

---

## (D) When to Use On-the-fly Quantization vs. Pre-quantized FP8

### D.1 Decision rule

| Overlap pattern | A input to GEMM | Quantization strategy |
|-----------------|-----------------|----------------------|
| **GEMM → AllGather** | A is pre-quantized fp8 (quantized before GEMM) | Use `_w8a8_block_fp8_matmul` directly with fp8 A + A_scale. **No on-the-fly quant.** |
| **AllGather → GEMM** | A arrives as bf16 from AllGather (symmetric memory) | Use on-the-fly quantization (§C.2) inside the GEMM kernel. A is loaded as bf16 and quantized per K-tile. |

**Key principle:** On-the-fly quantization is ONLY needed when the GEMM kernel's A input is bf16 (because it came from a communication collective that operates on bf16). If A is already fp8 (pre-quantized before the overlap begins), the GEMM kernel should directly load fp8 and use the pre-computed scale — do NOT re-quantize.

### D.2 Constructing FP8 test data

Use `deep_gemm.utils` to produce correctly-shaped FP8 tensors and scales for testing:

```python
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

# Activation: per-token-group quantization (group_k=128 fixed)
# Input: a_bf16 [M, K], dtype=torch.bfloat16
# Output: a_fp8 [M, K], dtype=torch.float8_e4m3fn
#         a_scale [M, K // 128], dtype=torch.float32
a_fp8, a_scale = per_token_cast_to_fp8(a_bf16, use_ue8m0=False)

# Weight: per-block quantization (group_n=128, group_k=128)
# Input: weight_bf16 [N, K], dtype=torch.bfloat16
# Output: b_fp8 [N, K], dtype=torch.float8_e4m3fn
#         b_scale [N // 128, K // 128], dtype=torch.float32
b_fp8, b_scale = per_block_cast_to_fp8(weight_bf16, use_ue8m0=False)
```

**Important notes:**
- `per_token_cast_to_fp8` always uses group_k=128 internally (hardcoded). The returned `a_scale` shape is `[M, K // 128]`.
- `per_block_cast_to_fp8` uses 128×128 blocks. The returned `b_scale` shape is `[N // 128, K // 128]`.
- Both functions require `use_ue8m0=False` parameter (mandatory positional arg in current version).
- `per_block_cast_to_fp8` does NOT take `group_n`/`group_k` parameters — it always uses 128×128 blocks.