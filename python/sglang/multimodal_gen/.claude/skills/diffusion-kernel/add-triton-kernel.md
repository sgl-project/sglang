---
name: add-triton-kernel
description: Step-by-step guide for adding a new Triton kernel to SGLang Diffusion's jit_kernel module. Use when implementing fused elementwise ops, norm variants, RoPE variants, or any other lightweight GPU kernel for diffusion models using Triton JIT. Covers kernel authoring, autotune, torch.compile compatibility, layer integration, and tests.
---

# Adding a Triton Kernel to SGLang Diffusion

This guide walks through adding a Triton kernel to `python/sglang/jit_kernel/diffusion/triton/`.
We use a fused elementwise operation as the running example: `y = x * (1 + scale) + shift` (AdaLN modulation).

---

## Directory Layout

```
python/sglang/jit_kernel/diffusion/
├── triton/
│   ├── scale_shift.py          # AdaLN scale/shift fused kernels
│   ├── norm.py                 # LayerNorm / RMSNorm fused kernels
│   ├── rmsnorm_onepass.py      # One-pass RMSNorm for small hidden size
│   └── rotary.py               # RoPE kernel
└── cutedsl/
    └── ...                     # CuTe DSL kernels (see use-efficient-diffusion-kernels.md)
```

New Triton kernels go into `triton/<op_name>.py`.

---

## Step 1: Write the Triton Kernel

Create `python/sglang/jit_kernel/diffusion/triton/<op_name>.py`.

### 1a. Imports

```python
import torch
import triton          # type: ignore
import triton.language as tl  # type: ignore
```

Always use `# type: ignore` on triton imports — the stubs are incomplete.

### 1b. The `@triton.jit` Kernel Function

Follow the naming convention `_<op_name>_kernel` (private, underscore prefix).

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 64},  num_warps=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4),
        triton.Config({"BLOCK_C": 256}, num_warps=4),
        triton.Config({"BLOCK_C": 512}, num_warps=8),
    ],
    key=["C"],   # re-tune when hidden dim changes
)
@triton.jit
def _fused_scale_shift_kernel(
    # Pointers — always pass raw tensors; Triton takes .data_ptr() internally
    x_ptr,
    scale_ptr,
    shift_ptr,
    y_ptr,
    # Dimensions
    B,        # batch size
    L,        # sequence length
    C,        # hidden / channel dim
    # Strides — pass every stride separately; do NOT assume contiguous
    stride_xb, stride_xl, stride_xc,
    stride_sb, stride_sc,
    stride_yb, stride_yl, stride_yc,
    # Compile-time constants (tl.constexpr)
    BLOCK_C: tl.constexpr,
):
    # Grid: (cdiv(L, 1), B) — one program per (batch, token)
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)

    c_offs = tl.arange(0, BLOCK_C)
    mask   = c_offs < C

    x_row = pid_b * stride_xb + pid_l * stride_xl
    y_row = pid_b * stride_yb + pid_l * stride_yl
    s_row = pid_b * stride_sb

    x     = tl.load(x_ptr     + x_row + c_offs * stride_xc, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + s_row + c_offs * stride_sc,  mask=mask, other=0.0)
    shift = tl.load(shift_ptr + s_row + c_offs * stride_sc,  mask=mask, other=0.0)

    y = x * (1.0 + scale) + shift
    tl.store(y_ptr + y_row + c_offs * stride_yc, y, mask=mask)
```

**Rules:**
- All pointer arguments are raw (Triton extracts `.data_ptr()` internally when called via `kernel[grid](...)`).
- Pass every stride as a separate scalar — never assume a tensor is contiguous inside the kernel.
- Use `tl.constexpr` for block sizes and boolean flags (`HAS_RESIDUAL`, `IS_RMS_NORM`, etc.).
- Use `mask=mask, other=0.0` on every `tl.load` to avoid out-of-bounds reads.
- Compute in `tl.float32` when precision matters (`x.to(tl.float32)`), then cast back to output dtype before `tl.store`.
- Use `tl.fma(a, b, c)` (`a*b + c`) for fused multiply-add — avoids rounding errors and maps to a single instruction.

### 1c. `@triton.autotune` Guidelines

| `key` entry | When to include |
|-------------|-----------------|
| `"C"` / `"hidden_dim"` | Always — block tile size depends on C |
| `"IS_RMS_NORM"` | When the kernel has a `constexpr` boolean flag that changes code paths |
| `"HAS_RESIDUAL"` | Same — constexpr path branching |
| Shape / batch / seq | Usually NOT — autotune cost outweighs benefit |

Keep configs in ascending `BLOCK_C` order with matching `num_warps` (warp × 32 threads ≤ 1024).

### 1d. `torch.compile` Compatibility

When the kernel is called inside a `torch.compile`-d region, wrap the launch with `torch.library.wrap_triton`:

```python
with torch.get_device_module().device(x.device):
    torch.library.wrap_triton(_fused_scale_shift_kernel)[grid](
        x, scale, shift, y,
        B, L, C,
        x.stride(0), x.stride(1), x.stride(2),
        scale.stride(0), scale.stride(1),
        y.stride(0), y.stride(1), y.stride(2),
    )
```

Use `wrap_triton` when the kernel is called from a layer that runs under `torch.compile`.
Skip it for utility kernels called only at Python graph boundaries.

---

## Step 2: Write the Python Launcher

The launcher is a regular Python function (public, no underscore) in the same file.

```python
def fused_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """
    Fused AdaLN modulation: y = x * (1 + scale) + shift.

    Args:
        x:     [B, L, C], CUDA, contiguous
        scale: [B, C],    CUDA
        shift: [B, C],    CUDA (same shape as scale)

    Returns:
        y: same shape and dtype as x
    """
    # --- Precondition checks ---
    assert x.is_cuda,           "x must be on CUDA"
    assert x.is_contiguous(),   "x must be contiguous"
    assert scale.is_cuda and shift.is_cuda
    assert x.ndim == 3,         f"x must be 3D [B, L, C], got {x.shape}"
    assert scale.shape == shift.shape
    B, L, C = x.shape

    # Allocate output
    y = torch.empty_like(x)

    # Grid: one program per token
    grid = (L, B)

    _fused_scale_shift_kernel[grid](
        x, scale, shift, y,
        B, L, C,
        x.stride(0),     x.stride(1),     x.stride(2),
        scale.stride(0), scale.stride(1),
        y.stride(0),     y.stride(1),     y.stride(2),
    )
    return y
```

**Rules:**
- Validate CUDA placement and shape/dtype **before** launching — use `assert` with a helpful message.
- Call `.contiguous()` on inputs that the kernel requires contiguous **before** the launch, not inside it.
- Allocate the output with `torch.empty_like(x)` — never reuse input buffers unless the op is explicitly in-place.
- The `grid` is a tuple or a lambda `(META)` when block sizes are auto-tuned:

```python
# Static grid (block size fixed)
grid = (triton.cdiv(L, BLOCK_L), triton.cdiv(C, BLOCK_C), B)

# Dynamic grid (block size comes from autotune)
grid = lambda META: (triton.cdiv(L, META["BLOCK_C"]), B)
```

### Handling Non-Contiguous Inputs

Never call `.contiguous()` silently — it copies data. Instead, pass strides to the kernel and let it handle arbitrary layouts. Only call `.contiguous()` when the kernel genuinely requires it (e.g., after a reshape):

```python
# OK: reshape + contiguous needed for 2D view trick
x_2d = x.view(B * L, C)             # view only works on contiguous
if not x.is_contiguous():
    x = x.contiguous()
    x_2d = x.view(B * L, C)
```

---

## Step 3: Integrate into the Layer

Call the new kernel from the appropriate layer file in
`python/sglang/multimodal_gen/runtime/layers/` (typically `layernorm.py` or `elementwise.py`).

```python
# In layernorm.py or elementwise.py
import torch

def apply_scale_shift(x, scale, shift):
    if x.is_cuda:
        from sglang.jit_kernel.diffusion.triton.my_op import fused_scale_shift
        return fused_scale_shift(x, scale, shift)
    # Pure-PyTorch fallback for non-CUDA execution
    return x * (1.0 + scale) + shift
```

**Rules:**
- Gate on `x.is_cuda` — the Triton kernel only runs on CUDA; the fallback handles everything else.
- The launcher raises `AssertionError` on invalid inputs (wrong shape, CPU tensor, etc.) — do **not** silently catch these. Let them propagate so bugs are visible during development.
- Add `logger.warning_once(...)` only when falling back due to a **known hardware limitation** (e.g., unsupported SM compute capability), not for wrong-input errors.

---

## Step 4: Write Tests

Create `python/sglang/jit_kernel/tests/test_<op_name>.py`.

```python
import pytest
import torch

from sglang.jit_kernel.diffusion.triton.my_op import fused_scale_shift


def _ref_fused_scale_shift(x, scale, shift):
    """PyTorch reference implementation."""
    # Broadcast scale/shift from [B, C] to [B, L, C]
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.mark.parametrize("B,L,C", [
    (1, 6,    3072),   # Qwen (small batch)
    (1, 1024, 1536),   # Wan
    (2, 512,  3072),   # typical training shape
    (1, 1,    256),    # edge: L=1
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_scale_shift_correctness(B, L, C, dtype):
    torch.manual_seed(0)
    x     = torch.randn(B, L, C, dtype=dtype, device="cuda")
    scale = torch.randn(B, C,    dtype=dtype, device="cuda") * 0.1
    shift = torch.randn(B, C,    dtype=dtype, device="cuda") * 0.1

    out = fused_scale_shift(x, scale, shift)
    ref = _ref_fused_scale_shift(x.float(), scale.float(), shift.float()).to(dtype)

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol,
                                msg=f"Mismatch at B={B} L={L} C={C} dtype={dtype}")


def test_fused_scale_shift_non_cuda_raises():
    x     = torch.randn(1, 4, 64)
    scale = torch.randn(1, 64)
    shift = torch.randn(1, 64)
    with pytest.raises(AssertionError, match="CUDA"):
        fused_scale_shift(x, scale, shift)


def test_fused_scale_shift_output_dtype_preserved():
    x     = torch.randn(1, 8, 128, dtype=torch.bfloat16, device="cuda")
    scale = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
    shift = torch.zeros(1, 128, dtype=torch.bfloat16, device="cuda")
    out   = fused_scale_shift(x, scale, shift)
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run:

```bash
pytest python/sglang/jit_kernel/tests/test_<op_name>.py -v
```

**Test coverage requirements:**
1. Reference comparison against pure-PyTorch for all supported dtypes (fp16, bf16, fp32).
2. Edge shapes: `L=1`, `C` not a multiple of the largest BLOCK_C, large `B`.
3. Error cases: CPU tensor, wrong shape.
4. Output dtype and shape preservation.

---

## Step 5: Add a Benchmark (required)

Create `python/sglang/jit_kernel/benchmark/bench_<op_name>.py`.

```python
import torch
import triton.testing

from sglang.jit_kernel.diffusion.triton.my_op import fused_scale_shift


SHAPES = [
    # (B, L, C)  — representative diffusion shapes
    (1, 6,    3072),   # Qwen image
    (1, 1024, 1536),   # Wan video
    (1, 4096, 3072),   # FLUX double-stream
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "L", "C"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton Fused", "PyTorch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="µs (median)",
        plot_name="fused-scale-shift",
        args={},
    )
)
def benchmark(B, L, C, provider):
    dtype = torch.bfloat16
    x     = torch.randn(B, L, C, dtype=dtype, device="cuda")
    scale = torch.randn(B, C,    dtype=dtype, device="cuda")
    shift = torch.randn(B, C,    dtype=dtype, device="cuda")

    if provider == "triton":
        fn = lambda: fused_scale_shift(x, scale, shift)
    else:
        fn = lambda: x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    ms, *_ = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5, 0.2, 0.8])
    return ms * 1000  # µs


if __name__ == "__main__":
    benchmark.run(print_data=True)
```

Run:

```bash
python python/sglang/jit_kernel/benchmark/bench_<op_name>.py
```

---

## Step 6: Profile with Nsight Compute (optional but recommended)

After the kernel passes correctness tests, use **`nsight-profiler.md`** to measure its hardware-level efficiency. This step requires **Nsight Compute (`ncu`)** to be installed.

### Dependency Check

Before profiling, verify `ncu` is available:

```bash
ncu --version   # must print a version string, e.g. "NVIDIA Nsight Compute 2024.1.0"
```

If `ncu` is missing, install it via the CUDA Toolkit package or the standalone [Nsight Compute installer](https://developer.nvidia.com/nsight-compute).

### Quick Kernel Profile

```bash
# Profile the Triton kernel during the benchmark script
# --kernel-name: match the Triton-mangled name (check with nsys first if unsure)
ncu --set full \
    -o /tmp/triton_<op_name> \
    python python/sglang/jit_kernel/benchmark/bench_<op_name>.py
```

To profile only the Triton kernel (skip PyTorch reference and warmup launches), add `--launch-skip N --launch-count M`:

```bash
# Skip first 2 launches (warmup), capture 3 kernel invocations
ncu --set full --launch-skip 2 --launch-count 3 \
    -o /tmp/triton_<op_name> \
    python python/sglang/jit_kernel/benchmark/bench_<op_name>.py
```

### Key Metrics to Check for a Triton Kernel

| Metric | Healthy Range | Action if Low |
|--------|--------------|---------------|
| **Achieved Occupancy** | ≥ 50% | Reduce register usage or shared memory; try smaller block sizes |
| **Memory Throughput** | ≥ 70% of peak BW | Check for non-coalesced access (pass contiguous strides) |
| **Compute Throughput** | ≥ 50% of peak | Increase arithmetic intensity; fuse more ops per load |
| **Warp Efficiency (No Stall)** | ≥ 60% | Reduce branch divergence; avoid `tl.atomic_*` on hot paths |
| **L1/L2 Hit Rate** | L2 ≥ 40% | Reorder loads for locality; check broadcast patterns |

### CLI-Only Analysis Workflow

```bash
# 1. Collect profile (--csv is NOT valid here; only -o to save .ncu-rep)
ncu --set full \
    --launch-skip 2 --launch-count 1 \
    -o /tmp/prof_<op_name> \
    python python/sglang/jit_kernel/benchmark/bench_<op_name>.py

# 2. Export key metrics to CSV from the saved .ncu-rep (--csv is valid here)
ncu --import /tmp/prof_<op_name>.ncu-rep \
    --page details --csv \
    > /tmp/prof_<op_name>_details.csv

# 3. Quick summary — top bottleneck sections
python3 - << 'EOF'
import csv, sys
rows = list(csv.DictReader(open("/tmp/prof_<op_name>_details.csv")))
# print section names and their achieved % of peak
for r in rows:
    name = r.get("Metric Name", "")
    val  = r.get("Metric Value", "")
    if any(k in name for k in ["sol", "Occupancy", "Throughput"]):
        print(f"{name:60s} {val}")
EOF
```

### Compare Two Kernel Versions

```bash
# Profile baseline
ncu --set full --launch-skip 2 --launch-count 1 \
    -o /tmp/baseline python .../bench_<op_name>.py

# Profile optimized version (after your changes)
ncu --set full --launch-skip 2 --launch-count 1 \
    -o /tmp/optimized python .../bench_<op_name>.py

# Diff (CSV, no GUI)
ncu --import /tmp/baseline.ncu-rep \
    --import /tmp/optimized.ncu-rep \
    --page diff --csv > /tmp/diff_<op_name>.csv

python3 -c "
import csv
rows = list(csv.DictReader(open('/tmp/diff_<op_name>.csv')))
for r in rows[:30]:
    print(r.get('Metric Name','')[:55], r.get('Baseline',''), '->', r.get('Comparison',''))
"
```

> For a complete guide to Nsight Compute metrics, occupancy analysis, roofline model interpretation, and warp efficiency optimization, refer to **`nsight-profiler.md`** in this directory.

---

## Common Patterns Reference

### Pattern 1: Autotune over a 2D tile (L × C)

Used in `scale_shift.py` (`fuse_scale_shift_kernel_blc_opt`):

```python
@triton.jit
def _kernel(..., BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)
    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (l_offs[:, None] < L) & (c_offs[None, :] < C)
    ...

# Launch:
grid = (triton.cdiv(L, BLOCK_L), triton.cdiv(C, BLOCK_C), B)
_kernel[grid](..., BLOCK_L=block_l, BLOCK_C=block_c, num_warps=4, num_stages=2)
```

### Pattern 2: One-pass RMSNorm for small hidden size

Used in `rmsnorm_onepass.py`:

```python
@triton.jit
def _rms_norm_tiled_onepass(y_ptr, x_ptr, w_ptr,
                              SEQ: tl.constexpr, DIM: tl.constexpr, EPS: tl.constexpr,
                              BLOCK_SIZE_SEQ: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr):
    seq_blk_id = tl.program_id(0)
    seq_id     = seq_blk_id * BLOCK_SIZE_SEQ
    seq_offset = seq_id + tl.arange(0, BLOCK_SIZE_SEQ)[:, None]
    d_offset   = tl.arange(0, BLOCK_SIZE_DIM)[None, :]
    ...
    x = tl.load(x_ptr + seq_offset * DIM + d_offset, mask=..., other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=1, keep_dims=True) / DIM
    rstd    = tl.math.rsqrt(mean_sq + EPS)
    tl.store(y_ptr + ..., x * rstd * w, mask=...)

# Launch with wrap_triton for torch.compile compat:
with torch.get_device_module().device(x.device):
    torch.library.wrap_triton(_rms_norm_tiled_onepass)[grid](
        y_view, x_view, w,
        S, D, eps,
        BLOCK_SIZE_DIM=triton.next_power_of_2(D),
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
    )
```

### Pattern 3: `tl.constexpr` boolean flags for conditional paths

Used in `norm.py` and `scale_shift.py`:

```python
@triton.jit
def _kernel(...,
            IS_RMS_NORM:   tl.constexpr,
            HAS_RESIDUAL:  tl.constexpr,
            SCALE_IS_SCALAR: tl.constexpr):
    ...
    if IS_RMS_NORM:
        var = tl.sum(x * x, axis=0) / N
    else:
        mean = tl.sum(x, axis=0) / N
        var  = tl.sum((x - mean) ** 2, axis=0) / N

    if HAS_RESIDUAL:
        x = x + tl.load(residual_ptr + ...)

    if SCALE_IS_SCALAR:
        scale_val = tl.load(scale_ptr)
        scale = tl.full([BLOCK_N], scale_val, dtype=scale_val.dtype)
    else:
        scale = tl.load(scale_ptr + col_offsets, mask=mask, other=0.0)
```

Autotune key must include these booleans so the compiler generates separate specializations.

### Pattern 4: Computing in fp32, storing in original dtype

Always up-cast to `tl.float32` for reductions and math, then down-cast before storing:

```python
x_f32    = x.to(tl.float32)
scale_f32 = scale.to(tl.float32)
y_f32    = x_f32 * (1.0 + scale_f32) + shift_f32
tl.store(y_ptr + offsets, y_f32.to(x.dtype), mask=mask)
```

---

## Checklist Before Submitting

### Prerequisites
- [ ] `ncu --version` prints a valid Nsight Compute version (required for Step 7 profiling)

### Implementation
- [ ] Kernel file at `python/sglang/jit_kernel/diffusion/triton/<op_name>.py`
- [ ] All pointer arguments passed with separate stride scalars
- [ ] Every `tl.load` uses `mask=` and `other=`
- [ ] Autotune `key` includes all `constexpr` flags that change code paths
- [ ] `torch.library.wrap_triton` used if kernel runs inside `torch.compile` region
- [ ] PyTorch fallback path in the layer integration (see Step 4)

### Validation
- [ ] Tests pass: `pytest python/sglang/jit_kernel/tests/test_<op_name>.py -v`
- [ ] Benchmark runs: `python python/sglang/jit_kernel/benchmark/bench_<op_name>.py`
- [ ] **Correctness verified**: Triton output matches PyTorch reference within tolerance
- [ ] Nsight Compute profile collected (`ncu --set full`); achieved occupancy ≥ 50% and memory throughput ≥ 70% of peak (or bottleneck documented)

---

## Summary of Files Created/Modified

```
python/sglang/jit_kernel/diffusion/triton/<op_name>.py      # NEW: Triton kernel + launcher
python/sglang/jit_kernel/tests/test_<op_name>.py            # NEW: correctness tests
python/sglang/jit_kernel/benchmark/bench_<op_name>.py       # NEW: performance benchmark
python/sglang/multimodal_gen/runtime/layers/layernorm.py    # MODIFIED: integrate into layer
  (or elementwise.py, depending on op type)
```

## References

- `python/sglang/jit_kernel/diffusion/triton/scale_shift.py` — 2D tile pattern, scalar broadcast, 4D shape handling
- `python/sglang/jit_kernel/diffusion/triton/rmsnorm_onepass.py` — `wrap_triton`, tiled one-pass reduction
- `python/sglang/jit_kernel/diffusion/triton/norm.py` — complex autotune with many `constexpr` flags
- `python/sglang/jit_kernel/diffusion/triton/rotary.py` — per-head grid, interleaved RoPE
- `nsight-profiler.md` — full Nsight Compute guide: occupancy analysis, roofline model, warp efficiency, kernel comparison
- `diffusion-benchmark-and-profile.md` — how to verify the kernel's impact on denoise latency
- `use-efficient-diffusion-kernels.md` — overview of existing fused kernel entry points
