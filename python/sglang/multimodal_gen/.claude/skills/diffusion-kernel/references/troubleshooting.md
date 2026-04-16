# Troubleshooting Guide — SGLang Diffusion JIT CUDA Kernels

Common issues and solutions when writing and integrating JIT CUDA kernels for SGLang Diffusion.

> **Adapted from**: [HuggingFace kernels cuda-kernels skill](https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels)

---

## Build / Compile Issues

### 1. JIT compilation fails: "No such file or directory"

**Problem:** `load_jit` cannot find your `.cuh` file.

```
FileNotFoundError: .../jit_kernel/csrc/diffusion/your_op.cuh not found
```

**Fix:** Ensure the file is under `python/sglang/jit_kernel/csrc/diffusion/`. The path passed to `cuda_files` is relative to `csrc/`:

```python
# CORRECT — file lives at csrc/diffusion/your_op.cuh
load_jit(..., cuda_files=["diffusion/your_op.cuh"])
# resolves to: python/sglang/jit_kernel/csrc/diffusion/your_op.cuh

# ALSO CORRECT — absolute path (pathlib replaces the csrc/ prefix)
load_jit(..., cuda_files=["/full/absolute/path/to/your_op.cuh"])
```

### 2. Type conversion errors (FP16/BF16)

**Problem:** Implicit FP16/BF16 conversion fails because PyTorch compiles with `-D__CUDA_NO_HALF_OPERATORS__`:

```
error: no suitable conversion function from "__half" to "float" exists
```

**Fix:** SGLang's `static_cast<float>` works because `fp16_t` and `bf16_t` are typedef'd with proper conversion operators. Always use explicit casts:

```cpp
// CORRECT — explicit cast
float val = static_cast<float>(v[i]);   // fp16_t / bf16_t → float
v[i] = static_cast<T>(fp32_result);    // float → T

// WRONG — implicit conversion (disabled by PyTorch build flags)
float val = v[i];           // compile error
v[i] = fp32_result;         // compile error
```

If you need the raw intrinsics for packed types:
```cpp
// bf16x2_t → two floats
bf16x2_t packed = ...;
float v0 = __bfloat162float(packed.x);
float v1 = __bfloat162float(packed.y);
```

### 3. Template instantiation explodes / slow first compile

**Problem:** Many template combinations makes the first JIT compile very slow.

**Fix:** Reduce template argument combinations. Move compile-time constants to runtime if they don't affect performance critically:

```cpp
// Fewer template args = fewer instantiations
template <typename T>  // only dtype varies
void my_op(tvm::ffi::TensorView dst, tvm::ffi::TensorView src, int block_size);
```

### 4. SM check: kernel requires sm_90 but device is sm_80

**Problem:** Kernel uses H100-only features on A100.

**Fix:** Add a Python guard before calling `load_jit`:

```python
cap = torch.cuda.get_device_capability()
if cap[0] < 9:
    raise RuntimeError(
        f"This kernel requires H100 (sm_90+). "
        f"Got compute capability {cap[0]}.{cap[1]}. "
        f"Use the Triton fallback instead: diffusion_triton_<op>()"
    )
```

---

## Performance Issues

### 5. Kernel is slower than Triton / PyTorch baseline

**Steps to diagnose:**

1. Check dtype: are you using `bf16_t` on T4? (T4 has no BF16 — silently falls back to slow emulation)
2. Check vectorization: is `hidden_size` divisible by `kVecN = 16/sizeof(T)` (8 for bf16, 4 for fp32)?
3. Profile with `ncu`:
   ```bash
   ncu --set full --csv -o metrics.csv \
     python -c "from sglang.jit_kernel.diffusion.rmsnorm import diffusion_rmsnorm; ..."
   ```
   Look at `dram__throughput.avg.pct_of_peak_sustained_elapsed` — if < 30%, check coalescing.

4. Check occupancy: run with `--ptxas-options=-v` in `extra_cuda_cflags` to see register usage.

### 6. Shared memory bank conflicts

**Problem:** `ncu` reports high `l1tex__data_bank_conflicts_pipe_lmem_op_st.sum`.

**Fix:** Add padding to shared memory arrays:

```cpp
// Conflict (all threads hit same bank when stride=32)
__shared__ float data[32][32];

// Fixed with padding
__shared__ float data[32][33];  // 33 instead of 32
```

### 7. Low occupancy from too many registers

**Problem:** `nvcc --ptxas-options=-v` shows high register count; occupancy < 25%.

**Fix:** Add `--maxrregcount=N` to limit registers:

```python
extra_cuda_cflags=["-O3", "--use_fast_math", "--maxrregcount=64"]
```

Reduces registers per thread at the cost of possible register spilling to local memory.

---

## Integration Issues

### 8. RMSNorm weight is None (`elementwise_affine=False`)

**Problem:**
```
AttributeError: 'NoneType' object has no attribute 'data_ptr'
```

**Root Cause:** DiT transformer blocks often use `RMSNorm(dim, elementwise_affine=False)` — no learnable weight.

**Fix in Python wrapper:** pass an empty tensor when weight is absent; the kernel launcher checks `data_ptr == nullptr`:

```python
w = weight if weight is not None else torch.empty(0, dtype=src.dtype, device=src.device)
module.rmsnorm(out, src, w, eps)
```

**Fix in `.cuh` launcher:**

```cpp
const T* w_ptr = (weight.data_ptr() != nullptr)
    ? static_cast<const T*>(weight.data_ptr()) : nullptr;
// ... pass w_ptr to kernel ...
```

**Fix in module patching:**

```python
has_weight = hasattr(module, "weight") and module.weight is not None
if has_weight:
    def _fwd(mod, eps):
        def forward(x): return diffusion_rmsnorm(x, weight=mod.weight, eps=eps)
        return forward
    module.forward = _fwd(module, module.eps)
else:
    def _fwd_noweight(eps):
        def forward(x): return diffusion_rmsnorm(x, weight=None, eps=eps)
        return forward
    module.forward = _fwd_noweight(module.eps)
```

### 9. `isinstance(module, torch.nn.RMSNorm)` misses diffusion variants

**Problem:** Patching doesn't apply because diffusers / sglang diffusion models define their own `RMSNorm` class that is **not** a subclass of `torch.nn.RMSNorm`.

**Fix:** Match by class name string:

```python
# WRONG — misses diffusers/sglang RMSNorm
if isinstance(module, torch.nn.RMSNorm):

# CORRECT — catches all variants
if type(module).__name__ == "RMSNorm":
# or for broader matching:
if "RMSNorm" in type(module).__name__:
```

### 10. Kernel patching doesn't persist after CPU offloading

**Problem:** After calling `pipe.enable_model_cpu_offload()`, patched modules revert.

**Fix:** Always inject **after** moving to CUDA, **before** enabling any offloading:

```python
pipe = load_pipeline(...)
pipe.to("cuda")                  # 1. Move to CUDA
inject_optimized_kernels(pipe)   # 2. Patch modules
pipe.enable_model_cpu_offload()  # 3. Now safe to enable offloading
```

### 11. Kernel patched after `torch.compile`

**Problem:** Module is already compiled; patching its `forward` after compilation has no effect.

**Fix:** Apply patches **before** any `torch.compile` call:

```python
inject_optimized_kernels(pipe)          # FIRST: patch
pipe.transformer = torch.compile(...)   # SECOND: compile
```

---

## `torch.compile` Compatibility

### 12. Custom CUDA kernel causes graph break

**Problem:**
```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
```
or:
```
torch._dynamo.exc.TorchRuntimeError: Cannot access data pointer of Tensor (FakeTensor)
```

**Root Cause:** `torch.compile` traces with "fake tensors" that have no real data. Any kernel that calls `.data_ptr()` during tracing fails.

**Options:**

**Option A (simplest):** Don't use `torch.compile` with CUDA JIT kernels — use Triton instead:
```python
# Triton kernels are torch.compile compatible
from sglang.jit_kernel.diffusion.triton.norm import fused_rmsnorm
```

**Option B:** Register as a `@torch.library.custom_op` (advanced):
```python
import torch

@torch.library.custom_op("diffusion_jit::rmsnorm", mutates_args={"out"})
def _rmsnorm_op(out: torch.Tensor, src: torch.Tensor,
                weight: torch.Tensor, eps: float) -> None:
    module = _jit_rmsnorm_module(src.dtype)
    module.rmsnorm(out, src, weight, eps)

@_rmsnorm_op.register_fake
def _(out, src, weight, eps):
    pass  # no shape changes; output already allocated in 'out'
```

**Performance trade-off:**

| Approach | Speedup (denoise) | torch.compile | Notes |
|----------|-------------------|---------------|-------|
| CUDA JIT kernel | best | Yes (via `torch.library.custom_op`) | Performance-optimal regardless of whether `torch.compile` is enabled; use `custom_op` + `register_fake` for compile compatibility |
| Triton kernel | good | Yes | Use when you need faster iteration/portability, or when you do not have a well-tuned CUDA kernel yet |
| Triton + compile | good | Yes | Use for end-to-end `torch.compile` integration convenience; typically slower than a well-tuned CUDA kernel |

### 13. Unstable benchmark results from JIT timing

**Problem:** First few runs are slow due to JIT compilation; timing is noisy.

**Fix:** Use `triton.testing.do_bench` / `run_benchmark` which use CUDA-graph-based timing automatically. Always do a warmup run first:

```python
# Pre-compile by running once before timing
diffusion_rmsnorm(dummy_src, weight=dummy_w, eps=1e-6)
torch.cuda.synchronize()
# Now time
result = run_benchmark(lambda: diffusion_rmsnorm(src, weight=w, eps=1e-6))
```

---

## Debugging Checklist

```bash
# 1. Verify CUDA device and compute capability
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"

# 2. Force synchronous CUDA execution to get real error location
CUDA_LAUNCH_BLOCKING=1 python scripts/bench_diffusion_rmsnorm.py

# 3. Run memory sanitizer to catch illegal accesses
compute-sanitizer --tool memcheck python scripts/bench_diffusion_rmsnorm.py

# 4. Check register and shared memory usage
# Add to extra_cuda_cflags: "--ptxas-options=-v"

# 5a. Kernel-level profiling — full metrics
ncu --set full -o metrics.ncu-rep \
  python scripts/bench_diffusion_rmsnorm.py

# 5b. Kernel-level profiling — targeted bandwidth + occupancy check
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python scripts/bench_diffusion_rmsnorm.py

# Key metrics to interpret:
# - sm__throughput  : compute utilization % of peak
# - dram__throughput: memory bandwidth % of peak (target ≥ 30% on H100/A100)
# - smsp__warp_issue_stalled_*: warp stall breakdown (memory_dependency / math_pipe)

# 6. System-level profiling (per-op breakdown inside sglang generate)
nsys profile -o denoise_profile \
  sglang generate --model-path=black-forest-labs/FLUX.1-dev \
    --width=1024 --height=1024 --num-inference-steps=50 \
    --seed=42 --enable-torch-compile --warmup

# 7. Verify a patched module produces correct output
python - << 'EOF'
import torch
from sglang.jit_kernel.diffusion.rmsnorm import diffusion_rmsnorm

x = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
w = torch.ones(2048, dtype=torch.bfloat16, device="cuda")

out_jit = diffusion_rmsnorm(x, weight=w, eps=1e-6)
out_ref = torch.nn.functional.rms_norm(x.float(), (2048,), w.float(), eps=1e-6).to(torch.bfloat16)

max_diff = (out_jit - out_ref).abs().max().item()
print(f"Max diff: {max_diff:.2e} ({'PASS' if max_diff < 0.02 else 'FAIL'})")
EOF
```
