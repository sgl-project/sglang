---
name: debug-cuda-crash
description: Call this skill when you need to debug CUDA crashes in SGLang using kernel API logging
---

# Tutorial: Debugging CUDA Crashes with Kernel API Logging

This tutorial shows you how to debug CUDA crashes and errors in SGLang using the `@debug_kernel_api` logging decorator.

## Goal

When your code crashes with CUDA errors such as illegal memory access, device-side assert, out-of-bounds, or NaN/Inf, use kernel API logging to:
- Capture input tensors BEFORE the crash occurs
- Understand what data caused the problem
- Track tensor shapes, dtypes, and values through the call boundary that triggered the crash
- Detect numerical issues such as NaN, Inf, or obviously wrong shapes

## Why Use Kernel API Logging?

**Problem**: CUDA errors often crash the program before normal debugging output is flushed.

**Solution**: SGLang's `@debug_kernel_api` decorator logs inputs before execution, so you can still see what caused the crash even after the program aborts.

## What Is Covered?

The current logging coverage focuses on the highest-value kernel boundaries in SGLang:
- Custom ops registered through `register_custom_op(...)`
- External custom ops registered through `register_custom_op_from_extern(...)`
- LLM attention, linear, quantization, and multi-platform wrapper entry points
- Diffusion attention impl, linear, rotary, and custom-op wrapper entry points
- Selected direct `torch.ops.sglang.*` hotspots and model-specific bypasses

This means the logging is useful for both LLM and diffusion kernel debugging, but it does not automatically cover every pure PyTorch call in the repository.

## Step 1: Enable Kernel API Logging

### Basic Logging (Function Names Only)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=1
export SGLANG_KERNEL_API_LOGDEST=stdout

python my_script.py
```

Output:
```
================================================================================
[2026-03-19 00:47:06] SGLang Kernel API Call: RMSNorm.forward
================================================================================
[2026-03-19 00:47:06] SGLang Kernel API Call: sglang.quant_method.UnquantizedLinearMethod.apply
================================================================================
[2026-03-19 00:47:06] SGLang Kernel API Call: sglang.custom_op.fused_inplace_qknorm
```

This is a real level-1 excerpt captured from `Qwen/Qwen3-0.6B`.

### Detailed Logging (Inputs with Metadata)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug.log

python my_script.py
```

Output in `debug.log`:
```
================================================================================
[2026-03-19 00:47:30] SGLang Kernel API Call: sglang.quant_method.UnquantizedLinearMethod.apply
Positional input arguments:
  arg[0]=QKVParallelLinear(
      repr=QKVParallelLinear(in_features=1024, output_features=4096, bias=False, tp_size=1, gather_output=False)
    )
  arg[1]=Tensor(
      shape=(1, 1024)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
    )
  arg[2]=None
Output:
  return=Tensor(
      shape=(1, 4096)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
    )
```

This is a real level-3 excerpt captured from `Qwen/Qwen3-0.6B`.

### Full Logging (With Tensor Statistics)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=5
export SGLANG_KERNEL_API_LOGDEST=debug.log

python my_script.py
```

Additional output:
```
================================================================================
[2026-03-19 01:00:42] SGLang Kernel API Call: diffusion.quant_method.UnquantizedLinearMethod.apply
Positional input arguments:
  arg[1]=Tensor(
      shape=(1, 77, 768)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
      min=-27.250000
      max=28.500000
      mean=0.011723
      nan_count=0
      inf_count=0
    )
Output:
  return=Tensor(
      shape=(1, 77, 2304)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
      min=-8.937500
      max=9.375000
      mean=0.009460
      nan_count=0
      inf_count=0
    )
```

This is a real level-5 excerpt captured from `black-forest-labs/FLUX.1-dev`.

### Crash-Safe Dumps (Inputs Saved Before Execution)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=10
export SGLANG_KERNEL_API_LOGDEST=debug.log
export SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_kernel_api_dumps

python my_script.py
```

At level 10, SGLang saves the inputs before execution. If the kernel crashes, the dump directory still contains the inputs and exception metadata.

If CUDA graph capture is active, tensor dumps are skipped automatically to avoid capture-time CUDA errors. In that case, you still get the kernel API call log, but not `inputs.pt` / `outputs.pt`.

Level-10 dumps are best understood as crash-safe call snapshots. They always preserve the observed call boundary. They do not guarantee one-click replay for every method, because some methods depend on module state that is not serialized into the dump.

Real level-10 dump layout from `Qwen/Qwen3-0.6B`:

```text
/tmp/sglang_kernel_api_validation/qwen_qwen3_0_6b_level10_dumps
/tmp/sglang_kernel_api_validation/qwen_qwen3_0_6b_level10_dumps/20260319_004821_182_pid919286_RotaryEmbedding.forward_call0001
/tmp/sglang_kernel_api_validation/qwen_qwen3_0_6b_level10_dumps/20260319_004821_182_pid919286_RotaryEmbedding.forward_call0001/inputs.pt
/tmp/sglang_kernel_api_validation/qwen_qwen3_0_6b_level10_dumps/20260319_004821_182_pid919286_RotaryEmbedding.forward_call0001/metadata.json
/tmp/sglang_kernel_api_validation/qwen_qwen3_0_6b_level10_dumps/20260319_004821_182_pid919286_RotaryEmbedding.forward_call0001/outputs.pt
```

Real `metadata.json` excerpt:

```json
{
  "function_name": "RotaryEmbedding.forward",
  "timestamp": "20260319_004821_182",
  "process_id": 919286,
  "execution_status": "completed",
  "input_tensor_keys": ["arg_0", "arg_1", "arg_2"],
  "output_tensor_keys": ["result_0", "result_1"]
}
```

## Step 2: Reproduce an LLM CUDA Crash

Create a temporary reproducer:

```bash
python3 - <<'PY'
from pathlib import Path
Path("/tmp/sglang_llm_crash.py").write_text(
    "import torch\\n"
    "import torch.nn.functional as F\\n"
    "from sglang.srt.utils.custom_op import register_custom_op\\n\\n"
    "def _fake_embedding(indices, table):\\n"
    "    return torch.empty((*indices.shape, table.shape[-1]), device=table.device, dtype=table.dtype)\\n\\n"
    "@register_custom_op(op_name='mock_llm_cuda_crash', fake_impl=_fake_embedding)\\n"
    "def mock_llm_cuda_crash(indices, table):\\n"
    "    out = F.embedding(indices, table)\\n"
    "    torch.cuda.synchronize()\\n"
    "    return out\\n\\n"
    "table = torch.randn(4, 8, device='cuda', dtype=torch.float16)\\n"
    "indices = torch.tensor([0, 7], device='cuda', dtype=torch.long)\\n"
    "mock_llm_cuda_crash(indices, table)\\n"
)
PY

SGLANG_KERNEL_API_LOGLEVEL=1 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_llm_level1.log \
python3 /tmp/sglang_llm_crash.py
```

What to expect:
- The script exits with a CUDA `device-side assert`
- The log still contains the last API boundary before the crash

Try the same example at level 3:

```bash
SGLANG_KERNEL_API_LOGLEVEL=3 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_llm_level3.log \
python3 /tmp/sglang_llm_crash.py
```

Now the log shows tensor metadata before the crash.

Try level 10:

```bash
SGLANG_KERNEL_API_LOGLEVEL=10 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_llm_level10.log \
SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_llm_level10_dumps \
python3 /tmp/sglang_llm_crash.py
```

Now you should see:
- A log entry for `sglang.custom_op.mock_llm_cuda_crash`
- A dump directory with `inputs.pt`
- `metadata.json` showing `execution_status: "exception"`
- No `outputs.pt`, because the kernel crashed before producing output

For real-model success-path level-10 dumps, it is often easier to temporarily disable CUDA graph and piecewise CUDA graph for the debug run.

## Step 3: Reproduce a Diffusion CUDA Crash

Create a temporary diffusion-side reproducer:

```bash
python3 - <<'PY'
from pathlib import Path
Path("/tmp/sglang_diffusion_crash.py").write_text(
    "import torch\\n"
    "import torch.nn.functional as F\\n"
    "from sglang.multimodal_gen.runtime.layers.utils import register_custom_op\\n\\n"
    "def _fake_embedding(positions, cache):\\n"
    "    return torch.empty((*positions.shape, cache.shape[-1]), device=cache.device, dtype=cache.dtype)\\n\\n"
    "@register_custom_op(op_name='mock_diffusion_cuda_crash', fake_impl=_fake_embedding)\\n"
    "def mock_diffusion_cuda_crash(positions, cache):\\n"
    "    out = F.embedding(positions, cache)\\n"
    "    torch.cuda.synchronize()\\n"
    "    return out\\n\\n"
    "cache = torch.randn(4, 64, device='cuda', dtype=torch.float16)\\n"
    "positions = torch.tensor([0, 9], device='cuda', dtype=torch.long)\\n"
    "mock_diffusion_cuda_crash(positions, cache)\\n"
)
PY

SGLANG_KERNEL_API_LOGLEVEL=1 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_diffusion_level1.log \
python3 /tmp/sglang_diffusion_crash.py
```

Try level 3:

```bash
SGLANG_KERNEL_API_LOGLEVEL=3 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_diffusion_level3.log \
python3 /tmp/sglang_diffusion_crash.py
```

Try level 10:

```bash
SGLANG_KERNEL_API_LOGLEVEL=10 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_diffusion_level10.log \
SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_diffusion_level10_dumps \
python3 /tmp/sglang_diffusion_crash.py
```

If your local environment has unrelated FlashInfer import issues, resolve them in the shell before running the example. The example itself does not set any `FLASHINFER_*` environment variable.

## Step 4: Multi-Process Debugging

When running with multiple GPUs or worker processes, use `%i` in the log path:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug_rank_%i.log

torchrun --nproc_per_node=4 my_script.py
```

This creates separate logs such as:
- `debug_rank_12345.log`
- `debug_rank_12346.log`
- `debug_rank_12347.log`
- `debug_rank_12348.log`

Real multi-process example from a 2-GPU `Qwen/Qwen2.5-0.5B-Instruct` run:

```text
/tmp/sglang_kernel_api_validation_multi/qwen_qwen2_5_0_5b_instruct_level3_950201.log
/tmp/sglang_kernel_api_validation_multi/qwen_qwen2_5_0_5b_instruct_level3_950349.log
/tmp/sglang_kernel_api_validation_multi/qwen_qwen2_5_0_5b_instruct_level3_950350.log
/tmp/sglang_kernel_api_validation_multi/qwen_qwen2_5_0_5b_instruct_level3_950351.log
```

You should usually do the same for level-10 dump directories:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=10
export SGLANG_KERNEL_API_LOGDEST=debug_rank_%i.log
export SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_kernel_api_dumps_%i
```

This avoids multiple ranks writing into the same dump directory tree.

## Step 5: Filter Level-10 Dumps

If level 10 is too noisy, restrict dumps to specific APIs:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=10
export SGLANG_KERNEL_API_LOGDEST=debug.log
export SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_kernel_api_dumps
export SGLANG_KERNEL_API_DUMP_INCLUDE='sglang.custom_op.*'
export SGLANG_KERNEL_API_DUMP_EXCLUDE='*.fake_impl'
```

`SGLANG_KERNEL_API_DUMP_INCLUDE` and `SGLANG_KERNEL_API_DUMP_EXCLUDE` use shell-style wildcard matching.

## Step 6: Common CUDA Errors and What to Check

### Illegal Memory Access or Device-Side Assert

**Typical errors**:
```
RuntimeError: CUDA error: an illegal memory access was encountered
torch.AcceleratorError: CUDA error: device-side assert triggered
```

Use:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
```

Check in the logs:
- ✅ Tensor shapes
- ✅ Tensor dtypes
- ✅ CUDA vs CPU device placement
- ✅ Tensor stride / contiguity
- ✅ Whether the failing call has inputs logged but no outputs logged

Typical shape-mismatch pattern:

```text
SGLang Kernel API Call: ...
arg[0]=Tensor(shape=(..., 128), ...)   # ✅ expected dimension
arg[1]=Tensor(shape=(..., 64), ...)    # ❌ mismatch
```

This often points to head-dim, hidden-dim, or cache-layout mismatch rather than a random CUDA failure.

### NaN or Inf

Use:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=5
```

Check:
- `min`
- `max`
- `mean`
- `nan_count`
- `inf_count`

Typical bad pattern:

```text
Tensor(
  ...
  min=-1234567.000000   # ❌ suspiciously large
  max=9876543.000000    # ❌ suspiciously large
  mean=nan              # ❌ bad
  nan_count=128         # ❌ found NaNs
  inf_count=0           # ✅ no Infs here
)
```

This usually means the bad values were already present before the crashing kernel.

### Out of Memory

Use:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
```

Check:
- Unexpectedly large tensor shapes
- Batch size
- Sequence length
- Frame count or image resolution in diffusion workloads

Also check whether a supposedly per-token or per-frame tensor accidentally became full-sequence or full-image sized.

Typical bad pattern:

```text
Tensor(
  shape=(1024, 8192, 128, 128)   # ❌ way too large
  ...
)
```

### Example: Spot a Shape Bug from the Log

Suppose the failing API log looks like this:

```text
[2026-03-19 00:47:30] SGLang Kernel API Call: RotaryEmbedding.forward
Positional input arguments:
  arg[0]=Tensor(shape=(1, 8), dtype=torch.int64, ...)
  arg[1]=Tensor(shape=(1, 8, 8, 256), dtype=torch.bfloat16, ...)    # ✅ query
  arg[2]=Tensor(shape=(1, 8, 4, 64), dtype=torch.bfloat16, ...)     # ❌ key head_dim mismatch
```

What this tells you:
- ✅ positions look reasonable
- ✅ query looks plausible
- ❌ key last dimension is inconsistent with the expected rotary/head dimension

That usually means the bug is in projection layout, head packing, or cache format rather than in the rotary kernel itself.

## Step 7: Combine with compute-sanitizer

For harder bugs, combine kernel API logging with CUDA memory checking:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug.log

compute-sanitizer --tool memcheck python3 /tmp/sglang_llm_crash.py
```

Use `debug.log` to see the exact inputs that reached the crashing API boundary.

Typical `compute-sanitizer` output:

```text
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x1234 in SomeKernel
=========     by thread (256,0,0) in block (10,0,0)
=========     Address 0x... is out of bounds
```

Use the sanitizer output to identify the failing kernel and use `debug.log` to identify the exact tensors that reached the API boundary right before it.

If you need more synchronous host-side error reporting, you can try `CUDA_LAUNCH_BLOCKING=1` as a separate follow-up experiment. It is not part of the default workflow because it changes execution timing and can hide concurrency-related behavior.

## Step 8: Combine with cuda-gdb

For crashes that need a stack trace instead of only memory diagnostics:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug.log

cuda-gdb --args python3 /tmp/sglang_llm_crash.py
```

Inside `cuda-gdb`:

```text
(cuda-gdb) run
(cuda-gdb) where
```

Then correlate the backtrace with `debug.log`.

## Step 9: Kernel-Level Debugging with printf()

When you own the CUDA kernel, `printf()` is still useful for narrowing down bad indices, bad launch geometry, or broken state propagation.

Basic pattern:

```cpp
__global__ void MyKernel(const float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("n=%d input0=%f\n", n, input[0]);
  }

  if (idx < n) {
    output[idx] = input[idx] * 2.0f;
  }
}
```

After launch, force the output to flush:

```python
my_kernel(...)
torch.cuda.synchronize()
```

For warp-specialized kernels, do not blindly print only on `threadIdx.x == 0`. Pick one representative thread per warp or per specialization group instead.

### Warp-Specialized Kernels: Choosing the Right Print Thread

Problem:
- `threadIdx.x == 0` only prints from the first warp in the block
- for warp-specialized kernels, that often misses the warp or group that is actually wrong

Better pattern:

```cpp
__global__ void WarpSpecializedKernel(...) {
  // Example: first lane of each warp
  if ((threadIdx.x % 32) == 0) {
    printf("warp=%d\n", threadIdx.x / 32);
  }
}
```

Or, if the kernel is organized in larger specialization groups, print once per group instead of once per block.

Common mistake:

```cpp
// Only warp 0 prints
if (threadIdx.x == 0) {
  printf("warp=%d\n", threadIdx.x / 32);
}
```

### Quick Reference

| Kernel Type | Print Condition | Notes |
|----------|----------|-------------|
| Simple kernel | `threadIdx.x == 0` | One thread per block is usually enough |
| Warp-specialized kernel | one representative lane per warp | e.g. `threadIdx.x % 32 == 0` |
| Group-specialized kernel | one representative lane per group | choose based on the kernel's scheduling layout |

### Other Kernel Debugging Tools

```cpp
assert(value >= 0.0f && "value must be non-negative");
static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be warp aligned");
```

## Environment Variables Reference

| Variable | Values | Description |
|----------|--------|-------------|
| `SGLANG_KERNEL_API_LOGLEVEL` | `0` | No logging (default) |
|  | `1` | Function names only |
|  | `3` | Inputs and outputs with metadata |
|  | `5` | Level 3 plus tensor statistics |
|  | `10` | Level 5 plus crash-safe tensor dumps |
| `SGLANG_KERNEL_API_LOGDEST` | `stdout` | Log to stdout |
|  | `stderr` | Log to stderr |
|  | `<path>` | Log to file |
|  | `log_%i.txt` | `%i` expands to process ID |
| `SGLANG_KERNEL_API_DUMP_DIR` | `<path>` | Directory for level-10 dumps |
| `SGLANG_KERNEL_API_DUMP_INCLUDE` | wildcard list | Only dump matching API names |
| `SGLANG_KERNEL_API_DUMP_EXCLUDE` | wildcard list | Skip matching API names |

## Best Practices

### 1. Start with Level 3

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
```

Level 3 is usually enough to catch wrong shapes, wrong dtypes, and wrong devices.

### 2. Use Level 5 for Numerical Issues

```bash
export SGLANG_KERNEL_API_LOGLEVEL=5
```

Use it when you suspect NaN or Inf values.

### 3. Use Level 10 for Crash Reproduction

```bash
export SGLANG_KERNEL_API_LOGLEVEL=10
```

This is the most useful mode when the process crashes before you can inspect live tensors.

If you need successful input/output dumps from a real model run, temporarily disable CUDA graph for that debug session.

When level 10 is too noisy, pair it with `SGLANG_KERNEL_API_DUMP_INCLUDE` / `SGLANG_KERNEL_API_DUMP_EXCLUDE` instead of dumping every covered API.

### 4. Log to File for Crashes

```bash
export SGLANG_KERNEL_API_LOGDEST=crash.log
```

File logs are safer than stdout when the process aborts.

### 5. Disable Logging in Production

```bash
unset SGLANG_KERNEL_API_LOGLEVEL
```

When disabled, the decorator returns the original callable and adds no runtime logging overhead.

## Troubleshooting

### No Logs Appear

Check:
1. `echo $SGLANG_KERNEL_API_LOGLEVEL`
2. `echo $SGLANG_KERNEL_API_LOGDEST`
3. Whether the failing path goes through a covered API boundary

### Too Much Output

Reduce the level:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
```

### Statistics Are Skipped During CUDA Graph Capture

If you see:
```text
statistics=[skipped: CUDA graph capture in progress]
```

That is expected. Level-5 statistics are intentionally skipped during CUDA graph capture to avoid synchronization side effects.

### Tensor Dumps Are Skipped During CUDA Graph Capture

If you see:
```text
Tensor dump skipped: CUDA graph capture in progress
```

That is also expected. Level-10 dumps require copying tensors to CPU, which is not allowed during CUDA graph capture.
