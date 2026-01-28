# A01_B14: GitHub Issue 描述模板

## Issue 标题

```
Bug: Missing `accelerate` dependency in SGLang dev Docker image breaks Diffusion models
```

---

## Issue 正文（可直接复制使用）

```markdown
## Bug Description

The SGLang dev Docker image (`lmsysorg/sglang:dev`) is missing the `accelerate` library, 
which is **required** for Diffusion models to use `device_map=cuda`. This causes all 
Diffusion model runs to crash with a `NotImplementedError`.

## Error Message

```
Loading diffusers pipeline with dtype=torch.bfloat16, device_map=cuda
Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. 
Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster 
and less memory-intense model loading. You can do so with:
```
pip install accelerate
```

[01-28 04:18:09] Worker 0: Shutdown complete.
...
NotImplementedError: Using `device_map` requires the `accelerate` library. Please install it using: `pip install accelerate`.
[rank0]:[W128 04:18:09.073114819 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[01-28 04:18:10] Rank 0 scheduler is dead. Please check if there are relevant logs.
[01-28 04:18:11] Exit code: 1
...
EOFError
```

## Steps to Reproduce

1. Pull the latest dev image:
   ```bash
   docker pull lmsysorg/sglang:dev
   ```

2. Run a Diffusion model:
   ```bash
   docker run --gpus all \
     lmsysorg/sglang:dev \
     sglang generate \
     --model-path <any-diffusion-model> \
     --prompt "test" \
     --save-output
   ```

3. Observe the crash with `NotImplementedError`

## Expected Behavior

The Diffusion model should load successfully with `device_map=cuda` and run on GPU.

## Actual Behavior

The process crashes immediately with `NotImplementedError` because the `accelerate` library 
is missing from the Docker image.

## Root Cause Analysis

1. **Code explicitly uses `device_map=cuda`**:
   - In `diffusers_pipeline.py`, the code sets `device_map=cuda` when loading models
   - This is the correct approach for GPU loading

2. **But the Docker image doesn't have `accelerate`**:
   - `accelerate` is a **required** dependency for `device_map` functionality
   - It's a core Hugging Face library for memory management

3. **The contradiction**:
   - SGLang's code assumes `accelerate` is available
   - But the Docker image doesn't include it
   - This is a **dependency mismatch** between code and image

## Impact

- **Severity**: 🔴 **Critical** - All Diffusion models are completely broken
- **Affected users**: Anyone trying to use Diffusion models with the official Docker image
- **Workaround exists**: Yes (install `accelerate` manually), but this shouldn't be necessary

## Environment

- **Docker image**: `lmsysorg/sglang:dev` (latest as of 2025-01-28)
- **GPU**: RTX 4090
- **CUDA**: 12.9
- **OS**: Windows + Docker Desktop + WSL2
- **Python**: 3.12 (inside container)

## Workaround

Install `accelerate` inside the container:

```bash
# Option 1: Install in running container
docker exec -it <container_id> pip install accelerate

# Option 2: Install during docker run
docker run --gpus all \
  lmsysorg/sglang:dev \
  bash -c "pip install accelerate && sglang generate --model-path <model> --prompt 'test' --save-output"
```

## Suggested Fix

Add `accelerate` to the Docker image dependencies:

### Option 1: Update Dockerfile
```dockerfile
RUN uv pip install accelerate
```

### Option 2: Update pyproject.toml
```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "accelerate",  # Add this line
]
```

## Additional Context

This is part of a larger investigation into SGLang's Diffusion module stability. 
We've also identified issues with:
- File system handling in Windows + Docker environments (see #17671)
- Model download and validation logic

However, this `accelerate` dependency issue is the **most critical** because it 
completely prevents any Diffusion model from running, regardless of environment.

## Related Issues

- #17671 - Original issue about Diffusion models not working
- This issue is a **root cause** of the problems described in #17671

---

**Note**: This error message is a "textbook case" of a dependency mismatch. The code 
assumes a library exists, but the Docker image doesn't provide it. This should be 
a simple fix - just add `accelerate` to the dependencies.
```

---

## 简化版本（如果 Issue 太长）

```markdown
## Bug: Missing `accelerate` dependency breaks Diffusion models

The SGLang dev Docker image (`lmsysorg/sglang:dev`) is missing the `accelerate` library, 
which is required for Diffusion models to use `device_map=cuda`.

**Error**: `NotImplementedError: Using device_map requires the accelerate library`

**Reproduce**:
```bash
docker run --gpus all lmsysorg/sglang:dev \
  sglang generate --model-path <diffusion-model> --prompt "test" --save-output
```

**Fix**: Add `accelerate` to Docker image dependencies or `pyproject.toml`

**Workaround**: `pip install accelerate` inside container

Related to #17671
```

---

## 使用说明

1. **完整版本**: 适合详细的技术讨论，包含完整的错误分析和建议
2. **简化版本**: 适合快速报告，重点突出核心问题

**建议**: 先使用简化版本，如果维护者需要更多信息，再补充完整版本的内容。

---

**最后更新**: 2025年1月28日
