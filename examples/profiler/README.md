# SGLang Shape Profiler

Standalone debug tool to profile tensor shapes during model inference.

## Important Limitations

⚠️ **With TP > 1 (multi-GPU), this tool will capture 0 operations.** This is expected behavior because tensor operations happen in worker processes that PyTorch dispatch mode cannot intercept from the main process.

**For TP > 1 profiling, use:**
- NVIDIA Nsight Systems (`nsys profile`)
- AMD ROCm Profiler (`rocprof`)
- PyTorch Profiler
- Native framework profilers

**This tool only works properly with TP=1** (single GPU) where all operations occur in the main process.

## Usage

### Single GPU (Works - Captures All Operations)

```bash
# This will successfully capture tensor shapes
python profile_shapes.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tp-size 1 \
  --num-prompts 3 \
  --max-tokens 50
```

### Multi-GPU (Will Capture 0 Operations - Expected)

```bash
# This will run successfully but capture 0 operations (expected behavior)
python profile_shapes.py \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --tp-size 8 \
  --num-prompts 3 \
  --max-tokens 50
```

Takes the same parameters as `launch_server`.

## Files

- `profile_shapes.py` - Main profiler tool
- `torch_shape_logger.py` - Shape logging implementation  
- `analyze_shapes.py` - Analysis utilities

## Technical Note

This is a **standalone debug tool** that uses PyTorch's dispatch mode to intercept tensor operations. It doesn't modify SGLang's core code.

**Why TP > 1 doesn't work:**
- With tensor parallelism, model layers are distributed across multiple GPU processes
- PyTorch dispatch mode only intercepts operations in the current Python process
- Worker processes execute independently and their operations aren't visible to the main process
- The profiler can only see high-level coordination logic, not the actual tensor computations

**Best practice:** Use TP=1 with a smaller model to understand operation patterns, then extrapolate to multi-GPU setups.
