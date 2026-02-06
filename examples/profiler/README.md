# Torch Dispatch Shape Profiler

A minimal torch dispatch-based shape logger for profiling tensor shapes and dtypes in SGLang workers.

## Features

- **Shape + Dtype Logging**: Captures input/output tensor shapes AND dtypes for each torch operation
- **Error Capture**: Logs operation shapes even when kernel errors occur (critical for debugging)
- **Deduplication**: For diffusion models that run same ops 50 times, only logs unique operations
- **Forward Pass Tracking**: Markers for prefill/decode phases in LLMs
- **Crash-Safe**: Immediate flush after every operation ensures data is saved before crashes

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SGLANG_PROFILE_SHAPES` | Enable shape profiling (`1` to enable) | `0` |
| `SGLANG_PROFILE_SHAPES_RANK` | Only profile on this TP rank | `0` |
| `SGLANG_PROFILE_SHAPES_FILE` | Output file path | `shapes.jsonl` |
| `SGLANG_PROFILE_SHAPES_SKIP_WARMUP` | Skip first N forward passes (warmup) | `0` |
| `SGLANG_PROFILE_SHAPES_LOG_N_PASSES` | Only log first N passes after warmup (0=unlimited) | `0` |
| `SGLANG_PROFILE_SHAPES_DEDUPE` | Deduplicate ops with same signature (for diffusion) | `0` (LLM), `1` (diffusion) |

## Usage Examples

### Example 1: LLM Model (DeepSeek-V3)

```bash
SGLANG_PROFILE_SHAPES=1 \
SGLANG_PROFILE_SHAPES_RANK=0 \
SGLANG_PROFILE_SHAPES_FILE=deepseek_shapes.jsonl \
SGLANG_PROFILE_SHAPES_SKIP_WARMUP=26 \
SGLANG_PROFILE_SHAPES_LOG_N_PASSES=2 \
python3 -m sglang.bench_one_batch_server \
  --model-path /data/DeepSeek-V3-0324/ \
  --tp 8 \
  --batch-size 1 \
  --input-len 8192 \
  --output-len 8 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --trust-remote-code
```

### Example 2: Debug Kernel Error (GLM-4.5)

```bash
SGLANG_PROFILE_SHAPES=1 \
SGLANG_PROFILE_SHAPES_RANK=0 \
SGLANG_PROFILE_SHAPES_FILE=glm4_shapes.jsonl \
SGLANG_PROFILE_SHAPES_SKIP_WARMUP=0 \
SGLANG_PROFILE_SHAPES_LOG_N_PASSES=2 \
python3 -m sglang.bench_one_batch_server \
  --model-path zai-org/GLM-4.5-Air-FP8 \
  --tp 1 \
  --batch-size 1 \
  --input-len 1024 \
  --output-len 512 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --trust-remote-code
```

### Example 3: Diffusion Model (with deduplication)

```bash
SGLANG_PROFILE_SHAPES=1 \
SGLANG_PROFILE_SHAPES_RANK=0 \
SGLANG_PROFILE_SHAPES_FILE=diffusion_shapes.jsonl \
SGLANG_PROFILE_SHAPES_DEDUPE=1 \
python your_diffusion_script.py
```

## Output Format

The profiler outputs JSONL format with one operation per line:

```json
{
  "call_id": 1,
  "forward_pass": 3,
  "operation": "aten.mm.default",
  "inputs": {
    "self": {"shape": [128, 256], "dtype": "torch.float16"},
    "mat2": {"shape": [256, 512], "dtype": "torch.float16"}
  },
  "outputs": {"shape": [128, 512], "dtype": "torch.float16"}
}
```

### Error Capture

When a kernel error occurs, the operation is logged with error info:

```json
{
  "call_id": 42,
  "forward_pass": 1,
  "operation": "aten.mm.default",
  "inputs": {...},
  "outputs": null,
  "error": {
    "error_type": "RuntimeError",
    "error_message": "CUDA error: invalid configuration argument"
  }
}
```

## Summary File

A summary file (`*_summary.json`) is generated with statistics:

```json
{
  "rank": 0,
  "total_operations": 1234,
  "forward_passes": 2,
  "unique_operations": 45,
  "operation_counts": {
    "aten.mm.default": 100,
    "aten.add.Tensor": 50
  },
  "dedupe_enabled": true,
  "dedupe_skipped_count": 49000,
  "unique_op_signatures": 1000
}
```

## Supported Workers

- **LLM**: `python/sglang/srt/managers/tp_worker.py`
- **Diffusion**: `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py` (deduplication enabled by default)

## Notes

- For **diffusion models**, deduplication is enabled by default (`SGLANG_PROFILE_SHAPES_DEDUPE=1`) since the same operations run ~50 times during inference
- For **LLM models**, deduplication is disabled by default since prefill and decode have different shapes
- Set `SGLANG_PROFILE_SHAPES_DEDUPE=0` to disable deduplication if needed
