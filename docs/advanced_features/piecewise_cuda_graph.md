# Piecewise CUDA Graph

## Motivation

In LLM serving, the forward pass involves many small GPU kernels launched sequentially — attention projections, layer norms, MLPs, residual connections, etc. The overhead of launching each kernel individually can become a significant bottleneck.

Standard CUDA graph addresses this by capturing the entire forward pass as a single graph and replaying it. However, capturing the full forward pass in one graph creates high memory pressure, since the entire execution must use fixed-size buffers.

Piecewise CUDA graph takes a different approach: it splits the forward pass into per-layer pieces and captures each piece as an independent CUDA graph. This enables fine-grained memory management — pieces share a single memory pool, and intermediate outputs are released immediately via weak tensor references.

## How It Works

### Graph splitting

The forward pass is decomposed at layer boundaries using PyTorch FX's `split_module`. Each transformer layer (or attention/MoE block) becomes its own CUDA graph piece. A "stitching graph" orchestrates the execution of all pieces at runtime, preserving data flow dependencies.

### Three-phase execution

Each piece goes through three phases:

1. **Compile** — The piece is compiled using `torch.compile` with the selected compiler backend (`eager` or `inductor`).
2. **Warmup** — The compiled piece runs once to ensure CUDA kernels are loaded and ready for graph capture.
3. **Capture and replay** — A `torch.cuda.CUDAGraph` is created, the piece runs within the graph context to capture all kernel launches, and subsequent calls replay the captured graph directly.

### Memory optimization

- **Shared memory pool**: All pieces across all capture sizes share a single global memory pool, allocated once at initialization.
- **Large-to-small capture order**: Capture sizes are processed from largest to smallest. Larger graphs allocate memory in the pool first; smaller graphs reuse it with minimal additional allocation.
- **Weak tensor references**: Intermediate piece outputs are converted to weak references immediately after capture, freeing memory for reuse by subsequent pieces.
- **GC disabled after first piece**: Garbage collection is disabled during capture of non-first pieces to avoid fragmentation and slowdown from repeated collections.

### Shape scheduling

Piecewise CUDA graph pre-captures graphs for a set of token counts. At runtime, the actual token count is rounded up to the nearest captured size (via binary search), and the corresponding graph is replayed. If the token count exceeds the largest captured size, the runtime falls back to the normal (non-graph) forward path.

The default capture schedule is auto-generated with increasing granularity:

| Token range | Step size |
|-------------|-----------|
| 4 – 32      | 4         |
| 48 – 256    | 16        |
| 288 – 512   | 32        |
| 576 – 1024  | 64        |
| 1280 – 4096 | 256       |
| 4608+       | 512       |

Sizes are capped at `--piecewise-cuda-graph-max-tokens`.

## Quick Start

Enable piecewise CUDA graph with a single flag:

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-piecewise-cuda-graph
```

With explicit compiler and max token configuration:

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-compiler eager \
  --piecewise-cuda-graph-max-tokens 4096
```

Using the Engine API:

```python
from sglang import Engine

engine = Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    enable_piecewise_cuda_graph=True,
    piecewise_cuda_graph_compiler="eager",
)
```

### What to expect in logs

During startup, you will see capture progress messages:

```
Capture piecewise CUDA graph begin. avail mem=XX.XX GB
Capture cuda graph num tokens [4, 8, 12, ..., 4096]
Capture piecewise CUDA graph end. Time elapsed: XX.XX s. mem usage=XX.XX GB. avail mem=XX.XX GB.
```

## Configuration Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--enable-piecewise-cuda-graph` | bool flag (set to enable) | `False` | Enable piecewise CUDA graph for extend/prefill. Must be explicitly set. |
| `--piecewise-cuda-graph-tokens` | `List[int]` (space-separated on CLI) | auto-generated | Custom capture sizes (must be ascending). Example: `--piecewise-cuda-graph-tokens 128 256 512 1024` |
| `--piecewise-cuda-graph-compiler` | `eager` or `inductor` | `eager` | Compiler backend for graph pieces. |
| `--piecewise-cuda-graph-max-tokens` | int | dynamic | Maximum token count for capture. Default is `chunked_prefill_size` for non-MLA models, `2048` for MLA backend models. |

For the full list of server arguments, see [Server Arguments](server_arguments.md).

## Compatibility

### Works with

- **ViT CUDA graph**: Can be combined with `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` for multimodal models (see [CUDA Graph for Multi-Modal Encoder](cuda_graph_for_multi_modal_encoder.md)):

```bash
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-max-tokens 4096 \
  --piecewise-cuda-graph-compiler eager
```

### Model-specific behavior

When explicitly enabled, MLA backend models (DeepSeek V3/R1/V3.1, Qwen MLA variants) auto-configure:
- Max tokens defaults to `2048` (to avoid kernel dispatch differences compared to the original mode)
- DeepSeek attention method switches to MLA for prefill paths

### Disabled when

Piecewise CUDA graph is automatically disabled under these conditions (some paths emit log messages):

- **Draft workers** — disabled on draft workers
- **`torch.compile`** — has fundamental conflicts with piecewise CUDA graph
- **Pipeline parallelism (PP > 1)** — not yet supported
- **DeepEP or Mooncake MOE A2A backends** — compilation errors prevent usage
- **Non-standard GQA layers** — all layers must use standard Grouped Query Attention

## Constraints

- **Extend/prefill only** — piecewise CUDA graph applies to extend (prefill) operations, not decode.
- **Experimental feature** — the API and behavior may change in future releases.

## Code References

- Graph splitting and compilation backend: `python/sglang/srt/compilation/backend.py`
- Per-piece CUDA graph capture and replay: `python/sglang/srt/compilation/cuda_piecewise_backend.py`
- Warmup, capture orchestration, and runtime replay: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- Context state tracking: `python/sglang/srt/compilation/piecewise_context_manager.py`
- Compilation config: `python/sglang/srt/compilation/compilation_config.py`

## Troubleshooting

If CUDA graph capture fails during startup (e.g., out-of-memory errors), try:

1. Lower `--piecewise-cuda-graph-max-tokens` (e.g., `512`) to reduce the number and size of captured graphs.
2. Lower `--mem-fraction-static` (e.g., `0.8` or `0.7`) to reserve more memory for graph capture.
3. Disable the feature by removing `--enable-piecewise-cuda-graph`.
