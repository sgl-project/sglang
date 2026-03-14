# UVM-Based Expert Weight Offloading

SGLang supports UVM (CUDA Unified Memory) based expert weight offloading for MoE models. This allows running large MoE models (e.g., DeepSeek-V3, GLM-5) on fewer GPUs by keeping only a subset of experts in GPU VRAM while offloading the rest to CPU DRAM.

## Overview

Expert weights are stored in CUDA Unified Memory (`cudaMallocManaged`):

- **Resident experts** are advised `PREFER_GPU` -- their pages stay in GPU VRAM and are accessed at full VRAM bandwidth.
- **Offloaded experts** are advised `PREFER_CPU + ACCESSED_BY_GPU` -- their pages live in CPU DRAM and the GPU reads them transparently via PCIe read-through, with no page fault overhead.

All computation stays on GPU. There is no CPU inference path, no ID remapping, no assembly buffer, and no LRU cache. UVM handles everything transparently, including full compatibility with CUDA graphs.

### Comparison with KTransformers

Now SGLang provides two approaches for expert offloading. They are **mutually exclusive** -- use one or the other.

| | UVM Expert Offloading | KTransformers |
|---|---|---|
| Flag | `--expert-offload-num-resident` | `--kt-weight-path` |
| Compute location | All on GPU | GPU (resident) + CPU (offloaded, via AMX/AVX) |
| Offloaded expert access | PCIe read-through (transparent) | CPU inference with quantized weights |
| CUDA graph support | Fully compatible | Requires special handling |

## Quick Start

Enable expert offloading by passing `--expert-offload-num-resident N`, where `N` is the number of experts to keep permanently on GPU per MoE layer:

```bash
python3 -m sglang.launch_server --model-path /models/GLM-5-FP8 \
  --tensor-parallel-size 8 \
  --expert-offload-num-resident 200
```

In this example, each MoE layer has 257 local experts. 200 are kept resident in GPU VRAM (accessed at full bandwidth), and the remaining 57 are offloaded to CPU DRAM (accessed via PCIe).

## Resident Expert Selection

The `--expert-offload-resident-selection` flag controls how the resident set is chosen.

### `first_n` (default)

The first N experts (IDs 0 through N-1) are kept resident. Simple and deterministic:

```bash
--expert-offload-num-resident 200 \
--expert-offload-resident-selection first_n
```

### `frequency`

Initially uses the same assignment as `first_n`. During early prefill passes, collects per-layer expert routing frequencies. After accumulating enough routed tokens (default 4096), recomputes the optimal resident set based on actual usage and re-advises the UVM pages -- promoting frequently-used experts to GPU and demoting rarely-used ones to CPU:

```bash
--expert-offload-num-resident 200 \
--expert-offload-resident-selection frequency
```

### `manual`

Explicitly specify which expert IDs to keep resident via a comma-separated list:

```bash
--expert-offload-num-resident 3 \
--expert-offload-resident-selection manual \
--expert-offload-resident-ids "0,5,10"
```

If fewer IDs are provided than `--expert-offload-num-resident`, the remaining slots are filled from the beginning of the expert ID range.

## Example: GLM-5-FP8 on 8x H20 (96 GB)

```bash
python3 -m sglang.launch_server --model-path /models/GLM-5-FP8 \
  --host 0.0.0.0 --port 8080 \
  --mem-fraction-static 0.85 --tensor-parallel-size 8 \
  --attention-backend flashinfer \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --expert-offload-num-resident 200 \
  --expert-offload-resident-selection frequency \
  --enable-flashinfer-allreduce-fusion
```

## How It Works

1. **Weight loading**: Expert weights are loaded onto GPU normally via the standard checkpoint loader.

2. **UVM migration**: After loading, `ExpertOffloadManager` replaces each expert-indexed parameter with a `cudaMallocManaged` tensor. The original GPU tensor is freed, reclaiming VRAM.

3. **Memory advice**: `cudaMemAdvise` is called per expert slice:
   - Resident experts: `PREFER_GPU + ACCESSED_BY_GPU` (pages pinned to VRAM)
   - Offloaded experts: `PREFER_CPU + ACCESSED_BY_GPU` (pages in CPU DRAM, GPU reads via PCIe)

4. **Prefetch**: `cudaMemPrefetchAsync` migrates resident pages to GPU and offloaded pages to CPU to enforce the initial placement.

5. **Inference**: The MoE kernel indexes the full managed tensor directly -- `layer.w13_weight[expert_id]` works for any expert ID. Resident experts are read at VRAM speed; offloaded experts are read at PCIe speed. No special code paths needed.

6. **CUDA graphs**: Fully compatible. Kernels are recorded (not executed) during capture, so page placement is irrelevant at capture time. During replay, each expert is accessed at whichever bandwidth tier its pages are currently in.

7. **Memory reporting**: Offloaded bytes are registered with SGLang's memory reporting so that `init_memory_pool` correctly accounts for VRAM that the UVM driver will free on demand.

## CLI Reference

| Argument | Description | Default |
|---|---|---|
| `--expert-offload-num-resident` | Number of experts kept resident on GPU per MoE layer. `-1` disables offloading. | `-1` |
| `--expert-offload-resident-selection` | Strategy for choosing resident experts: `first_n`, `frequency`, or `manual`. | `first_n` |
| `--expert-offload-resident-ids` | Comma-separated expert IDs for `manual` selection mode. | `None` |
