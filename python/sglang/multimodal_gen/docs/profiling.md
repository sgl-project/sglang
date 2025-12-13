# Profiling Multimodal Generation

This guide covers profiling techniques for multimodal generation pipelines in SGLang.

## PyTorch Profiler

PyTorch Profiler provides detailed kernel execution time, call stack, and GPU utilization metrics.

### Denoising Stage Profiling

Profile the denoising stage with sampled timesteps (default: 5 steps after 1 warmup step):

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile
```

**Parameters:**
- `--profile`: Enable profiling for the denoising stage
- `--num-profiled-timesteps N`: Number of timesteps to profile after warmup (default: 5)
  - Smaller values reduce trace file size
  - Example: `--num-profiled-timesteps 10` profiles 10 steps after 1 warmup step

### Full Pipeline Profiling

Profile all pipeline stages (text encoding, denoising, VAE decoding, etc.):

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile \
  --profile-all-stages
```

**Parameters:**
- `--profile-all-stages`: Used with `--profile`, profile all pipeline stages instead of just denoising

### Output Location

By default, trace files are saved in the ./logs/ directory. The exact output file path will be shown in the console output, for example:

```bash
[mm-dd hh:mm:ss] Saving profiler traces to: /sgl-workspace/sglang/logs/mocked_fake_id_for_offline_generate-5_steps-global-rank0.trace.json.gz
```

### View Traces

Load and visualize trace files at:
- https://ui.perfetto.dev/ (recommended)
- chrome://tracing (Chrome only)

For large trace files, reduce `--num-profiled-timesteps` or avoid using `--profile-all-stages`.

## Nsight Systems

Nsight Systems provides low-level CUDA profiling with kernel details, register usage, and memory access patterns.

### Installation

See the [SGLang profiling guide](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md#profile-with-nsight) for installation instructions.

### Basic Profiling

Profile the entire pipeline execution:

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o QwenImage \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

### Targeted Stage Profiling

Use `--delay` and `--duration` to capture specific stages and reduce file size:

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --delay 10 \
  --duration 30 \
  -o QwenImage_denoising \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

**Parameters:**
- `--delay N`: Wait N seconds before starting capture (skip initialization overhead)
- `--duration N`: Capture for N seconds (focus on specific stages)
- `--force-overwrite`: Overwrite existing output files

## Notes

- **Reduce trace size**: Use `--num-profiled-timesteps` with smaller values or `--delay`/`--duration` with Nsight Systems
- **Stage-specific analysis**: Use `--profile` alone for denoising stage, add `--profile-all-stages` for full pipeline
- **Multiple runs**: Profile with different prompts and resolutions to identify bottlenecks across workloads
