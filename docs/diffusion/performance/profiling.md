# Profiling Multimodal Generation

This guide covers profiling techniques for multimodal generation pipelines in SGLang.

## Benchmarking

Benchmarking provides an end-to-end view of overall performance, including latency, throughput, and scalability.

### Server Benchmarking

We run benchmarking against a running server to obtain user-side metrics.

Firstly, start a server on your machine with your target model:
```bash
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --backend sglang
```
For detailed documentation on `sglang.serve` please see [cli.md](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md#serve)

With a server fully started and serving models, we can then trigger `bench_serving` as:
```bash
python3 -m sglang.multimodal_gen.benchmarks.bench_serving --dataset vbench --num-prompts 10 --width 512 --height 512 --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --num-frames 16
```

**Parameters:**
- `--base-url`: Base URL of the server (e.g., http://myhost:12345). Overrides host/port if provided.
- `--host`: Server host. (default: `localhost`)
- `--port`: Server port. (default: `3000`)
- `--model`: Name of model to be profiled, must match the model served from the running server.
- `--dataset`: Dataset to use. (Supported values: `vbench`, `random`)
- `--task`: The task will be inferred from huggingface pipeline_tag. When huggingface pipeline_tag is not provided, --task will be used. (Supported values: `text-to-video`, `image-to-video`, `text-to-image`, `image-to-image`, `video-to-video`)
- `--dataset-path`: Path to local dataset file (optional).
- `--num-prompts`: Number of prompts to benchmark. (Default: 10)
- `--max-concurrency`: Maximum number of concurrent requests, default to `1`. This can be used to help simulate an environment where a higher level component is enforcing a maximum number of concurrent requests. While the --request-rate argument controls the rate at which requests are initiated, this argument will control how many are actually allowed to execute at a time. This means that when used in combination, the actual request rate may be lower than specified with --request-rate, if the server is not processing requests fast enough to keep up.
- `--request-rate`: Number of requests per second. If this is inf, then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.
- `--width`: Width of image/video. If not provided, will use the model's default setting.
- `--height`: Height of image/video. If not provided, will use the model's default setting.
- `--num-frames`: Number of frames (for video). If not provided, will use the model's default setting.
- `--fps`: Frames Per Second (FPS) (for video). If not provided, will use the model's default setting.
- `--output-file`: Output JSON file for metrics.
- `--disable-tqdm`: Disable progress bar.
- `--log-level`: Log level. (Supported values: `DEBUG`, `INFO`, `WARNING`, `ERROR`)


### Offline Benchmarking

We can also run offline benchmarking without starting a server, so that we can focus on model inference performance.

To do this, use the bench_offline_throughput.py instead as:
```bash
python3 -m sglang.multimodal_gen.benchmarks.bench_offline_throughput --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --backend sglang --num-prompts 1 --batch-size 1 --num-runs 1 --num-inference-steps 10 --height 512 --width 512
```

**Parameters:**
- All server arguments can also be specified for setting up offline model, for details please refer to [`serve` documentation](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md#serve)
- `--model-path`: The path of the model weights. This can be a local folder or a Hugging Face repo ID.
- `--num-inference-steps`: Number of denoising steps (Default: 20)
- `--guidance-scale`: Classifier-free guidance scale (Default: 7.5)
- `--seed`: Random seed (Default: 42)
- `--disable-safety-checker`: Disable NSFW detection. (Default: False)
- `--width`: Width of image/video. If not provided, will use the model's default setting.
- `--height`: Height of image/video. If not provided, will use the model's default setting.
- `--num-frames`: Number of frames (for video). If not provided, will use the model's default setting.
- `--fps`: Frames Per Second (FPS) (for video). If not provided, will use the model's default setting.
- `--dataset`: Dataset to use. (Supported values: `vbench`, `random`)
- `--dataset-path`: Path to user-provided dataset (prompts file or image directory)
- `--num-prompts`: Total number of prompts to benchmark. (Default: 10)
- `--batch-size`: Batch size per generation call (Default: 1)
- `--skip-warmup`: Skip warmup batch. (Default: False)
- `--profile`: Enable torch profiler (use env var SGLANG_TORCH_PROFILER_DIR)
- `--num-runs`: Number of benchmark runs (Default: 1)
- `--output-file`: Output JSON file for results (append mode)
- `--disable-tqdm`: Disable progress bar


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

By default, trace files are saved in the ./logs/ directory.

The exact output file path will be shown in the console output, for example:

```bash
[mm-dd hh:mm:ss] Saved profiler traces to: /sgl-workspace/sglang/logs/mocked_fake_id_for_offline_generate-5_steps-global-rank0.trace.json.gz
```

### View Traces

Load and visualize trace files at:
- https://ui.perfetto.dev/ (recommended)
- chrome://tracing (Chrome only)

For large trace files, reduce `--num-profiled-timesteps` or avoid using `--profile-all-stages`.


### `--perf-dump-path` (Stage/Step Timing Dump)

Besides profiler traces, you can also dump a lightweight JSON report that contains:
- stage-level timing breakdown for the full pipeline
- step-level timing breakdown for the denoising stage (per diffusion step)

This is useful to quickly identify which stage dominates end-to-end latency, and whether denoising steps have uniform runtimes (and if not, which step has an abnormal spike).

The dumped JSON contains a `denoise_steps_ms` field formatted as an array of objects, each with a `step` key (the step index) and a `duration_ms` key.

Example:

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "<PROMPT>" \
  --perf-dump-path perf.json
```

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

## FAQ

- If you are profiling `sglang generate` with Nsight Systems and find that the generated profiler file did not capture any CUDA kernels, you can resolve this issue by increasing the model's inference steps to extend the execution time.
