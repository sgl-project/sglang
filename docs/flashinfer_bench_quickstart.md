# FlashInfer-Bench Quick Start

Get automatic kernel optimization for SGLang in 3 simple steps.

## Install FlashInfer-Bench

```bash
pip install flashinfer-bench
```

## Collect Workloads

Run your server with tracing enabled:

```bash
FIB_ENABLE_TRACING=1 python -m sglang.launch_server --model-path meta-llama/Llama-3-8b
```

Use your application normally. Workloads are saved to `~/.cache/flashinfer_bench/dataset/`

## Deploy Optimizations

Enable automatic kernel substitution:

```bash
FIB_ENABLE_APPLY=1 python -m sglang.launch_server --model-path meta-llama/Llama-3-8b
```

That's it! SGLang now automatically uses optimized kernels.

## Expected Results

- 15-30% latency reduction for typical workloads
- No code changes required
- Works with any model

## Next Steps

See [full documentation](./flashinfer_bench_integration.md) for:
- Benchmarking collected workloads
- Custom kernel development
- Advanced configuration
- Troubleshooting

## One-Line Example

```bash
# Collect and optimize in one command
FIB_ENABLE_TRACING=1 FIB_ENABLE_APPLY=1 python -m sglang.launch_server --model-path meta-llama/Llama-3-8b
```
