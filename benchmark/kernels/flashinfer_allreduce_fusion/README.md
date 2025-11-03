# FlashInfer Fused AllReduce + RMSNorm Benchmark

This benchmark script is modified from the [original implementation](https://github.com/vllm-project/vllm/blob/237e1fb887c7f5a579420fa0295097f24b006594/benchmarks/kernels/benchmark_fused_collective.py) by the vLLM community. It aims to compare the performance differences between FlashInfer fused operators in SGLang (trtllm_allreduce_fusion: AllReduce + Residual Add + RMSNorm + optional quantization) and conventional implementations (standard `tensor_model_parallel_all_reduce` + separate RMSNorm/quantization). Specifically, this script tests the timing performance of two implementation paths: 1) Standard AllReduce and RMSNorm executed separately; 2) FlashInfer's fused operator combining AllReduce, Residual Add, RMSNorm, and optional quantization operations.

This benchmark script helps us tune the ipc workspace size of the `flashinfer_allreduce_residual_rmsnorm` operator in SGLang and prepare for applications with FP8/FP4 quantized fused operators.

Script path: `benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py`

## Feature Overview

- Compare average execution time (ms) and calculate speedup ratios for the following paths:
  - standard_allreduce_rmsnorm (Standard AllReduce + RMSNorm)
  - flashinfer_fused_allreduce_rmsnorm (Fused AllReduce + RMSNorm), including oneshot and twoshot modes
  - Optionally compare FP8/FP4 quantized fused paths with standard paths
- Use CUDA Graph capture and batch replay to reduce measurement noise
- Automatically select the faster "standard baseline" (native/compiled version) as the denominator for speedup calculation
- Optionally export results in Markdown format

## Runtime Environment and Prerequisites

- At least 2 GPUs, and launch multi-process distributed training using `torchrun` (NCCL backend)
- Properly install/compile sglang along with sgl-kernel and custom operators

## Quick Start (Command Examples)

The following examples use world_size=2. You can modify `--nproc_per_node` and parameters according to your machine:

- Regular paths only (no quantization):
```
torchrun --nproc_per_node=2 \
benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py \
--no-quant --hidden-dim 1024 --seq-lens 512 1024 2048 4096 --trials 100
```

- FP8 quantization paths only:
```
torchrun --nproc_per_node=2 \
benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py \
--quant-fp8 --hidden-dim 1024 --seq-lens 512 1024 2048 4096 --trials 100
```

- FP4 quantization paths only:
```
torchrun --nproc_per_node=2 \
benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py \
--quant-fp4 --hidden-dim 1024 --seq-lens 512 1024 2048 4096 --trials 100
```

- Larger hidden dimensions:
```
torchrun --nproc_per_node=2 \
benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py \
--no-quant  --hidden-dim 4096 --seq-lens 512 1024 2048 4096 --trials 100
```

## Parameter Description
- `--seq-lens`: List of sequence lengths to test (default: 128 512 1024 2048)
- `--hidden-dim`: Hidden dimension (default: 8192)
- `--dtypes`: Data type list, `float16|bfloat16|float32` (default: bfloat16)
- `--no-residual`: Only test "no residual" scenarios (default tests both "with/without residual")
- Mutually exclusive quantization options:
  - `--no-quant`: No quantization testing
  - `--quant-fp8`: Only FP8 quantization testing
  - `--quant-fp4`: Only FP4 quantization testing
  - `--quant-all`: Test all (default)
- FlashInfer related:
  - `--disable-oneshot`: Disable oneshot mode (default enables oneshot and tests twoshot simultaneously)
- Runtime configuration:
  - `--warmup`: Warmup count before graph capture and before graph replay (default 5)
  - `--trials`: Benchmark iteration count (default 20; internally each `graph.replay()` will batch replay multiple times)
  - `--output-file`: Save results as Markdown file (only rank0 takes effect)

## Output Example

Each configuration group prints a table showing average execution time and relative speedup ratios (baseline is the faster standard implementation). For example:
```
================================================================================
Results: seq_len=1024, hidden_dim=1024
dtype=torch.bfloat16, residual=yes, quant_mode=none
================================================================================
Operation                                          Time (ms)    Speedup
--------------------------------------------------------------------------------
standard_allreduce_rmsnorm                         0.024        0.98x
standard_allreduce_rmsnorm_native_compiled         0.023        baseline
flashinfer_fused_allreduce_rmsnorm_oneshot         0.011        2.19x
flashinfer_fused_allreduce_rmsnorm_twoshot         0.041        0.57x
```

If `--output-file` is specified, all configurations will be summarized in Markdown tables in that file.

## Important Notes and Recommendations

- Distributed: The script uses `torchrun` environment variables to initialize distributed training and binds tensors/communication groups to the current rank's corresponding device.
- World size: Requires `WORLD_SIZE > 1` to perform communication operator benchmarks. Otherwise, the script will error and prompt.
- FlashInfer:
  - If not installed or interfaces are missing, the script will only run standard paths and provide prompts in the logs.
  - The fused operator internally uses "oneshot"/"twoshot" two trigger methods; oneshot is enabled by default and twoshot is tested simultaneously.
- FP8/FP4:
  - FP8 uses sglang's FP8 tools and dtype, with underlying platform selection of `e4m3`/`e4m3fnuz` etc.
  - FP4 uses sgl-kernel's `scaled_fp4_quant`, requiring corresponding platform support.
- CUDA Graph:
  - Uses sglang's `graph_capture()` to prepare capture-ready state for communication, then uses `torch.cuda.graph` to capture kernels, reducing measurement jitter.
