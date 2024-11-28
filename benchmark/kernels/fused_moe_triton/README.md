## Benchmark Kernels

This directory contains benchmarking tools for MoE (Mixture of Experts) kernels.

### Tuning Tool

- `tuning_fused_moe_triton.py`: A tool for tuning the `fused_moe_triton` kernel. Adapted from [vllm's benchmark_moe.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py), with added support for various model architectures.

Example usage:
```bash
# Tune Qwen2-57B with FP8 and TP=4
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model Qwen/Qwen2-57B-A14B-Instruct-FP8 \
    --tp-size 4 \
    --dtype fp8_w8a8 \
    --tune

# Tune Mixtral-8x7B with default settings
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tune
```

After tuning, a configuration file (e.g., `E=64,N=640,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8.json`) will be generated in the current directory. You can move this file to `sglang/srt/layers/fused_moe_triton/configs/` to use it in `sglang`.

### Performance Comparison Tool

- `benchmark_vllm_vs_sglang_fused_moe_triton.py`: A tool for comparing the performance of fused MoE kernels between vllm and sglang implementations. Supports various model architectures and data types.

Example usage:
```bash
# Compare with default settings (Mixtral model)
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py

# Compare with FP8 mode for Qwen2-57B
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py \
    --model Qwen/Qwen2-57B-A14B-Instruct-FP8 \
    --use-fp8

# Compare with custom TP size
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py \
    --tp-size 4
```

The benchmark results will be saved as plots and data files in the specified output directory (default: `./configs/benchmark_ops/vllm_sglang_fused_moe/`).
