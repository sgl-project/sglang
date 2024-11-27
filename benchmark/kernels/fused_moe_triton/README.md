## benchmark kernels

- `tuning_fused_moe_triton.py`: tuning the `fused_moe_triton` kernel. It's adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py , and add support for qwen2-57b tuning.

For example, to tune the `Qwen/Qwen2-57B-A14B-Instruct-FP8` model's `fused_moe_triton` fp8_w8a8 kernel with TP4, run:

```bash
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --model Qwen/Qwen2-57B-A14B-Instruct-FP8 --tp-size 4 --dtype fp8_w8a8 --tune
```

And you can get `E=64,N=640,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8.json` in current directory, then you can put it in `sglang/srt/layers/fused_moe_triton/configs/` and use it in `sglang`.

- `benchmark_qwen2_57b_vllm_vs_sglang_fused_moe_triton.py`: benchmark the `Qwen/Qwen2-57B-A14B-Instruct-FP8` model's `fused_moe_triton` fp8_w8a8 kernel with vllm and sglang.

