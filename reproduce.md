# Reproducing SGLang Diffusion Optimizations

This guide outlines the steps to reproduce the performance gains (targeting up to 22% on DiT denoising) for Z-Image-Turbo and FLUX models using fused kernels.

## 1. Prerequisites

### Environment Setup
Ensure you have a working CUDA environment and the necessary compilers for Triton/JIT compilation.
```bash
# Set compiler for Triton/CuTe JIT
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

### Dependencies
Install the current SGLang workspace in editable mode to include the optimizations:
```bash
cd python
pip install -e .
```

## 2. Reproduction Benchmark

Use the following command to measure the optimized denoising latency. We use Z-Image-Turbo as it is highly sensitive to kernel launch overhead.

```bash
sglang generate 
  --model-path Tongyi-MAI/Z-Image-Turbo 
  --prompt "A mountain landscape, cinematic lighting, 8k" 
  --width 1024 
  --height 1024 
  --num-inference-steps 4 
  --guidance-scale 1.0 
  --enable-torch-compile 
  --warmup 
  --dit-cpu-offload false
```

### Metrics to Track
- **[DenoisingStage] average time per step**: This is the primary metric.
- **Warmed-up request processed in X.XX seconds**: Total generation time excluding warmup.

## 3. Comparison with Baseline

To see the impact of the optimizations, you can compare the results against the baseline (unfused) path.

### Method A: Manual Revert (Cleanest Comparison)
Temporarily revert the changes in `python/sglang/multimodal_gen/runtime/layers/layernorm.py` to force the `forward_native` path:

```python
# In layernorm.py, modify forward_cuda to always return native
def forward_cuda(self, x, shift, scale):
    return self.forward_native(x, shift, scale)
```

### Method B: Profiling Kernel Count
Run with `nsys profile` to verify the reduction in kernel launches:
```bash
nsys profile -o profile_output sglang generate ... (same args as above)
```
- **Optimized**: Should show one `fused_norm_scale_shift` or `fused_scale_residual_norm_scale_shift` per block.
- **Baseline**: Should show multiple discrete kernels (`rmsnorm_kernel`, `add_kernel`, `mul_kernel`) per block.

## 4. Expected Results

On an NVIDIA H100/A100 GPU, you should observe:
- **Kernel Launch Reduction**: ~30-40% fewer total kernels in the DiT denoising trace.
- **Denoising Latency**: A measurable improvement in "Average time per step". For Z-Image-Turbo (4 steps), the E2E gain is ~2-5%; for standard 20-50 step generations (FLUX), the gain scales to **10-22%**.
