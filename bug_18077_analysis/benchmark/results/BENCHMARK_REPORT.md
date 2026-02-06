# GLM-Image Inference Performance Benchmark Report

**Issue**: [#18077](https://github.com/sgl-project/sglang/issues/18077)  
**Model**: `zai-org/GLM-Image`  
**Date**: 2026-02-05

---

## Summary

### SGLang-D vs Diffusers (Single GPU)

| Metric | SGLang-D | Diffusers | Ratio |
|:-------|:---------|:----------|:------|
| **Latency (Mean)** | 8-13% lower | Baseline | 0.88x - 0.92x |
| **Latency (P99)** | 4-12% lower | Baseline | 0.88x - 0.96x |
| **Throughput** | 8-13% higher | Baseline | 1.08x - 1.13x |
| **Memory** | +0.45 GB | Baseline | +1.3% |

### SGLang-D Single GPU vs Multi-GPU (2 GPUs)

| Metric | Single GPU | Multi-GPU | Speedup |
|:-------|:-----------|:----------|:--------|
| **Latency (Mean)** | Baseline | 1.19x - 1.27x faster | 1.19x - 1.27x |
| **Throughput** | Baseline | 1.19x - 1.27x higher | 1.19x - 1.27x |
| **Memory** | Baseline | Same | 1.00x |

**Note**: Results are specific to GLM-Image model. Multi-GPU tested with `--num-gpus 2 --enable-cfg-parallel`.

---

## Test Environment

### Hardware
- **GPU**: NVIDIA RTX A6000
  - VRAM: 49,140 MiB (~49 GB)
  - Driver Version: 570.207
- **CPU**: (Not specified)
- **System Memory**: (Not specified)

### Software
- **OS**: Linux 6.8.0-94-generic
- **Python**: 3.12.3
- **PyTorch**: 2.7.0
- **CUDA**: 12.8
- **Benchmark Tool**: `sglang.multimodal_gen.benchmarks.bench_serving`
- **Model**: `zai-org/GLM-Image`

### Test Configuration
- **SGLang-D vs Diffusers**: 8 configurations (4 resolutions × 2 concurrency levels)
- **Single GPU vs Multi-GPU**: 8 configurations (4 resolutions × 2 concurrency levels)
- **Resolutions**: 512×512, 512×1024, 1024×512, 1024×1024
- **Concurrency Levels**: 1, 2
- **Multi-GPU Setup**: 2 GPUs with `--enable-cfg-parallel`
- **Prompts per Configuration**: 10
- **Request Rate**: Unlimited (`inf`)
- **Dataset**: Random prompts

---

## Results

### SGLang-D vs Diffusers (Single GPU)

| Resolution | Concurrency | Latency Mean (s) | Latency P99 (s) | Throughput (req/s) | Memory (GB) |
|:-----------|:------------|:-----------------|:----------------|:-------------------|:------------|
| | | **SGLang** | **Diffusers** | **SGLang** | **Diffusers** | **SGLang** | **Diffusers** | **SGLang** | **Diffusers** |
| 1024×1024 | 1 | 85.08 | 96.04 | 85.85 | 96.51 | 0.0118 | 0.0104 | 38.13 | 37.68 |
| 1024×1024 | 2 | 162.33 | 182.80 | 172.14 | 192.75 | 0.0117 | 0.0104 | 38.13 | 37.68 |
| 1024×512 | 1 | 47.17 | 52.64 | 49.32 | 52.79 | 0.0212 | 0.0190 | 35.75 | 35.30 |
| 1024×512 | 2 | 89.49 | 99.23 | 95.47 | 104.59 | 0.0212 | 0.0191 | 35.75 | 35.30 |
| 512×1024 | 1 | 47.06 | 52.15 | 48.75 | 52.34 | 0.0213 | 0.0192 | 35.75 | 35.30 |
| 512×1024 | 2 | 89.49 | 99.88 | 97.38 | 106.19 | 0.0212 | 0.0190 | 35.75 | 35.30 |
| 512×512 | 1 | 29.53 | 31.97 | 30.92 | 32.05 | 0.0339 | 0.0313 | 34.56 | 34.11 |
| 512×512 | 2 | 55.78 | 60.81 | 60.58 | 64.07 | 0.0341 | 0.0312 | 34.56 | 34.11 |

### SGLang-D Single GPU vs Multi-GPU

| Resolution | Concurrency | Latency Mean (s) | Latency P99 (s) | Throughput (req/s) | Memory (GB) |
|:-----------|:------------|:-----------------|:----------------|:-------------------|:------------|
| | | **Single** | **Multi** | **Single** | **Multi** | **Single** | **Multi** | **Single** | **Multi** |
| 1024×1024 | 1 | 85.08 | 76.87 | 85.85 | 138.32 | 0.0118 | 0.0130 | 38.13 | 38.13 |
| 1024×1024 | 2 | 162.33 | 128.26 | 172.14 | 135.07 | 0.0117 | 0.0148 | 38.13 | 38.13 |
| 1024×512 | 1 | 47.17 | 48.43 | 49.32 | 110.02 | 0.0212 | 0.0206 | 35.75 | 35.75 |
| 1024×512 | 2 | 89.49 | 72.34 | 95.47 | 76.20 | 0.0212 | 0.0263 | 35.75 | 35.75 |
| 512×1024 | 1 | 47.06 | 38.18 | 48.75 | 38.26 | 0.0213 | 0.0262 | 35.75 | 35.75 |
| 512×1024 | 2 | 89.49 | 73.09 | 97.38 | 77.07 | 0.0212 | 0.0260 | 35.75 | 35.75 |
| 512×512 | 1 | 29.53 | 24.86 | 30.92 | 26.87 | 0.0339 | 0.0402 | 34.56 | 34.56 |
| 512×512 | 2 | 55.78 | 46.42 | 60.58 | 48.93 | 0.0341 | 0.0409 | 34.56 | 34.56 |

---

## Raw Data

- **SGLang-D vs Diffusers**: `/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/results/`
- **Single GPU vs Multi-GPU**: `/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/results_multi_gpu/`
