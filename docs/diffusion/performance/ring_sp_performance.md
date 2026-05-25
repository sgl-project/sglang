# Ring SP Benchmark: Wan2.2-TI2V-5B (u1r2 vs Baseline)

This page reports Ring-SP performance for `Wan2.2-TI2V-5B-Diffusers` using:

- Parallel config: `sp=2, ulysses=1, ring=2` (short: `u1r2`)
- Baseline config: `sp=1, ulysses=1, ring=1` (short: `u1r1`)

## Benchmark Setup

- Model: `Wan2.2-TI2V-5B-Diffusers`
- GPU: `48G RTX40 series * 2`

## Online Serving

### Ring SP (`u1r2`)

```bash
sglang serve \
  --model-type diffusion \
  --model-path /model/HuggingFace/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --num-gpus 2 --sp-degree 2 --ulysses-degree 1 --ring-degree 2 \
  --port 8898
```

### Baseline (`u1r1`)

```bash
sglang serve \
  --model-type diffusion \
  --model-path /model/HuggingFace/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --num-gpus 1 --sp-degree 1 --ulysses-degree 1 --ring-degree 1 \
  --port 8898
```

## Benchmarks

### Benchmark Disclaimer

These benchmarks are provided for reference under one specific setup and command configuration. Actual performance may vary with model settings, runtime environment, and request patterns.

### Stage Time Breakdown

| Stage / Metric | `u1r2` (s) | `u1r1` baseline (s) | Speedup |
|---|---:|---:|---:|
| InputValidation | 0.1060 | 0.1029 | 0.97x |
| TextEncoding | 1.3965 | 2.2261 | 1.59x |
| LatentPreparation | 0.0002 | 0.0002 | 1.00x |
| TimestepPreparation | 0.0003 | 0.0004 | 1.33x |
| Denoising | 52.6358 | 71.6785 | 1.36x |
| Decoding | 7.6708 | 13.4314 | 1.75x |
| **Total** | **63.74** | **90.63** | **1.42x** |

### Memory Usage

| Memory Metric | `u1r2` (GB) | `u1r1` baseline (GB) | Delta |
|---|---:|---:|---:|
| Peak GPU Memory | 20.07 | 27.40 | -7.33 |
| Peak Allocated | 13.35 | 20.40 | -7.05 |
| Memory Overhead | 6.72 | 7.00 | -0.28 |
| Overhead Ratio | 33.5% | 25.6% | +7.9pp |

## Summary

- End-to-end latency improves from `90.63s` to `63.74s` (`1.42x`).
- Main gains come from `Denoising` (`1.36x`) and `Decoding` (`1.75x`).
- Absolute memory usage drops noticeably on Ring-SP (`Peak GPU Memory -7.33GB`, `Peak Allocated -7.05GB`).
- Overhead ratio rises (`+7.9pp`), so future tuning can focus on reducing communication/runtime overhead while preserving the latency gain.
