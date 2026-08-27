# Ascend NPU Ring-SP Performance (Wan2.1-T2V-1.3B)

This page reports Ring-SP performance on Ascend NPU with `torch_npu==2.10.0`.

- Baseline config: `ulysses=1, ring=1` (short: `u1r1`)
- Ring-SP config: `ulysses=1, ring=2` (short: `u1r2`)

## Benchmark Setup

- Model: `Wan2.1-T2V-1.3B-Diffusers`
- Prompt: `"a cat is playing piano"`
- Framework command: `sglang generate`
- Runtime: `torch_npu==2.10.0`

## Generate Commands

### Baseline (`u1r1`)

```bash
sglang generate --model-path /nas/disk1/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "a cat is playing piano" --num-gpus 1 --ring-degree 1 \
    --save-output
```

### Ring-SP (`u1r2`)

```bash
sglang generate --model-path /nas/disk1/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "a cat is playing piano" --num-gpus 2 --ring-degree 2 \
    --save-output
```

## Benchmarks

Benchmark Disclaimer

These numbers are from one fixed setup and one prompt case. Actual performance may vary by model settings, environment, and workload.

### Stage Time Breakdown

| Stage / Metric | `u1r2` (s) | `u1r1` baseline (s) | Speedup |
|---|---:|---:|---:|
| InputValidation | 0.0003 | 0.0002 | 0.67x |
| TextEncoding | 3.5936 | 3.5820 | 1.00x |
| LatentPreparation | 0.0007 | 0.0055 | 7.86x |
| TimestepPreparation | 0.0008 | 0.0007 | 0.88x |
| Denoising | 121.2788 | 239.2580 | 1.97x |
| Decoding | 13.8685 | 16.4969 | 1.19x |
| **Total (Pixel data generated)** | **141.86** | **266.50** | **1.88x** |

## Summary

- With `torch_npu==2.10.0`, Ring-SP (`u1r2`) runs successfully on NPU for this case.
- End-to-end generation time improves from `266.50s` to `141.86s` (`1.88x`).
- The main gain comes from `DenoisingStage` (`1.97x`), while decoding also improves (`1.19x`).
