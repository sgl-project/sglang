# Add Diffusion ModelOpt FP8 Support and Validation Workflow

## Summary

This PR adds diffusion-side support for loading NVIDIA ModelOpt FP8 checkpoints in SGLang, plus a conversion tool for ModelOpt diffusers FP8 exports and a reusable accuracy-validation workflow for quantized diffusion models.

The core idea is:
- use ModelOpt's official diffusers PTQ flow as the source of truth for quantization metadata
- convert the exported FP8 diffusers checkpoint into the SGLang-native diffusion FP8 checkpoint layout
- load the converted checkpoint through the diffusion runtime's native FP8 linear path
- validate both speed and quality against the BF16 reference with fixed prompts, seeds, and profiler traces

## What This PR Changes

### Runtime / Loader

- Register diffusion-side `modelopt` / `modelopt_fp8` quantization methods
- Resolve flat ModelOpt configs such as:
  - `quant_method=modelopt`
  - `quant_algo=FP8`
  - `quant_algo=NVFP4`
- Add diffusion ModelOpt FP8 linear handling in the runtime
- Auto-disable `dit_cpu_offload` and `dit_layerwise_offload` when loading ModelOpt FP8 checkpoints, since both break the FP8 CUTLASS weight-layout assumptions
- Keep FLUX.2 packed-QKV handling constrained to checkpoints that actually use the packed NVFP4 layout

### Tools / Validation

- Add `convert_modelopt_fp8_checkpoint.py`
  - materializes `weight_scale` / `input_scale` from `backbone.pt`
  - converts eligible weights to `float8_e4m3fn`
  - preserves ModelOpt `ignore` layers as BF16
- Add `compare_diffusion_trajectory_similarity.py`
  - runs BF16 reference and quantized candidate with identical sampling args
  - captures denoising trajectories via `return_trajectory_latents=True`
  - reports per-step cosine / MAE / RMSE and final frame PSNR / MAE

### Tests

- Unit tests for diffusion ModelOpt FP8 quant-config resolution and offload guards
- Unit tests for ModelOpt FP8 conversion
- Unit tests for the new trajectory/frame similarity helpers

## Supported Scope

### Safe claims

- FLUX.2 end-to-end ModelOpt FP8 support in SGLang
- WAN2.2 A14B primary-transformer FP8 override recipe with exact-nightly benchmark parity on H100

### Important non-claims

- This PR does not claim LTX-2 ModelOpt FP8 support
- This PR does not claim WAN2.2 exact-nightly dual-transformer FP8 benchmarking
- WAN2.2 exact-nightly benchmark numbers below are for the currently validated benchmarked path:
  - quantized `transformer`
  - BF16 `transformer_2`

## Implementation Notes

### ModelOpt -> SGLang FP8 chain

```text
BF16 diffusers transformer
    -> ModelOpt PTQ (official quantize.py)
    -> diffusers HF export + backbone.pt
    -> SGLang converter materializes FP8 weights + scales
    -> SGLang diffusion runtime detects modelopt_fp8
    -> static FP8 activation quant
    -> FP8 linear GEMM on Hopper CUTLASS path
```

Important details:
- quantization semantics are static per-tensor FP8 E4M3 for both weights and activations
- SGLang repeats the scalar scales into kernel-friendly shapes at runtime, but this does not change the quantization semantics into true per-channel quantization
- ModelOpt `ignore` layers must remain BF16; converting them to FP8 causes broken outputs

## Benchmark Results

### FLUX.2, 2x H100 TP2, same BF16 / FP8 command shape

| Variant | Total | Text | Denoising | Decoding | Total delta | Denoise delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | `13.75 s` | `0.51 s` | `12.84 s` | `0.01 s` | - | - |
| FP8 | `9.84 s` | `0.51 s` | `8.93 s` | `0.01 s` | `-28.4%` | `-30.5%` |

5-step profiler rerun:

| Variant | Total | Text | Denoising | Decoding | Total delta | Denoise delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 profile | `27.67 s` | `0.51 s` | `26.72 s` | `0.01 s` | - | - |
| FP8 profile | `21.29 s` | `0.51 s` | `20.34 s` | `0.01 s` | `-23.1%` | `-23.9%` |

### WAN2.2 A14B, 4x H100 exact-nightly payload, no torch.compile

Both BF16 and FP8 were pinned to:
- same `720p`, `81` frames, `40` steps, prompt, seed, and GPU topology
- same `--dit-cpu-offload false`
- same `--dit-layerwise-offload false`
- same allocator environment:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

| Variant | Total | Text | Denoising | Decoding | Total delta | Denoise delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | `212.19 s` | `1.00 s` | `204.09 s` | `6.91 s` | - | - |
| FP8 | `204.38 s` | `1.00 s` | `196.28 s` | `6.92 s` | `-3.68%` | `-3.83%` |

5-step profiler rerun:

| Variant | Total | Text | Denoising | Decoding | Total delta | Denoise delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 profile | `227.77 s` | `1.00 s` | `219.62 s` | `6.96 s` | - | - |
| FP8 profile | `223.82 s` | `1.00 s` | `215.69 s` | `7.02 s` | `-1.74%` | `-1.79%` |

## Accuracy Validation

### Final-output checks

Validated final-output comparisons already collected:
- FLUX.2 single-image check:
  - `MAE ~= 1.90`
  - `PSNR ~= 39.06 dB`
- WAN2.2 repaired-output sanity check after honoring ModelOpt `ignore` layers:
  - frame 0: `MAE ~= 2.55`, `PSNR ~= 36.58 dB`
  - mid frame: `MAE ~= 2.72`, `PSNR ~= 36.15 dB`

### Trajectory-latent checks

This PR also adds a reusable trajectory comparison tool:
- `python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py`

It is intended for reduced deterministic smoke configs, not exact-nightly video payloads.

The tool is validated by remote unit tests and H100 smoke runs.
The produced JSON summaries live under:
- `/tmp/modelopt_accuracy/flux2_similarity.json`
- `/tmp/modelopt_accuracy/wan22_similarity.json`

Observed H100 smoke numbers:

| Model | Reduced config | Selected step | Latent cosine | Latent MAE | Frame-0 PSNR | Frame-0 MAE | All-frame PSNR | All-frame MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLUX.2 | `512x512`, `8` steps | `7` | `0.9971` | `0.0405` | `34.67 dB` | `2.43` | `34.67 dB` | `2.43` |
| WAN2.2 | `704x512`, `21` frames, `2` steps | `1` | `0.9755` | `0.2310` | `21.28 dB` | `18.05` | `21.41 dB` | `17.77` |

## Artifact Inventory

### FLUX.2

- Images:
  - `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/flux2_bf16.png`
  - `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/flux2_fp8.png`
- Perf:
  - `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/bf16_perf.json`
  - `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/fp8_perf.json`
- Profile traces:
  - `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/profile/bf16/trace.json.gz`
  - `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/profile/fp8/trace.json.gz`

### WAN2.2

- Full-run videos:
  - `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/bf16_nocompile/wan22_bf16_nocompile.mp4`
  - `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/fp8_nocompile/wan22_fp8_nocompile.mp4`
- Perf:
  - `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/bf16_nocompile/perf.json`
  - `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/fp8_nocompile/perf.json`
- Profile traces:
  - `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/profile/bf16/trace.json.gz`
  - `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/profile/fp8/trace.json.gz`
- Profile videos:
  - `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/profile/bf16/wan22_bf16_profile.mp4`
  - `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/profile/fp8/wan22_fp8_profile.mp4`

## Tests

Remote H100 validation:
- `PYTHONPATH=python /tmp/bench_venv/bin/python3 -m pytest -q python/sglang/multimodal_gen/test/unit/test_compare_diffusion_trajectory_similarity.py`
- `PYTHONPATH=python /tmp/bench_venv/bin/python3 -m pytest -q python/sglang/multimodal_gen/test/unit/test_compare_diffusion_trajectory_similarity.py python/sglang/multimodal_gen/test/unit/test_convert_modelopt_fp8_checkpoint.py python/sglang/multimodal_gen/test/unit/test_diffusion_modelopt_quant.py`

Current observed result:
- `14 passed` for the combined remote quant/accuracy unit-test set

## Known Limitations

- Diffusion ModelOpt FP8 currently requires `dit_cpu_offload=false` and `dit_layerwise_offload=false`
- The benchmarked WAN2.2 nightly result currently quantizes the primary `transformer`; `transformer_2` remains BF16 in that exact benchmarked path
- LTX-2 ModelOpt FP8 still needs additional upstream compatibility work
